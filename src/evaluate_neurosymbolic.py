"""Evaluation and rule extraction for the Neuro-Symbolic branch across all 4 configurations."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
from scipy.stats import spearmanr

from . import config
from .data_loader import load_dataset
from .model import ModelAdapter
from .xai.neurosymbolic import fit_surrogate, rule_path_attribution

EXPERIMENTS = {
    "A": ("cicids", "cicids"),
    "B": ("unsw", "unsw"),
    "C": ("cicids", "unsw"),
    "D": ("unsw", "cicids"),
}

LOGGER = logging.getLogger(__name__)


def extract_rules(surrogate: DecisionTreeClassifier, feature_names: list[str]) -> str:
    """Export the full IF-THEN rule set (all levels of the tree, nothing truncated)."""
    return export_text(surrogate, feature_names=feature_names, max_depth=surrogate.get_depth())


def black_box_threshold(model_path: Path) -> float:
    """Validation-tuned decision threshold saved next to the model (falls back to 0.5)."""
    sidecar = model_path.with_suffix(".json")
    if sidecar.exists():
        return float(json.loads(sidecar.read_text(encoding="utf-8")).get("val_threshold", 0.5))
    return 0.5


def explained_indices(experiment: str, n_rows: int) -> np.ndarray:
    """Same class-balanced test rows that SHAP/LIME explained, so the comparison is like-for-like."""
    path = config.RESULTS_DIR / "xai" / f"instances_{experiment}.json"
    if path.exists():
        return np.asarray(json.loads(path.read_text(encoding="utf-8"))["indices"], dtype=int)
    LOGGER.warning("%s missing; falling back to the first 1000 test rows", path)
    return np.arange(min(1000, n_rows))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    records_perf = []
    rules_text = []
    rule_attributions = []

    # 1. Train surrogates on Source datasets (CICIDS and UNSW) for best neural model mlp_a
    surrogates = {}
    black_boxes = {}
    for source in ["cicids", "unsw"]:
        X_train, y_train = load_dataset(source, "train")
        model_path = config.MODELS_DIR / source / "mlp_a" / "seed42.keras"
        if not model_path.exists():
            LOGGER.warning("Missing %s, skipping", model_path)
            continue
        adapter = ModelAdapter.load(model_path, name="mlp_a")
        threshold = black_box_threshold(model_path)
        preds = adapter.predict(X_train.to_numpy(float), threshold=threshold)

        # Fit surrogate tree on the black box's own decisions (not the true labels)
        surrogate = fit_surrogate(X_train.to_numpy(float), preds, max_depth=5, seed=42)
        surrogates[source] = surrogate
        black_boxes[source] = (adapter, threshold, float((surrogate.predict(X_train.to_numpy(float)) == preds).mean()))

        # Document human-readable rule trace
        rules_text.append(f"==================== NEURO-SYMBOLIC RULES (Source: {source.upper()}) ====================\n")
        rules_text.append(f"# Surrogate of mlp_a seed42 at threshold {threshold:.2f}; train fidelity {black_boxes[source][2]:.4f}\n")
        rules_text.append("# Thresholds are standardized (z-score) values; raw units cannot be recovered (scaler not preserved).\n")
        rules_text.append(extract_rules(surrogate, config.FEATURES))
        rules_text.append("\n\n")

    # 2. Evaluate across the 4 configurations (Exp A, B, C, D)
    for exp, (source, target) in EXPERIMENTS.items():
        if source not in surrogates:
            continue
        surrogate = surrogates[source]
        X_test, y_test = load_dataset(target, "test")
        values = np.asarray(X_test, dtype=float)
        labels = np.asarray(y_test, dtype=int)

        # Predictions
        probs = surrogate.predict_proba(values)[:, 1]
        preds = (probs >= 0.5).astype(int)

        # Fidelity: how often the surrogate agrees with the MLP it is meant to explain
        adapter, bb_threshold, train_fidelity = black_boxes[source]
        bb_preds = adapter.predict(values, threshold=bb_threshold)
        test_fidelity = float((preds == bb_preds).mean())
        attack_mask = bb_preds == 1
        attack_fidelity = float((preds[attack_mask] == 1).mean()) if attack_mask.any() else float("nan")

        auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
        cm = confusion_matrix(labels, preds)
        fpr = cm[0, 1] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0.0

        records_perf.append({
            "experiment": exp,
            "source": source,
            "target": target,
            "model": "neuro_symbolic_surrogate",
            "roc_auc": round(auc, 4),
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1_score": round(f1, 4),
            "fpr": round(fpr, 4),
            "fidelity_train": round(train_fidelity, 4),
            "fidelity_test": round(test_fidelity, 4),
            "fidelity_on_mlp_attacks": round(attack_fidelity, 4),
            "mlp_attack_rate_test": round(float(attack_mask.mean()), 4),
            "n_surrogate_leaves": int(surrogate.get_n_leaves()),
        })

        # 3. Rule path attributions for the configuration
        path_attrs = rule_path_attribution(surrogate, values[explained_indices(exp, len(values))])
        mean_attrs = np.mean(path_attrs, axis=0)
        for feat_idx, feat in enumerate(config.FEATURES):
            rule_attributions.append({
                "experiment": exp,
                "model": "neuro_symbolic",
                "feature": feat,
                "importance": float(mean_attrs[feat_idx]),
            })

    # Save outputs
    res_dir = config.RESULTS_DIR
    res_dir.mkdir(exist_ok=True)

    perf_df = pd.DataFrame(records_perf)
    perf_df.to_csv(res_dir / "neurosymbolic_metrics.csv", index=False)
    LOGGER.info("Saved %s", res_dir / "neurosymbolic_metrics.csv")

    with open(res_dir / "neurosymbolic_rules.txt", "w", encoding="utf-8") as f:
        f.writelines(rules_text)
    LOGGER.info("Saved %s", res_dir / "neurosymbolic_rules.txt")

    attr_df = pd.DataFrame(rule_attributions)
    attr_df.to_csv(res_dir / "neurosymbolic_attributions.csv", index=False)
    LOGGER.info("Saved %s", res_dir / "neurosymbolic_attributions.csv")

    # 4. Compare Neuro-symbolic explanations vs. Post-hoc SHAP & LIME
    summary_path = res_dir / "xai_summary.csv"
    if summary_path.exists():
        xai_summary = pd.read_csv(summary_path)
        comp_records = []
        for exp in ["A", "B", "C", "D"]:
            sym_exp = attr_df[attr_df["experiment"] == exp]
            if sym_exp.empty:
                continue
            for method in ["shap", "lime"]:
                post_exp = xai_summary[(xai_summary["experiment"] == exp) & (xai_summary["model"] == "mlp_a") & (xai_summary["method"] == method)]
                if post_exp.empty:
                    continue
                m = pd.merge(sym_exp, post_exp, on="feature")
                rho, pval = spearmanr(m["importance"], m["mean_abs_attribution"])
                top3_sym = sym_exp.sort_values(by="importance", ascending=False)["feature"].head(3).tolist()
                top3_post = post_exp.sort_values(by="mean_abs_attribution", ascending=False)["feature"].head(3).tolist()
                jaccard = len(set(top3_sym).intersection(set(top3_post))) / len(set(top3_sym).union(set(top3_post)))

                comp_records.append({
                    "experiment": exp,
                    "comparison": f"NeuroSymbolic_vs_{method.upper()}",
                    "spearman_rho": round(float(rho), 4),
                    "p_value": round(float(pval), 4),
                    "top3_jaccard": round(jaccard, 4),
                })
        comp_df = pd.DataFrame(comp_records)
        comp_df.to_csv(res_dir / "neurosymbolic_vs_posthoc_comparison.csv", index=False)
        LOGGER.info("Saved %s", res_dir / "neurosymbolic_vs_posthoc_comparison.csv")
        print("\nNeuro-Symbolic vs. Post-Hoc (SHAP & LIME) Comparison Preview:")
        print(comp_df.to_string())

    print("\nNeuro-Symbolic Performance Preview:")
    print(perf_df.to_string())


if __name__ == "__main__":
    main()
