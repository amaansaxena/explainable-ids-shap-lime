# Explainable IDS: SHAP, LIME and Neuro-Symbolic Rule Reliability Across Datasets

Do SHAP, LIME and neuro-symbolic rule explanations survive a change of dataset? This repository trains binary
intrusion-detection models (benign vs attack) on nine flow features shared by CICIDS2017 and UNSW-NB15, tests them
in-domain and cross-dataset, and measures how much SHAP, LIME and a neuro-symbolic rule surrogate (IF-THEN rules
extracted from the neural network) agree with each other and how stable they stay under dataset shift. A Streamlit dashboard shows all results and gives live predictions with explanations.



## Quick start (use the committed models)

Trained models and results are committed, so the dashboard runs without retraining.

```bash
git lfs pull                              # the aligned CSVs are stored with Git LFS
python -m pip install -r requirements.txt # versions tested with Python 3.13
streamlit run app.py
```

If the data CSVs are missing or are un-pulled Git LFS pointers, the Live Prediction page falls back to manual input.

## Full pipeline

| Step | Command | Output |
|---|---|---|
| 1. Quick test on small subsets | `python run_all.py --smoke` | `smoke_output/` (real models and results untouched) |
| 2. Full run: preprocess, train, evaluate, explain, summarise | `python run_all.py` | overwrites `models/` and `results/` |
| 3. Explanation reliability metrics | `python -m src.metrics_xai` | `results/explanation_reliability_metrics.csv` |
| 4. Rule surrogate of MLP-A | `python -m src.evaluate_neurosymbolic` | `results/neurosymbolic_*`, rule trees in `results/neurosymbolic/` |
| 5. Report figures | `python -m src.visualize_results` | `results/figures/fig1-4*.png` |
| 6. Check finalized data | `python validate_aligned.py` | row, feature and attack counts |
| 7. Unit tests | `pytest` | data loader and model tests |

`run_all.py` runs `src.preprocessing`, `src.train`, `src.evaluate`, `src.explain` and `src.compare_xai` in order.
Steps 3 to 5 depend on its outputs. Explanations are cached in `results/xai_cache/` (gitignored); LIME on the Keras
models is the slowest stage.

## Data

| Dataset | Source used | Train rows (attacks) | Test rows after de-duplication (attacks) |
|---|---|---|---|
| CICIDS2017 | Thursday-WorkingHours-Morning-WebAttacks, stratified 80/20 split | 131,440 (1.3%) | 24,683 (400, 1.6%) |
| UNSW-NB15 | Official training and testing files | 175,341 (68.1%) | 45,816 (17,617, 38.5%) |

**Nine shared features:** `flow_duration`, `fwd_packets`, `bwd_packets`, `fwd_bytes`, `bwd_bytes`,
`fwd_to_bwd_pkt_ratio`, `fwd_to_bwd_byte_ratio`, `bytes_per_packet`, `packets_per_second`.
Dataset-specific columns (TCP flags, TTL) were dropped. Each dataset was standardized separately on its own training split
in the preprocessing notebooks.

**Leakage fix:** many test rows had exactly the same nine-feature vector as a training row. `src/preprocessing.py`
removes them from the test split (originals are preserved): CICIDS 32,860 → 24,683 (8,177 removed, 29 attacks) and
UNSW-NB15 82,332 → 45,816 (36,516 removed, 27,715 attacks), which lowers the UNSW-NB15 test attack rate from 55% to 38.5%.

## Experiments

| Exp | Train | Test | Type |
|---|---|---|---|
| A | CICIDS2017 | CICIDS2017 | In-domain |
| B | UNSW-NB15 | UNSW-NB15 | In-domain |
| C | CICIDS2017 | UNSW-NB15 | Cross-dataset, no fine-tuning |
| D | UNSW-NB15 | CICIDS2017 | Cross-dataset, no fine-tuning |

## Models

5 model types × 5 seeds (42-46) × 2 training datasets = 50 trained models.

| Model | Architecture | Params |
|---|---|---|
| MLP-A | 9 → 64 → 32 → 16 → 1, ReLU, sigmoid output, no regularization | 3,265 |
| MLP-B | Gaussian noise (σ 0.05) → 2 × [Dense 64/32, L2 1e-4, LayerNorm, ReLU, Dropout 0.3] → 16 → 1 | 3,457 |
| MLP-C | Dense 64 projection → 2 residual blocks [LayerNorm, Dense 64 ReLU, Dropout 0.2, Dense 64] → 16 → 1 | 18,593 |
| RF | Random forest, 200 trees, max depth 12, balanced class weights | - |
| DT | Decision tree, max depth 5, balanced class weights | - |

MLPs: Adam (lr 1e-3), binary cross-entropy, batch 256, 50 epochs, balanced class weights. Every model gets a decision
threshold tuned on a stratified 15% validation split (best F1), stored in a JSON sidecar next to the model with the seed,
validation ROC-AUC, training time and git commit. Evaluation reports results at both the tuned threshold and a fixed 0.5.

## Explainability

- **Samples:** the same class-balanced test flows for every model (500 benign + 500 attack; 900 for CICIDS targets,
  which have 400 attacks), sampled with a fixed seed and saved to `results/xai/instances_*.json`. Seed-42 models only.
- **SHAP** (`src/xai/shap_explainer.py`): model-agnostic `shap.Explainer` on the attack probability, 100 source training
  rows as background; exact Shapley values with nine features.
- **LIME** (`src/xai/lime_explainer.py`): `LimeTabularExplainer` on source training statistics, continuous features
  not discretized, 1,000 perturbations per flow.
- **Reliability** (`src/metrics_xai.py`): Spearman ρ and top-3/top-5 Jaccard overlap between feature rankings
  (mean |attribution|). *Agreement* compares SHAP with LIME for one model and experiment; *stability* compares one method
  in-domain vs cross-dataset (A vs C, B vs D). Random baselines: top-3 Jaccard 0.23, top-5 0.40.
- **Rule surrogate** (`src/evaluate_neurosymbolic.py`, `src/xai/neurosymbolic.py`): a depth-5 decision tree fitted to
  MLP-A's (seed 42) predictions, not the true labels, exported as IF-THEN rules. Scored by fidelity (agreement with MLP-A),
  with rule-path attributions on the same flows SHAP and LIME explain. This is post-hoc global rule extraction; the
  rules do not change how the network makes decisions.

## Dashboard

`streamlit run app.py` opens seven pages: Overview, Model Performance, XAI Attributions, Reliability Analysis,
Neuro-Symbolic, Live Prediction and Visual Gallery. Sidebar filters select experiments, models and threshold type.

Live Prediction draws a real held-out test flow (from either dataset, so cross-dataset predictions can be tried), shows the
attack probability against the model's tuned threshold, compares all five model types on the same flow, and explains it
with SHAP, LIME or the surrogate rule, including whether the rule agrees with MLP-A and its test fidelity. Inputs are in
standardized (z-score) units because the original scaler was not preserved. For speed on single flows the dashboard uses
GradientExplainer (MLPs) and TreeExplainer (RF/DT) for SHAP and a 200-row background sample for LIME, so its values can
differ slightly from the pipeline's.

## Key results

Means over five seeds, validation-tuned threshold.

| ROC-AUC | A (CIC→CIC) | B (UNSW→UNSW) | C (CIC→UNSW) | D (UNSW→CIC) |
|---|---|---|---|---|
| MLP-A | 0.997 | 0.907 | 0.522 | 0.720 |
| MLP-B | 0.994 | 0.814 | 0.487 | 0.863 |
| MLP-C | 0.997 | 0.913 | 0.525 | 0.720 |
| RF | 0.997 | 0.943 | 0.470 | 0.143 |
| DT | 0.996 | 0.813 | 0.466 | 0.462 |

- In-domain detection is strong, though CICIDS precision is only 0.36-0.73 because attacks are 1.6% of test flows.
- Cross-dataset detection collapses: CIC → UNSW is at chance for every model, and RF is worse than random in UNSW → CIC.
- SHAP and LIME barely agree: mean ρ 0.30, 2 of 20 pairs significant, top-3 overlap 0.31 vs 0.23 random.
- LIME rankings barely change across datasets (mean ρ 0.94) even when the models fail; SHAP shifts more (0.60).
- The surrogate copies MLP-A with 99.8% fidelity in-domain (CICIDS) but only 26.9% in UNSW → CIC.

Full tables: `results/metrics.csv`, `results/metrics_summary.csv`, `results/explanation_reliability_metrics.csv`,
`results/neurosymbolic_metrics.csv`.

## Project structure

```
app.py                      Streamlit dashboard
run_all.py                  Pipeline runner (--smoke for a quick test)
validate_aligned.py         Sanity checks on finalized data
src/
  config.py                 Paths, features, seeds, scaling mode (single source of truth)
  preprocessing.py          Validates aligned data, removes train/test duplicates, writes finalized files
  data_loader.py            Loads splits, stratified train/validation split per seed
  model.py                  Model zoo and ModelAdapter (one interface for Keras and sklearn)
  train.py                  Training and F1 threshold tuning
  evaluate.py               Experiments A-D at tuned and 0.5 thresholds
  explain.py                Runs and caches SHAP/LIME
  compare_xai.py            Mean |attribution| per feature
  metrics_xai.py            Agreement and stability metrics
  evaluate_neurosymbolic.py Rule surrogate, fidelity, rule-path attributions
  visualize_results.py      Report figures
  model_da.py               Placeholder for domain adaptation (not implemented)
  xai/                      SHAP, LIME, sampling and surrogate helpers
data/
  preprocess/               Feature selection outputs
  preprocessed/aligned/     Aligned train/test CSVs (Git LFS) and finalized representations
  preprocessed/raw/         Preprocessed UNSW-NB15 files (Git LFS)
notebooks/                  Dataset preprocessing and alignment
models/<dataset>/<model>/   seed42-46 .keras/.joblib files with JSON sidecars
results/                    Metrics, explanation summaries, surrogate outputs, figures
docs/audit_report.md        Data audit
tests/                      pytest tests for the data loader and models
```

## Limitations

- Only nine shared features, several strongly correlated (counters and their ratios).
- CICIDS2017 covers one morning with three web-attack types and 400 test attacks.
- Raw source tables and the scaler are unavailable, so the `raw`, `log` and `scaled` finalized files hold the same
  recovered model-space values; attributions and rule thresholds cannot be reported in raw units.
- Source-vs-domain scaling is exposed in the code, but source scaling cannot be reconstructed without the original scaler.
- The explanation analysis uses seed 42 only, and its samples are class-balanced rather than prevalence-balanced.
- No early stopping; the SHAP background is the first 100 training rows rather than a random sample.

## Future work

- A larger standard feature set (e.g. NetFlow-based features) shared across more datasets.
- Domain adaptation for the cross-dataset experiments (`src/model_da.py`).
- Explanation analysis over all five seeds.
- Saving the scaler so results can be reported in bytes and seconds.
- A direct faithfulness test: remove each method's top features and measure the change in the model's output.
