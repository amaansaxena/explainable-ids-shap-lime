"""
Explainable IDS Dashboard — SHAP/LIME Cross-Dataset Reliability
Beautiful Streamlit web application for IDS model performance and XAI explanations.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Explainable IDS · SHAP & LIME",
        layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — dark glassmorphism, bright readable text
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family:'Inter',sans-serif; }

/* Background */
.stApp {
    background: linear-gradient(135deg,#0d0f1a 0%,#111827 50%,#0d1b2a 100%);
}

/* Hide chrome */
#MainMenu,footer,header {visibility:hidden;}

/* ALL TEXT — bright white by default */
*, p, label, div, span, li, td, th {
    color: #f1f5f9 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0f172a 0%,#1e293b 100%);
    border-right: 1px solid rgba(99,102,241,.25);
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg,rgba(99,102,241,.18) 0%,rgba(139,92,246,.12) 100%);
    border: 1px solid rgba(99,102,241,.35);
    border-radius: 16px;
    padding: 1rem 1.25rem;
    backdrop-filter: blur(12px);
    transition: transform .2s ease, box-shadow .2s ease;
}
div[data-testid="metric-container"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(99,102,241,.35);
}
div[data-testid="metric-container"] label {
    color: #cbd5e1 !important;
    font-size: .75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: .05em;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 1.7rem !important;
    font-weight: 800 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(15,23,42,.7);
    border-radius: 12px;
    padding: 4px;
    border: 1px solid rgba(99,102,241,.25);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 600;
    color: #cbd5e1 !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,#6366f1,#8b5cf6) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 14px rgba(99,102,241,.45);
}

/* Inputs */
.stSelectbox [data-baseweb="select"]>div,
.stMultiSelect [data-baseweb="select"]>div {
    background: rgba(30,41,59,.9) !important;
    border: 1px solid rgba(99,102,241,.35) !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
}
.stNumberInput input, .stSlider [data-testid="stTickBar"] {
    color: #f1f5f9 !important;
}
input[type="number"] { color: #f1f5f9 !important; }

/* Radio */
.stRadio [data-testid="stMarkdownContainer"] p { color: #e2e8f0 !important; }

/* Expander */
.streamlit-expanderHeader { color: #e2e8f0 !important; font-weight: 600; }

/* Dataframe */
[data-testid="stDataFrame"] * { color: #f1f5f9 !important; }

/* Code */
code { color: #c4b5fd !important; }
pre { background: rgba(15,23,42,.8) !important; }

/* Scrollbar */
::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background:#0d0f1a; }
::-webkit-scrollbar-thumb { background:rgba(99,102,241,.5); border-radius:3px; }

/* Hero */
.hero-header {
    background: linear-gradient(135deg,rgba(99,102,241,.18) 0%,rgba(139,92,246,.12) 50%,rgba(59,130,246,.12) 100%);
    border: 1px solid rgba(99,102,241,.3);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    backdrop-filter: blur(20px);
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content:'';
    position:absolute; top:-50%; right:-10%;
    width:40%; height:200%;
    background:radial-gradient(ellipse,rgba(139,92,246,.18) 0%,transparent 70%);
    pointer-events:none;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg,#818cf8,#c084fc,#60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 .5rem 0;
    line-height: 1.2;
}
.hero-subtitle { color:#cbd5e1 !important; font-size:1rem; font-weight:400; margin:0; line-height:1.6; }
.hero-badge {
    display:inline-block;
    background:linear-gradient(135deg,rgba(99,102,241,.35),rgba(139,92,246,.35));
    border:1px solid rgba(99,102,241,.45);
    border-radius:20px;
    padding:4px 14px;
    font-size:.75rem; font-weight:700;
    color:#c4b5fd !important;
    margin-bottom:1rem;
    letter-spacing:.06em; text-transform:uppercase;
}

/* Section headers */
.section-header { display:flex; align-items:center; gap:.75rem; margin:1.5rem 0 1rem 0; }
.section-title { font-size:1.2rem; font-weight:700; color:#f1f5f9 !important; margin:0; }
.section-line { flex:1; height:1px; background:linear-gradient(90deg,rgba(99,102,241,.5),transparent); }

/* Prediction result box */
.result-safe {
    background: rgba(34,197,94,.12);
    border: 2px solid rgba(34,197,94,.5);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
}
.result-attack {
    background: rgba(239,68,68,.12);
    border: 2px solid rgba(239,68,68,.5);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
}

/* Feature input card */
.feat-card {
    background: rgba(15,23,42,.7);
    border: 1px solid rgba(99,102,241,.2);
    border-radius: 14px;
    padding: 1.25rem;
    margin-bottom: .5rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants & helpers
# ─────────────────────────────────────────────────────────────────────────────
BASE         = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:  # so `src` imports work whatever directory streamlit is launched from
    sys.path.insert(0, BASE)
FEATURES     = ["flow_duration","fwd_packets","bwd_packets","fwd_bytes","bwd_bytes",
                "fwd_to_bwd_pkt_ratio","fwd_to_bwd_byte_ratio","bytes_per_packet","packets_per_second"]
FEATURE_LABELS = {
    "flow_duration":        "Flow Duration",
    "fwd_packets":          "Fwd Packets",
    "bwd_packets":          "Bwd Packets",
    "fwd_bytes":            "Fwd Bytes",
    "bwd_bytes":            "Bwd Bytes",
    "fwd_to_bwd_pkt_ratio": "Fwd/Bwd Packet Ratio",
    "fwd_to_bwd_byte_ratio":"Fwd/Bwd Byte Ratio",
    "bytes_per_packet":     "Bytes per Packet",
    "packets_per_second":   "Packets per Second",
}
MODEL_COLORS = {"mlp_a":"#6366f1","mlp_b":"#8b5cf6","mlp_c":"#c084fc","rf":"#22c55e","dt":"#f59e0b"}
EXP_COLORS   = {"A":"#22c55e","B":"#3b82f6","C":"#f59e0b","D":"#ef4444"}
EXP_LABELS   = {"A":"A · CICIDS→CICIDS","B":"B · UNSW→UNSW","C":"C · CICIDS→UNSW","D":"D · UNSW→CICIDS"}
METHOD_COLORS= {"shap":"#6366f1","lime":"#f59e0b"}

PAPER_BG="rgba(0,0,0,0)"; PLOT_BG="rgba(15,23,42,.6)"; GRID_COLOR="rgba(99,102,241,.12)"

def chart_layout(fig, title="", height=420):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14,color="#f1f5f9",family="Inter"), x=.02),
        template="plotly_dark", paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(family="Inter",color="#e2e8f0"),
        height=height, margin=dict(l=16,r=16,t=48,b=16),
        legend=dict(bgcolor="rgba(15,23,42,.85)",bordercolor="rgba(99,102,241,.35)",
                    borderwidth=1,font=dict(size=11,color="#f1f5f9")),
    )
    fig.update_xaxes(gridcolor=GRID_COLOR,zeroline=False,tickfont=dict(color="#e2e8f0"))
    fig.update_yaxes(gridcolor=GRID_COLOR,zeroline=False,tickfont=dict(color="#e2e8f0"))
    return fig

def section(title):
    st.markdown(f"""
    <div class="section-header">
        <span class="section-title">{title}</span>
        <span class="section-line"></span>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_metrics():
    return pd.read_csv(os.path.join(BASE,"results","metrics.csv"))

@st.cache_data
def load_metrics_summary():
    df = pd.read_csv(os.path.join(BASE,"results","metrics_summary.csv"))
    df.columns = df.columns.str.strip()
    return df[df["experiment"].notna() & (df["experiment"]!="")]

@st.cache_data
def load_xai_metrics():
    return pd.read_csv(os.path.join(BASE,"results","xai_metrics.csv"))

@st.cache_data
def load_reliability():
    return pd.read_csv(os.path.join(BASE,"results","explanation_reliability_metrics.csv"))

@st.cache_data
def load_neurosymbolic():
    return pd.read_csv(os.path.join(BASE,"results","neurosymbolic_metrics.csv"))

def load_result_images():
    imgs = {}
    img_dir  = os.path.join(BASE,"results")
    figs_dir = os.path.join(img_dir,"figures")
    for f in os.listdir(img_dir):
        if f.endswith(".png"): imgs[f] = os.path.join(img_dir,f)
    if os.path.isdir(figs_dir):
        for f in os.listdir(figs_dir):
            if f.endswith(".png"): imgs[f"figures/{f}"] = os.path.join(figs_dir,f)
    return imgs

# ─────────────────────────────────────────────────────────────────────────────
# Model loading (prediction page)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_sklearn_model(dataset, model_name, seed):
    import joblib
    path = os.path.join(BASE,"models",dataset,model_name,f"seed{seed}.joblib")
    return joblib.load(path)

@st.cache_resource
def load_keras_model(dataset, model_name, seed):
    import tensorflow as tf
    path = os.path.join(BASE,"models",dataset,model_name,f"seed{seed}.keras")
    return tf.keras.models.load_model(path)

def load_sidecar(dataset, model_name, seed):
    ext = ".keras" if model_name.startswith("mlp") else ".joblib"
    sc  = os.path.join(BASE,"models",dataset,model_name,f"seed{seed}.json")
    if os.path.exists(sc):
        with open(sc) as f:
            return json.load(f)
    return {}

def _finalized_path(dataset):
    name = "CICIDS" if dataset == "cicids" else "UNSW_NB15"
    return os.path.join(BASE, "data", "preprocessed", "aligned", f"{name}_scaled.csv")

class DataUnavailable(RuntimeError):
    """Flow CSVs are missing, unreadable, or un-pulled Git LFS pointer files."""

def _read_flows(path):
    rel = os.path.relpath(path, BASE)
    if not os.path.exists(path):
        raise DataUnavailable(f"`{rel}` not found.")
    with open(path, "rb") as f:
        if f.read(40).startswith(b"version https://git-lfs"):
            raise DataUnavailable(f"`{rel}` is a Git LFS pointer, not data. Run `git lfs pull`.")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise DataUnavailable(f"`{rel}` could not be read: {e}") from e
    missing = [c for c in FEATURES + ["label"] if c not in df.columns]
    if missing:
        raise DataUnavailable(f"`{rel}` is missing columns: {', '.join(missing)}")
    return df

@st.cache_data
def load_split(dataset, split):
    """Rows of the finalized (standardized, leakage-checked) data for one split.

    Models were trained on these standardized values, so every input to the
    Live Prediction page and every XAI background must come from this space.
    Raises DataUnavailable if no readable data exists.
    """
    path = _finalized_path(dataset)
    if os.path.exists(path):
        df = _read_flows(path)
        df = df[df["split"] == split]
    else:  # fall back to the committed aligned split if the pipeline has not been run
        name = "CICIDS" if dataset == "cicids" else "UNSW_NB15"
        df = _read_flows(os.path.join(BASE, "data", "preprocessed", "aligned", f"{name}_{split}_aligned.csv"))
    return df[FEATURES + ["label"]].reset_index(drop=True)

def data_problem(dataset, split="train"):
    """Why a split cannot be loaded, or None if it loads fine."""
    try:
        load_split(dataset, split)
        return None
    except DataUnavailable as e:
        return str(e)

@st.cache_data
def background_sample(dataset, n=200):
    """Class-stratified background from the source training split.

    If the data cannot be loaded, fall back to standard-normal noise: the features
    are z-scored on the training split (mean 0, std 1), so this approximates the
    training distribution. Using copies of the input instead would zero out SHAP.
    """
    try:
        train = load_split(dataset, "train")
    except DataUnavailable:
        return np.random.default_rng(42).standard_normal((n, len(FEATURES))).astype(np.float32)
    parts = [g.sample(min(len(g), n // 2), random_state=42) for _, g in train.groupby("label")]
    return pd.concat(parts)[FEATURES].to_numpy(np.float32)

def _attack_prob(model, model_name, X):
    X = np.asarray(X, dtype=np.float32)
    if model_name.startswith("mlp"):
        return model.predict(X, verbose=0).reshape(-1)
    return model.predict_proba(X)[:, 1]

def _load(dataset, model_name, seed):
    if model_name.startswith("mlp"):
        return load_keras_model(dataset, model_name, seed)
    return load_sklearn_model(dataset, model_name, seed)

def predict(dataset, model_name, seed, x_arr):
    """Return (attack_prob, label, threshold) for a single standardized sample."""
    try:
        model = _load(dataset, model_name, seed)
        prob  = float(_attack_prob(model, model_name, x_arr)[0])
        sc    = load_sidecar(dataset, model_name, seed)
        thr   = sc.get("val_threshold", 0.5)
        label = int(prob >= thr)
        return prob, label, thr
    except Exception:
        return None, None, None

def _attack_slice(vals):
    """Reduce any SHAP output layout to a (n_rows, n_features) attack-class array."""
    if isinstance(vals, list):          # old SHAP: one array per class
        vals = vals[-1]
    vals = np.asarray(vals)
    if vals.ndim == 3:                  # new SHAP: (rows, features, outputs)
        vals = vals[:, :, -1]
    return vals.reshape(vals.shape[0], -1)

def shap_explanation(dataset, model_name, seed, x_arr):
    """Return list of (feature, shap_val) pairs for the attack class (may raise)."""
    import shap
    model = _load(dataset, model_name, seed)
    if model_name.startswith("mlp"):
        # Background must be real training data; using the input itself makes every value 0.
        explainer = shap.GradientExplainer(model, background_sample(dataset))
        vals = explainer.shap_values(np.asarray(x_arr, dtype=np.float32))
    else:
        vals = shap.TreeExplainer(model).shap_values(x_arr)
    return list(zip(FEATURES, _attack_slice(vals)[0].tolist()))

def lime_explanation(dataset, model_name, seed, x_arr):
    """Return list of (feature, lime_weight) pairs for the attack class."""
    from lime import lime_tabular
    model = _load(dataset, model_name, seed)
    def predict_fn(X):
        p = _attack_prob(model, model_name, X)
        return np.column_stack([1 - p, p])
    explainer = lime_tabular.LimeTabularExplainer(
        background_sample(dataset), feature_names=FEATURES, class_names=["Normal","Attack"],
        mode="classification", discretize_continuous=False, random_state=42
    )
    exp = explainer.explain_instance(np.asarray(x_arr[0], dtype=float), predict_fn,
                                     num_features=len(FEATURES), num_samples=1000, labels=(1,))
    # local_exp is keyed by feature index, so it works whatever the label text looks like.
    weights = dict(exp.local_exp[1])
    return [(f, float(weights.get(i, 0.0))) for i, f in enumerate(FEATURES)]

@st.cache_resource
def load_surrogate(source):
    """Decision-tree surrogate of mlp_a seed42 saved by src.evaluate_neurosymbolic (or None)."""
    import joblib
    path = os.path.join(BASE, "results", "neurosymbolic", f"surrogate_{source}.joblib")
    return joblib.load(path) if os.path.exists(path) else None

def surrogate_rule(tree, x_arr):
    """The IF-THEN conditions this flow satisfies on its path through the surrogate tree."""
    t = tree.tree_
    path = tree.decision_path(x_arr).indices
    conds = []
    for node, nxt in zip(path[:-1], path[1:]):
        # Direction comes from the child actually taken, so it always matches the tree's own routing.
        op = "≤" if nxt == t.children_left[node] else ">"
        f = t.feature[node]
        conds.append((FEATURES[f], op, float(t.threshold[node]), float(x_arr[0, f])))
    return conds, int(t.n_node_samples[path[-1]])

def neurosymbolic_explanation(source, target, x_arr):
    """Surrogate verdict, fired rule, path attributions and fidelity for one flow (None if no surrogate)."""
    from src.xai.neurosymbolic import rule_path_attribution
    tree = load_surrogate(source)
    if tree is None:
        return None
    x = np.asarray(x_arr, dtype=np.float32)
    prob = float(tree.predict_proba(x)[0, 1])
    conds, leaf_samples = surrogate_rule(tree, x)
    attrs = rule_path_attribution(tree, x)[0]
    ns = load_neurosymbolic()
    row = ns[(ns["source"] == source) & (ns["target"] == target)]
    fidelity = float(row["fidelity_test"].iloc[0]) if len(row) and "fidelity_test" in row else None
    return {"prob": prob, "label": int(prob >= 0.5), "conditions": conds, "leaf_samples": leaf_samples,
            "attributions": list(zip(FEATURES, attrs)), "fidelity": fidelity,
            "experiment": row["experiment"].iloc[0] if len(row) else None}

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 .5rem;">
        <div style="display:inline-block;padding:3px 10px;background:rgba(99,102,241,0.2);border:1px solid rgba(99,102,241,0.4);border-radius:20px;font-size:0.7rem;font-weight:700;color:#a5b4fc;margin-bottom:0.4rem;letter-spacing:0.08em;text-transform:uppercase;">IDS Security</div>
        <div style="font-size:1rem;font-weight:800;color:#f1f5f9;">Explainable IDS</div>
        <div style="font-size:.72rem;color:#818cf8;font-weight:700;letter-spacing:.08em;
                    text-transform:uppercase;margin-top:.25rem;">SHAP · LIME · XAI</div>
    </div>
    <hr style="border-color:rgba(99,102,241,.3);margin:1rem 0;">
    """, unsafe_allow_html=True)

    st.markdown("**Navigation**")
    page = st.radio("Go to", [
        "Overview","Model Performance","XAI Attributions",
        "Reliability Analysis","Neuro-Symbolic",
        "Live Prediction","Visual Gallery"
    ], label_visibility="collapsed")

    st.markdown("<hr style='border-color:rgba(99,102,241,.2);margin:1.2rem 0;'>", unsafe_allow_html=True)
    st.markdown("**Global Filters**")

    mdf = load_metrics()
    all_exp    = sorted(mdf["experiment"].unique())
    all_models = sorted(mdf["model"].unique())

    sel_exp = st.multiselect("Experiments", all_exp, default=all_exp,
                              format_func=lambda x: EXP_LABELS.get(x, x))
    sel_models = st.multiselect("Models", all_models, default=all_models)
    thr_type   = st.selectbox("Threshold",["validation","fixed_0_5"],
                               format_func=lambda x: "Validation-tuned" if x=="validation" else "Fixed 0.5")

    st.markdown("<hr style='border-color:rgba(99,102,241,.2);margin:1.2rem 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:.73rem;color:#94a3b8;line-height:2;">
        <b style="color:#cbd5e1;">Datasets</b><br>
        · CICIDS2017 &nbsp; · UNSW-NB15<br>
        <b style="color:#cbd5e1;">Models</b><br>
        · MLP-A / B / C &nbsp; · RF &nbsp; · DT<br>
        <b style="color:#cbd5e1;">Explainers</b><br>
        · SHAP &nbsp; · LIME &nbsp; · Neuro-Symbolic
    </div>
    """, unsafe_allow_html=True)

def filt(df, threshold=True):
    m = df["experiment"].isin(sel_exp) & df["model"].isin(sel_models)
    if threshold and "threshold_type" in df.columns:
        m &= (df["threshold_type"] == thr_type)
    return df[m]

# Pages driven by the sidebar filters show nothing useful with an empty selection.
if page not in ("Live Prediction", "Visual Gallery") and (not sel_exp or not sel_models):
    st.warning("Select at least one experiment and one model in the sidebar filters.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE — Overview
# ─────────────────────────────────────────────────────────────────────────────
if page == "Overview":
    st.markdown("""
    <div class="hero-header">
        <div class="hero-badge">Research Dashboard</div>
        <h1 class="hero-title">Explainable Intrusion Detection</h1>
        <p class="hero-subtitle">
            Evaluating SHAP &amp; LIME explanation reliability across in-domain and cross-dataset
            IDS experiments on <strong style="color:#c084fc;">CICIDS2017</strong>
            and <strong style="color:#60a5fa;">UNSW-NB15</strong>.
        </p>
    </div>""", unsafe_allow_html=True)

    df = load_metrics()
    fd = filt(df)
    vd = fd[fd["threshold_type"]=="validation"]

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: st.metric("Total Runs", f"{len(fd):,}")
    with c2: st.metric("Avg ROC-AUC", f"{vd['roc_auc'].mean():.3f}" if len(vd) else "—")
    with c3: st.metric("Avg F1",      f"{vd['f1'].mean():.3f}"      if len(vd) else "—")
    with c4: st.metric("Avg Recall",  f"{vd['recall'].mean():.3f}"  if len(vd) else "—")
    with c5: st.metric("Models",      fd["model"].nunique())

    st.markdown("<br>", unsafe_allow_html=True)
    section("Experiment Cards")
    cols = st.columns(4)
    info = [
        ("A","CICIDS → CICIDS","In-domain on CICIDS2017. Same-distribution train/test.","22c55e"),
        ("B","UNSW → UNSW",    "In-domain on UNSW-NB15. High class imbalance.","3b82f6"),
        ("C","CICIDS → UNSW",  "Cross-dataset: train CICIDS, test UNSW. Distribution shift.","f59e0b"),
        ("D","UNSW → CICIDS",  "Cross-dataset: train UNSW, test CICIDS. Reverse transfer.","ef4444"),
    ]
    for col,(exp,title,desc,hex_) in zip(cols,info):
        n = len(fd[fd["experiment"]==exp])
        r,g,b = int(hex_[:2],16),int(hex_[2:4],16),int(hex_[4:],16)
        with col:
            st.markdown(f"""
            <div style="background:rgba({r},{g},{b},.1);border:1px solid rgba({r},{g},{b},.35);
                        border-radius:14px;padding:1.25rem;height:185px;">
                <div style="font-size:1.6rem;font-weight:800;color:#{hex_};">Exp {exp}</div>
                <div style="font-size:.85rem;font-weight:700;color:#f1f5f9;margin:.3rem 0 .5rem;">{title}</div>
                <div style="font-size:.75rem;color:#cbd5e1;line-height:1.55;">{desc}</div>
                <div style="margin-top:.75rem;font-size:.75rem;color:#{hex_};font-weight:700;">{n} runs</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cl, cr = st.columns([3,2])

    with cl:
        section("ROC-AUC Heatmap")
        hm = vd.groupby(["experiment","model"])["roc_auc"].mean().reset_index()\
               .pivot(index="model",columns="experiment",values="roc_auc")
        if not hm.empty:
            fig = px.imshow(hm.round(3),
                color_continuous_scale=[(0,"#1e293b"),(0.4,"#4f46e5"),(0.7,"#7c3aed"),(1,"#a855f7")],
                text_auto=".3f", aspect="auto")
            fig.update_traces(textfont_size=12,textfont_color="#ffffff")
            fig = chart_layout(fig, height=340)
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(fig, width="stretch")

    with cr:
        section("Score Distribution")
        vd2 = vd[vd["roc_auc"].notna()].copy()
        fig2 = px.violin(vd2, x="experiment", y="roc_auc", color="experiment",
                         color_discrete_map=EXP_COLORS, box=True, points="all",
                         labels={"roc_auc":"ROC-AUC","experiment":"Experiment"})
        fig2 = chart_layout(fig2, height=340)
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, width="stretch")

    section("Summary Table")
    smry = vd.groupby(["experiment","model"])[["roc_auc","f1","recall","precision"]]\
              .mean().round(4).reset_index()
    smry["experiment"] = smry["experiment"].map(EXP_LABELS).fillna(smry["experiment"])
    st.dataframe(smry.rename(columns={"experiment":"Experiment","model":"Model",
        "roc_auc":"ROC-AUC","f1":"F1","recall":"Recall","precision":"Precision"}),
        width="stretch", hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE — Model Performance
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Model Performance":
    st.markdown("""
    <div class="hero-header">
        <div class="hero-badge">Evaluation</div>
        <h1 class="hero-title">Model Performance Analysis</h1>
        <p class="hero-subtitle">Detailed classification metrics across models, seeds and experiments.</p>
    </div>""", unsafe_allow_html=True)

    df = filt(load_metrics())
    metric_map = {"ROC-AUC":"roc_auc","F1 Score":"f1","Recall":"recall","Precision":"precision",
                  "PR-AUC":"pr_auc","Accuracy":"accuracy","Balanced Accuracy":"balanced_accuracy","MCC":"mcc"}
    chosen_label = st.selectbox("Primary Metric", list(metric_map.keys()))
    chosen       = metric_map[chosen_label]

    section(f"Mean {chosen_label} by Model & Experiment")
    bar = df.groupby(["experiment","model"])[chosen].mean().reset_index()
    fig = px.bar(bar, x="model", y=chosen, color="experiment", barmode="group",
                 color_discrete_map=EXP_COLORS,
                 labels={chosen:chosen_label,"model":"Model","experiment":"Experiment"},
                 text=bar[chosen].round(3))
    fig.update_traces(textposition="outside",textfont_size=10,textfont_color="#f1f5f9",marker_line_width=0)
    fig = chart_layout(fig, height=420)
    fig.update_layout(legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
    st.plotly_chart(fig, width="stretch")

    cl,cr = st.columns(2)
    with cl:
        section("Seed Stability (Box)")
        fig2 = px.box(df, x="model", y=chosen, color="experiment",
                      color_discrete_map=EXP_COLORS, points="all",
                      labels={chosen:chosen_label,"model":"Model"})
        fig2 = chart_layout(fig2, height=400)
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, width="stretch")

    with cr:
        section("ROC-AUC vs Recall")
        sc = df[df["roc_auc"].notna() & df["recall"].notna()]
        fig3 = px.scatter(sc, x="recall", y="roc_auc", color="experiment", symbol="model",
                          color_discrete_map=EXP_COLORS, opacity=.85,
                          hover_data=["model","seed","f1"],
                          labels={"roc_auc":"ROC-AUC","recall":"Recall"})
        fig3.update_traces(marker_size=9)
        fig3 = chart_layout(fig3, height=400)
        st.plotly_chart(fig3, width="stretch")

    section("Multi-Metric Radar")
    radar_m = ["roc_auc","f1","recall","precision","balanced_accuracy"]
    radar_l = ["ROC-AUC","F1","Recall","Precision","Bal.Acc"]
    rdf = df.groupby("model")[radar_m].mean().reset_index()
    fig4 = go.Figure()
    for _,row in rdf.iterrows():
        v = [row[m] for m in radar_m]+[row[radar_m[0]]]
        fig4.add_trace(go.Scatterpolar(r=v, theta=radar_l+[radar_l[0]], fill="toself",
            name=row["model"], line_color=MODEL_COLORS.get(row["model"],"#6366f1"), opacity=.85))
    fig4.update_layout(
        polar=dict(radialaxis=dict(visible=True,range=[0,1],gridcolor=GRID_COLOR,
                                   tickfont=dict(size=9,color="#e2e8f0")),
                   angularaxis=dict(gridcolor=GRID_COLOR,tickfont=dict(color="#f1f5f9")),
                   bgcolor=PLOT_BG),
        template="plotly_dark", paper_bgcolor=PAPER_BG,
        font=dict(family="Inter",color="#e2e8f0"), height=440,
        margin=dict(l=60,r=60,t=40,b=40),
        legend=dict(bgcolor="rgba(15,23,42,.85)",bordercolor="rgba(99,102,241,.3)",borderwidth=1,
                    font=dict(color="#f1f5f9")),
    )
    st.plotly_chart(fig4, width="stretch")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE — XAI Attributions
# ─────────────────────────────────────────────────────────────────────────────
elif page == "XAI Attributions":
    st.markdown("""
    <div class="hero-header">
        <div class="hero-badge">Explainability</div>
        <h1 class="hero-title">Feature Attributions</h1>
        <p class="hero-subtitle">Mean absolute SHAP and LIME attributions per feature, model and experiment.</p>
    </div>""", unsafe_allow_html=True)

    xai = load_xai_metrics()
    xf  = xai[xai["experiment"].isin(sel_exp) & xai["model"].isin(sel_models)]
    feat_order = FEATURES

    c1,c2,c3 = st.columns(3)
    with c1: sel_exp_x = st.selectbox("Experiment", sorted(xf["experiment"].unique()), format_func=lambda x: EXP_LABELS.get(x,x))
    with c2: sel_mod_x = st.selectbox("Model", sorted(xf[xf["experiment"]==sel_exp_x]["model"].unique()))
    with c3: sel_meth  = st.selectbox("Method", ["shap","lime","both"], format_func=lambda x:{"shap":"SHAP","lime":"LIME","both":"SHAP & LIME"}[x])

    sub = xf[(xf["experiment"]==sel_exp_x) & (xf["model"]==sel_mod_x)]
    if sel_meth != "both": sub = sub[sub["method"]==sel_meth]

    section("Mean Absolute Attribution by Feature")
    agg = sub.groupby(["feature","method"])["mean_abs_attribution"].mean().reset_index()
    agg["feature"] = pd.Categorical(agg["feature"], categories=feat_order, ordered=True)
    agg = agg.sort_values("feature")
    fig = px.bar(agg, y="feature", x="mean_abs_attribution", color="method",
                 color_discrete_map=METHOD_COLORS, orientation="h", barmode="group",
                 labels={"mean_abs_attribution":"Mean |Attribution|","feature":"Feature","method":"Method"},
                 text=agg["mean_abs_attribution"].round(4))
    fig.update_traces(textposition="outside",textfont_size=9,textfont_color="#f1f5f9",marker_line_width=0)
    fig = chart_layout(fig, height=440)
    st.plotly_chart(fig, width="stretch")

    section("Attribution Heatmap across Experiments")
    hm_meth = st.radio("Method for heatmap",["shap","lime"], horizontal=True, format_func=str.upper)
    hm = (xf[xf["method"]==hm_meth].groupby(["experiment","feature"])["mean_abs_attribution"]
            .mean().reset_index().pivot(index="feature",columns="experiment",values="mean_abs_attribution"))
    hm = hm.reindex([f for f in feat_order if f in hm.index])
    if not hm.empty:
        fig2 = px.imshow(hm.round(4),
            color_continuous_scale=[(0,"#0f172a"),(0.3,"#312e81"),(0.6,"#7c3aed"),(1,"#c084fc")],
            text_auto=".4f", aspect="auto")
        fig2.update_traces(textfont_size=10,textfont_color="#ffffff")
        fig2 = chart_layout(fig2, height=400)
        st.plotly_chart(fig2, width="stretch")

    section("SHAP vs LIME Agreement Scatter")
    cmp = xf[(xf["experiment"]==sel_exp_x)&(xf["model"]==sel_mod_x)]\
            .groupby(["feature","method"])["mean_abs_attribution"].mean().reset_index()
    sv = cmp[cmp["method"]=="shap"].set_index("feature")["mean_abs_attribution"]
    lv = cmp[cmp["method"]=="lime"].set_index("feature")["mean_abs_attribution"]
    common = sv.index.intersection(lv.index)
    if len(common):
        scat = pd.DataFrame({"feature":common,"SHAP":sv[common].values,"LIME":lv[common].values})
        fig3 = px.scatter(scat, x="SHAP", y="LIME", text="feature", color="feature",
                          color_discrete_sequence=px.colors.qualitative.Set2,
                          labels={"SHAP":"SHAP |Attribution|","LIME":"LIME |Attribution|"})
        fig3.update_traces(textposition="top center",textfont_size=10,textfont_color="#f1f5f9",marker_size=12)
        mx = max(scat[["SHAP","LIME"]].max())
        fig3.add_shape(type="line",x0=0,y0=0,x1=mx*1.1,y1=mx*1.1,
                       line=dict(color="rgba(99,102,241,.5)",dash="dash",width=1.5))
        fig3 = chart_layout(fig3, height=420)
        fig3.update_layout(showlegend=False)
        st.plotly_chart(fig3, width="stretch")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE — Reliability Analysis
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Reliability Analysis":
    st.markdown("""
    <div class="hero-header">
        <div class="hero-badge">XAI Reliability</div>
        <h1 class="hero-title">Explanation Reliability</h1>
        <p class="hero-subtitle">Spearman ρ, Jaccard similarity, and in-domain vs cross-domain agreement.</p>
    </div>""", unsafe_allow_html=True)

    rel = load_reliability()
    svm = rel[(rel["comparison_type"]=="SHAP_vs_LIME_Agreement") &
               rel["model"].isin(sel_models) & rel["experiment_1"].isin(sel_exp)]

    section("SHAP vs LIME Agreement (Spearman ρ)")
    if not svm.empty:
        fig = px.bar(svm, x="model", y="spearman_rho", color="experiment_1", barmode="group",
                     color_discrete_map=EXP_COLORS,
                     labels={"spearman_rho":"Spearman ρ","model":"Model","experiment_1":"Experiment"},
                     text=svm["spearman_rho"].round(3))
        fig.update_traces(textposition="outside",textfont_size=10,textfont_color="#f1f5f9",marker_line_width=0)
        fig.add_hline(y=0,line_dash="dash",line_color="rgba(239,68,68,.5)",line_width=1.5)
        fig = chart_layout(fig, height=400)
        fig.update_layout(legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
        st.plotly_chart(fig, width="stretch")

    cl,cr = st.columns(2)
    with cl:
        section("Top-3 Jaccard")
        fig2 = px.scatter(svm, x="model", y="top3_jaccard", color="experiment_1",
                          color_discrete_map=EXP_COLORS,
                          labels={"top3_jaccard":"Jaccard (Top-3)","model":"Model"})
        fig2.update_traces(marker_size=12,marker_line_width=1.5,marker_line_color="#f1f5f9")
        fig2.add_hline(y=.5,line_dash="dot",line_color="rgba(99,102,241,.5)")
        fig2 = chart_layout(fig2, height=360)
        st.plotly_chart(fig2, width="stretch")

    with cr:
        section("Top-5 Jaccard")
        fig3 = px.scatter(svm, x="model", y="top5_jaccard", color="experiment_1",
                          color_discrete_map=EXP_COLORS,
                          labels={"top5_jaccard":"Jaccard (Top-5)","model":"Model"})
        fig3.update_traces(marker_size=12,marker_line_width=1.5,marker_line_color="#f1f5f9")
        fig3.add_hline(y=.5,line_dash="dot",line_color="rgba(99,102,241,.5)")
        fig3 = chart_layout(fig3, height=360)
        fig3.update_layout(showlegend=False)
        st.plotly_chart(fig3, width="stretch")

    section("In-Domain vs Cross-Domain Reliability")
    cross = rel[rel["comparison_type"].str.startswith("In_Domain_vs_Cross_Domain") &
                rel["model"].isin(sel_models)]
    if not cross.empty:
        fig4 = px.strip(cross, x="model", y="spearman_rho", color="method", facet_col="comparison_type",
                        color_discrete_map=METHOD_COLORS, stripmode="overlay",
                        labels={"spearman_rho":"Spearman ρ","model":"Model","method":"Method"})
        fig4.update_traces(marker_size=10,marker_line_width=1,marker_line_color="#f1f5f9",opacity=.85)
        fig4.add_hline(y=0,line_dash="dash",line_color="rgba(239,68,68,.4)")
        fig4 = chart_layout(fig4, height=380)
        fig4.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1].replace("_"," "),
                                                     font=dict(size=10,color="#f1f5f9")))
        st.plotly_chart(fig4, width="stretch")

    with st.expander("Raw Reliability Data"):
        st.dataframe(rel, width="stretch", hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE — Neuro-Symbolic
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Neuro-Symbolic":
    st.markdown("""
    <div class="hero-header">
        <div class="hero-badge">Neuro-Symbolic AI</div>
        <h1 class="hero-title">Neuro-Symbolic Surrogate</h1>
        <p class="hero-subtitle">A depth-5 decision-tree surrogate fitted to MLP-A's decisions (global rule extraction).
        Fidelity = share of test flows where the rules agree with the MLP they explain.</p>
    </div>""", unsafe_allow_html=True)

    ns  = load_neurosymbolic()
    m   = filt(load_metrics())
    vd  = m[m["threshold_type"]=="validation"]

    cols = st.columns(len(ns))
    for col,(_,row) in zip(cols,ns.iterrows()):
        exp = row["experiment"]
        c   = EXP_COLORS.get(exp,"#6366f1")
        with col:
            st.markdown(f"""
            <div style="background:rgba(0,0,0,.35);border:1px solid {c}55;
                        border-top:3px solid {c};border-radius:14px;
                        padding:1.25rem;text-align:center;">
                <div style="font-size:1.4rem;font-weight:800;color:{c};">Exp {exp}</div>
                <div style="font-size:.72rem;color:#94a3b8;text-transform:uppercase;
                            font-weight:600;margin-bottom:.8rem;">{row['source']} → {row['target']}</div>
                <div style="display:flex;justify-content:space-around;">
                    <div>
                        <div style="font-size:1.2rem;font-weight:700;color:#f1f5f9;">{row['roc_auc']:.3f}</div>
                        <div style="font-size:.7rem;color:#94a3b8;">ROC-AUC</div>
                    </div>
                    <div>
                        <div style="font-size:1.2rem;font-weight:700;color:#f1f5f9;">{row['f1_score']:.3f}</div>
                        <div style="font-size:.7rem;color:#94a3b8;">F1</div>
                    </div>
                    <div>
                        <div style="font-size:1.2rem;font-weight:700;color:#f1f5f9;">{row.get('fidelity_test', float('nan')):.3f}</div>
                        <div style="font-size:.7rem;color:#94a3b8;">Fidelity</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cl,cr = st.columns(2)
    with cl:
        section("Neuro-Symbolic Radar")
        nm = ["roc_auc","precision","recall","f1_score"]
        nl = ["ROC-AUC","Precision","Recall","F1"]
        fig = go.Figure()
        for _,row in ns.iterrows():
            v = [row[m] for m in nm]+[row[nm[0]]]
            fig.add_trace(go.Scatterpolar(r=v,theta=nl+[nl[0]],fill="toself",
                name=f"Exp {row['experiment']}",line_color=EXP_COLORS.get(row["experiment"],"#6366f1"),opacity=.85))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True,range=[0,1],gridcolor=GRID_COLOR,
                                       tickfont=dict(size=9,color="#e2e8f0")),
                       angularaxis=dict(gridcolor=GRID_COLOR,tickfont=dict(color="#f1f5f9")),
                       bgcolor=PLOT_BG),
            template="plotly_dark",paper_bgcolor=PAPER_BG,
            font=dict(family="Inter",color="#e2e8f0"),height=400,
            margin=dict(l=60,r=60,t=40,b=40),
            legend=dict(bgcolor="rgba(15,23,42,.85)",bordercolor="rgba(99,102,241,.3)",
                        borderwidth=1,font=dict(color="#f1f5f9")),
        )
        st.plotly_chart(fig, width="stretch")

    with cr:
        section("Surrogate vs Best Model (ROC-AUC)")
        best = vd.groupby("experiment")["roc_auc"].max().reset_index().rename(columns={"roc_auc":"Best"})
        cmp  = ns.merge(best,on="experiment",how="left")
        cmp["exp_label"] = cmp["experiment"].map(EXP_LABELS).fillna(cmp["experiment"])
        fig2 = go.Figure([
            go.Bar(x=cmp["exp_label"],y=cmp["roc_auc"],name="Neuro-Symbolic",
                   marker_color="#8b5cf6",text=cmp["roc_auc"].round(3),textposition="outside",
                   textfont=dict(color="#f1f5f9")),
            go.Bar(x=cmp["exp_label"],y=cmp["Best"],name="Best Standard Model",
                   marker_color="#22c55e",text=cmp["Best"].round(3),textposition="outside",
                   textfont=dict(color="#f1f5f9")),
        ])
        fig2.update_layout(barmode="group",template="plotly_dark",paper_bgcolor=PAPER_BG,
            plot_bgcolor=PLOT_BG,font=dict(family="Inter",color="#e2e8f0"),height=400,
            margin=dict(l=16,r=16,t=40,b=16),yaxis=dict(range=[0,1.12],gridcolor=GRID_COLOR),
            legend=dict(bgcolor="rgba(15,23,42,.85)",bordercolor="rgba(99,102,241,.3)",borderwidth=1,
                        font=dict(color="#f1f5f9")))
        st.plotly_chart(fig2, width="stretch")

    rules_path = os.path.join(BASE,"results","neurosymbolic_rules.txt")
    if os.path.exists(rules_path):
        with st.expander("Neuro-Symbolic Decision Rules"):
            st.code(open(rules_path).read(), language="text")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE — Live Prediction
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Live Prediction":
    st.markdown("""
    <div class="hero-header">
        <div class="hero-badge">Live Inference</div>
        <h1 class="hero-title">Real-Time Traffic Prediction</h1>
        <p class="hero-subtitle">
            Pick a real flow from a held-out test set (or enter values) and get an
            attack/normal prediction — with SHAP or LIME explanations.
        </p>
    </div>""", unsafe_allow_html=True)

    # ── Configuration row ──────────────────────────────────────────────────
    cfg1,cfg2,cfg3,cfg4 = st.columns(4)
    with cfg1:
        pred_dataset = st.selectbox("Dataset / Domain",["cicids","unsw"],
                                    format_func=lambda x:{"cicids":"CICIDS2017","unsw":"UNSW-NB15"}[x])
    with cfg2:
        pred_model = st.selectbox("Model",["rf","dt","mlp_a","mlp_b","mlp_c"])
    with cfg3:
        pred_seed = st.selectbox("Seed",[42,43,44,45,46])
    with cfg4:
        xai_choice = st.selectbox("XAI Explanation",["None","SHAP","LIME","Neuro-Symbolic"],
                                  help="Neuro-Symbolic shows the IF-THEN rule a decision-tree surrogate "
                                       "of MLP-A (seed 42) applies to this flow.")

    st.markdown("<br>", unsafe_allow_html=True)
    section("Input Network Flow Features")

    # ── Input source ───────────────────────────────────────────────────────
    # The models were trained on standardized (z-score) features and the original
    # scaler was not preserved, so raw units (bytes, seconds, pkt/s) cannot be fed in.
    # Inputs therefore come from real held-out test rows in the same standardized space.
    src1, src2, src3 = st.columns([2,2,1])
    with src1:
        sample_kind = st.selectbox("Input", [
            "Random attack flow (test set)",
            "Random benign flow (test set)",
            "Manual (standardized values)",
        ])
    with src2:
        sample_domain = st.selectbox("Draw flows from", ["cicids","unsw"],
                                     index=0 if pred_dataset=="cicids" else 1,
                                     format_func=lambda x:{"cicids":"CICIDS2017 test","unsw":"UNSW-NB15 test"}[x],
                                     help="Pick the other dataset to try a cross-domain prediction.")
    with src3:
        st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
        if st.button("New sample", width="stretch"):
            st.session_state["sample_draw"] = st.session_state.get("sample_draw", 0) + 1
    draw = st.session_state.get("sample_draw", 0)

    true_label = None
    pool_problem = data_problem(sample_domain, "test") if sample_kind.startswith("Random") else None
    if pool_problem:
        st.error(f"Can't draw test flows: {pool_problem} Falling back to manual input.")
        sample_kind = "Manual (standardized values)"
    if data_problem(pred_dataset) and xai_choice in ("SHAP", "LIME"):
        st.caption(f"Training data for {pred_dataset.upper()} is unavailable, so {xai_choice} uses a "
                   "synthetic standard-normal background. Explanations are approximate.")
    if sample_kind.startswith("Random"):
        pool = load_split(sample_domain, "test")
        want = 1 if "attack" in sample_kind else 0
        cand = pool[pool["label"] == want]
        row  = cand.sample(1, random_state=draw).iloc[0]
        defaults   = [float(row[f]) for f in FEATURES]
        true_label = int(row["label"])
        input_key  = f"{sample_domain}_{want}_{draw}"
    else:
        defaults  = [0.0] * len(FEATURES)   # 0 = training-set mean for every feature
        input_key = "manual"

    st.caption("Values are standardized (z-scores): 0 is the training mean and ±1 is one "
               "standard deviation. The original scaler was not preserved, so raw units "
               "cannot be converted.")

    # ── Feature inputs ────────────────────────────────────────────────────
    feat_vals = {}
    rows = [FEATURES[:3], FEATURES[3:6], FEATURES[6:]]
    for row_feats in rows:
        cols = st.columns(len(row_feats))
        for col, feat in zip(cols, row_feats):
            idx = FEATURES.index(feat)
            with col:
                feat_vals[feat] = st.number_input(
                    FEATURE_LABELS[feat] + " (z)",
                    value=float(defaults[idx]),
                    format="%.4f",
                    key=f"inp_{feat}_{input_key}",
                )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Run Prediction ────────────────────────────────────────────────────
    if st.button("Run Prediction", type="primary", width="stretch"):
        x_arr = np.array([[feat_vals[f] for f in FEATURES]], dtype=np.float32)

        with st.spinner("Running model inference…"):
            prob, label, threshold = predict(pred_dataset, pred_model, pred_seed, x_arr)

        if prob is None:
            st.error("Could not load the model. Check that model files exist.")
        else:
            st.markdown("<br>", unsafe_allow_html=True)

            # ── Result banner ──────────────────────────────────────────────
            r1,r2,r3 = st.columns([2,1,1])
            with r1:
                if label == 1:
                    st.markdown(f"""
                    <div class="result-attack">
                        <div style="font-size:0.85rem;font-weight:700;letter-spacing:0.1em;color:#ef4444;text-transform:uppercase;margin-bottom:0.25rem;">Threat Detected</div>
                        <div style="font-size:1.8rem;font-weight:800;color:#ef4444;margin:.2rem 0 .5rem;">
                            ATTACK DETECTED
                        </div>
                        <div style="font-size:1rem;color:#fca5a5;">
                            Model: <b>{pred_model.upper()}</b> on <b>{pred_dataset.upper()}</b> (seed {pred_seed})
                        </div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-safe">
                        <div style="font-size:0.85rem;font-weight:700;letter-spacing:0.1em;color:#22c55e;text-transform:uppercase;margin-bottom:0.25rem;">Traffic Normal</div>
                        <div style="font-size:1.8rem;font-weight:800;color:#22c55e;margin:.2rem 0 .5rem;">
                            NORMAL TRAFFIC
                        </div>
                        <div style="font-size:1rem;color:#86efac;">
                            Model: <b>{pred_model.upper()}</b> on <b>{pred_dataset.upper()}</b> (seed {pred_seed})
                        </div>
                    </div>""", unsafe_allow_html=True)

            with r2:
                st.metric("Attack Probability", f"{prob:.1%}")
                if true_label is not None:
                    verdict = "correct" if label == true_label else "wrong"
                    st.caption(f"True label: **{'Attack' if true_label else 'Normal'}** — model is {verdict}")
            with r3:
                st.metric("Decision Threshold", f"{threshold:.2f}")

            # ── Probability gauge ──────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            section("Attack Probability Gauge")
            gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob*100,
                delta={"reference": threshold*100, "suffix":"%",
                       "increasing":{"color":"#ef4444"},"decreasing":{"color":"#22c55e"}},
                number={"suffix":"%","font":{"size":36,"color":"#f1f5f9"}},
                gauge={
                    "axis":{"range":[0,100],"tickfont":{"color":"#e2e8f0"}},
                    "bar":{"color":"#ef4444" if label==1 else "#22c55e"},
                    "steps":[
                        {"range":[0,threshold*100],"color":"rgba(34,197,94,.15)"},
                        {"range":[threshold*100,100],"color":"rgba(239,68,68,.15)"},
                    ],
                    "threshold":{"line":{"color":"#f59e0b","width":3},"value":threshold*100},
                    "bgcolor":PLOT_BG,
                    "bordercolor":"rgba(99,102,241,.3)",
                },
            ))
            gauge.update_layout(template="plotly_dark",paper_bgcolor=PAPER_BG,
                                 font=dict(family="Inter",color="#e2e8f0"),
                                 height=280,margin=dict(l=30,r=30,t=20,b=20))
            st.plotly_chart(gauge, width="stretch")

            # ── All-model comparison ───────────────────────────────────────
            section("All Models — Quick Comparison")
            all_model_names = ["rf","dt","mlp_a","mlp_b","mlp_c"]
            results = []
            for mn in all_model_names:
                p,lbl,thr = predict(pred_dataset,mn,pred_seed,x_arr)
                if p is not None:
                    results.append({"Model":mn.upper(),"Attack Prob":p,"Label":"Attack" if lbl==1 else "Normal","Threshold":thr})
            if results:
                rdf = pd.DataFrame(results)
                fig_bar = px.bar(rdf, x="Model", y="Attack Prob", color="Label",
                                 color_discrete_map={"Attack":"#ef4444","Normal":"#22c55e"},
                                 text=rdf["Attack Prob"].apply(lambda v:f"{v:.1%}"))
                fig_bar.update_traces(textposition="outside",textfont_size=12,textfont_color="#f1f5f9",marker_line_width=0)
                fig_bar.add_hline(y=0.5,line_dash="dash",line_color="rgba(245,158,11,.6)",
                                  annotation_text="0.5 reference (labels use each model's own threshold)",annotation_font_color="#f1f5f9")
                fig_bar = chart_layout(fig_bar, height=320)
                fig_bar.update_layout(yaxis=dict(range=[0,1.15],tickformat=".0%"),showlegend=True)
                st.plotly_chart(fig_bar, width="stretch")

            # ── Neuro-Symbolic Explanation ─────────────────────────────────
            if xai_choice == "Neuro-Symbolic":
                section("Neuro-Symbolic Rule Explanation")
                target = sample_domain if true_label is not None else pred_dataset
                ns_exp = neurosymbolic_explanation(pred_dataset, target, x_arr)
                if ns_exp is None:
                    st.warning("No saved surrogate tree found. Run `python -m src.evaluate_neurosymbolic` "
                               "to create `results/neurosymbolic/`.")
                else:
                    mlp_prob, mlp_label, _ = predict(pred_dataset, "mlp_a", 42, x_arr)
                    n1, n2, n3 = st.columns(3)
                    with n1:
                        st.metric("Rule Verdict", "Attack" if ns_exp["label"] else "Normal",
                                  f"leaf attack share {ns_exp['prob']:.0%}", delta_color="off")
                    with n2:
                        if mlp_label is None:
                            st.metric("Agrees with MLP-A?", "—")
                        else:
                            st.metric("Agrees with MLP-A?", "Yes" if mlp_label == ns_exp["label"] else "No",
                                      f"MLP-A says {'Attack' if mlp_label else 'Normal'} ({mlp_prob:.0%})",
                                      delta_color="off")
                    with n3:
                        fid = ns_exp["fidelity"]
                        st.metric("Rule Fidelity (test)", f"{fid:.1%}" if fid is not None else "—",
                                  f"Exp {ns_exp['experiment']}" if ns_exp["experiment"] else None,
                                  delta_color="off")
                    if ns_exp["fidelity"] is not None and ns_exp["fidelity"] < 0.8:
                        st.warning(f"On {target.upper()} test flows these rules agree with MLP-A only "
                                   f"{ns_exp['fidelity']:.0%} of the time, so this rule is a weak "
                                   "description of what the network does here.")
                    if (pred_model, pred_seed) != ("mlp_a", 42):
                        st.caption(f"The surrogate imitates MLP-A seed 42 trained on {pred_dataset.upper()}, "
                                   f"not the selected {pred_model.upper()} seed {pred_seed}.")

                    # Merge repeated splits on one feature into a single range (same logic, easier to read)
                    bounds = {}
                    for f, op, thr, v in ns_exp["conditions"]:
                        lo, hi, _ = bounds.get(f, (None, None, v))
                        if op == ">":
                            lo = thr if lo is None else max(lo, thr)
                        else:
                            hi = thr if hi is None else min(hi, thr)
                        bounds[f] = (lo, hi, v)
                    conds = []
                    for f, (lo, hi, v) in bounds.items():
                        name = f"{FEATURE_LABELS[f]} (z)"
                        if lo is not None and hi is not None:
                            cond = f"{lo:+.4f} < {name} ≤ {hi:+.4f}"
                        elif lo is not None:
                            cond = f"{name} > {lo:+.4f}"
                        else:
                            cond = f"{name} ≤ {hi:+.4f}"
                        conds.append(f"{cond}    [this flow: {v:+.4f}]")
                    lines = ["IF"] + [f"   {'AND ' if i else '    '}{c}" for i, c in enumerate(conds)] \
                          + [f"THEN {'Attack' if ns_exp['label'] else 'Normal'}"
                         f"    ({ns_exp['leaf_samples']:,} training flows reach this leaf)"]
                    st.code("\n".join(lines), language="text")

                    attr_df = pd.DataFrame(ns_exp["attributions"], columns=["Feature","Importance"])
                    attr_df = attr_df[attr_df["Importance"] > 0].sort_values("Importance")
                    if not attr_df.empty:
                        fig_ns = px.bar(attr_df, x="Importance",
                                        y=attr_df["Feature"].map(FEATURE_LABELS), orientation="h",
                                        text=attr_df["Importance"].round(4),
                                        labels={"y":"Feature","Importance":"Split importance on this path"})
                        fig_ns.update_traces(marker_color="#8b5cf6", textposition="outside",
                                             textfont=dict(color="#f1f5f9",size=10))
                        fig_ns = chart_layout(fig_ns, height=300)
                        st.plotly_chart(fig_ns, width="stretch")
                        st.caption("Features the rule tests, weighted by how much each split reduces "
                                   "impurity across the whole tree. Unused features score 0.")

            # ── XAI Explanation ────────────────────────────────────────────
            elif xai_choice != "None":
                section(f"{xai_choice} Feature Explanation")
                with st.spinner(f"Computing {xai_choice} explanation…"):
                    try:
                        if xai_choice == "SHAP":
                            exp_pairs = shap_explanation(pred_dataset, pred_model, pred_seed, x_arr)
                        else:
                            exp_pairs = lime_explanation(pred_dataset, pred_model, pred_seed, x_arr)

                        exp_df = pd.DataFrame(exp_pairs, columns=["Feature", "Attribution"])
                        exp_df["|Attribution|"] = exp_df["Attribution"].abs()
                        exp_df = exp_df.sort_values("|Attribution|", ascending=True)
                        exp_df["Color"] = exp_df["Attribution"].apply(lambda v:"#ef4444" if v>0 else "#22c55e")
                        exp_df["Direction"] = exp_df["Attribution"].apply(
                            lambda v:"→ Pushes toward Attack" if v>0 else "→ Pushes toward Normal")

                        fig_xai = go.Figure()
                        for _,row in exp_df.iterrows():
                            fig_xai.add_trace(go.Bar(
                                y=[FEATURE_LABELS.get(row["Feature"],row["Feature"])],
                                x=[row["Attribution"]],
                                orientation="h",
                                marker_color=row["Color"],
                                name=row["Direction"],
                                showlegend=False,
                                text=f"{row['Attribution']:+.4f}",
                                textfont=dict(color="#f1f5f9",size=10),
                                textposition="outside",
                            ))
                        fig_xai.add_vline(x=0, line_color="rgba(255,255,255,.3)", line_width=1.5)
                        fig_xai = chart_layout(fig_xai, height=360)
                        fig_xai.update_layout(
                            xaxis_title=f"{xai_choice} Attribution Value",
                            yaxis_title="Feature",
                            barmode="overlay",
                        )
                        st.plotly_chart(fig_xai, width="stretch")

                        # legend
                        st.markdown("""
                        <div style="display:flex;gap:2rem;font-size:.82rem;color:#cbd5e1;margin-top:.5rem;">
                            <span><span style="color:#ef4444;font-weight:700;">■</span>
                                Positive → pushes prediction toward <b>Attack</b></span>
                            <span><span style="color:#22c55e;font-weight:700;">■</span>
                                Negative → pushes prediction toward <b>Normal</b></span>
                        </div>""", unsafe_allow_html=True)

                        with st.expander("Raw attribution values"):
                            st.dataframe(exp_df[["Feature","Attribution","Direction"]]
                                         .rename(columns={"Feature":"Feature","Attribution":"Attribution","Direction":"Direction"}),
                                         width="stretch", hide_index=True)
                    except Exception as e:
                        st.warning(f"{xai_choice} explanation failed: {e}")

    # ── Feature description table ──────────────────────────────────────────
    with st.expander("Feature Descriptions"):
        feat_info = pd.DataFrame([
            ["flow_duration","Total duration of the network flow in seconds"],
            ["fwd_packets","Number of packets sent in the forward direction"],
            ["bwd_packets","Number of packets sent in the backward direction"],
            ["fwd_bytes","Total bytes sent in the forward direction"],
            ["bwd_bytes","Total bytes sent in the backward direction"],
            ["fwd_to_bwd_pkt_ratio","Ratio of forward to backward packet counts"],
            ["fwd_to_bwd_byte_ratio","Ratio of forward to backward byte counts"],
            ["bytes_per_packet","Average bytes per packet across the flow"],
            ["packets_per_second","Average packet rate of the flow"],
        ], columns=["Feature","Description"])
        st.dataframe(feat_info, width="stretch", hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE — Visual Gallery
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Visual Gallery":
    st.markdown("""
    <div class="hero-header">
        <div class="hero-badge">Gallery</div>
        <h1 class="hero-title">Visual Gallery</h1>
        <p class="hero-subtitle">Pre-computed visualisations: feature distributions, correlations, attack analysis and publication figures.</p>
    </div>""", unsafe_allow_html=True)

    images = load_result_images()
    cats = {
        "Publication Figures": {k:v for k,v in images.items() if k.startswith("figures/")},
        "Feature Importance":  {k:v for k,v in images.items() if "feature_importance" in k},
        "Distributions":       {k:v for k,v in images.items() if "distribution" in k or "label" in k},
        "Correlations":        {k:v for k,v in images.items() if "correlation" in k},
        "Attack Analysis":     {k:v for k,v in images.items() if "attack" in k or "traffic" in k or "flag" in k},
    }
    shown = set()
    for cat_title, cat_imgs in cats.items():
        if not cat_imgs: continue
        section(cat_title)
        img_list = list(cat_imgs.items())
        for i in range(0, len(img_list), 2):
            cols = st.columns(2)
            for j,(fname,fpath) in enumerate(img_list[i:i+2]):
                shown.add(fname)
                label = fname.replace("figures/","").replace("_"," ").replace(".png","").title()
                with cols[j]:
                    st.image(fpath, caption=label, width="stretch")

    remaining = {k:v for k,v in images.items() if k not in shown}
    if remaining:
        section("Other Plots")
        img_list = list(remaining.items())
        for i in range(0,len(img_list),2):
            cols = st.columns(2)
            for j,(fname,fpath) in enumerate(img_list[i:i+2]):
                with cols[j]:
                    st.image(fpath, caption=fname.replace("_"," ").replace(".png","").title(), width="stretch")

    if not images:
        st.info("No PNG images found in `results/`. Run `python run_all.py` first.")

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border-color:rgba(99,102,241,.15);margin:3rem 0 1rem;">
<div style="text-align:center;color:#64748b;font-size:.75rem;padding-bottom:1rem;">
    Explainable IDS · SHAP &amp; LIME Cross-Dataset Reliability ·
    <span style="color:#6366f1;">CICIDS2017</span> &amp;
    <span style="color:#3b82f6;">UNSW-NB15</span>
</div>
""", unsafe_allow_html=True)
