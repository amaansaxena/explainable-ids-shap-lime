"""Single source of truth for the IDS pipeline configuration."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ALIGNED_DIR = DATA_DIR / "preprocessed" / "aligned"
# IDS_OUTPUT_DIR redirects models and results (run_all.py --smoke uses it so test runs never overwrite real outputs).
OUTPUT_ROOT = Path(os.environ["IDS_OUTPUT_DIR"]).resolve() if os.environ.get("IDS_OUTPUT_DIR") else ROOT
MODELS_DIR = OUTPUT_ROOT / "models"
SCALERS_DIR = MODELS_DIR / "scalers"
RESULTS_DIR = OUTPUT_ROOT / "results"
XAI_CACHE_DIR = RESULTS_DIR / "xai_cache"
DATASETS = ("cicids", "unsw")
FEATURES = ["flow_duration", "fwd_packets", "bwd_packets", "fwd_bytes", "bwd_bytes", "fwd_to_bwd_pkt_ratio", "fwd_to_bwd_byte_ratio", "bytes_per_packet", "packets_per_second"]
LABEL = "label"
SEEDS = [42, 43, 44, 45, 46]
SCALING_MODE = "per_domain"
ALLOWED_SCALING_MODES = ("per_domain", "source")
VAL_FRACTION = 0.15
HEAVY_TAILED_FEATURES = FEATURES[:]
SMALL_POSITIVE_WARNING = 30


def dataset_key(name: str) -> str:
    """Normalize public dataset names."""
    try:
        return {"cicids": "CICIDS", "unsw": "UNSW_NB15"}[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset: {name}") from exc


def aligned_path(name: str, split: str) -> Path:
    return ALIGNED_DIR / f"{dataset_key(name)}_{split}_aligned.csv"


def finalized_path(name: str, representation: str) -> Path:
    return ALIGNED_DIR / f"{dataset_key(name)}_{representation}.csv"