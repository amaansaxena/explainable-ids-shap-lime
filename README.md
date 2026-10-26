# Explainable IDS: SHAP/LIME Cross-Dataset Reliability

This repository evaluates binary intrusion-detection models on aligned CICIDS2017 and UNSW-NB15 features and caches SHAP/LIME explanations for in-domain and cross-dataset experiments.

## Setup
1. `git lfs pull` (the aligned CSVs are stored with Git LFS)
2. `python -m pip install -r requirements.txt`
3. Run the smoke pipeline: `python run_all.py --smoke`
4. Run the full pipeline: `python run_all.py`
5. Run the neuro-symbolic surrogate (needs `results/xai_summary.csv` from step 4): `python -m src.evaluate_neurosymbolic`
6. Regenerate figures: `python -m src.visualize_results`
7. Launch the dashboard: `streamlit run app.py`

The dashboard's Live Prediction page uses real held-out test flows in standardized (z-score) units,
because the models were trained on standardized features and the original scaler was not preserved.

## Datasets
- CICIDS2017
- UNSW-NB15

## Project Structure
- `data/raw/` — place downloaded datasets here (see Datasets section)
- `data/preprocessed/aligned/` — committed aligned inputs and finalized representations
- `notebooks/` — analysis notebooks, one per pipeline phase
- `src/` — reusable Python modules (data loading, preprocessing, model, explain)
- `models/` — saved trained model weights (gitignored)
- `results/` — output plots and metric tables

## Experiments

- A: CICIDS to CICIDS test
- B: UNSW to UNSW test
- C: CICIDS to UNSW test
- D: UNSW to CICIDS test

## Limitations

- The committed aligned data contains train/test feature-row overlaps. Finalization quarantines overlapping test rows in new files; original inputs are preserved.
  The overlap is small in unique feature vectors (CICIDS 3,512; UNSW-NB15 2,037) but these vectors repeat many times,
  so the number of removed test rows is much larger: CICIDS 32,860 → 24,683 (8,177 removed, 29 attacks) and
  UNSW-NB15 82,332 → 45,816 (36,516 removed, 27,715 attacks). The UNSW-NB15 test attack rate therefore falls from 55% to 38%.
- Raw source tables and scaler artifacts are unavailable, so `raw`, `log`, and `scaled` finalized files preserve recovered aligned model-space values. Raw-unit explanation claims are not valid until raw data is restored.
- CICIDS represents one day with few attacks.
- Only nine shared features are available and several are strongly correlated.
- Source-vs-domain scaling is exposed, but source scaling cannot be reconstructed without the original scaler.
- Explanation sampling is class-balanced rather than prevalence-balanced.

## Neuro-symbolic surrogate

`src/evaluate_neurosymbolic.py` fits a depth-5 decision tree to MLP-A's decisions (global rule extraction) and reports
fidelity (agreement with the MLP) alongside accuracy metrics. Rule-path attributions are computed on the same
class-balanced instances that SHAP/LIME explain, so the comparison is like-for-like.

## Results

Generated outputs are written under `results/`. Numeric interpretation is intentionally left to the report author.