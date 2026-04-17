# Causal Counterfactual Explanation Engine

A causal counterfactual explanation system for credit lending decisions, built with a trained XGBoost model, structural causal reasoning, and interactive UI support.

## What it does

- Predicts credit decision outcomes for applicant profiles.
- Generates candidate counterfactual profiles that are more likely to receive approval.
- Uses a structural causal model to enforce realistic feature propagation.
- Presents explanations with SHAP importance and counterfactual validity metadata.

## Features

- `demo/app.py` Streamlit UI with a rich decision dashboard.
- Gradio fallback interface via `python demo/app.py --ui gradio`.
- Counterfactual generation using an MILP-based candidate generator and DPP diversity selection.
- Structural causal propagation using `causal/structural_equations.py`.
- Evaluation utilities in `evaluation/benchmark.py` and `evaluation/metrics.py`.

## Repository structure

- `data/` - feature metadata, preprocessing logic, and sample CSV data.
- `models/` - model training logic, saved model artifact, and prediction wrapper.
- `causal/` - causal graph and structural equation model definitions.
- `counterfactuals/` - counterfactual generation logic and diversity selector.
- `demo/` - interactive UI entry point.
- `evaluation/` - benchmark and validation utilities.
- `tests/` - project tests.

## Setup

1. Create and activate a Python virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

3. Prepare data and train the model:
   ```powershell
   python -m data.preprocess
   python -m models.train
   ```

4. Estimate causal structure if needed:
   ```powershell
   python -m causal.scm_estimator
   ```

## Running the demo

- Streamlit UI (preferred):
  ```powershell
  streamlit run demo/app.py
  ```

- Python launcher with fallback:
  ```powershell
  python demo/app.py --ui streamlit
  python demo/app.py --ui gradio
  ```

## Testing

Run unit tests with:
```powershell
python -m pytest -q
```

## Notes

- The model artifact is stored in `models/xgb_model.joblib`.
- The `data/feature_meta.json` file defines target and bounds used by the UI and engine.
- If you use the Gradio UI, counterfactual generation remains available through the Streamlit interface.

## License

Licensed under the MIT License. See `LICENSE`.
