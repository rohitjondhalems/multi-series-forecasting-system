"""
FastAPI forecasting server.

Endpoints:
    GET  /health              - check if models are loaded
    GET  /models              - list available models and results
    POST /train               - retrain from data/raw/ files
    POST /forecast/global     - forecast using LightGBM (exog variant)
    POST /forecast/global/noexog - forecast using LightGBM (no covariates)
    POST /evaluate/global     - evaluate on test set

Usage:
    uvicorn api.main:app --reload --port 8000
    Then visit: http://localhost:8000/docs
"""
import torch  
import io
import json
import pickle
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from src import config as C
from src import features as F
from src.models import foundation_model as fm


# ──────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────
class ModelBundle:
    """Holds the 5 quantile models + feature columns for one variant."""
    def __init__(self, variant_dir):
        self.models = {}
        for q in C.QUANTILES:
            path = variant_dir / f"quantile_{q}.pkl"
            if not path.exists():
                raise FileNotFoundError(f"Model not found: {path}")
            with open(path, "rb") as f:
                self.models[q] = pickle.load(f)

        feat_path = variant_dir / "feature_columns.pkl"
        with open(feat_path, "rb") as f:
            self.feature_cols = pickle.load(f)

    def predict_row(self, X):
        """Predict all quantiles for one or more rows. Returns dict of arrays."""
        preds = {}
        for q in C.QUANTILES:
            preds[q] = self.models[q].predict(X)
        return preds


# Cache: loaded once at startup
_bundles = {}


def get_bundle(use_exog: bool) -> ModelBundle:
    tag = "exog" if use_exog else "noexog"
    if tag not in _bundles:
        variant_dir = C.MODELS_DIR / tag
        if not variant_dir.exists():
            raise HTTPException(409, f"Models not trained yet ({tag}). Hit POST /train first.")
        _bundles[tag] = ModelBundle(variant_dir)
    return _bundles[tag]


# ──────────────────────────────────────────────
# File parsing (same delimiter logic as prepare_data)
# ──────────────────────────────────────────────
def parse_upload(raw_bytes: bytes) -> pd.DataFrame:
    """Parse uploaded CSV/TSV, validate schema."""
    text = raw_bytes.decode("utf-8-sig")
    lines = text.strip().splitlines()
    if not lines:
        raise HTTPException(400, "Empty file")
    header = lines[0]
    sep = "\t" if header.count("\t") >= header.count(",") else ","
    df = pd.read_csv(io.StringIO(text), sep=sep)

    required = [C.TIME_COL, C.ID_COL, C.TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(422, f"Missing required columns: {missing}")

    df[C.TIME_COL] = pd.to_datetime(df[C.TIME_COL])
    df = df.sort_values([C.ID_COL, C.TIME_COL]).reset_index(drop=True)
    return df


# ──────────────────────────────────────────────
# Recursive forecasting (the key function)
# ──────────────────────────────────────────────
def recursive_forecast(bundle: ModelBundle, history: pd.DataFrame,
                       horizon: int, use_exog: bool) -> list:
    """
    Forecast the next `horizon` hours after the history ends.
    
    How it works:
    1. Take the last row's timestamp
    2. Append a new row with timestamp+1h, target=NaN
    3. Build features from the whole history + new row
    4. Predict all quantiles for the new row
    5. Fill the new row's target with the median prediction
    6. Repeat for the next hour
    
    This is "recursive" because each step feeds its own prediction
    back as history for the next step.
    """
    work = history.copy()
    sid = work[C.ID_COL].iloc[-1]
    last_ts = work[C.TIME_COL].max()
    step = pd.Timedelta(hours=1)

    # For covariates in the future: persist last known value
    last_covs = {}
    if use_exog:
        for c in C.COV_NUM_COLS:
            if c in work.columns:
                last_covs[c] = work[c].dropna().iloc[-1] if work[c].notna().any() else 0.0
        if C.COV_CAT_COL in work.columns:
            last_covs[C.COV_CAT_COL] = work[C.COV_CAT_COL].iloc[-1]

    forecasts = []

    for i in range(1, horizon + 1):
        ts = last_ts + i * step

        # Build a new row: future timestamp, unknown target
        new_row = {C.TIME_COL: ts, C.ID_COL: sid, C.TARGET: np.nan}

        # Persist covariates (or fill with last known)
        if use_exog:
            for c in C.COV_NUM_COLS:
                new_row[c] = last_covs.get(c, 0.0)
            new_row[C.COV_CAT_COL] = last_covs.get(C.COV_CAT_COL, "__missing__")

        # Aux columns (not used in features, but keep schema consistent)
        for c in C.AUX_COLS:
            if c in work.columns:
                new_row[c] = np.nan

        # Append future row to working history
        work = pd.concat([work, pd.DataFrame([new_row])], ignore_index=True)

        # Build features on the FULL working history (same function as training)
        featured = F.create_features(work.copy())

        # Get the last row's features (the one we just appended)
        last_featured = featured.iloc[[-1]]
        X = last_featured[bundle.feature_cols].astype("float32")

        # Predict all quantiles
        preds = bundle.predict_row(X)

        # Sort quantiles to prevent crossing (lower quantile should be <= higher)
        sorted_vals = np.sort([float(preds[q][0]) for q in C.QUANTILES])
        pred_dict = dict(zip(C.QUANTILES, sorted_vals))

        # Feed median back as the "known" target for next step's lags
        work.loc[work.index[-1], C.TARGET] = pred_dict[0.5]

        forecasts.append({
            "timestamp": str(ts),
            "step": i,
            "point": round(pred_dict[0.5], 4),
            "lower_90": round(pred_dict[0.05], 4),
            "upper_90": round(pred_dict[0.95], 4),
            "lower_50": round(pred_dict[0.25], 4),
            "upper_50": round(pred_dict[0.75], 4),
        })

    return forecasts


# ──────────────────────────────────────────────
# Response schemas
# ──────────────────────────────────────────────
class ForecastResponse(BaseModel):
    model: str
    series_id: str
    horizon: int
    interval: int
    use_exog: bool
    n_history_rows: int
    forecast: list


class EvalResponse(BaseModel):
    model: str
    use_exog: bool
    metrics: dict


# ──────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────
app = FastAPI(
    title="Multi-Series Forecasting API",
    version="1.0.0",
    description=(
        "Two-model forecasting system: LightGBM (global, with/without covariates).\n\n"
        "**Click order in /docs:**\n"
        "1. POST /train (if models not yet trained)\n"
        "2. POST /forecast/global (upload a CSV, get future predictions)\n"
    ),
)


@app.on_event("startup")
async def startup():
    """Try to load models at startup (non-fatal if not trained yet)."""
    for tag in ("exog", "noexog"):
        try:
            _bundles[tag] = ModelBundle(C.MODELS_DIR / tag)
            print(f"Loaded {tag} models ({len(_bundles[tag].feature_cols)} features)")
        except Exception as e:
            print(f"Models ({tag}) not loaded: {e}. Train via POST /train.")


# ──────────────────────────────────────────────
# Health + Info
# ──────────────────────────────────────────────
@app.get("/health", tags=["info"])
def health():
    return {
        "status": "ok",
        "exog_loaded": "exog" in _bundles,
        "noexog_loaded": "noexog" in _bundles,
    }


@app.get("/models", tags=["info"])
def models_info():
    results_path = C.MODELS_DIR / "results.json"
    results = None
    if results_path.exists():
        results = json.loads(results_path.read_text())
    return {
        "variants": list(_bundles.keys()),
        "quantiles": C.QUANTILES,
        "results": results,
    }


# ──────────────────────────────────────────────
# Train (triggers the same logic as train_global.py)
# ──────────────────────────────────────────────
@app.post("/train", tags=["train"])
def train():
    """Train LightGBM quantile models from data in data/raw/.
    Runs the same pipeline as `python -m src.train_global`."""
    from src import train_global
    train_global.main()

    # Reload into memory
    _bundles.clear()
    for tag in ("exog", "noexog"):
        _bundles[tag] = ModelBundle(C.MODELS_DIR / tag)

    return {"status": "trained", "variants": list(_bundles.keys())}


# ──────────────────────────────────────────────
# Forecast endpoints
# ──────────────────────────────────────────────
@app.post("/forecast/global", response_model=ForecastResponse, tags=["forecast"])
async def forecast_global(
    file: UploadFile = File(..., description="History CSV/TSV (must have timestamp, series_id, target)"),
    horizon: int = Form(24, description="Hours ahead to forecast (1-720)"),
    interval: int = Form(90, description="Interval width: 50 or 90"),
    use_exog: bool = Form(True, description="Use covariates (cov_1..5, cov_cat)?"),
):
    """
    Upload a CSV with historical data → get future forecasts with uncertainty bands.

    The model forecasts the next `horizon` hours AFTER the last timestamp in your file.
    It uses the same feature engineering as training (no train-serve skew).

    **What the response contains:**
    - point: median forecast (quantile 0.50)
    - lower_50 / upper_50: 50% prediction interval
    - lower_90 / upper_90: 90% prediction interval

    **Exogenous toggle:** set use_exog=false to see forecast WITHOUT covariates.
    Compare the interval widths — covariates should tighten the bands.
    """
    # Validate params
    if horizon < 1 or horizon > 720:
        raise HTTPException(422, "horizon must be between 1 and 720")
    if interval not in (50, 90):
        raise HTTPException(422, "interval must be 50 or 90")

    # Parse file
    raw = await file.read()
    history = parse_upload(raw)

    if len(history) < 50:
        raise HTTPException(422, f"Need at least 50 rows of history, got {len(history)}")

    # Get model
    bundle = get_bundle(use_exog)

    # Forecast
    forecasts = recursive_forecast(bundle, history, horizon, use_exog)

    # Filter to requested interval
    if interval == 50:
        for f in forecasts:
            f.pop("lower_90", None)
            f.pop("upper_90", None)
    else:
        for f in forecasts:
            f.pop("lower_50", None)
            f.pop("upper_50", None)

    series_id = str(history[C.ID_COL].iloc[-1])
    return ForecastResponse(
        model="global_lgbm",
        series_id=series_id,
        horizon=horizon,
        interval=interval,
        use_exog=use_exog,
        n_history_rows=len(history),
        forecast=forecasts,
    )


# ──────────────────────────────────────────────
# Evaluate endpoint
# ──────────────────────────────────────────────
@app.post("/evaluate/global", response_model=EvalResponse, tags=["evaluate"])
async def evaluate_global(
    use_exog: bool = Form(True),
):
    """Evaluate the global model on the saved test set."""
    test_path = C.DATA_PROCESSED / "test.csv"
    if not test_path.exists():
        raise HTTPException(409, "Test data not found. Run POST /train first.")

    test = pd.read_csv(test_path)
    bundle = get_bundle(use_exog)

    test_featured = F.create_features(test.copy())
    X = test_featured[bundle.feature_cols].astype("float32")
    y = test_featured[C.TARGET].astype("float32")

    # Point metrics
    y_pred = bundle.models[0.5].predict(X)
    mae = float(np.mean(np.abs(y - y_pred)))
    rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))

    # Coverage
    lo_90 = bundle.models[0.05].predict(X)
    hi_90 = bundle.models[0.95].predict(X)
    lo_50 = bundle.models[0.25].predict(X)
    hi_50 = bundle.models[0.75].predict(X)
    cov_90 = float(np.mean((y >= lo_90) & (y <= hi_90)))
    cov_50 = float(np.mean((y >= lo_50) & (y <= hi_50)))

    tag = "exog" if use_exog else "noexog"
    return EvalResponse(
        model=f"global_lgbm_{tag}",
        use_exog=use_exog,
        metrics={"mae": mae, "rmse": rmse, "cov_50": cov_50, "cov_90": cov_90},
    )
    
def recursive_forecast(bundle: ModelBundle, history: pd.DataFrame,
                horizon: int, use_exog: bool) -> list:
    """
    Forecast the next `horizon` hours by building features for each
    future step directly from recent history. Much faster than rebuilding
    all features each step, and avoids the NaN-drop bug.
    """
    work = history.copy()
    work[C.TIME_COL] = pd.to_datetime(work[C.TIME_COL])
    work = work.sort_values(C.TIME_COL).reset_index(drop=True)

    sid = work[C.ID_COL].iloc[-1]
    last_ts = work[C.TIME_COL].max()
    step = pd.Timedelta(hours=1)

    # Collect recent target values into a list we can append to
    # (much faster than slicing a dataframe each step)
    recent_targets = work[C.TARGET].tolist()

    # Last known covariate values (persisted into the future)
    last_covs = {}
    if use_exog:
        for c in C.COV_NUM_COLS:
            if c in work.columns:
                vals = work[c].dropna()
                last_covs[c] = float(vals.iloc[-1]) if len(vals) > 0 else 0.0
        if C.COV_CAT_COL in work.columns:
            last_covs[C.COV_CAT_COL] = work[C.COV_CAT_COL].iloc[-1]

    forecasts = []

    for i in range(1, horizon + 1):
        ts = last_ts + i * step

        # ── Build features for this ONE row directly ──

        row = {}

        # Calendar features (from the future timestamp)
        row["hour"] = ts.hour
        row["day_of_week"] = ts.dayofweek
        row["day_of_month"] = ts.day
        row["month"] = ts.month
        row["quarter"] = (ts.month - 1) // 3 + 1
        row["is_weekend"] = 1 if ts.dayofweek >= 5 else 0

        # Lag features (from the tail of recent_targets)
        n = len(recent_targets)
        row["lag_1"] = recent_targets[n - 1] if n >= 1 else np.nan
        row["lag_2"] = recent_targets[n - 2] if n >= 2 else np.nan
        row["lag_3"] = recent_targets[n - 3] if n >= 3 else np.nan
        row["lag_24"] = recent_targets[n - 24] if n >= 24 else np.nan
        row["lag_48"] = recent_targets[n - 48] if n >= 48 else np.nan

        # Rolling stats (from the last N values, shifted by 1)
        # "shifted by 1" means we exclude the current step (which doesn't
        # have a real target yet) — but lag_1 IS the previous step's value,
        # so the window starts from index n-1 going backward.
        def _rolling(window):
            start = max(0, n - window)
            vals = [v for v in recent_targets[start:n] if not np.isnan(v)]
            if len(vals) == 0:
                return np.nan, np.nan
            mean = np.mean(vals)
            std = np.std(vals, ddof=1) if len(vals) >= 2 else np.nan
            return mean, std

        row["rolling_mean_7"], row["rolling_std_7"] = _rolling(7)
        row["rolling_mean_24"], row["rolling_std_24"] = _rolling(24)

        # Covariates (persisted last known values)
        if use_exog:
            for c in C.COV_NUM_COLS:
                row[c] = last_covs.get(c, 0.0)
            row[C.COV_CAT_COL] = last_covs.get(C.COV_CAT_COL, "__missing__")

        # ── Predict ──

        row_df = pd.DataFrame([row])

        # Align to trained feature columns, fill missing with NaN
        for col in bundle.feature_cols:
            if col not in row_df.columns:
                row_df[col] = np.nan

        # Encode cov_cat as integer (LightGBM uses int codes internally)
        # This avoids the categorical metadata mismatch between train and predict
        if C.COV_CAT_COL in bundle.feature_cols and C.COV_CAT_COL in row_df.columns:
            row_df[C.COV_CAT_COL] = pd.Categorical(
                row_df[C.COV_CAT_COL].astype("string").fillna("__missing__")
            ).codes.astype("float32")

        X = row_df[bundle.feature_cols].astype("float32")

        # Predict all quantiles
        preds = bundle.predict_row(X)
        
        # Sort to prevent quantile crossing
        sorted_vals = np.sort([float(preds[q][0]) for q in C.QUANTILES])
        pred_dict = dict(zip(C.QUANTILES, sorted_vals))

        # Feed median prediction back as "known" target for next step's lags
        recent_targets.append(pred_dict[0.5])

        forecasts.append({
            "timestamp": str(ts),
            "step": i,
            "point": round(pred_dict[0.5], 4),
            "lower_90": round(pred_dict[0.05], 4),
            "upper_90": round(pred_dict[0.95], 4),
            "lower_50": round(pred_dict[0.25], 4),
            "upper_50": round(pred_dict[0.75], 4),
        })

    return forecasts


@app.get("/", tags=["info"])
def root():
    return {
        "name": "Multi-Series Forecasting API",
        "version": "1.0.0",
        "docs": "/docs",
        "click_order": [
            "1. POST /train (if models not trained)",
            "2. POST /forecast/global (upload CSV, get predictions)",
            "3. POST /evaluate/global (check metrics on test set)",
        ],
    }
    

# Add this endpoint (after your /forecast/global endpoint)
@app.post("/forecast/foundation", response_model=ForecastResponse, tags=["forecast"])
async def forecast_foundation(
    file: UploadFile = File(..., description="History CSV/TSV"),
    horizon: int = Form(24, description="Hours ahead to forecast (1-168)"),
    interval: int = Form(90, description="50 or 90"),
):
    """
    Forecast using Chronos-2 (pretrained transformer, zero-shot).

    Unlike LightGBM, Chronos-2:
    - Needs NO training — it's already pretrained by Amazon
    - Reads raw target values — no feature engineering
    - Works on ANY series, even ones it's never seen
    - Does NOT use covariates (it's univariate)

    First call downloads model weights (~500MB, one-time).
    Subsequent calls are faster (~5-30 seconds depending on history length).
    """
    if horizon < 1 or horizon > 168:
        raise HTTPException(422, "horizon must be between 1 and 168 for foundation model")
    if interval not in (50, 90):
        raise HTTPException(422, "interval must be 50 or 90")

    # Parse file
    raw = await file.read()
    history = parse_upload(raw)

    if len(history) < 50:
        raise HTTPException(422, f"Need at least 50 rows of history, got {len(history)}")

    # Extract just the target values (Chronos doesn't use features)
    target_values = history[C.TARGET].tolist()
    series_id = str(history[C.ID_COL].iloc[-1])
    last_ts = pd.to_datetime(history[C.TIME_COL]).max()

    # Forecast
    quantiles = fm.forecast(target_values, horizon)

    # Build response
    step = pd.Timedelta(hours=1)
    forecasts = []
    for i in range(horizon):
        ts = last_ts + (i + 1) * step
        point = round(quantiles[0.5][i], 4)

        row = {"timestamp": str(ts), "step": i + 1, "point": point}

        if interval == 90:
            row["lower_90"] = round(quantiles[0.05][i], 4)
            row["upper_90"] = round(quantiles[0.95][i], 4)
        else:
            row["lower_50"] = round(quantiles[0.25][i], 4)
            row["upper_50"] = round(quantiles[0.75][i], 4)

        forecasts.append(row)

    return ForecastResponse(
        model="chronos2",
        series_id=series_id,
        horizon=horizon,
        interval=interval,
        use_exog=False,    # Chronos-2 is univariate — no covariates
        n_history_rows=len(history),
        forecast=forecasts,
    )