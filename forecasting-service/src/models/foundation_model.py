"""
Chronos-2 foundation model wrapper.
Zero-shot: forecasts ANY series (including held-out) from raw history alone.
No training, no feature engineering — just feed it target values.

torch is imported here at module level (and this module is imported early
by api/main.py) so it loads its DLLs on the main thread before uvicorn's
event loop or any worker thread exists. On Windows, torch fails with
OSError WinError 1114 (c10.dll init failure) if its first import happens
after any thread has been spawned in the process — which happens on the
very first request once FastAPI/Starlette offloads blocking work (e.g.
UploadFile.read()) to a worker thread. chronos stays lazy since it's
heavy and only needed once the endpoint is actually called.
"""
import numpy as np
import torch
from src import config as C

# Cache pipelines per model name, since /tune can sweep across variants
# (e.g. chronos-bolt-tiny vs -small vs -base) within one process.
_pipelines = {}


def get_pipeline(model_name=None):
    model_name = model_name or C.CHRONOS_MODEL
    if model_name not in _pipelines:
        from chronos import BaseChronosPipeline
        print(f"Loading Chronos model '{model_name}' (first time downloads weights)...")
        torch.manual_seed(C.SEED)
        # BaseChronosPipeline dispatches to the pipeline class the model's own
        # config asks for (e.g. ChronosBoltPipeline for chronos-bolt-* models,
        # which uses direct quantile regression, not the older sampling API).
        _pipelines[model_name] = BaseChronosPipeline.from_pretrained(
            model_name,
            device_map=C.CHRONOS_DEVICE,
            dtype=torch.float32,
        )
        print(f"Chronos model '{model_name}' loaded.")
    return _pipelines[model_name]


def forecast(history_values, horizon, model_name=None, context_length=None):
    pipe = get_pipeline(model_name)
    context_length = context_length or C.CHRONOS_CONTEXT
    values = np.array(history_values, dtype=np.float32)
    mask = np.isnan(values)
    if mask.any():
        nans = np.where(mask)[0]
        ok = np.where(~mask)[0]
        if len(ok) > 0:
            values[nans] = np.interp(nans, ok, values[ok])
    if len(values) > context_length:
        values = values[-context_length:]
    context = torch.tensor(values).unsqueeze(0)
    with torch.no_grad():
        quantile_preds, _ = pipe.predict_quantiles(
            context, prediction_length=horizon, quantile_levels=C.QUANTILES,
        )
    quantile_preds = quantile_preds.squeeze(0).numpy()  # [horizon, len(QUANTILES)]
    result = {}
    for i, q in enumerate(C.QUANTILES):
        result[q] = quantile_preds[:, i].tolist()
    return result


def evaluate(model_name=None, context_length=None, horizon=24, max_series=None):
    """Zero-shot backtest on the held-out test split: for each series, forecast
    the last `horizon` points from the history before them and score against
    the actual values. Chronos has no trainable weights, so this is how its
    inference-time "hyperparameters" (model variant, context length) get
    tuned and compared in MLflow against LightGBM's metrics."""
    import pandas as pd

    test_path = C.DATA_PROCESSED / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(
            "Test data not found. Run POST /train or scripts/prepare_data first."
        )

    test = pd.read_csv(test_path)
    test[C.TIME_COL] = pd.to_datetime(test[C.TIME_COL])

    series_ids = sorted(test[C.ID_COL].unique())
    if max_series:
        series_ids = series_ids[:max_series]

    actuals, points, lo90, hi90, lo50, hi50 = [], [], [], [], [], []

    for sid in series_ids:
        s = test[test[C.ID_COL] == sid].sort_values(C.TIME_COL)
        if len(s) < horizon + 20:
            continue
        history = s[C.TARGET].values[:-horizon]
        actual = s[C.TARGET].values[-horizon:]

        quantiles = forecast(history, horizon, model_name=model_name, context_length=context_length)
        actuals.extend(actual.tolist())
        points.extend(quantiles[0.5])
        lo90.extend(quantiles[0.05])
        hi90.extend(quantiles[0.95])
        lo50.extend(quantiles[0.25])
        hi50.extend(quantiles[0.75])

    if not actuals:
        raise ValueError(
            f"No series had at least {horizon + 20} rows to backtest with horizon={horizon}."
        )

    actual = np.array(actuals)
    point = np.array(points)
    mae = float(np.mean(np.abs(actual - point)))
    rmse = float(np.sqrt(np.mean((actual - point) ** 2)))
    cov_90 = float(np.mean((actual >= np.array(lo90)) & (actual <= np.array(hi90))))
    cov_50 = float(np.mean((actual >= np.array(lo50)) & (actual <= np.array(hi50))))

    return {
        "mae": mae, "rmse": rmse, "cov_50": cov_50, "cov_90": cov_90,
        "n_series": len(series_ids),
    }