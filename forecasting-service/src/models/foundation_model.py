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

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from chronos import BaseChronosPipeline
        print("Loading Chronos-2 model (first time downloads weights)...")
        torch.manual_seed(C.SEED)
        # BaseChronosPipeline dispatches to the pipeline class the model's own
        # config asks for (e.g. ChronosBoltPipeline for chronos-bolt-* models,
        # which uses direct quantile regression, not the older sampling API).
        _pipeline = BaseChronosPipeline.from_pretrained(
            C.CHRONOS_MODEL,
            device_map=C.CHRONOS_DEVICE,
            dtype=torch.float32,
        )
        print("Chronos-2 loaded.")
    return _pipeline


def forecast(history_values, horizon):
    pipe = get_pipeline()
    values = np.array(history_values, dtype=np.float32)
    mask = np.isnan(values)
    if mask.any():
        nans = np.where(mask)[0]
        ok = np.where(~mask)[0]
        if len(ok) > 0:
            values[nans] = np.interp(nans, ok, values[ok])
    if len(values) > C.CHRONOS_CONTEXT:
        values = values[-C.CHRONOS_CONTEXT:]
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