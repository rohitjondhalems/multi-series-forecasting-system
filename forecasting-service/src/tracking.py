"""
MLflow tracking helpers (Azure ML backend).

Both models log to the same experiment with a "model" tag so LightGBM and
Chronos runs can be compared side by side in the MLflow UI.
"""
from contextlib import contextmanager
from typing import Optional

import mlflow

from src import config as C

EXPERIMENT_NAME = "multi-series-forecasting"

_initialized = False


def _init():
    global _initialized
    if _initialized:
        return
    if not C.MLFLOW_TRACKING_URI:
        raise RuntimeError("MLFLOW_TRACKING_URI is not set in .env")
    mlflow.set_tracking_uri(C.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    _initialized = True


@contextmanager
def start_run(run_name: Optional[str], model_tag: str, params: dict):
    """Start an MLflow run tagged with which model it belongs to, and log its
    hyperparameters. Log metrics from inside the `with` block via
    mlflow.log_metrics()."""
    _init()
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("model", model_tag)
        mlflow.log_params(params)
        yield run
