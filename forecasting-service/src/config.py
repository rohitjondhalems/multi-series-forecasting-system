"""
Configuration for the forecasting service.
"""
from pathlib import Path

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

# ──────────────────────────────────────────────
# DATA SCHEMA
# ──────────────────────────────────────────────
# Column names in the data
TIME_COL = "timestamp"      # timestamp column
ID_COL = "series_id"        # series/group identifier
TARGET = "target"           # target variable to forecast

# Numeric covariates (features)
COV_NUM_COLS = ["cov_1", "cov_2", "cov_3", "cov_4", "cov_5"]

# Categorical covariate
COV_CAT_COL = "cov_cat"

# Auxiliary columns (e.g., metadata, not used in modeling)
AUX_COLS = ["aux_1", "aux_2", "aux_3", "aux_4", "aux_5"]

# ──────────────────────────────────────────────
# TRAIN/VAL/TEST SPLIT
# ──────────────────────────────────────────────
TRAIN_FRAC = 0.70  # 70% for training
VAL_FRAC = 0.15    # 15% for validation
# Test: 1 - TRAIN_FRAC - VAL_FRAC = 0.15 (15% for testing)

# ──────────────────────────────────────────────
# RANDOM SEED
# ──────────────────────────────────────────────
SEED = 42

# ──────────────────────────────────────────────
# QUANTILE REGRESSION (Conformal Prediction)
# ──────────────────────────────────────────────
# Quantiles for prediction intervals
# 0.5 = point forecast (median)
# 0.25-0.75 = 50% prediction interval
# 0.05-0.95 = 90% prediction interval
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]

# --- Foundation model (Chronos-2) ---
CHRONOS_MODEL = "amazon/chronos-bolt-small"   # fast Chronos-2 variant, CPU-friendly
CHRONOS_DEVICE = "cpu"
CHRONOS_CONTEXT = 2048    # cap history length for speed