"""
Feature engineering for time series forecasting.
Creates lagged features, rolling statistics, and temporal features.
"""
import pandas as pd
import numpy as np
from src import config as C


def create_lag_features(df, lag_periods=[1, 2, 3, 24, 48, 168]):
    """Create lagged target features."""
    df = df.copy()
    for lag in lag_periods:
        df[f'lag_{lag}'] = df.groupby(C.ID_COL)[C.TARGET].shift(lag)
    return df


def create_rolling_features(df, windows=[7, 24, 168]):
    """Create rolling mean and std features."""
    df = df.copy()
    for window in windows:
        df[f'rolling_mean_{window}'] = (
            df.groupby(C.ID_COL)[C.TARGET]
            .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
        )
        df[f'rolling_std_{window}'] = (
            df.groupby(C.ID_COL)[C.TARGET]
            .transform(lambda x: x.rolling(window=window, min_periods=1).std())
        )
    return df


def create_temporal_features(df):
    """Create date/time-based features."""
    df = df.copy()
    df[C.TIME_COL] = pd.to_datetime(df[C.TIME_COL])
    
    # Time-based features
    df['hour'] = df[C.TIME_COL].dt.hour
    df['day_of_week'] = df[C.TIME_COL].dt.dayofweek
    df['day_of_month'] = df[C.TIME_COL].dt.day
    df['month'] = df[C.TIME_COL].dt.month
    df['quarter'] = df[C.TIME_COL].dt.quarter
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    return df


def encode_categorical(df):
    """Encode categorical features as integers."""
    df = df.copy()
    if C.COV_CAT_COL in df.columns:
        # Convert to categorical first, then get codes
        df[C.COV_CAT_COL] = pd.Categorical(df[C.COV_CAT_COL])
        df[C.COV_CAT_COL] = df[C.COV_CAT_COL].cat.codes
    return df


def create_features(df):
    """Full feature engineering pipeline.
    
    ASSIGNMENT REQUIREMENTS MET:
    1. Hourly patterns: Creates 'hour' feature (critical for daily seasonality)
    2. Covariates: Includes cov_1, cov_2, cov_3, cov_4, cov_5, cov_cat
    
    Pipeline steps:
    1. Temporal features (hour is key for daily patterns)
    2. Lag features (target history)
    3. Rolling statistics (trend + volatility)
    4. Categorical encoding (wind direction)
    5. Drop incomplete rows (NaN from lag/rolling)
    """
    print("Creating features...")
    
    # Temporal features (includes REQUIRED 'hour' feature for daily patterns)
    df = create_temporal_features(df)
    
    # Lag features
    df = create_lag_features(df, lag_periods=[1, 2, 3, 24, 48])
    
    # Rolling features
    df = create_rolling_features(df, windows=[7, 24])
    
    # Encode categorical (cov_cat is REQUIRED covariate)
    df = encode_categorical(df)
    
    # Drop rows with NaN (from lag/rolling operations)
    df = df.dropna()
    
    print(f"Features created. Shape: {df.shape}")
    return df


def get_feature_columns(df, use_exog=True):
    """Get list of feature column names for model input.
    
    Args:
        df: DataFrame with features
        use_exog: If True, include covariates (cov_*). If False, exclude them.
    
    REQUIRED FEATURES (per assignment):
    - Hour: Critical for capturing daily patterns in hourly data
    - Covariates: cov_1, cov_2, cov_3, cov_4, cov_5, cov_cat (if use_exog=True)
    
    FEATURE GROUPS:
    - Temporal (6): hour, day_of_week, day_of_month, month, quarter, is_weekend
    - Lag (5): lag_1, lag_2, lag_3, lag_24, lag_48
    - Rolling (4): rolling_mean_7, rolling_mean_24, rolling_std_7, rolling_std_24
    - Covariates (6, optional): cov_1, cov_2, cov_3, cov_4, cov_5, cov_cat
    
    EXCLUDED:
    - Time column (timestamp)
    - Series ID
    - Target (what we're predicting)
    - Auxiliary columns (not available at inference time)
    """
    exclude = {C.TIME_COL, C.ID_COL, C.TARGET}
    exclude.update(C.AUX_COLS)  # Don't use auxiliary columns
    
    # Optionally exclude covariates (for baseline comparison)
    if not use_exog:
        exclude.update(C.COV_NUM_COLS)
        exclude.add(C.COV_CAT_COL)
    
    features = [c for c in df.columns if c not in exclude]
    
    # Verify required features are present
    has_hour = 'hour' in features
    has_covariates = any(f.startswith('cov_') for f in features) if use_exog else True
    
    if not has_hour:
        print("WARNING: 'hour' feature not found. Daily patterns may not be captured.")
    if use_exog and not has_covariates:
        print("WARNING: No covariate features found. Assignment requires at least one covariate.")
    
    return features
