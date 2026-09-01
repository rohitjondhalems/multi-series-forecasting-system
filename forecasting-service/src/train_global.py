"""
Train global LightGBM quantile models on all time series.
Trains two variants (with/without covariates) × 5 quantiles = 10 models.

Usage:
    python -m src.train_global
"""
import json
import pickle
import platform
from pathlib import Path

import pandas as pd
import numpy as np
import lightgbm as lgb

from src import config as C
from src import features as F


def _run_data_prep():
    """Build data/processed/{train,val,test}.csv from the raw CSVs in
    data/raw/. Runs the same three steps as `python -m scripts.prepare_data`,
    called directly (not via its argparse main()) so this works from inside
    a running API process. Returns True on success, False if raw data is
    missing or fails validation."""
    from scripts import prepare_data as PD

    print("Prepared data not found — running data preparation from data/raw/ first...")
    df = PD.read_data()
    if df is None:
        return False
    if not PD.check_data(df):
        return False

    train, val, test = PD.prepare_data(df)
    C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    train.to_csv(C.DATA_PROCESSED / "train.csv", index=False)
    val.to_csv(C.DATA_PROCESSED / "val.csv", index=False)
    test.to_csv(C.DATA_PROCESSED / "test.csv", index=False)
    print("Data preparation complete.")
    return True


def load_prepared_data():
    """Load train/val/test splits from CSV files, preparing them from
    data/raw/ first if they don't exist yet."""
    train_path = C.DATA_PROCESSED / "train.csv"
    val_path = C.DATA_PROCESSED / "val.csv"
    test_path = C.DATA_PROCESSED / "test.csv"

    if not train_path.exists():
        if not _run_data_prep():
            print(f"ERROR: Could not prepare data (no valid CSVs in {C.DATA_RAW})")
            return None, None, None

    print("Loading prepared data...")
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    test = pd.read_csv(test_path)

    print(f"  Train: {len(train):,} rows")
    print(f"  Val:   {len(val):,} rows")
    print(f"  Test:  {len(test):,} rows")
    return train, val, test


def create_features(df, use_exog=True):
    """Create features and return X, y, feature column names."""
    df = F.create_features(df)
    feature_cols = F.get_feature_columns(df, use_exog=use_exog)

    X = df[feature_cols].copy()
    y = df[C.TARGET].astype("float32")

    # drop rows where target is NaN (can't train on those)
    mask = y.notna()
    return X[mask], y[mask], feature_cols


def train_quantile_models(X_train, y_train, X_val, y_val, feature_cols,
                           param_overrides=None, num_boost_round=500,
                           early_stopping_rounds=50):
    """Train 5 quantile models (0.05, 0.25, 0.50, 0.75, 0.95).
    The 0.50 model = point forecast (median).
    0.25-0.75 = 50% band. 0.05-0.95 = 90% band.

    param_overrides lets callers (e.g. the /tune API) sweep hyperparameters
    like num_leaves/learning_rate without touching objective/seed, which stay
    fixed so results remain deterministic (same input -> same output)."""

    base_params = {
        "objective": "quantile",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "n_jobs": 1,               # single thread = reproducible
        "seed": C.SEED,
        "deterministic": True,
        "force_row_wise": True,
    }
    if param_overrides:
        base_params.update(param_overrides)

    models = {}
    for q in C.QUANTILES:
        print(f"\n  Training quantile {q}...")
        params = {**base_params, "alpha": q}

        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            params, train_data,
            num_boost_round=num_boost_round,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=100),
            ],
        )
        models[q] = model
        print(f"  quantile {q} done (best iter: {model.best_iteration})")

    return models


def evaluate_models(models, X, y, set_name=""):
    """Evaluate the median (0.5) model and show coverage of bands."""
    y_pred = models[0.5].predict(X)

    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    mae = np.mean(np.abs(y - y_pred))

    # coverage: what fraction of actuals fall inside the band?
    lo_90, hi_90 = models[0.05].predict(X), models[0.95].predict(X)
    lo_50, hi_50 = models[0.25].predict(X), models[0.75].predict(X)
    cov_90 = np.mean((y >= lo_90) & (y <= hi_90))
    cov_50 = np.mean((y >= lo_50) & (y <= hi_50))

    print(f"  {set_name:6s} | MAE: {mae:8.2f} | RMSE: {rmse:8.2f} "
          f"| 50% cov: {cov_50:.2%} | 90% cov: {cov_90:.2%}")

    return {"mae": float(mae), "rmse": float(rmse),
            "cov_50": float(cov_50), "cov_90": float(cov_90)}


def feature_importance(models, top_n=15):
    """Print top features from the median model."""
    model = models[0.5]
    importance = model.feature_importance(importance_type="gain")
    names = model.feature_name()
    sorted_idx = np.argsort(importance)[::-1][:top_n]

    print(f"\nTop {top_n} features (median model):")
    for rank, idx in enumerate(sorted_idx, 1):
        marker = ""
        if names[idx] == "hour":
            marker = " ← daily patterns"
        elif names[idx].startswith("cov_"):
            marker = " ← covariate"
        print(f"  {rank:2d}. {names[idx]:25s} | Gain: {importance[idx]:12.0f}{marker}")


def tune(param_overrides: dict, use_exog: bool = True, num_boost_round: int = 500,
         early_stopping_rounds: int = 50, save: bool = False) -> dict:
    """Train one LightGBM variant with the given hyperparameters and return
    test-set metrics. Used by the /tune API for hyperparameter sweeps tracked
    in MLflow. Does not overwrite the deployed models unless save=True."""
    train, val, test = load_prepared_data()
    if train is None:
        raise FileNotFoundError(
            "Prepared data not found. Run POST /train or scripts/prepare_data first."
        )

    X_train, y_train, feature_cols = create_features(train, use_exog=use_exog)
    X_val, y_val, _ = create_features(val, use_exog=use_exog)
    X_test, y_test, _ = create_features(test, use_exog=use_exog)

    models = train_quantile_models(
        X_train, y_train, X_val, y_val, feature_cols,
        param_overrides=param_overrides,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
    )
    metrics = evaluate_models(models, X_test, y_test, "Test")

    if save:
        tag = "exog" if use_exog else "noexog"
        output_dir = C.MODELS_DIR / tag
        output_dir.mkdir(parents=True, exist_ok=True)
        for q, model in models.items():
            with open(output_dir / f"quantile_{q}.pkl", "wb") as f:
                pickle.dump(model, f)
        with open(output_dir / "feature_columns.pkl", "wb") as f:
            pickle.dump(feature_cols, f)

    return metrics


def main():
    # Seed everything
    np.random.seed(C.SEED)

    # Load data
    train, val, test = load_prepared_data()
    if train is None:
        return

    results = {}

    for use_exog in [True, False]:
        tag = "exog" if use_exog else "noexog"
        print("\n" + "=" * 60)
        print(f"TRAINING VARIANT: {tag.upper()}")
        print("=" * 60)

        # Create features
        X_train, y_train, feature_cols = create_features(train, use_exog=use_exog)
        X_val, y_val, _ = create_features(val, use_exog=use_exog)
        X_test, y_test, _ = create_features(test, use_exog=use_exog)

        print(f"Features ({len(feature_cols)}): {feature_cols}")

        # Train 5 quantile models
        models = train_quantile_models(X_train, y_train, X_val, y_val, feature_cols)

        # Evaluate
        print(f"\nEvaluation ({tag}):")
        evaluate_models(models, X_train, y_train, "Train")
        val_metrics = evaluate_models(models, X_val, y_val, "Val")
        test_metrics = evaluate_models(models, X_test, y_test, "Test")
        results[f"global_{tag}"] = test_metrics

        # Feature importance
        feature_importance(models)

        # Save models
        output_dir = C.MODELS_DIR / tag
        output_dir.mkdir(parents=True, exist_ok=True)
        for q, model in models.items():
            model_path = output_dir / f"quantile_{q}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        # Save feature columns
        features_path = output_dir / "feature_columns.pkl"
        with open(features_path, 'wb') as f:
            pickle.dump(feature_cols, f)
        
        print(f"\nModels saved to {output_dir}/")

    # Save all results
    results_path = C.MODELS_DIR / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()