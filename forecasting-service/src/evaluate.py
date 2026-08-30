"""
Evaluation module for forecasting model.
Computes metrics, generates plots, and analyzes performance.

Usage:
    python -m src.evaluate
"""
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,
    r2_score
)
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from src import config as C
from src import features as F


def load_model():
    """Load trained model and feature columns."""
    model_path = C.MODELS_DIR / "global_model.pkl"
    features_path = C.MODELS_DIR / "feature_columns.pkl"
    
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Run: python -m src.train_global")
        return None, None
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(features_path, 'rb') as f:
        feature_cols = pickle.load(f)
    
    return model, feature_cols


def load_test_data():
    """Load test data."""
    test_path = C.DATA_PROCESSED / "test.csv"
    if not test_path.exists():
        print(f"ERROR: Test data not found at {test_path}")
        return None
    
    test = pd.read_csv(test_path)
    return test


def compute_metrics(y_true, y_pred, set_name=""):
    """Compute evaluation metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2
    }
    
    print(f"\n{set_name} Metrics:")
    print(f"  RMSE:  {rmse:10.4f}")
    print(f"  MAE:   {mae:10.4f}")
    print(f"  MAPE:  {mape:10.4f}")
    print(f"  R²:    {r2:10.4f}")
    
    return metrics


def evaluate_by_series(test, model, feature_cols):
    """Evaluate performance per series."""
    print("\n" + "=" * 50)
    print("PERFORMANCE BY SERIES")
    print("=" * 50)
    
    results = []
    for series_id in sorted(test[C.ID_COL].unique()):
        series_data = test[test[C.ID_COL] == series_id].copy()
        
        if len(series_data) < 30:  # Skip very short series
            continue
        
        # Create features
        series_with_features = F.create_features(series_data)
        if len(series_with_features) == 0:
            continue
        
        X = series_with_features[feature_cols].astype('float32')
        y = series_with_features[C.TARGET].astype('float32')
        
        y_pred = model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        results.append({
            'series_id': series_id,
            'n_samples': len(series_with_features),
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        })
        
        print(f"{series_id:12s} | N={len(series_with_features):6d} | "
              f"RMSE={rmse:8.4f} | MAE={mae:8.4f} | R²={r2:8.4f}")
    
    results_df = pd.DataFrame(results)
    return results_df


def evaluate_residuals(y_true, y_pred):
    """Analyze residuals."""
    residuals = y_true - y_pred
    
    print("\n" + "=" * 50)
    print("RESIDUAL ANALYSIS")
    print("=" * 50)
    print(f"Mean:              {residuals.mean():10.4f}")
    print(f"Std:               {residuals.std():10.4f}")
    print(f"Min:               {residuals.min():10.4f}")
    print(f"Max:               {residuals.max():10.4f}")
    print(f"Skewness:          {residuals.skew():10.4f}")
    print(f"% within ±2*MAE:   {(np.abs(residuals) <= 2 * np.abs(y_true - y_pred).mean()).mean() * 100:10.2f}%")
    
    return residuals


def plot_predictions(y_true, y_pred, output_path=None, title="Predictions vs Actual"):
    """Plot actual vs predicted values."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Time series plot
    time_idx = np.arange(len(y_true))
    axes[0].plot(time_idx[:1000], y_true[:1000], 'o-', label='Actual', alpha=0.7, markersize=3)
    axes[0].plot(time_idx[:1000], y_pred[:1000], 's-', label='Predicted', alpha=0.7, markersize=3)
    axes[0].set_xlabel('Time Index')
    axes[0].set_ylabel('Value')
    axes[0].set_title(f'{title} (First 1000 points)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Scatter plot
    axes[1].scatter(y_true, y_pred, alpha=0.3, s=10)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect', linewidth=2)
    axes[1].set_xlabel('Actual')
    axes[1].set_ylabel('Predicted')
    axes[1].set_title('Actual vs Predicted')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"Plot saved: {output_path}")
    
    return fig


def plot_residuals(y_true, y_pred, output_path=None):
    """Plot residuals analysis."""
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Residuals over time
    axes[0, 0].plot(residuals[:1000], 'o-', alpha=0.7, markersize=3)
    axes[0, 0].axhline(0, color='r', linestyle='--', linewidth=2)
    axes[0, 0].set_ylabel('Residual')
    axes[0, 0].set_title('Residuals Over Time (First 1000)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Histogram
    axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Residual Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Residuals')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot (Normality Check)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Residuals vs Predicted
    axes[1, 1].scatter(y_pred, residuals, alpha=0.3, s=10)
    axes[1, 1].axhline(0, color='r', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Predicted')
    axes[1, 1].set_ylabel('Residual')
    axes[1, 1].set_title('Residuals vs Predicted')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"Plot saved: {output_path}")
    
    return fig


def main():
    print("=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)
    
    # Load model
    model, feature_cols = load_model()
    if model is None:
        return
    
    # Load test data
    test = load_test_data()
    if test is None:
        return
    
    # Create features
    print("\nCreating test features...")
    test_with_features = F.create_features(test)
    X_test = test_with_features[feature_cols].astype('float32')
    y_test = test_with_features[C.TARGET].astype('float32')
    
    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test)
    
    # Overall metrics
    print("\n" + "=" * 50)
    print("OVERALL METRICS")
    print("=" * 50)
    compute_metrics(y_test, y_pred, "Test")
    
    # Per-series metrics
    series_metrics = evaluate_by_series(test, model, feature_cols)
    print(f"\nAverage per-series R²: {series_metrics['r2'].mean():.4f}")
    
    # Residual analysis
    residuals = evaluate_residuals(y_test, y_pred)
    
    # Create output directory
    output_dir = C.BASE_DIR / "evaluation"
    output_dir.mkdir(exist_ok=True)
    
    # Generate plots
    print("\n" + "=" * 50)
    print("GENERATING PLOTS")
    print("=" * 50)
    plot_predictions(y_test.values, y_pred, 
                     output_path=output_dir / "predictions.png",
                     title="Test Set: Predictions vs Actual")
    plot_residuals(y_test.values, y_pred,
                   output_path=output_dir / "residuals.png")
    
    # Save results
    series_metrics.to_csv(output_dir / "series_metrics.csv", index=False)
    print(f"\nResults saved to {output_dir}/")
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
