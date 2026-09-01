# Multi-Series Forecasting System

A production-ready multi-series time series forecasting system with two models
(LightGBM + Chronos-2), uncertainty quantification (50% and 90% prediction
intervals), and an interactive demo UI.

Built for the Cloud ML Engineer take-home assignment.

## Quick start

```bash
docker compose up
```

- **Demo UI**: http://localhost:8501
- **Swagger API**: http://localhost:8000/docs

## Local development (without Docker)

### Setup and run

```bash
# 1. Put the OneDrive CSV files into data/raw/

# 2. Prepare data (validate, clean, split)
python -m scripts.prepare_data

# 3. Train the global model (both exog and noexog variants)
python -m src.train_global

# 4. Start the API (terminal 1)
uvicorn api.main:app --reload --port 8000

# 5. Start the UI (terminal 2)
streamlit run ui/app.py --server.port 8501
```

Steps 2 and 3 can also be triggered via the API — once the server is running,
hit `POST /train` in Swagger UI at http://localhost:8000/docs. This runs data
preparation and model training in one call, so no command-line execution is
needed after the initial server start. All other operations (forecast, evaluate)
are also available as API endpoints.

Alternatively, all steps (train, evaluate, forecast) can be driven entirely
from the Swagger UI at http://localhost:8000/docs — no command-line runs needed.

## Models

### 1. LightGBM — global model (trained)

A gradient-boosted model trained across all 10 series using quantile regression.
Five quantile models (0.05, 0.25, 0.50, 0.75, 0.95) provide the point forecast
(median) and both 50% and 90% prediction intervals.

**Key design choices:**

- **Global**: one model trained on pooled data from all series, so it generalises
  to unseen series using only the uploaded history
- **Deterministic**: single-threaded, seeded (`SEED=42`), pinned dependencies —
  same input produces the same forecast on every run
- **Two variants**: with covariates (exog) and without (noexog), enabling an
  honest ablation via the demo's toggle
- **Recursive forecasting**: each step feeds the median prediction back as
  "known" target for the next step's lag features, allowing a single trained
  model to serve any horizon from an uploaded history

### 2. Chronos-2 — foundation model (pretrained, zero-shot)

A pretrained transformer-based model (`amazon/chronos-bolt-small`)
used zero-shot — no training on the provided data. It reads raw target values
directly without any feature engineering.

**Trade-off vs LightGBM:** Chronos-2's prediction intervals are significantly
wider (band width ~30 vs ~3) because it has no prior exposure to this dataset
and does not use covariates in the current configuration. This reflects honest
uncertainty from a zero-shot model. LightGBM achieves tighter intervals because
it was trained on 245,000 rows with covariates. This contrast demonstrates the
fundamental trade-off: a trained model gives better accuracy on known data
patterns, while a foundation model provides immediate forecasting capability on
any new series without retraining.

## Results

Evaluation on the time-ordered held-out test window (15% of each series):

| Model | MAE | RMSE | 50% coverage | 90% coverage |
|-------|-----|------|-------------|-------------|
| LightGBM (with covariates) | **9.05** | **18.51** | 48.6% | 88.0% |
| LightGBM (no covariates) | 9.20 | 19.09 | 49.2% | 88.1% |
| Chronos-2 (zero-shot) | 19.16 | 43.84 | 60.1% | 93.7% |

**Interpretation:**

- MAE of 9.05 represents ~2% of the target range (3–433), indicating strong
  average accuracy for hourly data
- Covariates improve MAE by ~2% (9.05 vs 9.20) — all six covariates rank in
  the top 15 features by gain, confirming they carry signal beyond what lags
  and calendar features capture
- LightGBM's 90% coverage (88%) is slightly tight vs the 90% target; conformal
  calibration would close this gap (see "What I would do next")
- Chronos-2's wider intervals (93.7% coverage) reflect conservative uncertainty
  from a zero-shot model — honest but less precise
- The RMSE-to-MAE ratio (~2:1) indicates occasional large errors on volatile
  hours, which is expected for hourly frequency data

### Top features by importance (LightGBM median model, exog variant)

| Rank | Feature | Gain | Note |
|------|---------|------|------|
| 1 | lag_1 | 901,473 | Previous hour's value |
| 2 | lag_2 | 135,036 | Two hours ago |
| 3 | rolling_mean_7 | 68,231 | 7-step rolling average |
| 4 | rolling_std_7 | 42,644 | Recent volatility |
| 5 | lag_3 | 15,690 | Three hours ago |
| 6 | hour | 12,692 | Daily patterns |
| 7 | cov_5 | 8,604 | Covariate |
| 8 | cov_3 | 7,038 | Covariate |
| 9 | cov_1 | 5,728 | Covariate |
| 10 | cov_cat | 5,421 | Categorical covariate |

## Feature engineering

21 features for the covariate variant, 15 without:

**Calendar:** hour, day_of_week, day_of_month, month, quarter, is_weekend

**Target lags:** lag_1, lag_2, lag_3, lag_24, lag_48

**Rolling statistics:** rolling_mean_7, rolling_std_7, rolling_mean_24, rolling_std_24
(shifted by 1 step to avoid leaking the current value)

**Covariates (when enabled):** cov_1 through cov_5 (raw numeric), cov_cat
(native LightGBM categorical)

All features are computed by one shared module (`src/features.py`) imported by
both training and serving, guaranteeing no train-serve skew.

## Exogenous features at serving time

### Encoding choice

**Continuous covariates (cov_1–cov_5):** fed as raw numeric values with no
scaling. Tree-based models are scale-invariant, so normalization is unnecessary
and would add complexity without benefit.

**Categorical covariate (cov_cat):** uses LightGBM's native categorical handling
rather than one-hot encoding. With ~16 buckets (compass directions: N, NNE, NE,
...), native categorical splits are more efficient than 16 binary columns. The
category list is frozen at training time for reproducible serve-time encoding;
an unseen category degrades gracefully to a default value rather than breaking
the prediction.

### Serving-time assumption

Covariates are treated as **past-only by default**:

- **target** and **aux_1–aux_5** are observed-only — they enter the model
  exclusively as lagged features, never at the forecast timestamp, because at
  forecast time these values do not exist yet
- **cov_1–cov_5** and **cov_cat** over the forecast horizon: the last known
  value is persisted forward unless the caller supplies explicit future values
- The **recursive forecasting loop** feeds the median prediction back as the
  "known" target for the next step's lag features — this is what allows a
  single trained model to serve any horizon

### API contract

`POST /forecast/global` accepts:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| file | CSV/TSV | required | History file matching the training schema |
| horizon | int | 24 | Hours ahead to forecast (1–720) |
| interval | int | 90 | Prediction band width: 50 or 90 |
| use_exog | bool | true | Use covariates (cov_1–5, cov_cat) or not |

Features are recomputed from the uploaded history using the same shared module
used in training (`src/features.py`), guaranteeing no train-serve skew. The
endpoint works on series the model has never seen during training, because all
features are derived from each series' own history.

### What the response contains

```json
{
  "model": "global_lgbm",
  "series_id": "series_L",
  "horizon": 24,
  "interval": 90,
  "use_exog": true,
  "forecast": [
    {
      "timestamp": "2017-03-01 00:00:00",
      "step": 1,
      "point": 12.0003,
      "lower_90": 10.0336,
      "upper_90": 14.3254
    }
  ]
}
```

## Uncertainty quantification

### LightGBM

Five separate quantile regression models, each trained with
`objective="quantile"` at a different alpha:

| Quantile | Purpose |
|----------|---------|
| 0.05 | Bottom of 90% band |
| 0.25 | Bottom of 50% band |
| 0.50 | Point forecast (median) |
| 0.75 | Top of 50% band |
| 0.95 | Top of 90% band |

Quantile crossing is prevented by sorting the five predictions at each step.

**Calibration defense:** empirical coverage on the test window is 88% for the
90% interval (target: 90%) and 49% for the 50% interval (target: 50%). Both
are within 2 percentage points of ideal, indicating well-calibrated bands.

### Chronos-2

Chronos-2 generates probabilistic forecasts natively by drawing 200 sample
paths from its learned distribution, then computing quantiles from the samples.
Coverage on the test set is 93.7% for the 90% band — slightly conservative
(wider than necessary), which is expected for a zero-shot model.

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check and model loading status |
| GET | `/models` | List artifacts, results, and model info |
| POST | `/train` | Train LightGBM from data in data/raw/ |
| POST | `/forecast/global` | Forecast with LightGBM (exog toggle, horizon, interval) |
| POST | `/forecast/foundation` | Forecast with Chronos-2 (zero-shot) |
| POST | `/evaluate/global` | Evaluate LightGBM on the saved test set |

All endpoints are documented with interactive forms at `/docs` (Swagger UI).
Invalid inputs return well-formed HTTP errors (400/422/409).

## Demo UI

The Streamlit interface allows a reviewer to:

- **Upload** a CSV file matching the training schema
- **Plot** history + forecast + uncertainty band on one chart
- **Toggle** between LightGBM and Chronos-2 to compare models
- **Toggle** exogenous features on/off and observe the effect on forecast
  accuracy and interval width (covariates tighten the bands)
- **Adjust** forecast horizon with a slider (1–720 hours)
- **Switch** between 50% and 90% prediction intervals
- **View** a metrics card showing MAE and coverage for the uploaded series

The demo works on series the model has never seen during training — at the
debrief, upload one of the held-out series and the system forecasts from its
history alone.

## Experiment tracking

Model training and evaluation runs are tracked in Azure Machine Learning
with MLflow:

- **Workspace**: Multi-Forecasting-System (Azure ML)
- **Experiment**: multi-series-forecasting
- **Tracked per run**: model parameters, feature configuration, evaluation
  metrics (MAE, RMSE, 50%/90% coverage), training artifacts, feature importance
- **LightGBM**: versioned trained artifacts in the model registry
- **Chronos-2**: versioned as a pinned reference (model ID + package version),
  not as a trained artifact

## Data handling

### Schema

| Column | Type | Role |
|--------|------|------|
| timestamp | datetime | Observation time (hourly) |
| series_id | string | Series identifier |
| target | numeric | Value to forecast |
| aux_1–aux_5 | numeric | Auxiliary channels (observed-only, used as lags) |
| cov_1–cov_5 | numeric | Numeric covariates |
| cov_cat | string | Categorical covariate (~16 compass direction buckets) |

### Missing values

- LightGBM handles NaN in features natively — numeric covariates are not imputed
- Missing cov_cat values are mapped to an explicit `__missing__` bucket
- Chronos-2 requires continuous input — NaN values are interpolated before
  being passed to the model

### Train/validation/test split

Time-ordered per series: 70% train / 15% validation / 15% test. No data from
validation or test precedes the training boundary — no leakage.

## Determinism

The assignment requires "the same input must produce the same forecast on a
re-run." This is achieved through:

- `SEED = 42` applied to numpy, LightGBM, and PyTorch
- LightGBM: `deterministic=True`, `force_row_wise=True`, `n_jobs=1`
- All dependencies pinned with exact versions in `requirements.txt`
- Exact resolved environment in `requirements.lock.txt`
- Python 3.11, LightGBM 4.3.0, PyTorch 2.13.0+cpu

## Project structure

forecasting-service/
├── api/main.py # FastAPI serving endpoints
├── ui/app.py # Streamlit demo UI
├── src/
│ ├── config.py # Central config (seeds, schema, paths)
│ ├── features.py # Shared feature engineering (train + serve)
│ ├── train_global.py # LightGBM quantile training
│ ├── evaluate.py # Evaluation and metrics
│ └── models/
│ └── foundation_model.py # Chronos-2 wrapper
├── scripts/
│ └── prepare_data.py # Data loading, validation, splitting
├── models/ # Trained artifacts (gitignored)
├── data/ # Raw + processed data (gitignored)
├── docker-compose.yml
├── Dockerfile.api / Dockerfile.ui
├── requirements.txt # Pinned dependencies
├── README.md
└── MLOPS.md # Production deployment plan


## What would do next/Future Scope

1. **Conformal calibration**: Post-hoc scale prediction intervals on a
   calibration set to hit exactly 90% coverage (currently 88%)
2. **Feed covariates into Chronos-2**: Use its native covariate support
   to narrow the foundation model's prediction intervals
3. **Freeze cov_cat encoding**: Store the category-to-integer mapping at
   training time and reapply at serving time for exact reproducibility
   across unseen category values
4. **Cyclical encoding for cov_cat**: The compass-direction categories
   (N, NNE, NE, ...) are genuinely cyclical — a sin/cos encoding of the
   bearing angle would capture adjacency that nominal encoding misses
5. **Larger Chronos-2 variant**: Upgrade from `chronos-2-small` (28M params)
   to `chronos-2-base` (200M) for better zero-shot accuracy, with GPU serving
6. **Batch serving pipeline**: Nightly Azure ML Pipeline job to forecast
   all series and store results for dashboard consumption
7. **Add aux_* as lagged features**: The five auxiliary channels are currently
   unused — adding them as lag features could improve accuracy




<img width="1917" height="1030" alt="streamlit_1" src="https://github.com/user-attachments/assets/ef168204-0d57-4d30-a6f4-3c435c142045" />
<img width="1917" height="1027" alt="streamlit" src="https://github.com/user-attachments/assets/a6e4278e-2aaf-4bdc-9f03-ef142a943b85" />
<img width="1917" height="1026" alt="apis-1" src="https://github.com/user-attachments/assets/b7f852f1-2178-4008-b493-9d685f316b93" />
<img width="1917" height="972" alt="swagger_apis" src="https://github.com/user-attachments/assets/98418522-ae7e-46cb-a337-57a3621d6284" />
<img width="1917" height="972" alt="lightgbm_1" src="https://github.com/user-attachments/assets/bd5b9b73-6f3e-4859-8a49-7fe6af6d80f1" />
<img width="1917" height="977" alt="chronos-2" src="https://github.com/user-attachments/assets/4c2c437b-ac46-401f-a69c-2a4afdf9fb3b" />
<img width="1917" height="977" alt="lightgbm" src="https://github.com/user-attachments/assets/2b0c65f5-8ef0-45a4-9cdf-90e70adc9390" />
<img width="1917" height="907" alt="chronos-2-run" src="https://github.com/user-attachments/assets/fea5a445-6739-476b-9a44-ad240619ded5" />





