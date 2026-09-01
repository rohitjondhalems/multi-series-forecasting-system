# MLOps Production Plan

This document describes how to take the multi-series forecasting system to
production on a cloud-native environment (Azure), covering model registry,
serving strategy, drift monitoring, retraining triggers, and rollback.

## Architecture

# Azure ML Forecasting Architecture

## Architecture

| Layer | Component | Details |
|-------|-----------|---------|
| **Tracking** | Azure ML + MLflow | Model registry, experiment logs, metrics (MAE, RMSE, coverage) |
| **Serving (online)** | Azure Container Apps | FastAPI (LightGBM) — 0.5 vCPU / 1 GiB, scale-to-zero |
| **Serving (online)** | Azure Container Apps | Streamlit UI — 0.5 vCPU / 1 GiB, scale-to-zero |
| **Serving (batch)** | Azure ML Pipeline | Nightly forecast job → Azure Blob Storage |
| **Models** | LightGBM (trained) | 5 quantile models, versioned artifacts in registry |
| **Models** | Chronos-2 (pretrained) | Zero-shot, pinned reference in registry |

## Components

| Component                | Purpose                                                                  |
| ------------------------ | ------------------------------------------------------------------------ |
| **Azure ML (MLflow)**    | Model registry and experiment tracking                                   |
| **Model Registry**       | Stores LightGBM model versions and Chronos-2 reference                   |
| **Experiment Logs**      | Tracks MAE, RMSE, coverage, and other evaluation metrics                 |
| **Azure Container Apps** | Hosts the forecasting API and Streamlit UI                               |
| **FastAPI (LightGBM)**   | Serves forecasting predictions using the deployed LightGBM model         |
| **Streamlit UI**         | Provides an interactive interface for forecast visualization and testing |
| **Azure Blob Storage**   | Stores batch forecast outputs                                            |

## Resource Configuration

* **FastAPI:** 0.5 vCPU / 1 GiB memory
* **Streamlit:** 0.5 vCPU / 1 GiB memory
* **Container Apps:** Scale-to-zero enabled
* **Forecast Storage:** Azure Blob Storage
* **Model Management:** Azure ML with MLflow


## 1. Model Registry and Versioning

### LightGBM (trained model)

Each trained model is registered as a versioned artifact in Azure ML with MLflow:

- **Artifact bundle**: 5 quantile `.pkl` files + `feature_columns.pkl` + `metadata.json`
- **Versioning**: semantic versions (v1.0, v1.1, v2.0) with stage labels
  (Staging → Production → Archived)
- **Metadata logged per version**: seed, training window dates, feature list,
  Python/LightGBM versions, evaluation metrics (MAE, RMSE, 50%/90% coverage)
- **Reproducibility**: deterministic training (single-threaded, seeded, pinned
  dependencies) ensures any version can be exactly recreated

### Chronos-2 (foundation model)

Chronos-2 is not retrained — it is a pretrained foundation model. What we
version is the **reference**:

- Model ID: `autogluon/chronos-2-small`
- Package version: `chronos-forecasting==2.1.0`
- PyTorch version: `2.13.0+cpu`
- Configuration: context length (2048), num_samples (200)

A version bump means upgrading the model checkpoint (e.g., `chronos-2-small` →
`chronos-2-base`) or the package version, not retraining weights.

## 2. Serving Strategy

### Online serving (current)

- **What**: FastAPI REST endpoints (`/forecast/global`, `/forecast/foundation`)
- **Where**: Azure Container Apps, 0.5 vCPU / 1 GiB per container, scale-to-zero
- **When**: User uploads a CSV and gets a forecast in real-time
- **Latency**: LightGBM ~1s for 24-step forecast; Chronos-2 ~10-30s on CPU
- **Use case**: Ad-hoc analysis, held-out series evaluation, demo

### Batch serving (production addition)

- **What**: Scheduled Azure ML Pipeline job that forecasts all known series
- **When**: Nightly (or hourly, depending on business need)
- **Output**: Forecasts written to Azure Blob Storage / database table
- **Use case**: Routine operational forecasts, dashboard feeds, alerting
- **Benefit**: Pre-computed results avoid real-time inference latency;
  Chronos-2's slower CPU inference is acceptable in a batch window

### Recommendation

Run both: batch for routine forecasts (all series, nightly), online for
interactive use (new series, ad-hoc uploads, the demo UI). LightGBM is the
primary production model (fast, accurate with covariates). Chronos-2 serves
as a fallback for series with insufficient history for feature engineering.

## 3. Drift Monitoring

### What to monitor

| Signal | Metric | Threshold | Tool |
|--------|--------|-----------|------|
| **Input drift** | Covariate distribution shift (KL divergence) | >0.1 on any cov | Azure ML Data Drift Monitor |
| **Input quality** | Missing value rate in target / covariates | >10% in a window | Custom check in batch pipeline |
| **Prediction drift** | Forecast distribution shift vs training | >0.15 KL divergence | Evidently or custom |
| **Accuracy decay** | Rolling MAE (7-day window) | >15% increase vs baseline | Custom metric job |
| **Coverage decay** | Rolling 90% interval coverage | <80% (target: 90%) | Custom metric job |

### How it works

1. As actuals arrive (hourly), compare against stored forecasts
2. Compute rolling MAE and coverage over a trailing window
3. Log metrics to Azure ML for visibility
4. Alert via Azure Monitor / email when thresholds breach

### Chronos-2 monitoring

Since Chronos-2 is not retrained, monitor its **relative performance** vs
LightGBM. If the gap narrows significantly (Chronos-2 improves due to a new
checkpoint, or LightGBM degrades), that's a signal to re-evaluate model
selection.

## 4. Retraining Triggers

### Scheduled retraining

- **Frequency**: Weekly (Sunday night)
- **Process**: Pull latest data → run `train_global.py` → evaluate on held-out
  window → register new version in Staging
- **Automation**: Azure ML Pipeline with scheduled trigger

### Event-driven retraining

Triggered by drift monitoring alerts:

- Coverage drops below 80% for 3 consecutive days
- MAE increases beyond threshold vs baseline for 5 consecutive days
- New covariate category appears (cov_cat value unseen in training)

### Retraining pipeline

The forecasting system follows an automated validate → train → evaluate →
compare → stage → shadow test → promote workflow whenever new data arrives.


                    New data arrives
                          │
                          ▼
                ┌─────────────────────┐
                │   Prepare data      │
                │   Validate + split  │
                └─────────┬───────────┘
                          │
                          ▼
                ┌─────────────────────┐
                │   Train models      │
                │   LightGBM exog +   │
                │   noexog variants   │
                └─────────┬───────────┘
                          │
                          ▼
                ┌─────────────────────┐
                │   Evaluate on       │
                │   validation window │
                └─────────┬───────────┘
                          │
                ┌─────────┴──────────┐
                ▼                    ▼
         Better than           Worse than
         production?           production?
                │                    │
                ▼                    ▼
         Register as          Log results
         STAGING              Keep production
                │
                ▼
         Shadow test
         (staging vs production)
                │
                ▼
         Metrics hold?
         Yes → Promote to PRODUCTION
         No  → Reject, keep current


### Workflow steps

**1. New data arrives** — new forecasting data is received and made available
to the training pipeline.

**2. Prepare data** — `POST /train` on the FastAPI server triggers data
preparation and training in one call. It validates the incoming data, checks
required columns and data quality, handles preprocessing, and creates the
train/validation/test windows.

**3. Train candidate models** — the same `POST /train` endpoint trains both
model variants (LightGBM with and without exogenous variables). Each training
run is tracked in MLflow, including model parameters, feature configuration,
validation metrics, training artifacts, and feature importance. No manual
command-line execution required — everything is driven through the REST API
or Swagger UI at `/docs`.

**4. Evaluate on the validation window** — candidate models are evaluated
against the latest validation window on MAE, RMSE, and coverage metrics. The
candidate is compared against the currently deployed Production model.

**5. Model selection** — if the candidate outperforms the current production
model, it is registered in MLflow as a new version with the Staging stage.
If it underperforms, the candidate is retained for analysis but the
production model is not changed.

**6. Shadow testing** — a model promoted to Staging receives the same inputs
as Production in parallel. Predictions from both are compared on accuracy
(RMSE/MAE when actuals arrive), forecast stability, latency, error rate,
and business-specific thresholds.

            Incoming Request
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
Production Model        Staging Model
        │                     │
        ▼                     ▼
  Production              Staging
   Prediction             Prediction
        │                     │
        └──────────┬──────────┘
                   ▼
            Compare Results


**7. Promote to production** — if the Staging model passes the shadow test,
it becomes the new Production model. The previous Production version is
retained for rollback.

**8. Rollback / rejection** — if the Staging model fails the shadow test,
it is rejected and the existing Production model continues serving.

### Chronos-2 retraining

Chronos-2 is not retrained. Instead:

- **Re-evaluate** when Amazon releases a new checkpoint
- **Upgrade** the pinned model reference if the new version benchmarks better
- **Fine-tune** (future): when `chronos-forecasting` adds fine-tuning support
  (LoRA or QLoRA), fine-tune on domain data and version the fine-tuned weights
  separately

## 5. Rollback

### Strategy: Blue/Green with automatic rollback

- **Current Production** (blue) always stays warm and addressable
- **New candidate** (green) gets deployed alongside with a canary split
  (e.g., 10% traffic)
- **Promotion**: if canary metrics hold for 24h, green becomes the new blue
- **Rollback trigger**: MAE on canary traffic >20% worse than blue, or coverage
  <75%, or error rate >5%
- **Rollback action**: route 100% back to blue, archive the failed green version,
  alert the team

### Implementation on Azure Container Apps

Azure Container Apps supports revision-based traffic splitting, which maps
directly to blue/green deployment:

```bash
# Deploy new candidate as a new revision
az containerapp update \
  --name forecasting-api \
  --resource-group multi-forecasting \
  --image forecasting-api:v1.3 \
  --revision-suffix v1-3

# Split traffic: 90% current (blue), 10% candidate (green)
az containerapp ingress traffic set \
  --name forecasting-api \
  --resource-group multi-forecasting \
  --revision-weight forecasting-api--v1-2=90 forecasting-api--v1-3=10

# PROMOTE (canary metrics hold after 24h):
az containerapp ingress traffic set \
  --revision-weight forecasting-api--v1-3=100

# ROLLBACK (canary metrics degrade):
az containerapp ingress traffic set \
  --revision-weight forecasting-api--v1-2=100
```

### Rollback criteria

| Metric | Threshold | Action |
|--------|-----------|--------|
| Canary MAE vs production MAE | >20% worse | Auto-rollback |
| 90% interval coverage | <75% | Auto-rollback |
| API error rate (5xx) | >5% | Auto-rollback |
| Response latency (p99) | >60s | Alert + manual review |

### Registry stage lifecycle

| Stage | Meaning |
|-------|---------|
| **None** | Just trained, not yet evaluated |
| **Staging** | Passed offline evaluation, ready for canary |
| **Production** | Currently serving, backed by monitoring |
| **Archived** | Previous production, kept for instant rollback |

### Chronos-2 rollback

Since Chronos-2 is not a trained artifact, rollback means reverting the
pinned model reference in config and restarting the container:

```python
# Current
CHRONOS_MODEL = "autogluon/chronos-2-small"

# Rollback = revert config + redeploy
CHRONOS_MODEL = "amazon/chronos-bolt-small"   # previous reference
```

## 6. Current implementation status

| Component | Status |
|-----------|--------|
| Model registry (Azure ML + MLflow) | ✅ Live |
| Online serving (FastAPI + Docker) | ✅ Live |
| Streamlit demo UI | ✅ Live |
| Docker Compose single-command bringup | ✅ Live |
| Batch serving | 📋 not yet implemented |
| Drift monitoring | 📋 not yet implemented |
| Automated retraining pipeline | 📋 not yet implemented |
| Blue/green rollback | 📋 not yet implemented |

## 7. What I would do next

1. **Conformal calibration**: Post-hoc scale prediction intervals to hit
   exactly 90% coverage (currently 88%)
2. **Feed covariates into Chronos-2**: Use its native covariate support
   to narrow the foundation model's wider prediction intervals
3. **Freeze cov_cat encoding**: Store the category-to-integer mapping at
   training time and reapply at serving time for exact reproducibility
4. **Implement drift monitoring**: Azure ML Data Drift Monitor + Evidently
   reports integrated into the Streamlit dashboard
5. **Automated retraining**: Azure ML Pipeline with scheduled + event-driven
   triggers, gated by offline evaluation before promotion
6. **GPU serving for Chronos-2**: Move to GPU inference to reduce latency
   from ~30s to <2s per forecast
7. **Add aux_* as lagged features**: The five auxiliary channels are currently
   unused — adding them as lag features could improve accuracy
