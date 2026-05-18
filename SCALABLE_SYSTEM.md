# Scalable Insurtech System

This project now supports a production-style workflow across three layers:

1. Parallel PPO training for policy optimization
2. Concurrent inference for real-time and batch pricing decisions
3. Parallel driver-stat aggregation for telemetry analytics

## Training

Run the hybrid trainer with configurable PPO parallelism:

```bash
python train_hybrid_risk_model.py --data preprocessed_telemetry_dataset.csv --ppo-timesteps 500000 --ppo-envs 8 --ppo-vec-mode subproc
```

Key outputs in `artifacts/hybrid_risk/`:

- `lightgbm_risk_model.pkl`
- `ppo_premium_policy.zip`
- `ppo_observation_scaler.pkl`
- `training_summary.json`

If subprocess environments are unavailable on the local machine, training falls back to `DummyVecEnv` automatically instead of failing.

## Real-Time API

Start the FastAPI pricing service:

```bash
uvicorn pricing_engine:create_app --factory --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /health`
- `POST /score`
- `POST /score/batch`

The API loads LightGBM, PPO, and the saved PPO scaler once at startup and returns risk scores, policy actions, and adjusted premium decisions.

## Python Inference Service

Use the reusable scoring service directly:

```python
from pricing_engine import HybridPricingInferenceService
from telematics_data_generator import get_preprocessed_dataset

service = HybridPricingInferenceService()
df = get_preprocessed_dataset().head(1000)
scored = service.score_frame_concurrently(df, partition_size=250, max_workers=4)
```

## Streamlit Background Analysis

The Streamlit app now uses the saved LightGBM + PPO artifacts in the background for:

- live risk scoring
- policy pricing
- profitability analysis
- uploaded portfolio batch analysis

Open the `Portfolio Analysis` page in Streamlit to:

- monitor the latest background-scored live portfolio
- upload a CSV/XLSX/Parquet portfolio file
- let scoring run asynchronously in the background
- download scored analysis and premium schedule outputs once the job finishes

## Spark Batch Scoring

For distributed pricing decisions:

```python
from pricing_engine import HybridPricingInferenceService

service = HybridPricingInferenceService()
scored_spark_df = service.score_spark_dataframe(input_spark_df)
```

This uses Spark `mapInPandas` so each partition is scored with the same LightGBM + PPO pipeline.

## Driver Statistics

Capture telemetry history and aggregate rolling driver statistics:

```python
from telematics_data_generator import capture_telematics_history, get_parallel_driver_statistics

history = capture_telematics_history(cycles=12, sleep_seconds=0.0)
driver_stats = get_parallel_driver_statistics(history_df=history, max_workers=4)
```

Spark aggregation is also available:

```python
from telematics_data_generator import get_parallel_driver_statistics_spark

driver_stats_spark = get_parallel_driver_statistics_spark()
```
