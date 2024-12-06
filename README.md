# Model Health Monitor

Production monitoring for tabular classifiers. Detects when your model starts making bad decisions because the data changed, not because the model broke.

## What This Does

Most ML monitoring tools tell you the model is wrong. This one tells you why. It watches feature distributions drift, tracks performance decay, and stops serving when drift gets critical. The circuit breaker halts predictions at p-value < 0.001—better to return 503 than garbage.

Built for the Adult income classification dataset (RandomForest by default), but the monitoring logic works for any tabular classifier.

## Quick Start

Train, serve, monitor:

```bash
# 1. Train (expects data/raw/adult_train.csv)
python scripts/train_model.py

# 2. Start API
python scripts/run_local.py
# Runs on http://127.0.0.1:8000

# 3. Make a prediction
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "age": 39,
      "workclass": "State-gov",
      "education": "Bachelors",
      "hours-per-week": 40,
      ...
    },
    "include_explanation": true
  }'
```

The dashboard at `http://localhost:8000/` shows drift status, metrics, and alerts. Auto-refreshes every 30 seconds.

## Architecture Decisions

**Async event queue**: Predictions return immediately; monitoring happens in background threads. Tradeoff: if the queue fills (10k max), events drop silently. Serving continues, but you lose history.

**Drift detection**: KS test for numeric features, Chi-square for categorical. Runs every 100 predictions or every 5 minutes, whichever comes first. Not real-time, but fast enough to catch distribution shifts before they cause damage.

**Circuit breaker**: When drift p-value drops below 0.001, the API returns 503. No fallback model—just stops. You can acknowledge drift signals to resume, or wait 24 hours for auto-recovery.

**SQLite by default**: Fine for single-instance deployments. Not production-ready for scale, but zero-config and sufficient for monitoring workloads.

## API

- `POST /predict` - Single prediction with optional explanation
- `POST /predict/batch` - Batch predictions
- `GET /metrics` - Current performance metrics (requires labeled predictions)
- `GET /drift` - Recent drift detection results
- `GET /health` - Health check (includes circuit breaker status)

See `/docs` for full OpenAPI spec.

## Monitoring

Tracks everything: predictions, probabilities, explanations, drift signals, alerts. Stores in SQLite (`monitor.db`).

**Important**: Metrics only update when you provide ground truth labels. Without labels, you're blind to performance degradation until labels arrive (could be days). This is a known limitation—the system assumes you have some way to get labels eventually.

## Project Structure

```
model-health-monitor/
├── model/          # Training, evaluation, prediction
├── monitoring/     # Drift detection, metrics, alerts, circuit breaker
├── service/        # FastAPI app with dashboard
├── storage/        # SQLite database interface
└── scripts/        # Training and monitoring workflows
```

## Requirements

Python 3.8+, scikit-learn, FastAPI, SQLAlchemy. See `requirements.txt`.

## License

MIT
