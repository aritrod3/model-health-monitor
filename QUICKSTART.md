# Quick Start Guide

## Prerequisites
- Python 3.8+
- Data files in `data/raw/` (adult_train.csv, adult_test.csv)

## Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

## Step 2: Train Model (First Time)
```bash
python scripts/train_model.py
```
**What happens:**
- Loads training data from `data/raw/adult_train.csv`
- Trains RandomForest classifier
- Saves model to `model/model.pkl`
- Evaluates on test data
- Stores baseline metrics in database

**Expected output:**
- Model file created: `model/model.pkl`
- Database created: `monitor.db`
- Logs showing training progress and metrics

## Step 3: Start API Service
```bash
python scripts/run_local.py
```

**Or with auto-reload:**
```bash
python scripts/run_local.py --reload
```

**What happens:**
- FastAPI server starts
- Database connects
- Model loads (if available)
- Server runs on `http://127.0.0.1:8000`

**Expected output:**
```
INFO:     Started server process
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

## Step 4: Access Dashboard
Open browser: **http://localhost:8000/**

**What you'll see:**
- Dashboard with stats cards (model status, predictions, alerts)
- Current metrics table
- Recent alerts table
- Drift status table
- Auto-refreshes every 30 seconds

## What You Can Do

### 1. View Dashboard
- Visit `http://localhost:8000/`
- See model status, metrics, alerts, drift status
- Click "Refresh" to update manually

### 2. Make Predictions
- Use FastAPI docs: `http://localhost:8000/docs`
- POST to `/predict` with feature data
- Get predictions with probabilities and explanations

### 3. Check Metrics
- Dashboard shows latest metrics
- Or: `GET http://localhost:8000/metrics`

### 4. Monitor System
- Run `python scripts/monitor.py` to check monitoring
- View alerts and drift in dashboard

## Expected State (After Training)

**Files created:**
- `model/model.pkl` - Trained model
- `monitor.db` - SQLite database
- `logs/` - Log files (if configured)

**Database contains:**
- Baseline metrics (accuracy, precision, F1, etc.)
- No predictions yet (until you make API calls)
- No alerts yet (until drift/degradation detected)

## Testing the System

### Make a Prediction (via curl):
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "age": 39,
      "workclass": "State-gov",
      "fnlwgt": 77516,
      "education": "Bachelors",
      "education-num": 13,
      "marital-status": "Never-married",
      "occupation": "Adm-clerical",
      "relationship": "Not-in-family",
      "race": "White",
      "sex": "Male",
      "capital-gain": 2174,
      "capital-loss": 0,
      "hours-per-week": 40,
      "native-country": "United-States"
    },
    "include_explanation": true
  }'
```

### Check Health:
```bash
curl http://localhost:8000/health
```

## Troubleshooting

**Model not loading?**
- Check `model/model.pkl` exists
- Run `python scripts/train_model.py` first

**Database errors?**
- Check `monitor.db` exists
- Delete `monitor.db` and re-run training script

**Port already in use?**
- Change port: `python scripts/run_local.py --port 8001`
- Or kill process using port 8000

**Dashboard empty?**
- Make some predictions first
- Check database has data
- Look at browser console for errors
