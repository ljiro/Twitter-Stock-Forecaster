from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import pandas as pd

app = FastAPI(title="Nvidia Stock Predictor API")

# Allow frontend to access this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# File Paths (Must match main_orchestrator.py)
LATEST_PREDICTION_FILE = "data/latest_prediction.json"
PREDICTIONS_CSV = "data/live_predictions.csv"
METRICS_CSV = "data/model_performance.csv"

@app.get("/")
def home():
    return {"status": "online", "service": "NVDA Predictor"}

@app.get("/predict")
def get_prediction():
    """Returns the latest JSON prediction with model details."""
    if not os.path.exists(LATEST_PREDICTION_FILE):
        return {"error": "System initializing. Please wait."}
    
    try:
        with open(LATEST_PREDICTION_FILE, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        return {"error": f"Failed to load prediction: {str(e)}"}

@app.get("/history")
def get_history():
    """Returns historical price vs prediction data for charts."""
    if not os.path.exists(PREDICTIONS_CSV):
        return []
    
    try:
        # Read CSV and return as list of dicts (JSON array)
        df = pd.read_csv(PREDICTIONS_CSV)
        # Limit to last 100 points to keep API fast
        df = df.tail(100) 
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/performance")
def get_performance_metrics():
    """Returns the training performance history (Winkler scores)."""
    if not os.path.exists(METRICS_CSV):
        return []
    
    try:
        # Returns [timestamp, model_name, coverage, winkler]
        df = pd.read_csv(METRICS_CSV, names=["timestamp", "model", "coverage", "winkler"])
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run command: uvicorn src.api:app --reload