from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import pandas as pd
import numpy as np  # <--- CRITICAL IMPORT

app = FastAPI(title="Nvidia Market Oracle API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# File Paths
LATEST_PREDICTION_FILE = "data/latest_prediction.json"
PREDICTIONS_CSV = "data/live_predictions.csv"
METRICS_CSV = "data/model_performance.csv"

@app.get("/")
def home():
    return {"status": "online", "service": "NVDA Predictor"}

@app.get("/predict")
def get_prediction():
    if not os.path.exists(LATEST_PREDICTION_FILE):
        return {"error": "System initializing. Please wait."}
    try:
        with open(LATEST_PREDICTION_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": f"Failed to load prediction: {str(e)}"}

@app.get("/history")
def get_history():
    """Returns historical price vs prediction data for charts."""
    if not os.path.exists(PREDICTIONS_CSV):
        return []
    
    try:
        df = pd.read_csv(PREDICTIONS_CSV)
        
        if df.empty: return []

        # Limit to last 100 points
        df = df.tail(100)
        
        # --- CRITICAL FIX: SANITIZE DATA ---
        # 1. Replace Infinite values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # 2. Replace NaN with None (which becomes JSON 'null')
        df = df.where(pd.notnull(df), None)
        
        return df.to_dict(orient="records")
        
    except Exception as e:
        print(f"⚠️ API Error reading history: {e}")
        return []

@app.get("/performance")
def get_performance_metrics():
    if not os.path.exists(METRICS_CSV):
        return []
    try:
        df = pd.read_csv(METRICS_CSV, names=["timestamp", "model", "coverage", "winkler"])
        
        # Sanitize metrics too
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.where(pd.notnull(df), None)
        
        return df.to_dict(orient="records")
    except Exception as e:
        print(f"⚠️ API Error reading performance: {e}")
        return []