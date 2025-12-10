from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json
import os

app = FastAPI(title="Nvidia Stock Predictor API")

# Allow frontend to access this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

LATEST_PREDICTION_FILE = "data/latest_prediction.json"

@app.get("/")
def home():
    return {"status": "online", "service": "NVDA Predictor"}

@app.get("/predict")
def get_prediction():
    """Returns the latest prediction for Today, Tomorrow, and +3 Days"""
    if not os.path.exists(LATEST_PREDICTION_FILE):
        return {"error": "No prediction available yet. Wait for next sync."}
    
    with open(LATEST_PREDICTION_FILE, "r") as f:
        data = json.load(f)
    return data

# To run: uvicorn src.api:app --reload