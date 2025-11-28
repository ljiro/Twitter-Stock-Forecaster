import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import os

MODEL_PATH = "models/qr_model.pkl"

class NvidiaQuantileModel:
    def __init__(self):
        # We need 3 models: Lower (10%), Median (50%), Upper (90%)
        self.q10 = GradientBoostingRegressor(loss='quantile', alpha=0.1, n_estimators=100)
        self.q50 = GradientBoostingRegressor(loss='quantile', alpha=0.5, n_estimators=100)
        self.q90 = GradientBoostingRegressor(loss='quantile', alpha=0.9, n_estimators=100)
        self.is_trained = False

    def train(self, data_path="data/processed/training_data.csv"):
        """
        Expects a CSV with columns: ['sentiment', 'price', 'target_price_3d']
        """
        if not os.path.exists(data_path):
            print("⚠️ No training data found. Skipping training.")
            return

        df = pd.read_csv(data_path)
        X = df[['sentiment', 'price']]
        y = df['target_price_3d']

        self.q10.fit(X, y)
        self.q50.fit(X, y)
        self.q90.fit(X, y)
        self.is_trained = True
        
        # Save the object
        joblib.dump(self, MODEL_PATH)
        print("✅ Model retrained and saved.")

    def predict(self, current_sentiment, current_price):
        """Returns (Lower, Median, Upper)"""
        if not os.path.exists(MODEL_PATH):
            print("⚠️ No model found on disk.")
            return None
        
        # Load model if not in memory
        loaded_model = joblib.load(MODEL_PATH)
        
        X_new = pd.DataFrame([[current_sentiment, current_price]], columns=['sentiment', 'price'])
        
        lower = loaded_model.q10.predict(X_new)[0]
        median = loaded_model.q50.predict(X_new)[0]
        upper = loaded_model.q90.predict(X_new)[0]
        
        return {
            "lower_bound": round(lower, 2),
            "predicted_price": round(median, 2),
            "upper_bound": round(upper, 2)
        }