import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from src.database import load_training_data, DB_PATH
from src.features import add_technical_indicators # <--- NEW IMPORT

MODEL_PATH = "models/qr_model.pkl"

class NvidiaQuantileModel:
    def __init__(self):
        # We increase n_estimators slightly as data is more complex now
        self.q10 = GradientBoostingRegressor(loss='quantile', alpha=0.1, n_estimators=200)
        self.q50 = GradientBoostingRegressor(loss='quantile', alpha=0.5, n_estimators=200)
        self.q90 = GradientBoostingRegressor(loss='quantile', alpha=0.9, n_estimators=200)
        
        # Define the exact feature list so we never mismatch columns
        self.features = [
            'close_price', 'avg_sentiment', 
            'price_lag_1', 'price_lag_3', 'price_lag_7',
            'sentiment_ma_3', 'sentiment_ma_7',
            'volatility_7', 'roc_5'
        ]

    def train(self):
        print("📊 Loading training data...")
        df_raw = load_training_data()
        
        # --- FEATURE ENGINEERING STEP ---
        df_processed = add_technical_indicators(df_raw)
        
        if len(df_processed) < 50:
            print("⚠️ Not enough data (need >50 days for lags).")
            return

        X = df_processed[self.features]
        y = df_processed['target_price_3d']

        self.q10.fit(X, y)
        self.q50.fit(X, y)
        self.q90.fit(X, y)
        
        joblib.dump(self, MODEL_PATH)
        print(f"✅ Model trained on {len(df_processed)} samples with {len(self.features)} features.")

    def predict_from_features(self, df_features):
        """
        Expects a DataFrame that already has the technical indicators calculated
        """
        try:
            model = joblib.load(MODEL_PATH)
            
            # Ensure order matches training
            X = df_features[self.features]
            
            return {
                "lower": round(model.q10.predict(X)[0], 2),
                "predicted": round(model.q50.predict(X)[0], 2),
                "upper": round(model.q90.predict(X)[0], 2)
            }
        except FileNotFoundError:
            print("⚠️ Model file not found.")
            return None