import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from src.database import load_history_for_features
from src.features import add_technical_indicators

MODEL_PATH = "models/qr_model.pkl"

class NvidiaQuantileModel:
    def __init__(self):
        self.q10 = GradientBoostingRegressor(loss='quantile', alpha=0.1, n_estimators=100)
        self.q50 = GradientBoostingRegressor(loss='quantile', alpha=0.5, n_estimators=100)
        self.q90 = GradientBoostingRegressor(loss='quantile', alpha=0.9, n_estimators=100)
        
        self.features = [
            'close_price', 'avg_sentiment', 
            'price_lag_1', 'price_lag_3', 'price_lag_7',
            'sentiment_ma_3', 'sentiment_ma_7', 'volatility_7'
        ]

    def train(self):
        df = load_history_for_features()
        df = add_technical_indicators(df).dropna()
        if len(df) < 50: return
        
        X = df[self.features]
        y = df['target_price_3d']
        
        self.q10.fit(X, y)
        self.q50.fit(X, y)
        self.q90.fit(X, y)
        
        joblib.dump(self, MODEL_PATH)
        print(f"✅ Trained on {len(df)} samples.")

    def predict(self, df_features):
        try:
            model = joblib.load(MODEL_PATH)
            X = df_features[self.features]
            return {
                "lower": round(model.q10.predict(X)[0], 2),
                "pred": round(model.q50.predict(X)[0], 2),
                "upper": round(model.q90.predict(X)[0], 2)
            }
        except: return None