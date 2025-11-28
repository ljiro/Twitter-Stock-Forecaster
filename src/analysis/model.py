import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from src.database import load_training_data

MODEL_PATH = "models/qr_model.pkl"

class NvidiaQuantileModel:
    def __init__(self):
        # 10th, 50th, 90th percentiles
        self.q10 = GradientBoostingRegressor(loss='quantile', alpha=0.1, n_estimators=100)
        self.q50 = GradientBoostingRegressor(loss='quantile', alpha=0.5, n_estimators=100)
        self.q90 = GradientBoostingRegressor(loss='quantile', alpha=0.9, n_estimators=100)

    def train(self):
        df = load_training_data()
        if len(df) < 50:
            print("⚠️ Not enough data to train.")
            return

        X = df[['avg_sentiment', 'close_price']]
        y = df['target_price_3d']

        self.q10.fit(X, y)
        self.q50.fit(X, y)
        self.q90.fit(X, y)
        
        joblib.dump(self, MODEL_PATH)
        print("✅ Model retrained & saved.")

    def predict(self, sentiment, price):
        try:
            model = joblib.load(MODEL_PATH)
            X_new = pd.DataFrame([[sentiment, price]], columns=['avg_sentiment', 'close_price'])
            return {
                "lower": round(model.q10.predict(X_new)[0], 2),
                "predicted": round(model.q50.predict(X_new)[0], 2),
                "upper": round(model.q90.predict(X_new)[0], 2)
            }
        except Exception:
            print("⚠️ Model file not found. Please train first.")
            return None