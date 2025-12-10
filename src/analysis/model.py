import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss

class NvidiaQuantileModel:
    def __init__(self, target_col='target_price_3d', model_name='model_3d'):
        self.target_col = target_col
        self.model_path = f"models/{model_name}.pkl"
        
        params = {
            'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.05,
            'min_samples_leaf': 20, 'min_samples_split': 20
        }
        
        self.q10 = GradientBoostingRegressor(loss='quantile', alpha=0.1, **params)
        self.q50 = GradientBoostingRegressor(loss='quantile', alpha=0.5, **params)
        self.q90 = GradientBoostingRegressor(loss='quantile', alpha=0.9, **params)
        
        self.features = [
            'close_price', 'avg_sentiment', 'pct_change', 
            'volatility_7', 'momentum_7', 'rsi_14', 
            'sentiment_ma_3', 'sent_x_vol', 'price_lag_1', 'price_lag_7',
            'market_fear_index', 'analyst_rating', 'smart_money_divergence' 
        ]

    def _calculate_metrics(self, y_true, q10_pred, q50_pred, q90_pred):
        y_true, q10, q90 = np.array(y_true), np.array(q10_pred), np.array(q90_pred)
        inside = (y_true >= q10) & (y_true <= q90)
        coverage = np.mean(inside)
        
        alpha = 0.1
        width = q90 - q10
        score = width + (2/alpha) * np.maximum(0, q10 - y_true) + (2/alpha) * np.maximum(0, y_true - q90)
        
        return {
            "coverage": round(coverage, 3), 
            "winkler": round(np.mean(score), 3), 
            "pinball_loss": round(mean_pinball_loss(y_true, q50_pred, alpha=0.5), 3)
        }

    def train(self):
        try:
            df = pd.read_csv("data/processed/full_training_dataset.csv")
            df = df.dropna(subset=[self.target_col])
        except: return None

        valid_features = [f for f in self.features if f in df.columns]
        X = df[valid_features]
        # Calculate Returns Target
        y_returns = (df[self.target_col] - df['close_price']) / df['close_price']

        # === 1. PURGED SPLIT (The Fix) ===
        split = int(len(df) * 0.8)
        purge_gap = 20 # 20 days gap to prevent lookahead leakage
        
        X_train = X.iloc[:split]
        y_train = y_returns.iloc[:split]
        
        # Test Set starts AFTER the gap
        X_test = X.iloc[split+purge_gap:]
        y_test = y_returns.iloc[split+purge_gap:]
        
        if len(X_test) == 0: 
            print("⚠️ Not enough data for testing after purge.")
            return None

        # Validation Training
        self.q10.fit(X_train, y_train)
        self.q50.fit(X_train, y_train)
        self.q90.fit(X_train, y_train)
        
        # Metrics Calculation
        current_price_test = df['close_price'].iloc[split+purge_gap:]
        actual_price_test = df[self.target_col].iloc[split+purge_gap:]
        
        p10 = current_price_test * (1 + self.q10.predict(X_test))
        p50 = current_price_test * (1 + self.q50.predict(X_test))
        p90 = current_price_test * (1 + self.q90.predict(X_test))
        
        metrics = self._calculate_metrics(actual_price_test, p10, p50, p90)
        print(f"📉 {self.model_path} Performance (Purged): {metrics}")

        # === 2. FINAL PRODUCTION TRAIN (Full Data) ===
        # We retrain on ALL data for the live API
        self.q10.fit(X, y_returns)
        self.q50.fit(X, y_returns)
        self.q90.fit(X, y_returns)
        
        joblib.dump(self, self.model_path)
        return metrics

    def predict(self, df_features):
        try:
            model = joblib.load(self.model_path)
            valid_features = [f for f in self.features if f in df_features.columns]
            X = df_features[valid_features]
            current_price = X['close_price'].values[0]
            
            ret_10 = model.q10.predict(X)[0]
            ret_50 = model.q50.predict(X)[0]
            ret_90 = model.q90.predict(X)[0]
            
            lower = current_price * (1 + ret_10)
            pred = current_price * (1 + ret_50)
            upper = current_price * (1 + ret_90)
            
            # Safety Checks
            vol = X['volatility_7'].values[0] if 'volatility_7' in X else 0.02
            min_buf = current_price * (vol * 0.5)
            if (upper - lower) < min_buf:
                diff = (min_buf - (upper - lower))/2
                lower -= diff; upper += diff
            
            return {
                "lower": round(lower, 2), "pred": round(pred, 2), "upper": round(upper, 2),
                "return_pred": round(ret_50, 4)
            }
        except: return None