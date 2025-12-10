import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss

class NvidiaQuantileModel:
    def __init__(self, target_col='target_price_3d', model_name='model_3d'):
        self.target_col = target_col
        self.model_path = f"models/{model_name}.pkl"
        
        # --- ROBUST HYPERPARAMETERS ---
        # Optimized for small datasets (prevent overfitting)
        params = {
            'n_estimators': 100,      
            'max_depth': 2,           # Keep trees simple
            'learning_rate': 0.05,
            'min_samples_leaf': 20,   # Ensure statistical significance per leaf
            'min_samples_split': 20
        }
        
        self.q10 = GradientBoostingRegressor(loss='quantile', alpha=0.1, **params)
        self.q50 = GradientBoostingRegressor(loss='quantile', alpha=0.5, **params)
        self.q90 = GradientBoostingRegressor(loss='quantile', alpha=0.9, **params)
        
        self.features = [
            'close_price', 'avg_sentiment', 'pct_change', 'volatility_7',
            'momentum_7', 'rsi_14', 'sentiment_ma_3', 'sent_x_vol',
            'price_lag_1', 'price_lag_7'
        ]

    def _calculate_metrics(self, y_true, q10_pred, q50_pred, q90_pred):
        y_true, q10, q90 = np.array(y_true), np.array(q10_pred), np.array(q90_pred)
        
        # 1. Coverage (Did the actual price fall inside our range?)
        inside = (y_true >= q10) & (y_true <= q90)
        coverage = np.mean(inside)
        
        # 2. Winkler Score (Penalizes missing the target)
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
        except:
            print("❌ Training data missing.")
            return None

        X = df[self.features]
        # --- CRITICAL CHANGE: Train on RETURNS, not Prices ---
        # We predict the % move (e.g., 0.02 for +2%)
        # Formula: (Target - Current) / Current
        y_returns = (df[self.target_col] - df['close_price']) / df['close_price']

        # Validation Split (Last 20%)
        split = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y_returns.iloc[:split], y_returns.iloc[split:]
        
        # Actual Prices for Validation (to calc metrics)
        price_test = df[self.target_col].iloc[split:]
        current_price_test = df['close_price'].iloc[split:]
        
        # Train
        self.q10.fit(X_train, y_train)
        self.q50.fit(X_train, y_train)
        self.q90.fit(X_train, y_train)
        
        # Predict Returns on Test Set
        pred_ret_10 = self.q10.predict(X_test)
        pred_ret_50 = self.q50.predict(X_test)
        pred_ret_90 = self.q90.predict(X_test)
        
        # Convert Returns back to Prices for Metrics
        # Price = Current * (1 + Return)
        pred_price_10 = current_price_test * (1 + pred_ret_10)
        pred_price_50 = current_price_test * (1 + pred_ret_50)
        pred_price_90 = current_price_test * (1 + pred_ret_90)
        
        metrics = self._calculate_metrics(price_test, pred_price_10, pred_price_50, pred_price_90)
        print(f"📉 {self.model_path} Performance: {metrics}")

        # Final Training (Full Data)
        print(f"🏋️ Retraining full model on {len(df)} records...")
        self.q10.fit(X, y_returns)
        self.q50.fit(X, y_returns)
        self.q90.fit(X, y_returns)
        
        joblib.dump(self, self.model_path)
        return metrics

    def predict(self, df_features):
        try:
            model = joblib.load(self.model_path)
            X = df_features[self.features]
            current_price = X['close_price'].values[0]
            
            # 1. Predict Returns
            ret_10 = model.q10.predict(X)[0]
            ret_50 = model.q50.predict(X)[0]
            ret_90 = model.q90.predict(X)[0]
            
            # 2. Convert to Prices
            lower = current_price * (1 + ret_10)
            pred = current_price * (1 + ret_50)
            upper = current_price * (1 + ret_90)
            
            # 3. Safety Floor (Ensure minimum width based on volatility)
            # If the model predicts 0% move, force a small volatility buffer
            volatility = X['volatility_7'].values[0] if 'volatility_7' in X else 0.02
            min_buffer = current_price * (volatility * 0.5)
            
            if (upper - lower) < min_buffer:
                lower = pred - min_buffer
                upper = pred + min_buffer
            
            return {
                "lower": round(lower, 2),
                "pred": round(pred, 2),
                "upper": round(upper, 2)
            }
        except Exception as e:
            print(f"Prediction Error: {e}")
            return None