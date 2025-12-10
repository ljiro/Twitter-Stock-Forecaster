import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
import os

# ---------------------------------------------------------
# 1. SHARED HELPERS
# ---------------------------------------------------------
def custom_quantile_loss(quantiles):
    def quantile_loss_non_crossing(y_true, y_pred):
        total_loss = 0.0
        for i, q in enumerate(quantiles):
            error = y_true - y_pred[:, i:i+1]
            pinball = tf.reduce_mean(tf.maximum(q * error, (q - 1) * error))
            total_loss += pinball
        return total_loss
    return quantile_loss_non_crossing

def calculate_metrics(y_true, q10_pred, q50_pred, q90_pred):
    """
    Calculates Coverage and Winkler Score.
    """
    y_true, q10, q90 = np.array(y_true), np.array(q10_pred), np.array(q90_pred)
    
    # Coverage
    inside = (y_true >= q10) & (y_true <= q90)
    coverage = np.mean(inside)
    
    # Winkler Score
    alpha = 0.1
    width = q90 - q10
    # Penalty for missing the target
    penalty_low = (2/alpha) * np.maximum(0, q10 - y_true)
    penalty_high = (2/alpha) * np.maximum(0, y_true - q90)
    score = width + penalty_low + penalty_high
    
    return {
        "coverage": round(float(coverage), 3),
        "winkler": round(float(np.mean(score)), 3)
    }

# ---------------------------------------------------------
# 2. Linear Quantile Regression
# ---------------------------------------------------------
class NvidiaLinearQR:
    def __init__(self, target_col='target_price_3d', model_name='linear_qr'):
        self.model_path = f"models/{model_name}.pkl"
        self.target_col = target_col
        self.features = ['avg_sentiment', 'pct_change', 'volatility_7', 'rsi_14', 'momentum_7']

    def train(self, df):
        # Prepare Returns Data
        y = ((df[self.target_col] - df['close_price']) / df['close_price']).values
        X = df[self.features].fillna(0).values

        # --- VALIDATION STEP ---
        split = int(len(df) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Train Validation Models
        q10 = QuantileRegressor(quantile=0.1, solver='highs').fit(X_train, y_train)
        q50 = QuantileRegressor(quantile=0.5, solver='highs').fit(X_train, y_train)
        q90 = QuantileRegressor(quantile=0.9, solver='highs').fit(X_train, y_train)
        
        # Calculate Metrics (Convert Returns -> Price for fair comparison)
        current_prices = df['close_price'].iloc[split:].values
        actual_prices = df[self.target_col].iloc[split:].values
        
        pred_p10 = current_prices * (1 + q10.predict(X_test))
        pred_p50 = current_prices * (1 + q50.predict(X_test))
        pred_p90 = current_prices * (1 + q90.predict(X_test))
        
        metrics = calculate_metrics(actual_prices, pred_p10, pred_p50, pred_p90)
        
        # --- PRODUCTION RETRAINING ---
        model_dict = {
            'q10': QuantileRegressor(quantile=0.1, solver='highs').fit(X, y),
            'q50': QuantileRegressor(quantile=0.5, solver='highs').fit(X, y),
            'q90': QuantileRegressor(quantile=0.9, solver='highs').fit(X, y)
        }
        joblib.dump(model_dict, self.model_path)
        return metrics

    def predict(self, df_features):
        try:
            models = joblib.load(self.model_path)
            current_price = df_features['close_price'].values[0]
            X = df_features[self.features].fillna(0).values
            
            return {
                "lower": round(current_price * (1 + models['q10'].predict(X)[0]), 2),
                "pred": round(current_price * (1 + models['q50'].predict(X)[0]), 2),
                "upper": round(current_price * (1 + models['q90'].predict(X)[0]), 2)
            }
        except: return None

# ---------------------------------------------------------
# 3. MQLSTM (Keras)
# ---------------------------------------------------------
class NvidiaMQLSTM:
    def __init__(self, target_col='target_price_3d', model_name='mqlstm'):
        self.model_path = f"models/{model_name}.keras"
        self.scaler_path = f"models/{model_name}_scaler.pkl"
        self.target_col = target_col
        self.features = ['close_price', 'avg_sentiment', 'pct_change', 'volatility_7', 'rsi_14']
        self.scaler = StandardScaler()

    def _build_model(self, input_shape):
        model = Sequential([
            Input(shape=input_shape),
            LSTM(50, activation='relu', return_sequences=False),
            Dense(50, activation='relu'),
            Dense(3, activation='linear')
        ])
        model.compile(optimizer='adam', loss=custom_quantile_loss([0.1, 0.5, 0.9]))
        return model

    def train(self, df):
        y = ((df[self.target_col] - df['close_price']) / df['close_price']).values
        X = df[self.features].fillna(0).values
        
        # Validation Split
        split = int(len(df) * 0.8)
        
        # Scaling (Fit on Train ONLY to avoid leakage)
        X_train_raw, X_test_raw = X[:split], X[split:]
        self.scaler.fit(X_train_raw)
        
        X_train = np.expand_dims(self.scaler.transform(X_train_raw), axis=1)
        X_test = np.expand_dims(self.scaler.transform(X_test_raw), axis=1)
        y_train, y_test = y[:split], y[split:]
        
        # Train Validation Model
        val_model = self._build_model((1, len(self.features)))
        val_model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)
        
        # Calculate Metrics
        preds = val_model.predict(X_test, verbose=0) # [q10, q50, q90]
        
        current_prices = df['close_price'].iloc[split:].values
        actual_prices = df[self.target_col].iloc[split:].values
        
        pred_p10 = current_prices * (1 + preds[:, 0])
        pred_p50 = current_prices * (1 + preds[:, 1])
        pred_p90 = current_prices * (1 + preds[:, 2])
        
        metrics = calculate_metrics(actual_prices, pred_p10, pred_p50, pred_p90)
        
        # --- PRODUCTION RETRAINING ---
        # Retrain on full data
        self.scaler.fit(X) # Refit scaler on full data
        X_full = np.expand_dims(self.scaler.transform(X), axis=1)
        
        final_model = self._build_model((1, len(self.features)))
        final_model.fit(X_full, y, epochs=50, batch_size=16, verbose=0)
        
        final_model.save(self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        return metrics

    def predict(self, df_features):
        try:
            model = load_model(self.model_path, custom_objects={'quantile_loss_non_crossing': custom_quantile_loss([0.1, 0.5, 0.9])}, compile=False)
            scaler = joblib.load(self.scaler_path)
            
            current_price = df_features['close_price'].values[0]
            X = df_features[self.features].fillna(0).values
            X_scaled = scaler.transform(X)
            X_reshaped = np.expand_dims(X_scaled, axis=1)
            
            preds = model.predict(X_reshaped, verbose=0)[0]
            
            return {
                "lower": round(current_price * (1 + preds[0]), 2),
                "pred": round(current_price * (1 + preds[1]), 2),
                "upper": round(current_price * (1 + preds[2]), 2)
            }
        except: return None

# ---------------------------------------------------------
# 4. Keras QRNN
# ---------------------------------------------------------
class NvidiaQRNN_Keras:
    def __init__(self, target_col='target_price_3d', model_name='qrnn_keras'):
        self.model_path = f"models/{model_name}.keras"
        self.scaler_path = f"models/{model_name}_scaler.pkl"
        self.target_col = target_col
        self.features = ['close_price', 'avg_sentiment', 'pct_change', 'volatility_7', 'rsi_14']
        self.scaler = StandardScaler()

    def _build_model(self, input_dim):
        input_layer = Input(shape=(input_dim,))
        h1 = Dense(64, activation='relu')(input_layer)
        h2 = Dense(32, activation='relu')(h1)
        output_layer = Dense(3, activation='linear')(h2)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=Adam(0.001), loss=custom_quantile_loss([0.1, 0.5, 0.9]))
        return model

    def train(self, df):
        y = ((df[self.target_col] - df['close_price']) / df['close_price']).values
        X = df[self.features].fillna(0).values
        
        # Validation Split
        split = int(len(df) * 0.8)
        X_train_raw, X_test_raw = X[:split], X[split:]
        self.scaler.fit(X_train_raw)
        
        X_train = self.scaler.transform(X_train_raw)
        X_test = self.scaler.transform(X_test_raw)
        y_train, y_test = y[:split], y[split:]
        
        # Train Validation Model
        val_model = self._build_model(len(self.features))
        val_model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
        
        # Metrics
        preds = val_model.predict(X_test, verbose=0)
        current_prices = df['close_price'].iloc[split:].values
        actual_prices = df[self.target_col].iloc[split:].values
        
        pred_p10 = current_prices * (1 + preds[:, 0])
        pred_p50 = current_prices * (1 + preds[:, 1])
        pred_p90 = current_prices * (1 + preds[:, 2])
        
        metrics = calculate_metrics(actual_prices, pred_p10, pred_p50, pred_p90)
        
        # Retrain Full
        self.scaler.fit(X)
        X_full = self.scaler.transform(X)
        final_model = self._build_model(len(self.features))
        final_model.fit(X_full, y, epochs=100, batch_size=16, verbose=0)
        
        final_model.save(self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        return metrics

    def predict(self, df_features):
        try:
            model = load_model(self.model_path, compile=False)
            scaler = joblib.load(self.scaler_path)
            
            current_price = df_features['close_price'].values[0]
            X = df_features[self.features].fillna(0).values
            X_scaled = scaler.transform(X)
            
            preds = model.predict(X_scaled, verbose=0)[0]
            
            return {
                "lower": round(current_price * (1 + preds[0]), 2),
                "pred": round(current_price * (1 + preds[1]), 2),
                "upper": round(current_price * (1 + preds[2]), 2)
            }
        except: return None