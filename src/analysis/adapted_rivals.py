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

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

FEATURES = [
    'close_price', 'avg_sentiment', 'pct_change', 
    'volatility_7', 'momentum_7', 'rsi_14', 
    'sentiment_ma_3', 'sent_x_vol', 'price_lag_1', 'price_lag_7',
    'market_fear_index', 'analyst_rating', 'smart_money_divergence'
]

def custom_quantile_loss(quantiles):
    def quantile_loss_non_crossing(y_true, y_pred):
        total_loss = 0.0
        for i, q in enumerate(quantiles):
            error = y_true - y_pred[:, i:i+1]
            total_loss += tf.reduce_mean(tf.maximum(q * error, (q - 1) * error))
        return total_loss
    return quantile_loss_non_crossing

def calculate_metrics(y_true, q10_pred, q50_pred, q90_pred):
    y_true, q10, q90 = np.array(y_true), np.array(q10_pred), np.array(q90_pred)
    inside = (y_true >= q10) & (y_true <= q90)
    width = q90 - q10
    penalty = (20) * np.maximum(0, q10 - y_true) + (20) * np.maximum(0, y_true - q90) 
    return {"coverage": round(float(np.mean(inside)), 3), "winkler": round(float(np.mean(width + penalty)), 3)}

# --- 1. Linear QR ---
class NvidiaLinearQR:
    def __init__(self, target_col='target_price_3d', model_name='linear_qr'):
        self.model_path = f"models/{model_name}.pkl"
        self.target_col = target_col
        self.features = ['avg_sentiment', 'pct_change', 'volatility_7', 'rsi_14', 'market_fear_index'] 

    def train(self, df):
        try:
            y = ((df[self.target_col] - df['close_price']) / df['close_price']).values
            valid_feats = [f for f in self.features if f in df.columns]
            X = df[valid_feats].fillna(0).values

            split = int(len(df) * 0.8)
            purge = 20
            
            # Train on Purged Split for Metrics
            q10 = QuantileRegressor(quantile=0.1, solver='highs').fit(X[:split], y[:split])
            q50 = QuantileRegressor(quantile=0.5, solver='highs').fit(X[:split], y[:split])
            q90 = QuantileRegressor(quantile=0.9, solver='highs').fit(X[:split], y[:split])
            
            # Eval
            if len(X[split+purge:]) > 0:
                cp = df['close_price'].iloc[split+purge:].values
                ap = df[self.target_col].iloc[split+purge:].values
                p10 = cp * (1+q10.predict(X[split+purge:]))
                p50 = cp * (1+q50.predict(X[split+purge:]))
                p90 = cp * (1+q90.predict(X[split+purge:]))
                metrics = calculate_metrics(ap, p10, p50, p90)
            else:
                metrics = {"coverage": 0.0, "winkler": 0.0}

            # Retrain Full & Save
            final_models = {
                'q10': QuantileRegressor(0.1, solver='highs').fit(X, y),
                'q50': QuantileRegressor(0.5, solver='highs').fit(X, y),
                'q90': QuantileRegressor(0.9, solver='highs').fit(X, y),
                'features': valid_feats
            }
            joblib.dump(final_models, self.model_path)
            return metrics
        except Exception as e:
            print(f"LinearQR Train Error: {e}")
            return {"coverage": 0.0, "winkler": 0.0}

    def predict(self, df_features):
        try:
            m = joblib.load(self.model_path)
            X = df_features[m['features']].fillna(0).values
            cp = df_features['close_price'].values[0]
            return {
                "lower": round(cp * (1 + m['q10'].predict(X)[0]), 2),
                "pred": round(cp * (1 + m['q50'].predict(X)[0]), 2),
                "upper": round(cp * (1 + m['q90'].predict(X)[0]), 2)
            }
        except: return None

# --- 2. MQLSTM ---
class NvidiaMQLSTM:
    def __init__(self, target_col='target_price_3d', model_name='mqlstm'):
        self.model_path = f"models/{model_name}.keras"
        self.scaler_path = f"models/{model_name}_scaler.pkl"
        self.target_col = target_col
        self.features = FEATURES
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
        try:
            y = ((df[self.target_col] - df['close_price']) / df['close_price']).values
            valid_feats = [f for f in self.features if f in df.columns]
            X = df[valid_feats].fillna(0).values
            
            split = int(len(df) * 0.8)
            purge = 20
            
            self.scaler.fit(X[:split])
            X_train = np.expand_dims(self.scaler.transform(X[:split]), axis=1)
            y_train = y[:split]
            
            val_model = self._build_model((1, len(valid_feats)))
            val_model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)
            
            if len(X[split+purge:]) > 0:
                X_test = np.expand_dims(self.scaler.transform(X[split+purge:]), axis=1)
                preds = val_model.predict(X_test, verbose=0)
                cp = df['close_price'].iloc[split+purge:].values
                ap = df[self.target_col].iloc[split+purge:].values
                metrics = calculate_metrics(ap, cp*(1+preds[:,0]), cp*(1+preds[:,1]), cp*(1+preds[:,2]))
            else:
                metrics = {"coverage": 0.0, "winkler": 0.0}
            
            # Full Train
            self.scaler.fit(X)
            X_full = np.expand_dims(self.scaler.transform(X), axis=1)
            final_model = self._build_model((1, len(valid_feats)))
            final_model.fit(X_full, y, epochs=50, batch_size=16, verbose=0)
            
            final_model.save(self.model_path)
            joblib.dump({'scaler': self.scaler, 'features': valid_feats}, self.scaler_path)
            return metrics
        except Exception as e:
            print(f"MQLSTM Train Error: {e}")
            return {"coverage": 0.0, "winkler": 0.0}

    def predict(self, df_features):
        try:
            model = load_model(self.model_path, custom_objects={'quantile_loss_non_crossing': custom_quantile_loss([0.1, 0.5, 0.9])}, compile=False)
            meta = joblib.load(self.scaler_path)
            X = df_features[meta['features']].fillna(0).values
            X = np.expand_dims(meta['scaler'].transform(X), axis=1)
            preds = model.predict(X, verbose=0)[0]
            cp = df_features['close_price'].values[0]
            return {"lower": round(cp*(1+preds[0]),2), "pred": round(cp*(1+preds[1]),2), "upper": round(cp*(1+preds[2]),2)}
        except: return None

# --- 3. QRNN ---
class NvidiaQRNN_Keras:
    def __init__(self, target_col='target_price_3d', model_name='qrnn_keras'):
        self.model_path = f"models/{model_name}.keras"
        self.scaler_path = f"models/{model_name}_scaler.pkl"
        self.target_col = target_col
        self.features = FEATURES
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
        try:
            y = ((df[self.target_col] - df['close_price']) / df['close_price']).values
            valid_feats = [f for f in self.features if f in df.columns]
            X = df[valid_feats].fillna(0).values
            
            split = int(len(df) * 0.8)
            purge = 20
            
            self.scaler.fit(X[:split])
            X_train = self.scaler.transform(X[:split])
            y_train = y[:split]
            
            val_model = self._build_model(len(valid_feats))
            val_model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
            
            if len(X[split+purge:]) > 0:
                X_test = self.scaler.transform(X[split+purge:])
                preds = val_model.predict(X_test, verbose=0)
                cp = df['close_price'].iloc[split+purge:].values
                ap = df[self.target_col].iloc[split+purge:].values
                metrics = calculate_metrics(ap, cp*(1+preds[:,0]), cp*(1+preds[:,1]), cp*(1+preds[:,2]))
            else:
                metrics = {"coverage": 0.0, "winkler": 0.0}
            
            # Full Train
            self.scaler.fit(X)
            final_model = self._build_model(len(valid_feats))
            final_model.fit(self.scaler.transform(X), y, epochs=100, batch_size=16, verbose=0)
            
            final_model.save(self.model_path)
            joblib.dump({'scaler': self.scaler, 'features': valid_feats}, self.scaler_path)
            return metrics
        except Exception as e:
            print(f"QRNN Train Error: {e}")
            return {"coverage": 0.0, "winkler": 0.0}

    def predict(self, df_features):
        try:
            model = load_model(self.model_path, compile=False)
            meta = joblib.load(self.scaler_path)
            X = df_features[meta['features']].fillna(0).values
            X = meta['scaler'].transform(X)
            preds = model.predict(X, verbose=0)[0]
            cp = df_features['close_price'].values[0]
            return {"lower": round(cp*(1+preds[0]),2), "pred": round(cp*(1+preds[1]),2), "upper": round(cp*(1+preds[2]),2)}
        except: return None