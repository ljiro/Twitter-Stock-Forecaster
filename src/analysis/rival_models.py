import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from statsmodels.regression.quantile_regression import QuantReg
import statsmodels.api as sm

# ---------------------------------------------------------
# 1. HELPER: Quantile Loss Function (Fixed Dimensions)
# ---------------------------------------------------------
def quantile_loss(preds, target, quantiles=[0.1, 0.5, 0.9]):
    """
    preds: tensor of shape (batch_size, 3) -> [q10, q50, q90]
    target: tensor of shape (batch_size, 1) -> True Return
    """
    loss = 0
    for i, q in enumerate(quantiles):
        # Extract the i-th quantile prediction and reshape to (N, 1)
        pred_q = preds[:, i].unsqueeze(1)
        
        # Calculate Error
        error = target - pred_q
        loss += torch.max((q - 1) * error, q * error).mean()
        
    return loss

# ---------------------------------------------------------
# 2. QRNN (Quantile Regression Neural Network)
# ---------------------------------------------------------
class QRNN_Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1) # Reduced dropout for small data
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 3) # q10, q50, q90

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return self.output(x)

class NvidiaQRNN:
    def __init__(self, target_col='target_price_3d', model_name='qrnn_3d'):
        self.model_path = f"models/{model_name}.pkl"
        self.scaler = StandardScaler()
        self.model = None
        # Same features as GB model
        self.features = [
            'close_price', 'avg_sentiment', 'pct_change', 'volatility_7',
            'momentum_7', 'rsi_14', 'sentiment_ma_3', 'sent_x_vol',
            'price_lag_1', 'price_lag_7'
        ]
        self.target_col = target_col

    def train(self, df):
        try:
            # Prepare Data
            X = df[self.features].values
            
            # Predict Returns: (Target - Current) / Current
            y = ((df[self.target_col] - df['close_price']) / df['close_price']).values
            
            # Scale Features
            X_scaled = self.scaler.fit_transform(X)
            
            # Convert to Tensors
            X_tensor = torch.FloatTensor(X_scaled)
            y_tensor = torch.FloatTensor(y).view(-1, 1) # Force (N, 1) shape

            # Initialize Network
            self.model = QRNN_Net(input_dim=len(self.features))
            optimizer = optim.Adam(self.model.parameters(), lr=0.005) # Lower LR
            
            # Training Loop
            self.model.train()
            for epoch in range(300): # More epochs
                optimizer.zero_grad()
                preds = self.model(X_tensor)
                
                # FIXED: Pass y_tensor directly (N, 1)
                loss = quantile_loss(preds, y_tensor) 
                
                loss.backward()
                optimizer.step()
                
            # Save Scaler and State Dict together
            joblib.dump({'model_state': self.model.state_dict(), 'scaler': self.scaler}, self.model_path)
            return {"status": "trained"}
            
        except Exception as e:
            print(f"QRNN Training Error: {e}")
            return {"status": "failed"}

    def predict(self, df_features):
        try:
            checkpoint = joblib.load(self.model_path)
            self.scaler = checkpoint['scaler']
            
            # Re-init model structure
            self.model = QRNN_Net(input_dim=len(self.features))
            self.model.load_state_dict(checkpoint['model_state'])
            self.model.eval()

            current_price = df_features['close_price'].values[0]
            X = df_features[self.features].values
            X_scaled = self.scaler.transform(X)
            
            with torch.no_grad():
                preds_ret = self.model(torch.FloatTensor(X_scaled)).numpy()[0]
            
            # Convert Return -> Price
            return {
                "lower": round(current_price * (1 + preds_ret[0]), 2),
                "pred": round(current_price * (1 + preds_ret[1]), 2),
                "upper": round(current_price * (1 + preds_ret[2]), 2)
            }
        except: return None

# ---------------------------------------------------------
# 3. CAViaR (Econometric Proxy)
# ---------------------------------------------------------
class NvidiaCAViaR:
    """
    Uses Quantile Autoregression on Returns.
    """
    def __init__(self, target_col='target_price_3d', model_name='caviar_3d'):
        self.model_path = f"models/{model_name}.pkl"
        self.target_col = target_col
        self.models = {} 

    def train(self, df):
        try:
            # Target: Future Return
            y = ((df[self.target_col] - df['close_price']) / df['close_price'])
            
            # Feature: Absolute value of lagged returns (Volatility Proxy)
            X = df[['pct_change']].shift(1).fillna(0).abs()
            X = sm.add_constant(X)
            
            # Train 3 separate quantile regressions
            self.models['q10'] = QuantReg(y, X).fit(q=0.1)
            self.models['q50'] = QuantReg(y, X).fit(q=0.5)
            self.models['q90'] = QuantReg(y, X).fit(q=0.9)
            
            joblib.dump(self.models, self.model_path)
            return {"status": "trained"}
        except: 
            return {"status": "failed"}

    def predict(self, df_features):
        try:
            models = joblib.load(self.model_path)
            current_price = df_features['close_price'].values[0]
            
            # Create feature: Abs(Yesterday's Return)
            val = abs(df_features['pct_change'].values[0])
            exog = [1.0, val] # Constant + Feature
            
            ret_10 = models['q10'].predict(exog)[0]
            ret_50 = models['q50'].predict(exog)[0]
            ret_90 = models['q90'].predict(exog)[0]
            
            return {
                "lower": round(current_price * (1 + ret_10), 2),
                "pred": round(current_price * (1 + ret_50), 2),
                "upper": round(current_price * (1 + ret_90), 2)
            }
        except: return None