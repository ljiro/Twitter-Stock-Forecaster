import pandas as pd
import numpy as np

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def add_technical_indicators(df):
    """
    Transforms raw data into a Rich Quant Dataset.
    Now includes Institutional Features.
    """
    df = df.copy()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
    
    # --- 1. Price Derivatives ---
    df['pct_change'] = df['close_price'].pct_change()
    df['volatility_7'] = df['pct_change'].rolling(window=7).std()
    df['momentum_7'] = df['close_price'] / df['close_price'].rolling(window=7).mean()
    df['rsi_14'] = calculate_rsi(df['close_price'], period=14)

    # --- 2. Sentiment Derivatives ---
    df['sentiment_ma_3'] = df['avg_sentiment'].rolling(window=3).mean()
    df['sent_x_vol'] = df['avg_sentiment'] * df['volatility_7']

    # --- 3. Institutional Derivatives (NEW) ---
    # Put/Call Ratio is already raw, but we can look for spikes
    # If PCR exists in columns, use it, else default 1.0 (for safety)
    if 'put_call_ratio' not in df.columns: df['put_call_ratio'] = 1.0
    if 'analyst_rating' not in df.columns: df['analyst_rating'] = 3.0
    
    # "Smart Money Divergence": When Sentiment is High but Analysts are Selling (Rating > 3)
    # (Note: Yahoo Rating 1=Buy, 5=Sell. Sentiment 1=Positive)
    # If Rating is 5 (Sell) and Sentiment is 1 (Bullish) -> High Divergence Risk
    df['smart_money_divergence'] = df['avg_sentiment'] * (df['analyst_rating'] - 3)

    # --- 4. Lags ---
    df['price_lag_1'] = df['close_price'].shift(1)
    df['price_lag_7'] = df['close_price'].shift(7)

    return df.dropna()

def prepare_inference_features(conn):
    # Fetch latest 50 days (needed for RSI window)
    query = "SELECT * FROM daily_summary ORDER BY date DESC LIMIT 50"
    df = pd.read_sql(query, conn)
    
    if len(df) < 20: return None
    
    df = df.sort_values('date', ascending=True)
    
    # Fix Types
    cols = ['close_price', 'avg_sentiment', 'put_call_ratio', 'analyst_rating']
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c])
            
    df_features = add_technical_indicators(df)
    
    if df_features.empty: return None
        
    return df_features.iloc[[-1]]