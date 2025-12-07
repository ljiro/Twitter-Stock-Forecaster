import pandas as pd

def add_technical_indicators(df):
    df = df.copy()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
    
    # Lags
    df['price_lag_1'] = df['close_price'].shift(1)
    df['price_lag_3'] = df['close_price'].shift(3)
    df['price_lag_7'] = df['close_price'].shift(7)
    
    # Moving Averages
    df['sentiment_ma_3'] = df['avg_sentiment'].rolling(window=3).mean()
    df['sentiment_ma_7'] = df['avg_sentiment'].rolling(window=7).mean()
    
    # Volatility
    df['volatility_7'] = df['close_price'].rolling(window=7).std()
    
    return df

def prepare_inference_features(conn):
    query = "SELECT * FROM daily_summary ORDER BY date DESC LIMIT 40"
    df = pd.read_sql(query, conn)
    if len(df) < 15: return None
    
    df = df.sort_values('date', ascending=True)
    df_features = add_technical_indicators(df)
    return df_features.iloc[[-1]]