import pandas as pd
import yfinance as yf
import glob
import os
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.database import init_db, save_daily_summary, DB_PATH
from src.features import add_technical_indicators

def normalize_sentiment(row):
    try:
        label = str(row['Sentiment_Label']).lower().strip()
        score = float(row['Sentiment_Score'])
        if label == 'positive': return score
        elif label == 'negative': return -score
        return 0.0
    except: return 0.0

def fetch_historical_analyst_ratings(ticker_symbol, start_date, end_date):
    print("🏦 Fetching Historical Analyst Ratings...")
    try:
        ticker = yf.Ticker(ticker_symbol)
        upgrades = ticker.upgrades_downgrades
        
        if upgrades is None or upgrades.empty: return None
            
        grade_col = None
        candidates = ['To Grade', 'ToGrade', 'Grade', 'to_grade']
        for col in candidates:
            if col in upgrades.columns:
                grade_col = col
                break
        
        if not grade_col: return None

        upgrades.index = pd.to_datetime(upgrades.index).tz_localize(None)
        start_dt = pd.to_datetime(start_date).tz_localize(None)
        end_dt = pd.to_datetime(end_date).tz_localize(None)
        
        upgrades = upgrades[(upgrades.index >= start_dt) & (upgrades.index <= end_dt)]
        
        rating_map = {
            'Strong Buy': 1.0, 'Buy': 2.0, 'Overweight': 2.0, 'Outperform': 2.0,
            'Neutral': 3.0, 'Hold': 3.0, 'Market Perform': 3.0, 'Equal-Weight': 3.0,
            'Underperform': 4.0, 'Underweight': 4.0, 'Sell': 5.0
        }
        
        upgrades = upgrades.copy()
        upgrades.loc[:, 'numeric_rating'] = upgrades[grade_col].map(rating_map).fillna(3.0)
        
        return upgrades.resample('D')['numeric_rating'].mean()
        
    except: return None

def ingest_historical_data():
    print("⏳ Starting Advanced Ingestion...")
    init_db() 
    
    # 1. Load Tweets
    all_files = glob.glob("data/raw_history/*.csv")
    if not all_files:
        print("❌ No CSVs found in data/raw_history/")
        return

    daily_sentiments = []
    for filename in all_files:
        try:
            try: df = pd.read_csv(filename, low_memory=False)
            except: df = pd.read_csv(filename, sep='\t', low_memory=False)

            if 'Timestamp' in df.columns:
                df['date'] = pd.to_datetime(df['Timestamp'], utc=True).dt.date
            else: continue
            
            if 'Sentiment_Label' in df.columns:
                df['signal'] = df.apply(normalize_sentiment, axis=1)
                daily_grp = df.groupby('date')['signal'].mean().reset_index()
                daily_sentiments.append(daily_grp)
        except: pass

    if not daily_sentiments: return

    full_df = pd.concat(daily_sentiments).groupby('date')['signal'].mean().reset_index()
    full_df.rename(columns={'signal': 'avg_sentiment'}, inplace=True)
    full_df['date'] = pd.to_datetime(full_df['date'])
    
    # 2. Fetch External Data
    print("📈 Fetching Stock, Volatility & Institutional Data...")
    start_date = full_df['date'].min()
    end_date = full_df['date'].max()
    
    # A. Stock Price
    ticker = yf.Ticker("NVDA")
    stock_df = ticker.history(start=start_date, end=end_date + pd.Timedelta(days=15))
    stock_df.reset_index(inplace=True)
    stock_df['date'] = pd.to_datetime(stock_df['Date']).dt.date
    stock_df['date'] = pd.to_datetime(stock_df['date'])
    
    # B. Market Fear Index (^VXN - Nasdaq Volatility) <--- NEW
    vxn_ticker = yf.Ticker("^VXN")
    vxn_df = vxn_ticker.history(start=start_date, end=end_date + pd.Timedelta(days=15))
    vxn_df.reset_index(inplace=True)
    vxn_df['date'] = pd.to_datetime(vxn_df['Date']).dt.date
    vxn_df['date'] = pd.to_datetime(vxn_df['date'])
    vxn_df = vxn_df[['date', 'Close']].rename(columns={'Close': 'market_fear_index'})
    
    # C. Analyst Ratings
    ratings_series = fetch_historical_analyst_ratings("NVDA", start_date, end_date)
    try: inst_own = ticker.info.get('heldPercentInstitutions', 0.60) 
    except: inst_own = 0.60
    
    # Merge All
    merged = pd.merge(full_df, stock_df[['date', 'Close']], on='date', how='inner')
    merged.rename(columns={'Close': 'close_price'}, inplace=True)
    
    merged = pd.merge(merged, vxn_df, on='date', how='left') # Merge VXN
    merged['market_fear_index'] = merged['market_fear_index'].ffill().fillna(20.0)
    
    if ratings_series is not None:
        ratings_df = ratings_series.reset_index()
        ratings_df.columns = ['date', 'numeric_rating']
        ratings_df['date'] = pd.to_datetime(ratings_df['date'].dt.date)
        merged = pd.merge(merged, ratings_df, on='date', how='left')
        merged['numeric_rating'] = merged['numeric_rating'].ffill().fillna(3.0)
    else:
        merged['numeric_rating'] = 3.0
        
    merged['analyst_rating'] = merged['numeric_rating']
    merged['institutional_ownership'] = inst_own
    
    # 3. Targets
    merged['target_price_1d'] = merged['close_price'].shift(-1)
    merged['target_price_3d'] = merged['close_price'].shift(-3)
    
    # 4. Save
    base_df = merged.dropna().copy()
    base_df['date'] = base_df['date'].apply(lambda x: x.strftime('%Y-%m-%d')) # SQLite Fix
    
    save_daily_summary(base_df)
    
    print("🧠 Feature Engineering...")
    base_df['date'] = pd.to_datetime(base_df['date'])
    rich_df = add_technical_indicators(base_df)
    
    os.makedirs("data/processed", exist_ok=True)
    rich_df.to_csv("data/processed/full_training_dataset.csv", index=False)
    print(f"✅ Ingestion Complete. Saved {len(rich_df)} rows.")

if __name__ == "__main__":
    ingest_historical_data()