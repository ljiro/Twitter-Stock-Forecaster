import pandas as pd
import yfinance as yf
import glob
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from database import init_db, save_daily_summary
from features import add_technical_indicators

def normalize_sentiment(row):
    try:
        label = str(row['Sentiment_Label']).lower().strip()
        score = float(row['Sentiment_Score'])
        if label == 'positive': return score
        elif label == 'negative': return -score
        return 0.0
    except: return 0.0

def ingest_historical_data():
    print("⏳ Starting Fast Ingestion...")
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
    
    # 2. Fetch Stock Data
    print("📈 Fetching NVDA Stock History...")
    start_date = full_df['date'].min()
    end_date = full_df['date'].max()
    stock = yf.Ticker("NVDA")
    stock_df = stock.history(start=start_date, end=end_date + pd.Timedelta(days=15))
    stock_df.reset_index(inplace=True)
    stock_df['date'] = pd.to_datetime(stock_df['Date']).dt.date
    stock_df['date'] = pd.to_datetime(stock_df['date'])
    
    merged = pd.merge(full_df, stock_df[['date', 'Close']], on='date', how='inner')
    merged.rename(columns={'Close': 'close_price'}, inplace=True)
    
    # 3. CREATE MULTIPLE TARGETS
    merged['target_price_1d'] = merged['close_price'].shift(-1) # Tomorrow
    merged['target_price_3d'] = merged['close_price'].shift(-3) # 3 Days later
    
    # 4. Save
    base_df = merged.dropna().copy()
    base_df['date'] = base_df['date'].dt.strftime('%Y-%m-%d')
    
    # Save targets to DB (We might need to update database.py to handle extra columns, 
    # but for now we rely on the CSV export for training)
    db_df = base_df[['date', 'avg_sentiment', 'close_price', 'target_price_3d']]
    save_daily_summary(db_df)
    
    # Save Full Training Set with BOTH targets
    print("🧠 Saving Training Data...")
    base_df['date'] = pd.to_datetime(base_df['date'])
    rich_df = add_technical_indicators(base_df)
    rich_df.to_csv("data/processed/full_training_dataset.csv", index=False)
    print("✅ Ingestion Complete.")

if __name__ == "__main__":
    ingest_historical_data()