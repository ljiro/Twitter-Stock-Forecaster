import pandas as pd
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import glob
import os
from database import init_db, save_daily_summary

analyzer = SentimentIntensityAnalyzer()

def ingest_historical_data():
    print("⏳ Starting Historical Ingestion...")
    init_db()
    
    # 1. Load CSVs
    all_files = glob.glob("data/raw_history/*.csv")
    if not all_files:
        print("❌ No CSVs found in data/raw_history/")
        return

    print(f"📂 Processing {len(all_files)} files...")
    daily_sentiments = []
    
    for filename in all_files:
        try:
            df = pd.read_csv(filename)
            # Normalize Date Column
            col = 'Timestamp' if 'Timestamp' in df.columns else 'Date'
            if col not in df.columns: continue
            
            df['date'] = pd.to_datetime(df[col]).dt.date
            
            # Sentiment
            text_col = 'Text' if 'Text' in df.columns else 'text'
            df['score'] = df[text_col].astype(str).apply(lambda x: analyzer.polarity_scores(x)['compound'])
            
            # Daily Mean
            daily_grp = df.groupby('date')['score'].mean().reset_index()
            daily_sentiments.append(daily_grp)
        except Exception as e:
            print(f"⚠️ Error {filename}: {e}")

    # Combine
    full_df = pd.concat(daily_sentiments).groupby('date')['score'].mean().reset_index()
    full_df.columns = ['date', 'avg_sentiment']
    full_df['date'] = pd.to_datetime(full_df['date'])

    # 2. Stock Prices
    print("📈 Fetching NVDA Stock History...")
    start, end = full_df['date'].min(), full_df['date'].max()
    stock = yf.Ticker("NVDA")
    stock_df = stock.history(start=start, end=end + pd.Timedelta(days=10))
    stock_df.reset_index(inplace=True)
    stock_df['date'] = pd.to_datetime(stock_df['Date']).dt.date
    
    # 3. Merge & Create Target
    merged = pd.merge(full_df, stock_df[['date', 'Close']], on='date', how='inner')
    merged.rename(columns={'Close': 'close_price'}, inplace=True)
    
    # Target = Price 3 days in future (shift backwards)
    merged['target_price_3d'] = merged['close_price'].shift(-3)
    
    # Save to DB
    final_df = merged.dropna()
    final_df['date'] = final_df['date'].astype(str)
    
    save_daily_summary(final_df)
    print(f"✅ Ingested {len(final_df)} days into SQLite.")

if __name__ == "__main__":
    ingest_historical_data()