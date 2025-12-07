import pandas as pd
import yfinance as yf
import glob
from database import init_db, save_daily_summary

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
    
    all_files = glob.glob("data/raw_history/*.csv")
    if not all_files:
        print("❌ No CSVs found in data/raw_history/")
        return

    daily_sentiments = []
    
    for filename in all_files:
        try:
            try: df = pd.read_csv(filename)
            except: df = pd.read_csv(filename, sep='\t')

            if 'Timestamp' in df.columns:
                df['date'] = pd.to_datetime(df['Timestamp'], utc=True).dt.date
            else: continue
            
            # Use existing FinBERT scores
            df['signal'] = df.apply(normalize_sentiment, axis=1)
            daily_grp = df.groupby('date')['signal'].mean().reset_index()
            daily_sentiments.append(daily_grp)
        except Exception as e: print(f"⚠️ Error {filename}: {e}")

    full_df = pd.concat(daily_sentiments).groupby('date')['signal'].mean().reset_index()
    full_df.rename(columns={'signal': 'avg_sentiment'}, inplace=True)
    full_df['date'] = pd.to_datetime(full_df['date'])
    
    # Get Stock Data
    print("📈 Fetching NVDA Stock History...")
    start, end = full_df['date'].min(), full_df['date'].max()
    stock = yf.Ticker("NVDA")
    stock_df = stock.history(start=start, end=end + pd.Timedelta(days=15))
    stock_df.reset_index(inplace=True)
    stock_df['date'] = pd.to_datetime(stock_df['Date']).dt.date
    stock_df['date'] = pd.to_datetime(stock_df['date']) # Align types
    
    merged = pd.merge(full_df, stock_df[['date', 'Close']], on='date', how='inner')
    merged.rename(columns={'Close': 'close_price'}, inplace=True)
    merged['target_price_3d'] = merged['close_price'].shift(-3)
    
    final_df = merged.dropna().copy()
    final_df['date'] = final_df['date'].dt.strftime('%Y-%m-%d')
    
    save_daily_summary(final_df)
    print(f"✅ Ingestion Complete. Saved {len(final_df)} records.")

if __name__ == "__main__":
    ingest_historical_data()