import time
import logging
import sqlite3
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

# Imports
from src.data_collection.scraper import run_scraper
from src.data_preprocessing.processor import get_sentiment, get_current_stock_price
from src.analysis.model import NvidiaQuantileModel
from src.database import init_db
from src.features import prepare_inference_features, add_technical_indicators

# -------------------------
# 1. Configuration & Logging
# -------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# File Paths
JSON_OUTPUT = "data/latest_prediction.json"
PREDICTIONS_CSV = "data/live_predictions.csv"
TRAINING_CSV = "data/processed/full_training_dataset.csv"
METRICS_CSV = "data/model_performance.csv"

# -------------------------
# 2. Helper Functions
# -------------------------
def regenerate_training_data():
    """
    Reads the raw DB history, re-calculates all technical indicators (RSI, Volatility),
    re-aligns the targets (1d/3d), and overwrites the master training CSV.
    """
    logger.info("♻️ Regenerating Full Training Dataset...")
    conn = sqlite3.connect("data/nvda.db")
    
    # Fetch all raw history
    df = pd.read_sql("SELECT * FROM daily_summary ORDER BY date ASC", conn)
    conn.close()
    
    if df.empty:
        logger.warning("⚠️ Database is empty. Skipping regeneration.")
        return
    
    # Fix Data Types
    df['date'] = pd.to_datetime(df['date'])
    df['close_price'] = pd.to_numeric(df['close_price'])
    df['avg_sentiment'] = pd.to_numeric(df['avg_sentiment'])
    
    # Re-Calculate Targets
    df['target_price_1d'] = df['close_price'].shift(-1)
    df['target_price_3d'] = df['close_price'].shift(-3)
    
    # Feature Engineering
    rich_df = add_technical_indicators(df)
    
    # Save to Disk
    os.makedirs(os.path.dirname(TRAINING_CSV), exist_ok=True)
    rich_df.to_csv(TRAINING_CSV, index=False)
    logger.info(f"✅ Dataset Updated: {len(rich_df)} rows saved to {TRAINING_CSV}")

# -------------------------
# 3. Scheduled Jobs
# -------------------------
def job_inference():
    """
    Runs every 10 minutes: Scrapes -> Updates DB -> Predicts -> Updates API JSON
    """
    logger.info("🚀 Starting 10-Minute Inference Cycle...")
    today = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    # A. Scrape Twitter (Limit 50 tweets)
    csv_path = run_scraper(yesterday, today)
    
    if csv_path:
        try:
            # B. Process Data
            df = pd.read_csv(csv_path)
            sent = get_sentiment(df)
            price = get_current_stock_price("NVDA")
            
            # C. Save Raw State to DB
            conn = sqlite3.connect("data/nvda.db")
            conn.execute('''
                INSERT INTO daily_summary (date, avg_sentiment, close_price)
                VALUES (?, ?, ?) ON CONFLICT(date) DO UPDATE SET 
                avg_sentiment=excluded.avg_sentiment, close_price=excluded.close_price
            ''', (today, sent, price))
            conn.commit()
            
            # D. Generate Features
            df_feat = prepare_inference_features(conn)
            conn.close()
            
            if df_feat is not None:
                # E. Run Models
                model_1d = NvidiaQuantileModel(target_col='target_price_1d', model_name='model_1d')
                pred_1d = model_1d.predict(df_feat)
                
                model_3d = NvidiaQuantileModel(target_col='target_price_3d', model_name='model_3d')
                pred_3d = model_3d.predict(df_feat)
                
                # F. Build API Payload
                api_payload = {
                    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "current_price": price,
                    "sentiment_score": round(sent, 4),
                    "forecasts": {
                        "tomorrow": pred_1d,
                        "next_3_days": pred_3d
                    }
                }
                
                # G. Save JSON for API
                with open(JSON_OUTPUT, "w") as f:
                    json.dump(api_payload, f, indent=4)
                    
                # H. Append to Log
                row = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "sentiment": round(sent, 4),
                    "price": price,
                    "pred_1d_median": pred_1d['pred'] if pred_1d else 0,
                    "pred_3d_median": pred_3d['pred'] if pred_3d else 0
                }
                out = pd.DataFrame([row])
                write_header = not os.path.isfile(PREDICTIONS_CSV)
                out.to_csv(PREDICTIONS_CSV, mode='a', header=write_header, index=False)

                logger.info(f"✅ Cycle Complete. Next run in 10 mins.")
                
        except Exception as e:
            logger.error(f"❌ Inference Failed: {e}")

def job_train():
    """
    Runs every 1 hour (TEST MODE): Regenerates Data -> Retrains Models -> Logs Drift Metrics
    """
    logger.info("🏋️ Starting Retraining & Drift Check...")
    
    # 1. Update Data
    regenerate_training_data()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        # 2. Train Models
        m1 = NvidiaQuantileModel(target_col='target_price_1d', model_name='model_1d')
        metrics_1d = m1.train()
        
        m3 = NvidiaQuantileModel(target_col='target_price_3d', model_name='model_3d')
        metrics_3d = m3.train()
        
        # 3. Log Performance
        if metrics_1d and metrics_3d:
            row = {
                "timestamp": timestamp,
                "model_1d_coverage": metrics_1d['coverage'],
                "model_1d_winkler": metrics_1d['winkler'],
                "model_1d_loss": metrics_1d['pinball_loss'],
                "model_3d_coverage": metrics_3d['coverage'],
                "model_3d_winkler": metrics_3d['winkler'],
                "model_3d_loss": metrics_3d['pinball_loss']
            }
            
            df_log = pd.DataFrame([row])
            write_header = not os.path.isfile(METRICS_CSV)
            df_log.to_csv(METRICS_CSV, mode='a', header=write_header, index=False)
            
            # --- PRINT TO CONSOLE SO YOU SEE IT IMMEDIATELY ---
            logger.info(f"📊 FRESH METRICS (1D): Coverage={metrics_1d['coverage']}, Winkler={metrics_1d['winkler']}")
            logger.info(f"📊 FRESH METRICS (3D): Coverage={metrics_3d['coverage']}, Winkler={metrics_3d['winkler']}")
            logger.info(f"✅ Performance Metrics Logged: {METRICS_CSV}")
            
    except Exception as e:
        logger.error(f"❌ Training Failed: {e}")

# -------------------------
# 4. Main Entry Point
# -------------------------
if __name__ == "__main__":
    init_db()
    
    scheduler = BackgroundScheduler()
    
    # Schedule Jobs
    scheduler.add_job(job_inference, IntervalTrigger(minutes=10)) 
    scheduler.add_job(job_train, IntervalTrigger(hours=1))       
    
    scheduler.start()
    
    logger.info("✅ High-Frequency Pipeline Running...")
    logger.info("   - Inference: Every 10 mins")
    logger.info("   - Retraining: Every 1 hour (TEST MODE)")
    
    # --- IMMEDIATE STARTUP EXECUTION ---
    # Run both jobs ONCE right now so we don't have to wait
    logger.info("⚡ Running initial Training & Inference...")
    job_train()      # <--- Calculates Metrics NOW
    job_inference()  # <--- Runs Prediction NOW
    
    try:
        while True: time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        logger.info("🛑 Pipeline Stopped.")