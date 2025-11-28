import time
import logging
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

# Imports
from src.data_collection.scraper import run_scraper
from src.data_preprocessing.processor import get_sentiment, get_current_stock_price
from src.analysis.model import NvidiaQuantileModel
from src.database import save_daily_summary, update_past_target, init_db

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

def job_inference_30min():
    """Runs every 30 mins: Scrape -> Sentiment -> Predict"""
    logger.info("🚀 Starting 30-min Inference...")
    
    # 1. Scrape (Last 24h window for context)
    today = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    csv_path = run_scraper(start_date=yesterday, end_date=today)
    
    if csv_path:
        # 2. Process
        df = pd.read_csv(csv_path)
        sent_score = get_sentiment(df)
        curr_price = get_current_stock_price("NVDA")
        
        # 3. Predict
        model = NvidiaQuantileModel()
        preds = model.predict(sent_score, curr_price)
        
        if preds:
            logger.info(f"🔮 3-Day Forecast: {preds} (Sentiment: {sent_score:.2f})")
            
            # 4. Log Prediction to DB
            conn = sqlite3.connect("data/nvda.db")
            conn.execute(
                "INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?)",
                (datetime.now().isoformat(), curr_price, sent_score, 
                 preds['predicted'], preds['lower'], preds['upper'])
            )
            conn.commit()
            conn.close()

def job_daily_maintenance():
    """Runs at Midnight: Updates DB history and Retrains Model"""
    logger.info("🏋️ Starting Daily Maintenance...")
    
    # 1. Get Today's Final Stats
    # (In production, you might scrape specifically for the full day here)
    curr_price = get_current_stock_price("NVDA")
    
    # Update the DB record for 3 DAYS AGO (set the target)
    three_days_ago = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
    update_past_target(three_days_ago, curr_price)
    
    # 2. Retrain Model
    model = NvidiaQuantileModel()
    model.train()

if __name__ == "__main__":
    # Ensure DB exists
    init_db()
    
    scheduler = BackgroundScheduler()
    
    # Schedule Inference (Every 30 mins)
    scheduler.add_job(job_inference_30min, IntervalTrigger(minutes=30))
    
    # Schedule Retraining (Every 24 hours)
    scheduler.add_job(job_daily_maintenance, IntervalTrigger(hours=24))
    
    scheduler.start()
    logger.info("✅ Scheduler Running. Ctrl+C to stop.")
    
    try:
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()