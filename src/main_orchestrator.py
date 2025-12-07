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
from src.database import init_db, update_past_target
from src.features import prepare_inference_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

def job_inference():
    logger.info("🚀 Inference Started")
    today = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    csv_path = run_scraper(yesterday, today)
    
    if csv_path:
        df = pd.read_csv(csv_path)
        sent = get_sentiment(df)
        price = get_current_stock_price("NVDA")
        
        # Save state for features
        conn = sqlite3.connect("data/nvda.db")
        conn.execute('''
            INSERT INTO daily_summary (date, avg_sentiment, close_price)
            VALUES (?, ?, ?) ON CONFLICT(date) DO UPDATE SET 
            avg_sentiment=excluded.avg_sentiment, close_price=excluded.close_price
        ''', (today, sent, price))
        conn.commit()
        
        # Predict
        df_feat = prepare_inference_features(conn)
        conn.close()
        
        if df_feat is not None:
            model = NvidiaQuantileModel()
            preds = model.predict(df_feat)
            if preds:
                logger.info(f"🔮 3-Day Range: ${preds['lower']} - ${preds['upper']}")

def job_train():
    logger.info("🏋️ Training Started")
    past_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
    price = get_current_stock_price("NVDA")
    update_past_target(past_date, price)
    
    model = NvidiaQuantileModel()
    model.train()

if __name__ == "__main__":
    init_db()
    scheduler = BackgroundScheduler()
    scheduler.add_job(job_inference, IntervalTrigger(minutes=30))
    scheduler.add_job(job_train, IntervalTrigger(hours=24))
    scheduler.start()
    
    logger.info("✅ Pipeline Running...")
    job_inference() # Run once on startup
    
    try:
        while True: time.sleep(1)
    except: scheduler.shutdown()