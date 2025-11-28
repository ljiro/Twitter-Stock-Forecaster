import time
import os
import glob
import logging
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from pyvirtualdisplay import Display
from Scweet.scweet import Scweet

# Import your modules
from src.data_preprocessing.processor import get_sentiment, get_current_stock_price
from src.analysis.model import NvidiaQuantileModel

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Paths
RAW_DIR = "src/data_collection/outputs"
COOKIES_DIR = "src/data_collection/cookies"

def run_scraper_now():
    """Runs Scweet for the last 24 hours (for freshness)"""
    display = Display(visible=0, size=(1920, 1080))
    display.start()
    try:
        # Scrape last 1 day to ensure we have data
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        
        scweet = Scweet(cookies_path=COOKIES_DIR, disable_images=True, headless=False)
        
        # Determine filename to find it later
        csv_name = f"live_run_{start_date}_{end_date}.csv"
        
        scweet.scrape(
            words=['nvidia', 'NVDA'], since=start_date, until=end_date,
            from_account=None, interval=1, 
            display_type="Latest", save_images=False, 
            lang="en", limit=100, # Keep limit low for speed
            save_dir=RAW_DIR, custom_csv_name=csv_name
        )
        return os.path.join(RAW_DIR, csv_name)
    except Exception as e:
        logger.error(f"Scrape Error: {e}")
        return None
    finally:
        display.stop()

def job_inference():
    logger.info("🚀 Starting Inference Job...")
    
    # 1. Scrape Data
    csv_path = run_scraper_now()
    if not csv_path or not os.path.exists(csv_path):
        logger.warning("⚠️ Scraping failed or no data found.")
        return

    # 2. Process Sentiment
    try:
        df = pd.read_csv(csv_path)
        avg_sentiment = get_sentiment(df)
        current_price = get_current_stock_price("NVDA")
        
        logger.info(f"📊 Inputs -> Sentiment: {avg_sentiment:.3f}, Price: ${current_price}")

        # 3. Run Prediction
        model = NvidiaQuantileModel()
        prediction = model.predict(avg_sentiment, current_price)
        
        if prediction:
            logger.info(f"🔮 PREDICTION (3 Days): {prediction}")
            # Save prediction to log file
            with open("data/processed/predictions_log.csv", "a") as f:
                f.write(f"{datetime.now()},{current_price},{avg_sentiment},{prediction['predicted_price']}\n")
    except Exception as e:
        logger.error(f"❌ Inference Pipeline Error: {e}")

def job_training():
    logger.info("🏋️ Starting Retraining Job...")
    # In a real app, you would aggregate all raw CSVs in `outputs` into one big training file here.
    # For now, we assume data/processed/training_data.csv exists.
    model = NvidiaQuantileModel()
    model.train()

if __name__ == "__main__":
    scheduler = BackgroundScheduler()
    
    # Add Jobs
    scheduler.add_job(job_inference, IntervalTrigger(minutes=30))
    scheduler.add_job(job_training, IntervalTrigger(hours=24))
    
    scheduler.start()
    logger.info("✅ Scheduler Started. Waiting for jobs...")
    
    # Run inference once immediately on startup to test
    job_inference()
    
    try:
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()