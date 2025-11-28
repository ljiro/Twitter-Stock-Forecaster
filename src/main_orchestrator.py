import time
import logging
import sys
import os

# Ensure the parent directory is in path so we can import freely
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

# --- IMPORT YOUR MODULES HERE ---
# Adjust these imports to match your actual function names
from src.data_collection import scraper_script  # example
from src.data_preprocessing import cleaner_script # example
from src.analysis import quantile_model_script # example

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

def pipeline_inference_30min():
    """Runs every 30 mins: Scrape -> Clean -> Predict"""
    logger.info("⏰ Triggering Inference Pipeline...")
    try:
        # 1. Scrape
        # scraper_script.get_latest_tweets() 
        
        # 2. Preprocess
        # data = cleaner_script.clean_latest()
        
        # 3. Predict (Load model & infer)
        # quantile_model_script.predict(data)
        
        logger.info("✅ Inference successfully saved to data/processed/")
    except Exception as e:
        logger.error(f"❌ Inference Failed: {e}")

def pipeline_retraining_24hr():
    """Runs every 24 hrs: Train on all history -> Save Model"""
    logger.info("🗓️ Triggering Daily Retraining...")
    try:
        # 1. Train
        # quantile_model_script.train_new_model()
        
        logger.info("✅ Model retrained and saved to models/nvda_qr_model.pkl")
    except Exception as e:
        logger.error(f"❌ Retraining Failed: {e}")

if __name__ == "__main__":
    # Initialize Scheduler
    scheduler = BackgroundScheduler()
    
    # Add Jobs
    scheduler.add_job(pipeline_inference_30min, IntervalTrigger(minutes=30))
    scheduler.add_job(pipeline_retraining_24hr, IntervalTrigger(hours=24))
    
    # Start
    scheduler.start()
    logger.info("🚀 Pipeline is running. Press Ctrl+C to stop.")
    
    # Keep container alive
    try:
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()