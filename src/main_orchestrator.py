import time
import logging
import sqlite3
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

# Imports from your project modules
from src.data_collection.scraper import run_scraper
from src.data_preprocessing.processor import get_sentiment, get_current_stock_price
from src.database import init_db
from src.features import prepare_inference_features, add_technical_indicators

# Models (Champion & Rivals)
from src.analysis.model import NvidiaQuantileModel       # Champion (Gradient Boosting)
from src.analysis.adapted_rivals import NvidiaLinearQR, NvidiaMQLSTM, NvidiaQRNN_Keras # Rivals

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
    
    # Re-Calculate Targets (Prices)
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
    Runs every 10 minutes: Scrapes -> Updates DB -> Runs ENSEMBLE Prediction -> Updates API
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
                # --- E. BATTLE OF THE MODELS (3-Day Horizon) ---
                
                # 1. Champion: Gradient Boosting
                champion = NvidiaQuantileModel(target_col='target_price_3d', model_name='model_3d')
                pred_gb = champion.predict(df_feat)
                
                # 2. Rival: Linear QR
                rival_lin = NvidiaLinearQR(target_col='target_price_3d')
                pred_lin = rival_lin.predict(df_feat)
                
                # 3. Rival: MQLSTM
                rival_lstm = NvidiaMQLSTM(target_col='target_price_3d')
                pred_lstm = rival_lstm.predict(df_feat)
                
                # 4. Rival: QRNN
                rival_qrnn = NvidiaQRNN_Keras(target_col='target_price_3d')
                pred_qrnn = rival_qrnn.predict(df_feat)
                
                # F. Build API Payload (Comparison View)
                api_payload = {
                    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "current_price": price,
                    "sentiment_score": round(sent, 4),
                    "predictions_3d": {
                        "champion_gb": pred_gb,
                        "rival_linear": pred_lin,
                        "rival_lstm": pred_lstm,
                        "rival_qrnn": pred_qrnn
                    }
                }
                
                # G. Save JSON for API
                with open(JSON_OUTPUT, "w") as f:
                    json.dump(api_payload, f, indent=4)
                    
                # H. Append Champion to CSV Log
                row = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "sentiment": round(sent, 4),
                    "price": price,
                    "champion_median": pred_gb['pred'] if pred_gb else 0,
                    "linear_median": pred_lin['pred'] if pred_lin else 0,
                    "lstm_median": pred_lstm['pred'] if pred_lstm else 0,
                    "qrnn_median": pred_qrnn['pred'] if pred_qrnn else 0
                }
                out = pd.DataFrame([row])
                write_header = not os.path.isfile(PREDICTIONS_CSV)
                out.to_csv(PREDICTIONS_CSV, mode='a', header=write_header, index=False)

                logger.info(f"✅ Inference Complete. JSON updated.")
                
        except Exception as e:
            logger.error(f"❌ Inference Failed: {e}")

def job_train():
    """
    Runs every 1 hour (Test Mode): Retrains ALL models and logs their metrics.
    """
    logger.info("🏋️ Starting Battle Retraining...")
    
    regenerate_training_data()
    
    try:
        df = pd.read_csv(TRAINING_CSV).dropna()
    except:
        logger.error("❌ No training data found.")
        return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        # 1. Train Champion (Gradient Boosting)
        champion = NvidiaQuantileModel(target_col='target_price_3d', model_name='model_3d')
        metrics_gb = champion.train()
        logger.info(f"🏆 Champion GB: Coverage={metrics_gb['coverage']}, Winkler={metrics_gb['winkler']}")
        
        # 2. Train Rivals
        logger.info("⚔️ Training Rival: Linear QR...")
        metrics_lin = NvidiaLinearQR().train(df)
        logger.info(f"   Linear QR: Coverage={metrics_lin['coverage']}, Winkler={metrics_lin['winkler']}")
        
        logger.info("⚔️ Training Rival: MQLSTM...")
        metrics_lstm = NvidiaMQLSTM().train(df)
        logger.info(f"   MQLSTM:    Coverage={metrics_lstm['coverage']}, Winkler={metrics_lstm['winkler']}")
        
        logger.info("⚔️ Training Rival: QRNN...")
        metrics_qrnn = NvidiaQRNN_Keras().train(df)
        logger.info(f"   QRNN:      Coverage={metrics_qrnn['coverage']}, Winkler={metrics_qrnn['winkler']}")

        # 3. Log ALL to CSV
        with open(METRICS_CSV, "a") as f:
            # Format: Timestamp, ModelName, Coverage, Winkler
            f.write(f"{timestamp},GradientBoosting,{metrics_gb['coverage']},{metrics_gb['winkler']}\n")
            f.write(f"{timestamp},LinearQR,{metrics_lin['coverage']},{metrics_lin['winkler']}\n")
            f.write(f"{timestamp},MQLSTM,{metrics_lstm['coverage']},{metrics_lstm['winkler']}\n")
            f.write(f"{timestamp},QRNN,{metrics_qrnn['coverage']},{metrics_qrnn['winkler']}\n")
            
        logger.info(f"✅ All metrics saved to {METRICS_CSV}")

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
    
    logger.info("✅ Multi-Model Pipeline Running...")
    logger.info("   - Inference: Every 10 mins")
    logger.info("   - Retraining: Every 1 hour")
    
    # Immediate Run
    logger.info("⚡ Initializing Models...")
    job_train()
    job_inference()
    
    try:
        while True: time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        logger.info("🛑 Pipeline Stopped.")