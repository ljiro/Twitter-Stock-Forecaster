import time
import logging
import sqlite3
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

# Import your modules
from src.data_collection.scraper import run_scraper
from src.data_collection.institutional import fetch_institutional_data
from src.data_preprocessing.processor import get_sentiment, get_current_stock_price
from src.database import init_db, save_raw_tweets
from src.features import prepare_inference_features, add_technical_indicators

from src.analysis.model import NvidiaQuantileModel        
from src.analysis.adapted_rivals import NvidiaLinearQR, NvidiaMQLSTM, NvidiaQRNN_Keras 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# --- CONFIGURATION ---
JSON_OUTPUT = "data/latest_prediction.json"
METRICS_SNAPSHOT = "data/latest_metrics.json" # <--- NEW: Stores performance scores
PREDICTIONS_CSV = "data/live_predictions.csv"
TRAINING_CSV = "data/processed/full_training_dataset.csv"
METRICS_CSV = "data/model_performance.csv"
DB_PATH = "data/nvda.db"

def regenerate_training_data():
    """Fetches full history from DB and calculates indicators for training."""
    logger.info("♻️ Regenerating Full Training Dataset...")
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM daily_summary ORDER BY date ASC", conn)
        if df.empty: return
        
        df['date'] = pd.to_datetime(df['date'])
        # Ensure numeric types
        cols = ['close_price', 'avg_sentiment', 'market_fear_index', 'analyst_rating']
        for c in cols: 
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
        
        # Create Targets (Shifted)
        df['target_price_1d'] = df['close_price'].shift(-1)
        df['target_price_3d'] = df['close_price'].shift(-3)
        
        # Add Features
        rich_df = add_technical_indicators(df)
        
        os.makedirs(os.path.dirname(TRAINING_CSV), exist_ok=True)
        rich_df.to_csv(TRAINING_CSV, index=False)
        logger.info(f"✅ Dataset Updated: {len(rich_df)} rows saved.")
    finally:
        conn.close()

def job_inference():
    """Runs every 10 mins: Scrapes -> Features -> Predicts -> JSON Output"""
    logger.info("🚀 Starting Inference Cycle...")
    today = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    try:
        # 1. LIVE DATA COLLECTION
        csv_path = run_scraper(yesterday, today) 
        
        inst_data = fetch_institutional_data("NVDA") or {}
        fear = inst_data.get('market_fear_index', 20.0)
        rating = inst_data.get('analyst_rating', 3.0)
        ownership = inst_data.get('institutional_ownership', 0.60)
        
        # Process Sentiment
        batch_avg_sent = 0.0
        if csv_path:
            df_tweets = pd.read_csv(csv_path)
            batch_avg_sent = get_sentiment(df_tweets)
            save_raw_tweets(df_tweets)
        
        price = get_current_stock_price("NVDA")
        
        # Update Daily Summary
        conn = sqlite3.connect(DB_PATH)
        curr_sent_row = pd.read_sql(f"SELECT AVG(sentiment_score) as sent FROM raw_tweets WHERE timestamp LIKE '{today}%'", conn)
        daily_sent = curr_sent_row['sent'].values[0] if curr_sent_row['sent'].values[0] is not None else batch_avg_sent
        
        conn.execute('''
            INSERT INTO daily_summary (
                date, avg_sentiment, close_price, 
                market_fear_index, analyst_rating, institutional_ownership
            ) VALUES (?, ?, ?, ?, ?, ?) 
            ON CONFLICT(date) DO UPDATE SET 
            avg_sentiment=excluded.avg_sentiment, 
            close_price=excluded.close_price,
            market_fear_index=excluded.market_fear_index
        ''', (today, daily_sent, price, fear, rating, ownership))
        conn.commit()
        
        # 2. FEATURE ENGINEERING
        df_feat = prepare_inference_features(conn)
        conn.close()
        
        if df_feat is None:
            logger.warning("⚠️ Not enough data history to generate features.")
            return

        # 3. LOAD LATEST METRICS (The Fix)
        metrics = {}
        if os.path.exists(METRICS_SNAPSHOT):
            with open(METRICS_SNAPSHOT, "r") as f:
                metrics = json.load(f)

        # 4. GENERATE PREDICTIONS
        # We pass target_col='target_price_3d' to ensure correct model loading
        preds = {
            "champion_gb": NvidiaQuantileModel(target_col='target_price_3d').predict(df_feat),
            "rival_linear": NvidiaLinearQR(target_col='target_price_3d').predict(df_feat),
            "rival_lstm": NvidiaMQLSTM(target_col='target_price_3d').predict(df_feat),
            "rival_qrnn": NvidiaQRNN_Keras(target_col='target_price_3d').predict(df_feat)
        }
        
        # 5. CONSTRUCT API PAYLOAD
        api_payload = {
            "meta": {
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": "NVDA",
                "current_price": price
            },
            "signals": {
                "sentiment_score": round(daily_sent, 4),
                "market_fear_index": fear,
                "analyst_rating": rating
            },
            "models": {
                "gradient_boosting": {
                    "prediction": preds['champion_gb'],
                    "performance": metrics.get('GradientBoosting', "N/A")
                },
                "linear_qr": {
                    "prediction": preds['rival_linear'],
                    "performance": metrics.get('LinearQR', "N/A")
                },
                "mqlstm": {
                    "prediction": preds['rival_lstm'],
                    "performance": metrics.get('MQLSTM', "N/A")
                },
                "qrnn": {
                    "prediction": preds['rival_qrnn'],
                    "performance": metrics.get('QRNN', "N/A")
                }
            }
        }
        
        with open(JSON_OUTPUT, "w") as f: json.dump(api_payload, f, indent=4)
        
        # Log to CSV
        gb_pred = preds['champion_gb']['pred'] if preds['champion_gb'] else 0
        log_row = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "price": price, "gb_pred": gb_pred}
        pd.DataFrame([log_row]).to_csv(PREDICTIONS_CSV, mode='a', header=not os.path.isfile(PREDICTIONS_CSV), index=False)
        
        logger.info(f"✅ Inference Complete. JSON updated.")

    except Exception as e:
        logger.error(f"❌ Inference Cycle Failed: {e}")

def job_train():
    """Runs hourly: Retrains models and updates performance metrics."""
    logger.info("🏋️ Starting Model Retraining...")
    regenerate_training_data()
    
    try: 
        df = pd.read_csv(TRAINING_CSV).dropna()
        if len(df) < 50: 
            logger.warning("⚠️ Not enough data to train.")
            return
    except: return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        # Train & Get Metrics
        # These methods now return dicts like {'coverage': 0.9, 'winkler': 12.5}
        m_gb = NvidiaQuantileModel().train()
        m_lin = NvidiaLinearQR().train(df)
        m_lstm = NvidiaMQLSTM().train(df)
        m_qrnn = NvidiaQRNN_Keras().train(df)
        
        # 1. Save History to CSV
        with open(METRICS_CSV, "a") as f:
            f.write(f"{timestamp},GradientBoosting,{m_gb.get('coverage')},{m_gb.get('winkler')}\n")
            f.write(f"{timestamp},LinearQR,{m_lin.get('coverage')},{m_lin.get('winkler')}\n")
            f.write(f"{timestamp},MQLSTM,{m_lstm.get('coverage')},{m_lstm.get('winkler')}\n")
            f.write(f"{timestamp},QRNN,{m_qrnn.get('coverage')},{m_qrnn.get('winkler')}\n")
            
        # 2. Save Snapshot for Inference Job (The Critical Fix)
        snapshot = {
            "GradientBoosting": m_gb,
            "LinearQR": m_lin,
            "MQLSTM": m_lstm,
            "QRNN": m_qrnn,
            "updated_at": timestamp
        }
        with open(METRICS_SNAPSHOT, "w") as f:
            json.dump(snapshot, f, indent=4)
            
        logger.info(f"✅ Training Complete. Metrics snapshot saved.")
        
    except Exception as e:
        logger.error(f"❌ Training Failed: {e}")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    init_db()
    
    scheduler = BackgroundScheduler()
    scheduler.add_job(job_inference, IntervalTrigger(minutes=10)) 
    scheduler.add_job(job_train, IntervalTrigger(hours=1))        
    scheduler.start()
    
    logger.info("✅ Pipeline Active.")
    
    # Trigger immediately to populate missing files
    job_train() 
    job_inference()
    
    try:
        while True: time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()