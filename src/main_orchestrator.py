import time
import logging
import sqlite3
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

# --- IMPORT MODULES ---
from src.data_collection.scraper import run_scraper
from src.data_collection.institutional import fetch_institutional_data
from src.data_preprocessing.processor import get_sentiment, get_current_stock_price
from src.database import init_db, save_raw_tweets
from src.features import prepare_inference_features, add_technical_indicators

# Models
from src.analysis.model import NvidiaQuantileModel        
from src.analysis.adapted_rivals import NvidiaLinearQR, NvidiaMQLSTM, NvidiaQRNN_Keras 

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[logging.FileHandler("system.log"), logging.StreamHandler()]
)
logger = logging.getLogger()

# --- CONFIGURATION ---
JSON_OUTPUT = "data/latest_prediction.json"
METRICS_SNAPSHOT = "data/latest_metrics.json" 
PREDICTIONS_CSV = "data/live_predictions.csv"
TRAINING_CSV = "data/processed/full_training_dataset.csv"
METRICS_CSV = "data/model_performance.csv"
DB_PATH = "data/nvda.db"

# ==============================================================================
# 0. HISTORY SEEDER
# ==============================================================================
def seed_history_csv():
    """Pre-fills history CSV from DB if missing."""
    if os.path.exists(PREDICTIONS_CSV): return
    logger.info("🌱 Seeding History CSV...")
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT date, close_price, avg_sentiment FROM daily_summary ORDER BY date ASC", conn)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            seed = pd.DataFrame({
                "timestamp": df['date'].dt.strftime("%Y-%m-%d %H:%M:%S"),
                "price": df['close_price'],
                "sentiment": df['avg_sentiment'],
                "gb_pred": df['close_price'] 
            })
            seed.tail(60).to_csv(PREDICTIONS_CSV, index=False)
            logger.info(f"✅ Seeded {len(seed)} rows.")
    except Exception as e: logger.error(f"❌ Seeding Failed: {e}")
    finally: conn.close()

# ==============================================================================
# 1. TRAINING DATA (Updated for 1D, 2D, 3D targets)
# ==============================================================================
def regenerate_training_data():
    logger.info("♻️ Regenerating Data...")
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM daily_summary ORDER BY date ASC", conn)
        if df.empty: return
        
        df['date'] = pd.to_datetime(df['date'])
        for c in ['close_price', 'avg_sentiment', 'market_fear_index', 'analyst_rating']: 
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
        
        # --- Create Full Term Structure ---
        df['target_price_1d'] = df['close_price'].shift(-1)
        df['target_price_2d'] = df['close_price'].shift(-2)
        df['target_price_3d'] = df['close_price'].shift(-3)
        
        rich_df = add_technical_indicators(df)
        os.makedirs(os.path.dirname(TRAINING_CSV), exist_ok=True)
        rich_df.to_csv(TRAINING_CSV, index=False)
    except Exception as e: logger.error(f"❌ Data Gen Failed: {e}")
    finally: conn.close()

# ==============================================================================
# 2. INFERENCE JOB (Generates the Curve)
# ==============================================================================
def job_inference():
    logger.info("🚀 Starting Inference...")
    today = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    try:
        # A. Collect
        csv_path = run_scraper(yesterday, today)
        inst = fetch_institutional_data("NVDA") or {}
        fear, rating = inst.get('market_fear_index', 20.0), inst.get('analyst_rating', 3.0)
        
        batch_s = 0.0
        if csv_path:
            df_t = pd.read_csv(csv_path)
            batch_s = get_sentiment(df_t)
            save_raw_tweets(df_t)
        
        price = get_current_stock_price("NVDA")
        
        # B. DB Update
        conn = sqlite3.connect(DB_PATH)
        curr_s = pd.read_sql(f"SELECT AVG(sentiment_score) as s FROM raw_tweets WHERE timestamp LIKE '{today}%'", conn)['s'].values[0]
        final_s = curr_s if curr_s is not None else batch_s
        
        conn.execute('''INSERT INTO daily_summary (date, avg_sentiment, close_price, market_fear_index, analyst_rating) 
                        VALUES (?, ?, ?, ?, ?) ON CONFLICT(date) DO UPDATE SET 
                        avg_sentiment=excluded.avg_sentiment, close_price=excluded.close_price''', 
                        (today, final_s, price, fear, rating))
        conn.commit()
        
        # C. Features
        df_feat = prepare_inference_features(conn)
        conn.close()
        if df_feat is None: return

        # D. Predict Curve (Champion) & Rivals
        metrics = {}
        if os.path.exists(METRICS_SNAPSHOT):
            with open(METRICS_SNAPSHOT, "r") as f: metrics = json.load(f)

        # 1. Champion Curve (1D, 2D, 3D)
        curve_data = {}
        for h in [1, 2, 3]:
            # Load specific model for horizon (e.g., gbr_1d.pkl)
            m = NvidiaQuantileModel(target_col=f'target_price_{h}d', model_name=f"gbr_{h}d")
            pred = m.predict(df_feat)
            if pred: curve_data[f"{h}d"] = pred

        # 2. Rivals (3D Only)
        rivals = {
            "linear_qr": NvidiaLinearQR(target_col='target_price_3d').predict(df_feat),
            "mqlstm": NvidiaMQLSTM(target_col='target_price_3d').predict(df_feat),
            "qrnn": NvidiaQRNN_Keras(target_col='target_price_3d').predict(df_feat)
        }
        
        # E. Save JSON
        rsi = df_feat['rsi_14'].values[0] if 'rsi_14' in df_feat else 50.0
        vol = df_feat['volatility_7'].values[0] if 'volatility_7' in df_feat else 0.0

        payload = {
            "meta": {"last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "current_price": price},
            "signals": {"sentiment_score": round(final_s, 4), "market_fear_index": fear, "analyst_rating": rating},
            "technicals": {"rsi_14": round(float(rsi), 2), "volatility_7": round(float(vol*100), 2)},
            
            # THE CURVE DATA
            "forecast_curve": curve_data,
            
            # Rivals for comparison
            "models": {
                "gradient_boosting": {"prediction": curve_data.get("3d"), "performance": metrics.get("GradientBoosting", "N/A")},
                "linear_qr": {"prediction": rivals["linear_qr"], "performance": metrics.get("LinearQR", "N/A")},
                "mqlstm": {"prediction": rivals["mqlstm"], "performance": metrics.get("MQLSTM", "N/A")},
                "qrnn": {"prediction": rivals["qrnn"], "performance": metrics.get("QRNN", "N/A")}
            }
        }
        with open(JSON_OUTPUT, "w") as f: json.dump(payload, f, indent=4)
        
        # F. Save CSV (Use 1D prediction for accuracy tracking)
        next_day = curve_data.get("1d", {}).get('pred', 0)
        log_row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "price": price,
            "sentiment": round(final_s, 4),
            "gb_pred": next_day
        }
        pd.DataFrame([log_row]).to_csv(PREDICTIONS_CSV, mode='a', header=not os.path.exists(PREDICTIONS_CSV), index=False)
        logger.info(f"✅ Curve Generated. 1D: ${next_day}")

    except Exception as e: logger.error(f"❌ Inference Failed: {e}")

# ==============================================================================
# 3. TRAINING JOB (Trains 3 GBR Models)
# ==============================================================================
def job_train():
    logger.info("⚡ Training Forecast Curve (1D, 2D, 3D)...")
    regenerate_training_data()
    try: 
        df = pd.read_csv(TRAINING_CSV).dropna()
        if len(df) < 50: return
    except: return

    try:
        # Train Champion on ALL horizons
        # We save metrics for the 3D model as the "Battle Score"
        m_gb_1 = NvidiaQuantileModel(target_col='target_price_1d', model_name="gbr_1d").train()
        m_gb_2 = NvidiaQuantileModel(target_col='target_price_2d', model_name="gbr_2d").train()
        m_gb_3 = NvidiaQuantileModel(target_col='target_price_3d', model_name="gbr_3d").train()
        
        logger.info("🐢 Training Rivals (3D Only)...")
        m_lin = NvidiaLinearQR(target_col='target_price_3d').train(df)
        m_lstm = NvidiaMQLSTM(target_col='target_price_3d').train(df)
        m_qrnn = NvidiaQRNN_Keras(target_col='target_price_3d').train(df)
        
        t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        snapshot = {
            "GradientBoosting": m_gb_3, # Use 3D score for comparison
            "LinearQR": m_lin, "MQLSTM": m_lstm, "QRNN": m_qrnn, "updated_at": t
        }
        with open(METRICS_SNAPSHOT, "w") as f: json.dump(snapshot, f, indent=4)
        
        # Log 3D Performance
        with open(METRICS_CSV, "a") as f:
            f.write(f"{t},GradientBoosting,{m_gb_3.get('coverage')},{m_gb_3.get('winkler')}\n")
            f.write(f"{t},MQLSTM,{m_lstm.get('coverage')},{m_lstm.get('winkler')}\n")
            
        logger.info("✅ Curve Models Retrained.")
    except Exception as e: logger.error(f"❌ Training Failed: {e}")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    init_db()
    seed_history_csv()
    
    scheduler = BackgroundScheduler()
    scheduler.add_job(job_inference, IntervalTrigger(minutes=10)) 
    scheduler.add_job(job_train, IntervalTrigger(hours=1))        
    scheduler.start()
    
    logger.info("⚡ Force Running Initial Cycle...")
    job_train()
    job_inference()
    
    logger.info("✅ Pipeline Active.")
    try:
        while True: time.sleep(1)
    except: scheduler.shutdown()