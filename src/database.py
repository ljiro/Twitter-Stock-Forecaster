import sqlite3
import pandas as pd
import os

DB_PATH = "data/nvda.db"

def init_db():
    """Initializes tables for training data and predictions."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Table 1: Historical Training Data (One row per day)
    c.execute('''
        CREATE TABLE IF NOT EXISTS daily_summary (
            date TEXT PRIMARY KEY,
            avg_sentiment REAL,
            close_price REAL,
            target_price_3d REAL
        )
    ''')
    
    # Table 2: Live Predictions Log (Every 30 mins)
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            timestamp TEXT,
            current_price REAL,
            sentiment REAL,
            pred_price REAL,
            lower_bound REAL,
            upper_bound REAL
        )
    ''')
    conn.commit()
    conn.close()

def save_daily_summary(df):
    """Upserts daily summary data."""
    conn = sqlite3.connect(DB_PATH)
    # Using raw SQL for UPSERT capability
    for _, row in df.iterrows():
        conn.execute('''
            INSERT INTO daily_summary (date, avg_sentiment, close_price, target_price_3d)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                avg_sentiment=excluded.avg_sentiment,
                close_price=excluded.close_price,
                target_price_3d=excluded.target_price_3d
        ''', (row['date'], row['avg_sentiment'], row['close_price'], row['target_price_3d']))
    conn.commit()
    conn.close()

def update_past_target(date_str, actual_price):
    """
    Updates the 'target_price_3d' for a record 3 days ago.
    Example: On Friday, we update Tuesday's row with Friday's price.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        UPDATE daily_summary 
        SET target_price_3d = ? 
        WHERE date = ?
    ''', (actual_price, date_str))
    conn.commit()
    conn.close()

def load_training_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM daily_summary WHERE target_price_3d IS NOT NULL", conn)
    conn.close()
    return df