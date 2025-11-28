import sqlite3
import pandas as pd
import os

DB_PATH = "data/nvda.db"

def init_db():
    """Creates the necessary tables if they don't exist"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Table 1: Daily Summary (The Training Data)
    # Stores aggregated sentiment and stock price per day
    c.execute('''
        CREATE TABLE IF NOT EXISTS daily_summary (
            date TEXT PRIMARY KEY,
            avg_sentiment REAL,
            close_price REAL,
            target_price_3d REAL
        )
    ''')
    
    # Table 2: Predictions Log
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            timestamp TEXT,
            current_price REAL,
            sentiment REAL,
            pred_price REAL,
            pred_lower REAL,
            pred_upper REAL
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Database initialized at", DB_PATH)

def save_daily_summary(df):
    """Saves aggregated data to DB (Upsert: Update if exists, Insert if new)"""
    conn = sqlite3.connect(DB_PATH)
    # Pandas to_sql doesn't support upsert easily in SQLite, so we use standard SQL
    for _, row in df.iterrows():
        conn.execute('''
            INSERT OR REPLACE INTO daily_summary (date, avg_sentiment, close_price, target_price_3d)
            VALUES (?, ?, ?, ?)
        ''', (row['date'], row['avg_sentiment'], row['close_price'], row.get('target_price_3d')))
    conn.commit()
    conn.close()

def load_training_data():
    """Reads history for the model"""
    conn = sqlite3.connect(DB_PATH)
    # We only want rows where we actually know the future price (target_price_3d is not null)
    df = pd.read_sql("SELECT * FROM daily_summary WHERE target_price_3d IS NOT NULL", conn)
    conn.close()
    return df