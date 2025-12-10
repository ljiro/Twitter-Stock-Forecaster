import sqlite3
import pandas as pd
import os

DB_PATH = "data/nvda.db"

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 1. Daily Summary Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_summary (
            date TEXT PRIMARY KEY,
            avg_sentiment REAL,
            close_price REAL,
            target_price_1d REAL,
            target_price_3d REAL,
            market_fear_index REAL DEFAULT 20.0,   -- Replaces Put/Call Ratio
            analyst_rating REAL DEFAULT 3.0,
            institutional_ownership REAL DEFAULT 0.60
        )
    ''')
    
    # Migration: Add new columns if they don't exist
    cursor.execute("PRAGMA table_info(daily_summary)")
    columns = [info[1] for info in cursor.fetchall()]
    
    if 'market_fear_index' not in columns:
        print("⚙️ Migrating DB: Adding 'market_fear_index'...")
        cursor.execute("ALTER TABLE daily_summary ADD COLUMN market_fear_index REAL DEFAULT 20.0")
    if 'analyst_rating' not in columns:
        cursor.execute("ALTER TABLE daily_summary ADD COLUMN analyst_rating REAL DEFAULT 3.0")
    if 'institutional_ownership' not in columns:
        cursor.execute("ALTER TABLE daily_summary ADD COLUMN institutional_ownership REAL DEFAULT 0.60")
    
    # 2. Raw Tweets Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS raw_tweets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT,
            timestamp TEXT,
            text TEXT,
            sentiment_score REAL,
            UNIQUE(user, timestamp, text) ON CONFLICT IGNORE
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"✅ Database initialized at {DB_PATH}")

def save_daily_summary(df):
    """
    Saves daily aggregated stats to DB.
    """
    conn = sqlite3.connect(DB_PATH)
    
    # Ensure defaults
    if 'market_fear_index' not in df.columns: df['market_fear_index'] = 20.0
    if 'analyst_rating' not in df.columns: df['analyst_rating'] = 3.0
    if 'institutional_ownership' not in df.columns: df['institutional_ownership'] = 0.60
    
    for _, row in df.iterrows():
        conn.execute('''
            INSERT INTO daily_summary (
                date, avg_sentiment, close_price, target_price_1d, target_price_3d,
                market_fear_index, analyst_rating, institutional_ownership
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
            avg_sentiment=excluded.avg_sentiment,
            close_price=excluded.close_price,
            target_price_1d=excluded.target_price_1d,
            target_price_3d=excluded.target_price_3d,
            market_fear_index=excluded.market_fear_index,
            analyst_rating=excluded.analyst_rating,
            institutional_ownership=excluded.institutional_ownership
        ''', (
            row['date'], 
            row['avg_sentiment'], 
            row['close_price'], 
            row.get('target_price_1d'), 
            row.get('target_price_3d'),
            row.get('market_fear_index'), 
            row.get('analyst_rating'), 
            row.get('institutional_ownership')
        ))
    conn.commit()
    conn.close()

def save_raw_tweets(df_tweets):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    new_count = 0
    for _, row in df_tweets.iterrows():
        user = row.get('UserName', 'Unknown')
        timestamp = row.get('Timestamp', '')
        text = row.get('Text', '')
        score = row.get('sentiment_score', 0.0)
        
        cursor.execute('''
            INSERT INTO raw_tweets (user, timestamp, text, sentiment_score)
            VALUES (?, ?, ?, ?)
        ''', (user, timestamp, text, score))
        if cursor.rowcount > 0: new_count += 1
            
    conn.commit()
    conn.close()
    return new_count