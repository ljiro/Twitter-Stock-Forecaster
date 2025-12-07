import sqlite3
import pandas as pd

DB_PATH = "data/nvda.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Training Data Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS daily_summary (
            date TEXT PRIMARY KEY,
            avg_sentiment REAL,
            close_price REAL,
            target_price_3d REAL
        )
    ''')
    # Predictions Table
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
    conn = sqlite3.connect(DB_PATH)
    for _, row in df.iterrows():
        conn.execute('''
            INSERT INTO daily_summary (date, avg_sentiment, close_price, target_price_3d)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                avg_sentiment=excluded.avg_sentiment,
                close_price=excluded.close_price,
                target_price_3d=excluded.target_price_3d
        ''', (row['date'], row['avg_sentiment'], row['close_price'], row.get('target_price_3d')))
    conn.commit()
    conn.close()

def update_past_target(date_str, actual_price):
    conn = sqlite3.connect(DB_PATH)
    conn.execute('UPDATE daily_summary SET target_price_3d = ? WHERE date = ?', (actual_price, date_str))
    conn.commit()
    conn.close()

def load_history_for_features():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM daily_summary ORDER BY date ASC", conn)
    conn.close()
    return df