import pandas as pd
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re

analyzer = SentimentIntensityAnalyzer()

def clean_text(text):
    """Basic tweet cleaning"""
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text) # Remove URLs
    text = re.sub(r'@\w+', '', text)    # Remove mentions
    text = re.sub(r'#', '', text)       # Remove hashtag symbol
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    return text

def get_sentiment(df):
    """Adds a 'compound' sentiment score to the DataFrame"""
    if df.empty:
        return 0.0
    
    # Scweet often outputs a 'Text' or 'text' column
    col_name = 'Text' if 'Text' in df.columns else 'text'
    
    df['clean_text'] = df[col_name].apply(clean_text)
    df['sentiment'] = df['clean_text].apply(
        lambda x: analyzer.polarity_scores(x)['compound']
    )
    # Return average sentiment for this batch
    return df['sentiment'].mean()

def get_current_stock_price(ticker="NVDA"):
    """Gets the live market price"""
    try:
        stock = yf.Ticker(ticker)
        # Get extremely recent data
        df = stock.history(period="1d", interval="1m")
        if df.empty:
            # Fallback for weekends/closed market: get last close
            df = stock.history(period="5d")
        return df['Close'].iloc[-1]
    except Exception as e:
        print(f"⚠️ Stock API Error: {e}")
        return 0.0