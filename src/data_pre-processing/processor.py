import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re

analyzer = SentimentIntensityAnalyzer()

def get_sentiment(df):
    if df.empty: return 0.0
    
    # Handle Scweet column names
    col_name = 'Text' if 'Text' in df.columns else 'text'
    if col_name not in df.columns: return 0.0

    def clean(text):
        text = str(text).lower()
        text = re.sub(r'http\S+', '', text)
        return text
    
    scores = df[col_name].apply(lambda x: analyzer.polarity_scores(clean(x))['compound'])
    return scores.mean()

def get_current_stock_price(ticker="NVDA"):
    try:
        stock = yf.Ticker(ticker)
        # 1m interval for live checks
        df = stock.history(period="1d", interval="1m")
        if df.empty:
            # Market closed? Get last close
            df = stock.history(period="5d")
        return df['Close'].iloc[-1]
    except:
        return 0.0