import yfinance as yf
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

# Load FinBERT (Cached)
# This will trigger the download in logs on first run
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=-1)

def get_sentiment(df):
    if df.empty: return 0.0
    # Scweet v1.8 usually uses 'Text' or 'text'
    col = 'Text' if 'Text' in df.columns else 'text'
    if col not in df.columns: return 0.0
    
    texts = df[col].astype(str).tolist()
    # Batch processing
    results = nlp(texts, truncation=True, max_length=512)
    
    scores = []
    for res in results:
        if res['label'] == 'Positive': scores.append(res['score'])
        elif res['label'] == 'Negative': scores.append(-res['score'])
        else: scores.append(0.0)
    
    return sum(scores) / len(scores) if scores else 0.0

def get_current_stock_price(ticker="NVDA"):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1d", interval="1m")
        if df.empty: df = stock.history(period="5d")
        return df['Close'].iloc[-1]
    except: return 0.0