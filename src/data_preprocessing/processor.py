import os

# CRITICAL: Set Hugging Face cache directory BEFORE importing transformers
# This must be done at the VERY TOP of the file
os.environ['HF_HOME'] = '/tmp/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/tmp/huggingface'
os.environ['XDG_CACHE_HOME'] = '/tmp'

import yfinance as yf
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_finbert_model():
    """Load FinBERT model with explicit cache directory."""
    try:
        logger.info(f"Loading FinBERT model...")
        logger.info(f"Cache directory: {os.environ.get('TRANSFORMERS_CACHE')}")
        
        # Load with explicit cache directory
        tokenizer = BertTokenizer.from_pretrained(
            'yiyanghkust/finbert-tone',
            cache_dir='/tmp/huggingface'
        )
        model = BertForSequenceClassification.from_pretrained(
            'yiyanghkust/finbert-tone',
            cache_dir='/tmp/huggingface'
        )
        
        logger.info("FinBERT model loaded successfully")
        return tokenizer, model
    except Exception as e:
        logger.error(f"Failed to load FinBERT model: {e}")
        raise

# Load the model
try:
    tokenizer, model = load_finbert_model()
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=-1)
    logger.info("Sentiment analysis pipeline created")
except Exception as e:
    logger.error(f"Failed to initialize sentiment analysis: {e}")
    # Create dummy objects to prevent crashes
    tokenizer, model, nlp = None, None, None

def get_sentiment(df):
    """Calculate sentiment score from tweet dataframe."""
    if df.empty: 
        return 0.0
    
    # Check if model loaded successfully
    if nlp is None:
        logger.warning("Sentiment model not loaded, returning neutral sentiment")
        return 0.0
    
    # Scweet v1.8 usually uses 'Text' or 'text'
    col = 'Text' if 'Text' in df.columns else 'text'
    if col not in df.columns: 
        return 0.0
    
    try:
        texts = df[col].astype(str).tolist()
        if not texts:
            return 0.0
            
        # Batch processing
        results = nlp(texts, truncation=True, max_length=512)
        
        scores = []
        for res in results:
            if res['label'] == 'Positive': 
                scores.append(res['score'])
            elif res['label'] == 'Negative': 
                scores.append(-res['score'])
            else: 
                scores.append(0.0)
        
        avg_score = sum(scores) / len(scores) if scores else 0.0
        logger.info(f"Calculated sentiment score: {avg_score} from {len(scores)} tweets")
        return avg_score
        
    except Exception as e:
        logger.error(f"Error calculating sentiment: {e}")
        return 0.0

def get_current_stock_price(ticker="NVDA"):
    """Get current stock price using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1d", interval="1m")
        if df.empty: 
            df = stock.history(period="5d")
        price = df['Close'].iloc[-1]
        logger.info(f"Current {ticker} price: ${price:.2f}")
        return price
    except Exception as e:
        logger.error(f"Error getting stock price: {e}")
        return 0.0