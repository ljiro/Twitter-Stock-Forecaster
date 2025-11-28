from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import torch

# Load FinBERT (Cached so it doesn't reload every function call)
# We use the 'cpu' device unless you have passed a GPU into Docker
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')

# Create a pipeline
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=-1) # device=-1 for CPU

def get_sentiment(df):
    """
    Replaces VADER with FinBERT.
    Returns: A score between -1 (Negative) and 1 (Positive).
    """
    if df.empty:
        return 0.0
    
    col_name = 'Text' if 'Text' in df.columns else 'text'
    if col_name not in df.columns:
        return 0.0
    
    # 1. Prepare texts (Truncate to 512 tokens max to prevent crash)
    texts = df[col_name].astype(str).tolist()
    
    # 2. Run Inference (Batching is handled by pipeline, but we can do simple loop for clarity)
    # FinBERT outputs labels: 'Positive', 'Negative', 'Neutral'
    results = nlp(texts, truncation=True, max_length=512)
    
    # 3. Convert Labels to Score (-1 to 1)
    scores = []
    for res in results:
        label = res['label']
        score = res['score'] # Confidence score (0.0 to 1.0)
        
        if label == 'Positive':
            scores.append(score)        # ex: 0.95
        elif label == 'Negative':
            scores.append(-score)       # ex: -0.95
        else: # Neutral
            scores.append(0.0)
            
    # Return average sentiment for the batch
    return sum(scores) / len(scores) if scores else 0.0