import yfinance as yf

def fetch_institutional_data(ticker_symbol="NVDA"):
    """
    Fetches 'Smart Money' indicators using free sources.
    1. Market Fear Index (^VXN) - Replaces Put/Call Ratio
    2. Analyst Ratings
    3. Institutional Ownership
    """
    print(f"🏦 Fetching Institutional Data...")
    try:
        # 1. Fetch Market Fear Index (Nasdaq Volatility)
        vxn = yf.Ticker("^VXN")
        # Get latest price (fastest way)
        todays_data = vxn.history(period="1d")
        if not todays_data.empty:
            market_fear = todays_data['Close'].iloc[-1]
        else:
            market_fear = 20.0 # Default fallback
            
        # 2. Fetch NVDA Specifics
        ticker = yf.Ticker(ticker_symbol)
        try:
            info = ticker.info
            analyst_rating = info.get('recommendationMean', 3.0) 
            inst_ownership = info.get('heldPercentInstitutions', 0.60)
        except:
            analyst_rating = 3.0
            inst_ownership = 0.60

        print(f"   ✅ Fear Index: {round(market_fear, 2)} | Rating: {analyst_rating}")
        
        return {
            "market_fear_index": round(market_fear, 2),
            "analyst_rating": float(analyst_rating),
            "institutional_ownership": float(inst_ownership)
        }

    except Exception as e:
        print(f"❌ Institutional Data Error: {e}")
        return None