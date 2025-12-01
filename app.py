import streamlit as st
import plotly.graph_objects as go
# Import your backend modules
# from model_pipeline import get_prediction, get_sentiment_data

st.set_page_config(layout="wide", page_title="Market Sentiment Oracle")

# --- ZONE A: SIDEBAR ---
with st.sidebar:
    st.header("Configuration")
    ticker = st.selectbox("Select Asset", ["$TSLA", "$AAPL", "$NVDA"])
    bot_filter = st.checkbox("Apply Bot Cleaning", value=True)
    st.info(f"Model: FinBERT + Quantile Regression")

# --- ZONE B: HERO CHART ---
st.title(f"Sentiment Forecast: {ticker}")

# Create the Fan Chart (Quantile Regression)
fig = go.Figure()

# 1. Historical Data
fig.add_trace(go.Candlestick(name='Market Data'))

# 2. Sentiment Overlay
fig.add_trace(go.Scatter(name='Sentiment Trend', line=dict(color='orange', width=2)))

# 3. Prediction Fan (The "Cone" from your PDF)
# This visualizes the Quantile Regression risk bounds
fig.add_trace(go.Scatter(name='Upper Bound (90%)', line=dict(width=0), fill=None))
fig.add_trace(go.Scatter(name='Lower Bound (10%)', fill='tonexty', fillcolor='rgba(0,100,255,0.2)'))

st.plotly_chart(fig, use_container_width=True)

# --- ZONE C & D: METRICS & INSIGHTS ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Dominant Topics (LDA Analysis)")
    # Placeholder for Word Cloud or Topic List
    st.bar_chart({"Earnings": 80, "Product Launch": 45, "Legal": 20})

with col2:
    st.subheader("Live Feed")
    st.markdown("""
    * **@TraderJ**: $TSLA looking strong on support! (Score: 0.9)
    * **@BearTrap**: Selling the news. (Score: -0.8)
    """)
    st.metric(label="3-Day Forecast Return", value="+2.4%", delta="Bullish")