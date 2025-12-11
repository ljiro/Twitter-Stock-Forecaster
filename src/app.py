import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import requests
import datetime

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Nvidia Market Oracle")
API_URL = "http://127.0.0.1:8000"

# --- HELPER FUNCTIONS ---
def get_api_data(endpoint):
    """Safely fetch data from API."""
    try:
        response = requests.get(f"{API_URL}{endpoint}")
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.sidebar.error(f"⚠️ API Error: {e}")
        return None

def get_color_rgba(name, opacity=0.1):
    c_map = {"blue": f"rgba(0,0,255,{opacity})", "orange": f"rgba(255,165,0,{opacity})", "teal": f"rgba(0,128,128,{opacity})", "purple": f"rgba(128,0,128,{opacity})"}
    return c_map.get(name, f"rgba(100,100,100,{opacity})")

def calculate_technicals(prices):
    """Calculates Volatility and RSI from price history list for the Report."""
    if not prices or len(prices) < 15: return 0.0, 50.0
    series = pd.Series(prices)
    pct_change = series.pct_change()
    volatility = pct_change.tail(30).std() * 100
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return volatility, rsi.iloc[-1]

# --- ZONE A: SIDEBAR ---
with st.sidebar:
    st.header("🎛️ Oracle Control")
    status = get_api_data("/")
    if status:
        st.success(f"🟢 System Online: {status.get('service', 'API')}")
    else:
        st.error("🔴 System Offline")

    ticker = st.selectbox("Select Asset", ["$NVDA"])
    
    st.divider()
    
    st.subheader("Model Selection")
    # For the table comparison
    model_options = ["Gradient Boosting", "Linear QR", "MQLSTM", "QRNN"]
    selected_models = st.multiselect("Compare Models", model_options, default=["Gradient Boosting", "MQLSTM"])
    
    st.info("The 'Forecast Chart' always displays the Champion Model (GBR) for clarity.")
    
    st.divider()
    if st.button("🔄 Refresh Data"):
        st.rerun()

# --- FETCH & PARSE DATA ---
pred_data = get_api_data("/predict")
history_data = get_api_data("/history")
perf_data = get_api_data("/performance")

if not pred_data or "error" in pred_data:
    st.warning("⏳ Pipeline Initializing... Please wait for the next cycle.")
    st.stop()

# Parse JSON safely
meta = pred_data.get('meta', {})
signals = pred_data.get('signals', {})
raw_models = pred_data.get('models', {})
technicals = pred_data.get('technicals', {})

# Key Metrics
current_price = meta.get('current_price', 0.0)
last_updated = meta.get('last_updated', 'Unknown')
sent_score = signals.get('sentiment_score', 0.0)
fear_index = signals.get('market_fear_index', 0.0)

# Champion Model Data (GBR)
champ_key = 'gradient_boosting'
champ_data = raw_models.get(champ_key, {}).get('prediction', {})
if not champ_data: 
    champ_data = {'pred': current_price, 'lower': current_price, 'upper': current_price, 'return_pred': 0}

# Mapping
model_key_map = {
    "Gradient Boosting": "gradient_boosting",
    "Linear QR": "linear_qr",
    "MQLSTM": "mqlstm",
    "QRNN": "qrnn"
}

# --- TABS LAYOUT ---
tab1, tab2 = st.tabs(["📈 Live Dashboard", "📄 Executive Report"])

# ==============================================================================
# TAB 1: LIVE DASHBOARD
# ==============================================================================
with tab1:
    # 1. KPI ROW
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("NVDA Price", f"${current_price:,.2f}")
    c2.metric("AI Sentiment", f"{sent_score:.2f}", delta="Bullish" if sent_score > 0 else "Bearish")
    c3.metric("Market Fear", f"{fear_index:.1f}", delta_color="inverse")
    
    ret_pct = champ_data['return_pred'] * 100
    c4.metric("3-Day Forecast", f"${champ_data['pred']:.2f}", delta=f"{ret_pct:+.2f}%")

    # 2. HERO CHART: RESEARCH GRAPH (History + Cone)
    st.subheader(f"🔮 Strategic Forecast (Champion Model)")
    
    # Setup History Data
    if history_data:
        df_hist = pd.DataFrame(history_data)
        df_hist['timestamp'] = pd.to_datetime(df_hist['timestamp'])
        df_chart = df_hist.tail(100) # Last 100 points
        last_hist_date = df_chart['timestamp'].iloc[-1]
        last_hist_price = df_chart['price'].iloc[-1]
    else:
        df_chart = pd.DataFrame()
        last_hist_date = datetime.datetime.now()
        last_hist_price = current_price

    # Setup Forecast Data (Stitched)
    target_dt = last_hist_date + datetime.timedelta(days=3)
    x_forecast = [last_hist_date, target_dt]
    y_median = [last_hist_price, champ_data['pred']]
    y_upper = [last_hist_price, champ_data['upper']]
    y_lower = [last_hist_price, champ_data['lower']]

    fig = go.Figure()

    # A. Historical Line
    if not df_chart.empty:
        fig.add_trace(go.Scatter(
            x=df_chart['timestamp'], y=df_chart['price'],
            mode='lines', name='History (30D)',
            line=dict(color='white', width=2), hovertemplate='$%{y:.2f}'
        ))

    # B. The Cone (Upper Transparent + Lower Fill)
    fig.add_trace(go.Scatter(
        x=x_forecast, y=y_upper, mode='lines', line=dict(width=0),
        showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=x_forecast, y=y_lower, mode='lines', 
        fill='tonexty', fillcolor='rgba(0, 0, 255, 0.15)',
        line=dict(width=0), name='80% Risk Range', hoverinfo='skip'
    ))

    # C. Median Forecast
    fig.add_trace(go.Scatter(
        x=x_forecast, y=y_median, mode='lines+markers',
        name='Median Forecast', line=dict(color='blue', width=2, dash='dash'),
        marker=dict(size=8)
    ))

    # Annotation
    fig.add_annotation(
        x=target_dt, y=champ_data['pred'], text=f"${champ_data['pred']:.2f}",
        showarrow=True, arrowhead=1, ax=40, ay=0, font=dict(color="blue", size=12)
    )

    fig.update_layout(
        height=500, title="Price Trajectory & Risk Cone",
        hovermode="x unified", xaxis_title="Timeline", yaxis_title="Price ($)",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    # Add vertical "Now" line
    fig.add_vline(x=last_hist_date, line_width=1, line_dash="dot", line_color="gray")
    
    st.plotly_chart(fig, use_container_width=True)

    # 3. METRICS & BATTLE ARENA
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("⚔️ Efficiency Frontier")
        st.caption("Which model is best? (Top-Left = Better)")
        
        if perf_data:
            df_perf = pd.DataFrame(perf_data)
            if not df_perf.empty:
                df_latest = df_perf.sort_values('timestamp').groupby('model').tail(1)
                
                fig_eff = px.scatter(
                    df_latest, x="winkler", y="coverage", color="model",
                    size=[20]*len(df_latest), hover_data=["timestamp"],
                    color_discrete_map={"GradientBoosting": "blue", "LinearQR": "orange", "MQLSTM": "teal", "QRNN": "purple"}
                )
                fig_eff.add_hline(y=0.80, line_dash="dash", line_color="red", annotation_text="Target (80%)")
                fig_eff.update_layout(xaxis_title="Winkler Score (Lower is Better)", yaxis_title="Coverage %")
                st.plotly_chart(fig_eff, use_container_width=True)
            else:
                st.info("Performance data is empty.")
        else:
            st.info("Waiting for training metrics...")

    with col_right:
        st.subheader("📋 Latest Signal Details")
        table_rows = []
        for m_name in model_options:
            k = model_key_map[m_name]
            # Deep safe get
            m_payload = raw_models.get(k, {})
            p_val = m_payload.get('prediction')
            s_val = m_payload.get('performance')
            
            # Formats
            price_fmt = f"${p_val['pred']:.2f}" if p_val else "-"
            range_fmt = f"${p_val['lower']:.0f}-${p_val['upper']:.0f}" if p_val else "-"
            
            if isinstance(s_val, dict):
                wink_fmt = f"{s_val.get('winkler', 0):.2f}"
                cov_fmt = f"{s_val.get('coverage', 0):.1%}"
                # Using 'winkler' as the main error score proxy if 'mae' isn't explicitly saved
                # or use pinball_loss if available
                err_fmt = f"{s_val.get('pinball_loss', 0):.3f}"
            else:
                wink_fmt, cov_fmt, err_fmt = "-", "-", "-"

            table_rows.append({
                "Model": m_name,
                "Price": price_fmt,
                "Range": range_fmt,
                "Winkler": wink_fmt,
                "Coverage": cov_fmt,
                "Err Score": err_fmt
            })
        
        st.dataframe(
            pd.DataFrame(table_rows), 
            hide_index=True,
            column_config={
                "Winkler": st.column_config.NumberColumn("Winkler (Risk)", help="Lower is better"),
                "Coverage": st.column_config.TextColumn("Safety %", help="Target 80%"),
            }
        )
        st.caption(f"Last Pipeline Update: {last_updated}")

# ==============================================================================
# TAB 2: EXECUTIVE REPORT
# ==============================================================================
with tab2:
    # Recalculate Technicals for Report Text
    hist_prices = pd.DataFrame(history_data)['price'].tolist() if history_data else []
    vol_calc, rsi_calc = calculate_technicals(hist_prices)
    
    # Use API provided technicals if available, else calculated
    rsi_val = technicals.get('rsi_14', rsi_calc)
    vol_val = technicals.get('volatility_7', vol_calc)

    # NARRATIVE LOGIC
    if rsi_val < 30: rsi_text = "OVERSOLD (Bullish Signal)"
    elif rsi_val > 70: rsi_text = "OVERBOUGHT (Bearish Signal)"
    else: rsi_text = "NEUTRAL (Equilibrium)"

    if sent_score > 0.1: sent_text = "BULLISH (Positive Chatter)"
    elif sent_score < -0.1: sent_text = "BEARISH (Negative Chatter)"
    else: sent_text = "MIXED/NEUTRAL"

    direction = "UP" if champ_data['return_pred'] > 0 else "DOWN"
    
    # RENDER
    st.title("📄 AI Market Analysis Report")
    st.caption(f"Generated at: {last_updated}")
    st.divider()

    st.subheader("1. Market Context")
    rc1, rc2, rc3, rc4 = st.columns(4)
    rc1.metric("Price", f"${current_price:.2f}")
    rc2.metric("Volatility", f"{vol_val:.2f}%")
    rc3.metric("RSI (14)", f"{rsi_val:.1f}")
    rc4.metric("Sentiment", f"{sent_score:.3f}")

    st.markdown(f"""
    * **Technical Outlook:** RSI is at **{rsi_val:.1f}**, indicating the asset is **{rsi_text}**.
    * **Volatility:** Daily volatility is **{vol_val:.2f}%**, suggesting {'high' if vol_val > 3 else 'stable'} conditions.
    * **Crowd Intelligence:** Social sentiment is **{sent_text}** (Score: {sent_score}).
    """)

    st.divider()

    st.subheader(f"2. AI Forecast ({direction})")
    st.markdown(f"""
    The Champion Model (Gradient Boosting) projects a move **{direction}** to **${champ_data['pred']:.2f}**.
    
    * **Upside Scenario (90%):** Rally to **${champ_data['upper']:.2f}**
    * **Downside Scenario (10%):** Drop to **${champ_data['lower']:.2f}**
    """)

    # Risk Calculation
    down_risk = current_price - champ_data['lower']
    up_reward = champ_data['upper'] - current_price
    rr_ratio = up_reward / down_risk if down_risk > 0 else 0

    st.subheader("3. Risk Assessment")
    st.info(f"**Risk/Reward Ratio:** {rr_ratio:.2f}x (Upside vs Downside Width)")
    st.markdown(f"There is an **80% probability** the price remains between **${champ_data['lower']:.2f}** and **${champ_data['upper']:.2f}** over the next 3 days.")