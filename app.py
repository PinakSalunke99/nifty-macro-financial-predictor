import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go 
from data_engine import NiftyPredictorEngine
    
# Check if we are running in the cloud (Streamlit Secrets) or locally
if "FRED_API_KEY" in st.secrets:
    FRED_API_KEY = st.secrets["FRED_API_KEY"]
else:
# If running locally, it will look for this key or stay as a placeholder
    FRED_API_KEY = "your_local_key_here" 

engine = NiftyPredictorEngine(fred_api_key=FRED_API_KEY)
st.set_page_config(page_title="Pinak's Macro Terminal", layout="wide", page_icon="🏛️")

st.title("🏛️ Nifty 50 Macro-Financial Analytics Terminal")
st.caption("Developed by Pinak Salunke | Multi-Variable Regression Implementation (R²=0.9859)")

# Sidebar for Economic Stress Testing
st.sidebar.header("⚙️ Economic Stress Test")
use_live_macro = st.sidebar.toggle("Use Live FRED Macro Data", value=False)
s_repo = st.sidebar.slider("Simulate Repo Rate", 4.0, 10.0, 6.5)
s_infl = st.sidebar.slider("Simulate Inflation", 2.0, 12.0, 4.8)
with st.spinner("Processing Market Intelligence..."):
    market, m_status = engine.fetch_data()
    live_macro, l_status = engine.fetch_live_macro()
        
    # Final Macro Selection
    macro = live_macro.copy()
    if not use_live_macro:
        macro["Repo_Rate"] = s_repo
        macro["Inflation"] = s_infl
        
    st.sidebar.info(f"Macro Status: {l_status}")

if market:
    # Perform core prediction logic
    pred, impacts, weights, (top_var, w_val) = engine.predict(market, macro)
    actual = market["Nifty_Actual"]
    vix = market["VIX"]
    gap = actual - pred

    # 1. Executive Summary Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted Equilibrium", f"₹{pred:,.2f}")
    m2.metric("Market Actual", f"₹{actual:,.2f}")
    m3.metric("Basis Point Gap", f"{gap:,.2f}", delta=f"{round((gap/actual)*100, 1)}%", delta_color="inverse")
    m4.metric("India VIX (Fear Index)", f"{vix:.2f}")

    # 2. AI Diagnostic Box
    st.markdown("---")
    if vix > 18:
        st.warning(f"**High Volatility Regime:** VIX at {vix:.2f} suggests market price is currently suppressed by investor fear.")
    elif abs(gap) > 2500:
        st.info("**Structural Deviation:** Significant gap identified between macro-fundamentals and current price.")
    else:
        st.success("**Equilibrium:** Market is trading in line with your research model.")

    # 3. Visual Analytics (Charts)
    c_left, c_right = st.columns(2)
    with c_left:
        st.subheader("Normalized Factor Weightage Analysis")
        fig_pie = px.pie(names=list(weights.keys()), values=list(weights.values()), hole=0.4, 
                        color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with c_right:
        st.subheader("30-Day Historical Backtest")
        df_bt, mape, rmse, drift, rec_err = engine.get_backtest_with_drift(market["History"], macro)
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=df_bt['Date'], y=df_bt['Actual'], name="Actual Price", line=dict(color='#00CC96', width=3)))
        fig_bt.add_trace(go.Scatter(x=df_bt['Date'], y=df_bt['Predicted'], name="Model Equilibrium", line=dict(dash='dash', color='#EF553B')))
        fig_bt.update_layout(template="plotly_dark", height=350, margin=dict(l=0, r=0, t=0, b=0), hovermode="x unified")
        st.plotly_chart(fig_bt, use_container_width=True)

    # 4. Live Variable Matrix Table
    st.subheader("Live Variable Ingestion Matrix")
    combined = {**{"Nifty": actual, "SPY": market["SPY_Price"], "USDINR": market["USDINR"]}, **macro}
    st.table(pd.DataFrame([combined]))

    # 5. Advanced Statistics & Drift Monitoring (Bottom Section)
    st.markdown("---")
    st.subheader("🔬 Strategic Model Monitoring & Professional Metrics")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.write(f"**MAPE (Average % Error):** {mape}%")
        st.write(f"**RMSE (Root Mean Square):** ₹{rmse:,.0f}")
    with col_b:
        if drift == "Stable":
            st.success(f"**Model Drift Status: {drift}**")
        else:
            st.error(f"**Model Drift Status: {drift}**")
        st.write(f"Recent Tracking Error (5D): {rec_err}%")
    with col_c:
        st.info(f"**Top Concentration Factor:** {top_var} contributes {w_val}% to current valuation.")

else:
    st.error(f"Terminal Initialization Error: {m_status}")
