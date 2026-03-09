import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from data_engine import NiftyPredictorEngine

if "FRED_API_KEY" in st.secrets:
    FRED_API_KEY = st.secrets["FRED_API_KEY"]
else:
    FRED_API_KEY = "your_local_key_here" 

engine = NiftyPredictorEngine(fred_api_key=FRED_API_KEY)
st.set_page_config(page_title="Pinak's Macro Terminal", layout="wide", page_icon="🏛️")

st.title("🏛️ Nifty 50 Macro-Financial Analytics Terminal")
st.caption("Developed by Pinak Salunke | Multi-Variable Regression Implementation (R²=0.9859)")

# Sidebar for Controls
st.sidebar.header("⚙️ Terminal Controls")
# NEW: Timeline Selector
time_option = st.sidebar.selectbox(
    "Select Backtest Duration",
    ("1 Month", "3 Months", "6 Months", "1 Year", "2 Years")
)
time_map = {"1 Month": 30, "3 Months": 90, "6 Months": 180, "1 Year": 365, "2 Years": 730}
selected_days = time_map[time_option]

use_live_macro = st.sidebar.toggle("Use Live FRED Macro Data", value=False)
s_repo = st.sidebar.slider("Simulate Repo Rate", 4.0, 10.0, 6.5)
s_infl = st.sidebar.slider("Simulate Inflation", 2.0, 12.0, 4.8)

with st.spinner(f"Synchronizing {time_option} of Market Data..."):
    market, m_status = engine.fetch_data() # Now fetches 2y by default
    live_macro, l_status = engine.fetch_live_macro()
    macro = live_macro.copy()
    if not use_live_macro:
        macro["Repo_Rate"] = s_repo
        macro["Inflation"] = s_infl

if market:
    pred, impacts, weights, (top_var, w_val) = engine.predict(market, macro)
    actual = market["Nifty_Actual"]
    vix = market["VIX"]
    gap = actual - pred

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted Equilibrium", f"₹{pred:,.2f}")
    m2.metric("Current Market", f"₹{actual:,.2f}")
    m3.metric("Basis Point Gap", f"{gap:,.2f}", delta=f"{round((gap/actual)*100, 1)}%", delta_color="inverse")
    m4.metric("India VIX (Fear Index)", f"{vix:.2f}")

    st.markdown("---")
    
    # CHARTS SECTION
    c_left, c_right = st.columns(2)
    with c_left:
        st.subheader("Normalized Factor Weightage Analysis")
        fig_pie = px.pie(names=list(weights.keys()), values=list(weights.values()), hole=0.4, 
                         color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with c_right:
        st.subheader(f"📊 {time_option} Historical Backtest")
        # Uses the dynamic window selected in sidebar
        df_bt, mape, rmse, drift, rec_err = engine.get_backtest_with_drift(market["History"], macro, days_back=selected_days)
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=df_bt['Date'], y=df_bt['Actual'], name="Actual Price", line=dict(color='#00CC96', width=2)))
        fig_bt.add_trace(go.Scatter(x=df_bt['Date'], y=df_bt['Predicted'], name="Model Equilibrium", line=dict(dash='dash', color='#EF553B')))
        fig_bt.update_layout(template="plotly_dark", height=350, margin=dict(l=0, r=0, t=0, b=0), hovermode="x unified")
        st.plotly_chart(fig_bt, use_container_width=True)

    # Statistics Section
    st.markdown("---")
    st.subheader(f"🔬 Strategic Monitoring & Metrics ({time_option})")
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
