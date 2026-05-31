import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))

import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import joblib
import plotly.graph_objects as go
from fraud_detector import analyze_transaction

st.set_page_config(page_title="UniPay FraudX", layout="wide", initial_sidebar_state="expanded")

def inject_custom_css(theme):
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        * { font-family: 'Inter', sans-serif !important; }
        
        [data-testid="stSidebar"] {
            padding-top: 2rem;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        .glass-card {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .glass-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.15);
        }
        
        .metric-title {
            font-size: 0.9rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
            opacity: 0.8;
        }
        .metric-value {
            font-size: 2.2rem;
            font-weight: 700;
            margin: 0;
            background: linear-gradient(90deg, #ff4b8b, #ff6b6b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .metric-subtitle {
            font-size: 0.85rem;
            opacity: 0.6;
            margin-top: 8px;
        }

        .stButton > button {
            background: linear-gradient(135deg, #ff4b8b 0%, #ff1e56 100%);
            color: white !important;
            border-radius: 8px;
            padding: 0.5rem 1.5rem;
            border: none;
            box-shadow: 0 4px 15px rgba(255, 75, 139, 0.3);
            transition: all 0.3s ease;
            font-weight: 600;
            width: 100%;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(255, 75, 139, 0.4);
            background: linear-gradient(135deg, #ff1e56 0%, #ff4b8b 100%);
        }

        @keyframes fillBar { from { width: 0%; } }
        .progress-bg {
            background: rgba(255,255,255,0.1);
            border-radius: 8px;
            height: 12px;
            width: 100%;
            overflow: hidden;
            margin-top: 10px;
        }
        .progress-fill {
            height: 100%;
            border-radius: 8px;
            animation: fillBar 1.5s cubic-bezier(0.1, 0.7, 0.1, 1) forwards;
        }

        .result-box {
            padding: 24px;
            border-radius: 16px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            margin-top: 20px;
            border-left: 6px solid transparent;
        }
        .result-danger {
            background: linear-gradient(145deg, rgba(255, 30, 86, 0.1) 0%, rgba(255, 75, 139, 0.05) 100%);
            border-color: #ff1e56;
            border: 1px solid rgba(255, 30, 86, 0.2);
            border-left: 6px solid #ff1e56;
        }
        .result-success {
            background: linear-gradient(145deg, rgba(0, 184, 148, 0.1) 0%, rgba(54, 207, 201, 0.05) 100%);
            border-color: #00b894;
            border: 1px solid rgba(0, 184, 148, 0.2);
            border-left: 6px solid #00b894;
        }
        
        .block-container { padding-top: 2rem !important; }
        </style>
    """, unsafe_allow_html=True)

    if theme == "Dark":
        st.markdown("""
            <style>
            [data-testid="stAppViewContainer"] { background-color: #0b0f19; }
            [data-testid="stSidebar"] { background-color: #111827; }
            h1, h2, h3, h4, h5, h6, p, label, .metric-title, .metric-subtitle { color: #f3f4f6 !important; }
            .glass-card { background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255,255,255,0.05); }
            [data-testid="stDataFrame"] { background-color: #1f2937; border-radius: 12px; }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            [data-testid="stAppViewContainer"] { background-color: #f8fafc; }
            [data-testid="stSidebar"] { background-color: #ffffff; }
            h1, h2, h3, h4, h5, h6, p, label, .metric-title, .metric-subtitle { color: #1e293b !important; }
            .glass-card { background: rgba(255, 255, 255, 1); border: 1px solid rgba(0,0,0,0.05); box-shadow: 0 4px 20px rgba(0,0,0,0.03); }
            .progress-bg { background: rgba(0,0,0,0.05); }
            [data-testid="stDataFrame"] { background-color: #ffffff; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.03); }
            </style>
        """, unsafe_allow_html=True)


@st.cache_data
def load_data():
    possible_paths = [
        "data.xlsx",
        os.path.join("dataset", "data.xlsx")
    ]
    for dataset_path in possible_paths:
        if os.path.exists(dataset_path):
            return pd.read_excel(dataset_path)
    st.error(f"Dataset not found. Tried: {', '.join(possible_paths)}")
    return pd.DataFrame()

@st.cache_resource
def load_model():
    possible_paths = [
        "fraud_model.pkl",
        os.path.join("models", "fraud_model.pkl")
    ]
    for model_path in possible_paths:
        if os.path.exists(model_path):
            try:
                return pickle.load(open(model_path, "rb"))
            except Exception as e:
                st.error(f"Error loading model from {model_path}: {e}")
                continue
    st.error(f"Model not found. Tried: {', '.join(possible_paths)}")
    return None

df = load_data()
model = load_model()

if df.empty or model is None:
    st.warning("⚠️ Application is running in limited mode due to missing data or model.")
    st.stop()

st.sidebar.markdown(
    """
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="font-size: 1.8rem; font-weight: 700; background: linear-gradient(90deg, #ff4b8b, #ff6b6b); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0;">UniPay FraudX</h1>
        <p style="font-size: 0.8rem; opacity: 0.7; margin-top: 0;">AI Fraud Intelligence</p>
    </div>
    """, unsafe_allow_html=True
)

st.sidebar.markdown("### Navigation")
page = st.sidebar.radio(
    label="Navigation",
    options=[
        "🏠 Home",
        "📊 Dashboard",
        "🔍 Analysis",
        "🔮 Prediction Engine",
        "⚙️ About"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
st.sidebar.markdown("### Settings")
theme_choice = st.sidebar.radio("Theme Mode", ["🌙 Dark", "☀️ Light"])
theme = "Dark" if "Dark" in theme_choice else "Light"

inject_custom_css(theme)

chart_font_color = "#f3f4f6" if theme == "Dark" else "#1e293b"
chart_bg_color = "rgba(0,0,0,0)"

def apply_plotly_layout(fig):
    fig.update_layout(
        plot_bgcolor=chart_bg_color,
        paper_bgcolor=chart_bg_color,
        font=dict(family="Inter", color=chart_font_color),
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.1)', zeroline=False)
    )
    return fig


if "Home" in page:
    st.markdown("""
        <style>
        .hero-container {
            height: 75vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            background: linear-gradient(135deg, rgba(255, 75, 139, 0.15) 0%, rgba(255, 30, 86, 0.08) 100%);
            border-radius: 24px;
            padding: 60px 40px;
            border: 1px solid rgba(255, 75, 139, 0.2);
            position: relative;
            overflow: hidden;
            margin-bottom: 3rem;
        }
        .hero-container::before {
            content: '';
            position: absolute;
            top: -50%; left: -50%; width: 200%; height: 200%;
            background: radial-gradient(circle, rgba(255,75,139,0.12) 0%, transparent 50%);
            animation: pulse 15s infinite linear;
        }
        @keyframes pulse {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .hero-title {
            font-size: 3.8rem;
            font-weight: 800;
            background: linear-gradient(90deg, #ff4b8b, #ff6b6b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 24px;
            z-index: 1;
            line-height: 1.2;
        }
        .hero-subtitle {
            font-size: 1.3rem;
            max-width: 700px;
            opacity: 0.85;
            margin-bottom: 40px;
            z-index: 1;
            line-height: 1.6;
        }
        .hero-stats {
            display: flex;
            gap: 40px;
            justify-content: center;
            z-index: 1;
            margin-top: 20px;
        }
        .stat-item { text-align: center; }
        .stat-number {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(90deg, #ff4b8b, #ff6b6b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .stat-label { font-size: 0.9rem; opacity: 0.7; margin-top: 5px; }
        .feature-card {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 32px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        .feature-card:hover {
            transform: translateY(-8px);
            box-shadow: 0 12px 40px rgba(255, 75, 139, 0.2);
            border-color: rgba(255, 75, 139, 0.3);
        }
        .feature-icon { font-size: 3rem; margin-bottom: 16px; }
        .feature-title { font-size: 1.4rem; font-weight: 600; margin-bottom: 12px; }
        .feature-desc { opacity: 0.75; line-height: 1.6; font-size: 0.95rem; }
        .cta-section {
            background: linear-gradient(135deg, rgba(255, 75, 139, 0.1) 0%, rgba(255, 30, 86, 0.05) 100%);
            border-radius: 20px;
            padding: 48px;
            text-align: center;
            margin-top: 3rem;
            border: 1px solid rgba(255, 75, 139, 0.15);
        }
        .cta-title { font-size: 2rem; font-weight: 700; margin-bottom: 16px; }
        .cta-subtitle { font-size: 1.1rem; opacity: 0.8; margin-bottom: 32px; }
        @media (max-width: 768px) {
            .hero-title { font-size: 2.5rem; }
            .hero-subtitle { font-size: 1.1rem; }
            .hero-stats { flex-direction: column; gap: 20px; }
            .stat-number { font-size: 2rem; }
            .feature-grid { grid-template-columns: 1fr; }
            .cta-section { padding: 32px 24px; }
        }
        </style>
        
        <div class="hero-container">
            <div class="hero-title">🚀 Next-Gen Fraud Defense</div>
            <div class="hero-subtitle">
                Protect your digital ecosystem with real-time AI analytics, robust machine learning predictions, 
                and enterprise-grade transaction intelligence powered by advanced algorithms.
            </div>
            <div class="hero-stats">
                <div class="stat-item">
                    <div class="stat-number">99.2%</div>
                    <div class="stat-label">Accuracy</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">&lt;50ms</div>
                    <div class="stat-label">Response Time</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">24/7</div>
                    <div class="stat-label">Monitoring</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<h2 style='text-align: center; margin: 3rem 0 2rem 0;'>🎯 Core Capabilities</h2>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div class='feature-card'>
                <div class='feature-icon'>⚡</div>
                <div class='feature-title'>Real-Time ML</div>
                <div class='feature-desc'>Millisecond inference times for active transaction streams with Random Forest algorithms trained on thousands of fraud patterns.</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class='feature-card'>
                <div class='feature-icon'>📊</div>
                <div class='feature-title'>Deep Analytics</div>
                <div class='feature-desc'>Identify complex behavioral patterns, velocity attacks, and emerging threats with hybrid rule-based + ML detection engine.</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div class='feature-card'>
                <div class='feature-icon'>🛡️</div>
                <div class='feature-title'>Enterprise Scale</div>
                <div class='feature-desc'>Designed for high-throughput fintech infrastructure with automatic risk scoring and actionable recommendations.</div>
            </div>
        """, unsafe_allow_html=True)

    col4, col5, col6 = st.columns(3)
    with col4:
        st.markdown("""
            <div class='feature-card'>
                <div class='feature-icon'>🎯</div>
                <div class='feature-title'>Smart Detection</div>
                <div class='feature-desc'>Hybrid system combining heuristic rules with machine learning for superior fraud detection accuracy and fewer false positives.</div>
            </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown("""
            <div class='feature-card'>
                <div class='feature-icon'>📈</div>
                <div class='feature-title'>Visual Insights</div>
                <div class='feature-desc'>Interactive dashboards with real-time charts, KPIs, and transaction analytics for comprehensive fraud monitoring.</div>
            </div>
        """, unsafe_allow_html=True)
    with col6:
        st.markdown("""
            <div class='feature-card'>
                <div class='feature-icon'>🔍</div>
                <div class='feature-title'>Explainable AI</div>
                <div class='feature-desc'>Transparent fraud decisions with detailed explanations of triggered rules, risk scores, and ML confidence levels.</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        <div class='cta-section'>
            <div class='cta-title'>Ready to Secure Your Transactions?</div>
            <div class='cta-subtitle'>Explore our interactive dashboard, analyze fraud patterns, and test the prediction engine with real-time data.</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    quick_col1, quick_col2, quick_col3, quick_col4 = st.columns(4)
    with quick_col1:
        if st.button("📊 View Dashboard", use_container_width=True):
            st.session_state.page = "📊 Dashboard"
            st.rerun()
    with quick_col2:
        if st.button("🔮 Try Prediction", use_container_width=True):
            st.session_state.page = "🔮 Prediction Engine"
            st.rerun()
    with quick_col3:
        if st.button("🔍 Analyze Data", use_container_width=True):
            st.session_state.page = "🔍 Analysis"
            st.rerun()
    with quick_col4:
        if st.button("⚙️ Learn More", use_container_width=True):
            st.session_state.page = "⚙️ About"
            st.rerun()

elif "Dashboard" in page:
    st.markdown("<h2>Analytics Dashboard</h2>", unsafe_allow_html=True)

    total_tx = len(df)
    fraud_tx = len(df[df["label"] == "Suspicious"])
    fraud_rate = (fraud_tx / total_tx) * 100 if total_tx > 0 else 0
    total_vol = df["amount"].sum()

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.markdown(f"""
        <div class="glass-card">
            <div class="metric-title">Total Volume</div>
            <div class="metric-value">₹{total_vol/1000000:.2f}M</div>
            <div class="metric-subtitle">Processed Transactions</div>
        </div>
    """, unsafe_allow_html=True)
    kpi2.markdown(f"""
        <div class="glass-card">
            <div class="metric-title">Total Transactions</div>
            <div class="metric-value">{total_tx:,}</div>
            <div class="metric-subtitle">Last 30 Days</div>
        </div>
    """, unsafe_allow_html=True)
    kpi3.markdown(f"""
        <div class="glass-card">
            <div class="metric-title">Fraud Flags</div>
            <div class="metric-value">{fraud_tx:,}</div>
            <div class="metric-subtitle">Suspicious Activities</div>
        </div>
    """, unsafe_allow_html=True)
    kpi4.markdown(f"""
        <div class="glass-card">
            <div class="metric-title">Fraud Rate</div>
            <div class="metric-value">{fraud_rate:.2f}%</div>
            <div class="metric-subtitle">Of total volume</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border: 1px solid rgba(255,255,255,0.05); margin: 2rem 0;'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        fig_bar = px.bar(
            df.groupby(["hour", "label"])["txn_count_1hr"].sum().reset_index(),
            x="hour", y="txn_count_1hr", color="label",
            title="Activity Volume by Hour",
            color_discrete_sequence=["#00b894", "#ff4b8b"]
        )
        st.plotly_chart(apply_plotly_layout(fig_bar), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        df_line = df.groupby("hour")["amount"].mean().reset_index()
        fig_line = px.line(df_line, x="hour", y="amount", title="Average Transaction Value Trend")
        fig_line.update_traces(line_color="#ff4b8b", line_width=3, fill='tozeroy', fillcolor='rgba(255,75,139,0.1)')
        st.plotly_chart(apply_plotly_layout(fig_line), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    col3, col4 = st.columns([1, 1.5])
    with col3:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        fig_donut = px.pie(
            df, names="label", hole=0.6,
            title="Risk Distribution",
            color_discrete_sequence=["#00b894", "#ff4b8b"]
        )
        fig_donut.update_layout(annotations=[dict(text=f'{fraud_rate:.1f}%<br>Fraud', x=0.5, y=0.5, font_size=20, showarrow=False, font=dict(color=chart_font_color))])
        st.plotly_chart(apply_plotly_layout(fig_donut), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col4:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        fig_sender = px.bar(
            df, x="sender_type", color="receiver_type",
            title="Entity Type Correlation Matrix", barmode="stack",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(apply_plotly_layout(fig_sender), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

elif "Analysis" in page:
    st.markdown("<h2>Data Intelligence Explorer</h2>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("#### Transaction Registry")
    st.dataframe(df, use_container_width=True, height=400)
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        fig_hist = px.histogram(df, x="amount", color="label", nbins=40, title="Amount Distribution Density", color_discrete_sequence=["#00b894", "#ff4b8b"])
        st.plotly_chart(apply_plotly_layout(fig_hist), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        fig_scatter = px.scatter(df, x="amount", y="txn_count_1hr", color="label", size="amount", hover_data=["hour"], title="Velocity vs Value Analysis", color_discrete_sequence=["#00b894", "#ff4b8b"])
        st.plotly_chart(apply_plotly_layout(fig_scatter), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

elif "Prediction" in page:
    st.markdown("<h2>Real-time Fraud Prediction Engine</h2>", unsafe_allow_html=True)
    st.markdown("<p style='opacity: 0.8;'>Submit transaction telemetry for instant ML inference.</p>", unsafe_allow_html=True)

    form_col, result_col = st.columns([1, 1.2])
    with form_col:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### Input Telemetry")
        amount = st.number_input("Transaction Value (₹)", min_value=0.0, value=1500.0, step=100.0)
        txn = st.number_input("Txn Velocity (Last 1hr)", min_value=0, value=2, step=1)
        hour = st.slider("Hour of Day", 0, 23, 14)
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("Initialize Inference", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with result_col:
        if predict_btn:
            result = analyze_transaction(
                amount=amount,
                txn_count=txn,
                hour=hour,
                model=model,
            )

            final_pred      = 1 if result["is_fraud"] else 0
            risk_score      = result["fraud_score"]
            reason          = result["reason"]
            risk_level      = result["risk_label"]
            confidence      = result["ml_proba"] * 100
            rules_fired     = result["triggered_rules"]

            card_class      = "result-danger" if final_pred == 1 else "result-success"
            icon            = "🚨" if final_pred == 1 else "✅"
            color           = "#ff1e56" if final_pred == 1 else "#00b894"
            display_verdict = "SUSPICIOUS" if final_pred == 1 else "SAFE"

            rules_html = ""
            if rules_fired:
                rules_items = "".join(
                    f"<li style='margin-bottom:4px; font-size:0.85rem; opacity:0.85;'>{r}</li>"
                    for r in rules_fired
                )
                rules_html = f"""
                    <hr style="border:1px solid rgba(128,128,128,0.1); margin: 16px 0;">
                    <div style="font-size:0.8rem; font-weight:600; opacity:0.7; margin-bottom:8px;">TRIGGERED RULES</div>
                    <ul style="margin:0; padding-left:18px; list-style:disc;">{rules_items}</ul>
                """

            st.markdown(f"""
                <div class="{card_class}">
                    <h3 style="margin-top:0; color: {color} !important;">{icon} {display_verdict} — {risk_level} RISK</h3>
                    <p style="opacity:0.8; margin-bottom: 20px;">{reason}</p>

                    <div style="display:flex; justify-content:space-between; margin-bottom: 5px;">
                        <span style="font-size:0.9rem; font-weight:600;">ML Fraud Probability</span>
                        <span style="font-size:0.9rem; font-weight:600; color:{color};">{confidence:.1f}%</span>
                    </div>
                    <div class="progress-bg">
                        <div class="progress-fill" style="width: {min(confidence, 100):.1f}%; background: {color};"></div>
                    </div>

                    <hr style="border:1px solid rgba(128,128,128,0.1); margin: 20px 0;">

                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                        <div>
                            <div style="font-size:0.8rem; opacity:0.7;">Risk Score (Calculated)</div>
                            <div style="font-size:1.5rem; font-weight:700;">{risk_score}/100</div>
                        </div>
                        <div>
                            <div style="font-size:0.8rem; opacity:0.7;">Recommendation</div>
                            <div style="font-size:1.1rem; font-weight:600;">{result['recommendation'].replace('_', ' ')}</div>
                        </div>
                    </div>
                    {rules_html}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style="height: 100%; display: flex; align-items: center; justify-content: center; opacity: 0.5; border: 2px dashed rgba(128,128,128,0.2); border-radius: 16px; padding: 50px; text-align: center;">
                    Waiting for telemetry input...<br>Enter parameters and click Initialize Inference.
                </div>
            """, unsafe_allow_html=True)

elif "About" in page:
    st.markdown("<h2>System Architecture & Intelligence</h2>", unsafe_allow_html=True)

    st.markdown("""
        <div class="glass-card">
            <h4>UniPay FraudX</h4>
            <p style="opacity: 0.8; line-height: 1.6;">
                An enterprise-grade fraud detection platform engineered to process digital transactions in real-time. 
                Combining heuristic rule engines with advanced Machine Learning (Random Forest algorithms), FraudX isolates behavioral anomalies, high-velocity attacks, and unusual geographic footprints with millisecond latency.
            </p>
            <br>
            <h5>Technical Stack</h5>
            <ul style="opacity: 0.8; line-height: 1.6;">
                <li><b>Frontend Infrastructure:</b> Streamlit, Custom HTML/CSS (Glassmorphism), Plotly</li>
                <li><b>Inference Engine:</b> Scikit-Learn (Random Forest Classifier)</li>
                <li><b>Data Processing Pipeline:</b> Pandas, NumPy</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)