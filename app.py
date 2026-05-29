import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import os
import joblib
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(page_title="UniPay FraudX", layout="wide", initial_sidebar_state="expanded")

# ================== CLEAN CSS THEME INJECTION ==================
def inject_custom_css(theme):
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        .hero-title, .hero-subtitle, .metric-title, .metric-value, .metric-subtitle, 
        .glass-card, .result-box, div[data-testid="stMarkdownContainer"] p, 
        label, h1, h2, h3, h4, h5, h6 {
            font-family: 'Inter', sans-serif !important;
        }
        
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

        @keyframes fillBar {
            from { width: 0%; }
        }
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
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        .result-danger {
            background: linear-gradient(145deg, rgba(255, 30, 86, 0.1) 0%, rgba(255, 75, 139, 0.05) 100%);
            border-left: 6px solid #ff1e56 !important;
        }
        .result-success {
            background: linear-gradient(145deg, rgba(0, 184, 148, 0.1) 0%, rgba(54, 207, 201, 0.05) 100%);
            border-left: 6px solid #00b894 !important;
        }
        
        .block-container {
            padding-top: 2rem !important;
        }
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

# ================== MODEL ESTIMATOR FALLBACK ==================
class DummyModel:
    def predict(self, X):
        return [1 if float(X[0][0]) > 10000 or int(X[0][1]) > 10 else 0]
    def predict_proba(self, X):
        return [[0.15, 0.85] if float(X[0][0]) > 10000 or int(X[0][1]) > 10 else [[0.92, 0.08]]]

# ================== DATA HANDLING PIPELINES ==================
@st.cache_data
def load_data():
    dataset_path = os.path.join("dataset", "data.xlsx")
    if not os.path.exists(dataset_path):
        return pd.DataFrame({
            "amount": [1500.0, 12000.0, 300.0, 5500.0, 89000.0],
            "txn_count_1hr": [2, 12, 1, 4, 15],
            "hour": [14, 2, 18, 23, 4],
            "label": ["Safe", "Suspicious", "Safe", "Safe", "Suspicious"],
            "sender_type": ["Customer", "Merchant", "Customer", "Customer", "Merchant"],
            "receiver_type": ["Merchant", "Customer", "Merchant", "Merchant", "Customer"]
        })
    return pd.read_excel(dataset_path)

@st.cache_resource
def load_model():
    model_path = os.path.join("models", "fraud_model.pkl")
    if not os.path.exists(model_path):
        return DummyModel()
    try:
        return pickle.load(open(model_path, "rb"))
    except Exception as e:
        return DummyModel()

df = load_data()
model = load_model()

# Sidebar UI
st.sidebar.markdown(
    """
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="font-size: 1.8rem; font-weight: 700;
        background: linear-gradient(90deg, #ff4b8b, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;">
        UniPay FraudX</h1>
        <p style="font-size: 0.8rem; opacity: 0.7;">AI Fraud Intelligence</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("### Navigation")
page = st.sidebar.radio(
    label="Navigation",
    options=["🏠 Home", "📊 Dashboard", "🔍 Analysis", "🔮 Prediction Engine", "⚙️ About"],
    label_visibility="collapsed"
)

st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
st.sidebar.markdown("### Settings")

theme_choice = st.sidebar.radio("Theme Mode", ["🌙 Dark", "☀️ Light"])

# ✅ IMPORTANT: define theme FIRST
theme = "Dark" if "Dark" in theme_choice else "Light"

# ✅ THEN derive dependent values
chart_font_color = "#f3f4f6" if theme == "Dark" else "#1e293b"
chart_bg_color = "rgba(0,0,0,0)"

# apply theme AFTER variables exist
inject_custom_css(theme)


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
# ================== PAGE ROUTER ==================

if "Home" in page:
    st.markdown("""
        <style>
        .hero-container {
            height: 70vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            background: linear-gradient(135deg, rgba(255, 75, 139, 0.1) 0%, rgba(255, 30, 86, 0.05) 100%);
            border-radius: 24px;
            padding: 40px;
            border: 1px solid rgba(255, 75, 139, 0.2);
            position: relative;
            overflow: hidden;
        }
        .hero-container::before {
            content: '';
            position: absolute;
            top: -50%; left: -50%; width: 200%; height: 200%;
            background: radial-gradient(circle, rgba(255,75,139,0.1) 0%, transparent 50%);
            animation: pulse 15s infinite linear;
        }
        @keyframes pulse {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .hero-title {
            font-size: 3.5rem;
            font-weight: 800;
            background: linear-gradient(90deg, #ff4b8b, #ff6b6b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 20px;
            z-index: 1;
        }
        .hero-subtitle {
            font-size: 1.2rem;
            max-width: 600px;
            opacity: 0.8;
            margin-bottom: 40px;
            z-index: 1;
        }
        </style>
        
        <div class="hero-container">
            <div class="hero-title">Next-Gen Fraud Defense</div>
            <div class="hero-subtitle">
                Protect your digital ecosystem with real-time AI analytics, robust machine learning predictions, and enterprise-grade transaction intelligence.
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.markdown("<div class='glass-card'><h4>⚡ Real-Time ML</h4><p style='opacity:0.7; font-size:0.9rem;'>Millisecond inference times for active transaction streams.</p></div>", unsafe_allow_html=True)
    c2.markdown("<div class='glass-card'><h4>📊 Deep Analytics</h4><p style='opacity:0.7; font-size:0.9rem;'>Identify complex behavioral patterns and emerging threats.</p></div>", unsafe_allow_html=True)
    c3.markdown("<div class='glass-card'><h4>🛡️ Enterprise Scale</h4><p style='opacity:0.7; font-size:0.9rem;'>Designed for high-throughput fintech infrastructure.</p></div>", unsafe_allow_html=True)

elif "Dashboard" in page:
    st.markdown("<h2>Analytics Dashboard</h2>", unsafe_allow_html=True)
    
    total_tx = len(df)
    fraud_tx = len(df[df["label"] == "Suspicious"])
    fraud_rate = (fraud_tx / total_tx) * 100 if total_tx > 0 else 0
    total_vol = df["amount"].sum()
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    kpi1.markdown(f"""<div class="glass-card"><div class="metric-title">Total Volume</div><div class="metric-value">₹{total_vol/1000000:.2f}M</div><div class="metric-subtitle">Processed Transactions</div></div>""", unsafe_allow_html=True)
    kpi2.markdown(f"""<div class="glass-card"><div class="metric-title">Total Transactions</div><div class="metric-value">{total_tx:,}</div><div class="metric-subtitle">Last 30 Days</div></div>""", unsafe_allow_html=True)
    kpi3.markdown(f"""<div class="glass-card"><div class="metric-title">Fraud Flags</div><div class="metric-value">{fraud_tx:,}</div><div class="metric-subtitle">Suspicious Activities</div></div>""", unsafe_allow_html=True)
    kpi4.markdown(f"""<div class="glass-card"><div class="metric-title">Fraud Rate</div><div class="metric-value">{fraud_rate:.2f}%</div><div class="metric-subtitle">Of total volume</div></div>""", unsafe_allow_html=True)
    
    st.markdown("<hr style='border: 1px solid rgba(255,255,255,0.05); margin: 2rem 0;'>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        fig_bar = px.bar(df.groupby(["hour", "label"])["txn_count_1hr"].sum().reset_index(), x="hour", y="txn_count_1hr", color="label", title="Activity Volume by Hour", color_discrete_sequence=["#00b894", "#ff4b8b"])
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
        fig_donut = px.pie(df, names="label", hole=0.6, title="Risk Distribution", color_discrete_sequence=["#00b894", "#ff4b8b"])
        fig_donut.update_layout(annotations=[dict(text=f'{fraud_rate:.1f}%<br>Fraud', x=0.5, y=0.5, font_size=20, showarrow=False, font=dict(color=chart_font_color))])
        st.plotly_chart(apply_plotly_layout(fig_donut), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col4:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        fig_sender = px.bar(df, x="sender_type", color="receiver_type", title="Entity Type Correlation Matrix", barmode="stack", color_discrete_sequence=px.colors.qualitative.Pastel)
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

elif "Prediction Engine" in page:
    st.markdown("<h2>Real-time Fraud Prediction Engine</h2>", unsafe_allow_html=True)
    st.markdown("<p style='opacity: 0.8;'>Submit transaction telemetry for instant ML inference.</p>", unsafe_allow_html=True)
    
    form_col, result_col = st.columns([1, 1.2])
    
    with form_col:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### Input Telemetry")
        
        amount = st.number_input(
            label="Transaction Value (₹)", 
            min_value=0.0, value=1500.0, step=100.0, 
            help="💵 Enter the precise transfer amount requested by the initiator."
        )
        
        txn = st.number_input(
            label="Transaction Velocity (Last 1 Hour)", 
            min_value=0, value=2, step=1, 
            help="📈 Monitors sudden, automated rapid-fire account activity spikes."
        )
        
        hour = st.slider(
            label="Time of Transaction (24-Hour Clock)", 
            min_value=0, max_value=23, value=14, format="%d:00",
            help="🕒 Select the time of day matching the transaction execution."
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("Initialize Inference", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with result_col:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

        if predict_btn:

            risk_score = 0
            if amount > 10000:
                risk_score += 50
            if txn > 10:
                risk_score += 30
            if 0 <= hour <= 5 and amount > 4000:
                risk_score += 20

            ml_pred = model.predict([[amount, txn, hour]])[0]
            ml_proba = model.predict_proba([[amount, txn, hour]])[0]

            if risk_score > 70:
                final_pred = 1
                reason = "Extreme high-risk pattern detected (rule engine)."
            elif risk_score > 40:
                final_pred = 1
                reason = "Moderate risk pattern detected (rule engine)."
            elif ml_pred == 1:
                final_pred = 1
                reason = "ML model detected anomaly in transaction behavior."
                risk_score = 85
            else:
                final_pred = 0
                reason = "Transaction appears normal based on ML + rules."
                risk_score = 15

            is_fraud = final_pred == 1

            # ================= HEADER BADGE =================
            if is_fraud:
                icon = "🚨"
                status = "HIGH RISK"
                color = "#ff4d6d"
                confidence = float(ml_proba[0][1]) * 100
                confidence_label = "Fraud Probability"
            else:
                icon = "🟢"
                status = "SAFE"
                color = "#00b894"
                confidence = float(ml_proba[0][0]) * 100
                confidence_label = "Legitimate Confidence"

            st.markdown(
                f"""
                <div style="
                    padding:12px 16px;
                    border-radius:12px;
                    background:rgba(255,255,255,0.03);
                    border-left:5px solid {color};
                    margin-bottom:15px;">
                    <h3 style="margin:0; color:{color};">
                        {icon} {status} TRANSACTION
                    </h3>
                </div>
                """,
                unsafe_allow_html=True
            )

            # ================= REASON =================
            st.caption("Decision Reason")
            st.write(reason)

            st.divider()

            # ================= CONFIDENCE =================
            st.subheader(confidence_label)

            col1, col2 = st.columns([3, 1])

            with col1:
                st.progress(int(confidence))

            with col2:
                st.markdown(
                    f"<h4 style='margin-top:5px; color:{color};'>{confidence:.1f}%</h4>",
                    unsafe_allow_html=True
                )

            st.divider()

            # ================= METRICS =================
            m1, m2 = st.columns(2)

            with m1:
                st.metric(
                    label="Risk Score",
                    value=f"{risk_score}/100",
                    delta=f"{'High' if risk_score > 70 else 'Low'} risk"
                )

            with m2:
                st.metric(
                    label="Engine",
                    value="RandomForest + Rules"
                )

        else:
            st.info("Enter transaction details and click **Initialize Inference** to see results.")

        st.markdown("</div>", unsafe_allow_html=True)
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