import streamlit as st
import pandas as pd
import pickle
import plotly.express as px # pyright: ignore[reportMissingImports]

model = pickle.load(open("fraud_model.pkl", "rb"))
# ✅ 1. PAGE CONFIG (SABSE PEHLE)
st.set_page_config(page_title="UniPay FraudX", layout="wide")

# ✅ 2. CUSTOM CSS (USKE BAAD)
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f7e1ea, #fbc2d4);
}
</style>
""", unsafe_allow_html=True)

# ✅ 3. SIDEBAR NAVIGATION (USKE BAAD)
page = st.sidebar.radio("Navigation", ["Dashboard", "Prediction"])

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="UniPay FraudX", layout="wide")

if page == "Dashboard":

    st.title("📊 A system that detects fraudulent transactions in real-time")

    # 👉 CARDS
    col1, col2, col3 = st.columns(3)

    # total, fraud, normal yaha

    # 👉 CHARTS
    # pie chart
    # bar chart
    # scatter plot
    
elif page == "Prediction":
    st.title("🔍 Predict Transaction")

    st.markdown("### 💳 Enter Transaction Details")

    col1, col2, col3 = st.columns(3)

    amount = col1.slider("Amount", 0, 10000, 1000)
    txn = col2.slider("Txn Count (1hr)", 0, 20, 2)
    hour = col3.slider("Hour", 0, 23, 12)

    if st.button("Predict"):
        pred = model.predict([[amount, txn, hour]])[0]

        if pred == 1:
            st.error("🚨 Fraud Transaction")
        else:
            st.success("✅ Normal Transaction")
    
# ------------------ CUSTOM CSS (PINK VIBE) ------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f7e1ea, #fbc2d4);
}

.block-container {
    background-color: transparent;
}

h1, h2, h3 {
    color: #5c2a4a;
    text-align: center;
}

.stButton>button {
    background-color: #ff85a2;
    color: white;
    border-radius: 12px;
    padding: 10px 20px;
}

.stButton>button:hover {
    background-color: #ff5c8a;
}
</style>
""", unsafe_allow_html=True)



# ------------------ LOAD DATA ------------------
df = pd.read_excel("data.xlsx")

# ------------------ LOAD MODEL ------------------
model = pickle.load(open("fraud_model.pkl", "rb"))

# ------------------ TITLE ------------------
st.markdown("<h1 style='text-align:center;'>💳 UniPay FraudX Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Smart Fraud Detection System</p>", unsafe_allow_html=True)

# ------------------ CARDS ------------------
col1, col2, col3 = st.columns(3)


total = len(df)
fraud = int(df["label"].value_counts().get("Suspicious", 0))
normal = total - fraud

df["label_name"] = df["label"].map({
    0: "Normal",
    1: "Suspicious"
})


with col1:
    st.markdown(f"<div class='card'><h3>Total Transactions</h3><h2>{total}</h2></div>", unsafe_allow_html=True)

with col2:
    st.markdown(f"<div class='card'><h3>Fraud</h3><h2>{fraud}</h2></div>", unsafe_allow_html=True)

with col3:
    st.markdown(f"<div class='card'><h3>Normal</h3><h2>{normal}</h2></div>", unsafe_allow_html=True)

st.markdown("---")

# ------------------ CHARTS ------------------
col1, col2 = st.columns(2)

with col1:
    fig1 = px.pie(df, names="label", title="Fraud Distribution",
                  color_discrete_sequence=["#ff4d6d", "#4cc9f0"])
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.histogram(df, x="hour", title="Transactions by Hour",
                        color_discrete_sequence=["#a2d2ff"])
    st.plotly_chart(fig2, use_container_width=True)

# ------------------ SCATTER PLOT ------------------
fig3 = px.scatter(df, x="amount", y="txn_count_1hr",
                  color=df["label"].astype(str),
                  title="Transaction Scatter Plot",
                  color_discrete_sequence=["green", "red"])

st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# ------------------ PREDICTION SECTION ------------------
st.subheader("🔍 Predict Transaction")

col1, col2, col3 = st.columns(3)

amount = int(col1.number_input("Amount", min_value=0, step=1))
txn = int(col2.number_input("Txn Count (1hr)", min_value=0, step=1))
hour = int(col3.number_input("Hour (0-23)", min_value=0, max_value=23, step=1))

if st.button("Predict"):

    if amount > 8000 and txn > 10:
        pred = 1
    else:
        pred = model.predict([[amount, txn, hour]])[0]

    if pred == 1:
        st.error("🚨 Fraud Transaction")
    else:
        st.success("✅ Normal Transaction")