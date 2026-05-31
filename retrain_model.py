"""
Quick model retraining script for UniPay FraudX
Uses existing data.xlsx to retrain the model with current environment
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

print("🔄 Retraining fraud detection model...")

# Load dataset
print("📊 Loading data.xlsx...")
df = pd.read_excel("data.xlsx")
print(f"   Loaded {len(df)} transactions")

# Check label distribution
print(f"\n📈 Label distribution:")
print(df["label"].value_counts())

# Prepare features
X = df[["amount", "txn_count_1hr", "hour"]]

# Convert labels to binary (0 = Normal, 1 = Suspicious/Fraud)
y = df["label"].apply(lambda x: 1 if str(x).lower() in ["suspicious", "fraud", "1"] else 0)

print(f"\n🎯 Training Random Forest model...")
# Train model with same parameters
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
model.fit(X, y)

# Save model
print("💾 Saving model to fraud_model.pkl...")
with open("fraud_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved successfully!")
print("\n🚀 Now refresh your browser to reload the Streamlit app.")
