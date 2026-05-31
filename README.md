# 🚀 UniPay FraudX  
### AI-Powered Digital Payment Fraud Detection System

---

## 📌 Overview  
UniPay FraudX is a machine learning-based web application designed to detect fraudulent and suspicious digital transactions in university and community ecosystems.  

The system analyzes transaction patterns and classifies them as **Normal** or **Suspicious**, providing real-time insights through an interactive dashboard.

---

## ⚠️ Problem Statement  
With the rapid growth of digital payments in academic and local communities, there is an increasing risk of fraudulent transactions.  

Existing fraud detection systems are often complex, expensive, and not suitable for educational environments. There is a need for a simple, accessible, and intelligent system to identify suspicious activities and promote digital payment awareness.

---

## 🎯 Objectives  
- Analyze digital transaction patterns using Machine Learning  
- Identify and classify suspicious transactions  
- Provide clear and explainable results  
- Build an interactive and user-friendly dashboard  
- Promote awareness of digital payment risks  
- Enhance understanding of fraud detection concepts  

---

## 💡 Proposed Solution  
UniPay FraudX combines **rule-based detection** with **machine learning** to analyze transaction patterns in real-time.  

### Key Features:
- **Hybrid Detection Engine**: Combines heuristic rules with Random Forest ML model
- **Real-time Analysis**: Instant fraud probability scoring (0-100)
- **Risk Classification**: Automatic categorization (LOW/MEDIUM/HIGH/CRITICAL)
- **Interactive Dashboard**: Visual analytics with Plotly charts
- **Explainable AI**: Shows which rules triggered the fraud alert

The system processes transaction inputs, applies both rule-based checks and ML inference, then provides actionable recommendations (APPROVE/FLAG/BLOCK).

---

## 🛠️ Tech Stack  

**Backend & ML:**
- Python 3.11+
- Streamlit (Web Framework)
- Scikit-learn (Random Forest Classifier)
- Pandas & NumPy (Data Processing)

**Frontend & Visualization:**
- Streamlit UI Components
- Plotly (Interactive Charts)
- Custom CSS (Glassmorphism Design)

**Data:**
- Excel/XLSX (Dataset Storage)
- Pickle (Model Serialization)  

---

## 📊 Features  
✅ **Real-time Fraud Prediction** - Instant ML-powered transaction analysis  
✅ **Interactive Dashboard** - Live charts and KPI metrics  
✅ **Risk Scoring System** - 0-100 fraud probability with severity levels  
✅ **Rule Engine** - Detects high amounts, velocity attacks, odd-hour transactions  
✅ **Data Visualization** - Pie charts, bar graphs, line trends, scatter plots  
✅ **Dark/Light Theme** - Modern glassmorphism UI design  
✅ **Explainable Results** - Shows triggered rules and ML confidence  
✅ **Transaction Explorer** - Browse and analyze historical data  

---

## 🔄 System Workflow  
1. **User Input** → Transaction details (amount, velocity, time)
2. **Rule Engine** → Checks heuristic fraud patterns
3. **ML Model** → Random Forest probability inference
4. **Hybrid Scoring** → Combines rules (60%) + ML (40%)
5. **Risk Classification** → Assigns severity level
6. **Recommendation** → APPROVE / FLAG_FOR_REVIEW / BLOCK
7. **Dashboard Update** → Real-time visualization  

---

## 📂 Dataset  
- **Source**: AI-generated synthetic transaction data
- **Purpose**: Simulates real-world digital payment patterns
- **Privacy**: Safe for academic and educational use
- **Features**: Amount, transaction velocity, time, entity types, location
- **Labels**: Normal vs Suspicious transactions  

---

## 🌐 Deployment  
The application is deployed using Streamlit Cloud and can be accessed online.  

👉 **Live Demo:** *https://minor-project-sem-4-2wdxbhegndunimrnis6i5r.streamlit.app/* 

---

## 📌 Future Enhancements  
- 🔗 Real-time payment gateway API integration  
- 🤖 Advanced ML models (XGBoost, Neural Networks)  
- 📧 Email/SMS alert notifications  
- 🔐 User authentication & role-based access  
- 📱 Mobile-responsive PWA version  
- 🗄️ Database integration (PostgreSQL/MongoDB)  
- 📊 Advanced analytics & reporting dashboard  
- 🌐 Multi-language support  

---

## 👩‍💻 Author  
**Aabha Tomar**  
B.Tech CSE (Data Science)  

---

## 🤝 Contributors
- Aabha Tomar
- Simran
- Nishant Rajawat
- Sakshi Rajawat
- Chetna Sharma

---

## 📂 Project Structure
```text
UniPay-FraudX/
├── app.py                    # Main Streamlit application
├── retrain_model.py          # Model retraining script
├── data.xlsx                 # Transaction dataset
├── fraud_model.pkl           # Trained ML model
├── models/
│   ├── fraud_detector.py     # Hybrid fraud detection engine
│   ├── fraud_model.py        # Model inference utilities
│   └── model.py              # Advanced model training script
├── dataset/                  # Alternative dataset location
├── backend_archive/          # (Archived) Flask backend files
├── templates/                # (Archived) HTML templates
├── static/                   # (Archived) CSS files
├── requirements.txt          # Python dependencies
└── README.md                 # Documentation
```

## ⚙️ Installation & Setup

### Quick Start

1. **Clone the repository:**
```bash
git clone https://github.com/Aabhaaatomar/UniPay-FraudX
cd UniPay-FraudX
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Train the model** (first time only):
```bash
python retrain_model.py
```

4. **Run the application:**
```bash
streamlit run app.py
```

5. **Access the app:**
   - Open your browser and go to `http://localhost:8501`
   - The app will automatically open in your default browser

### Troubleshooting

**If you see "Model not found" error:**
```bash
python retrain_model.py
```

**If you see "Dataset not found" error:**
- Ensure `data.xlsx` exists in the root directory
- Or place your dataset in `dataset/data.xlsx`

---

## ⭐ Conclusion  
UniPay FraudX demonstrates how machine learning can be applied to detect fraud in digital payments while maintaining simplicity and usability for academic environments.
