from flask import Flask, render_template
import pandas as pd

app = Flask(__name__)

# Load Dataset (Excel file)
df = pd.read_excel(
    "unipay_fraudx_dataset.xlsx",
    engine="openpyxl"
)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    transactions = df.head(15).to_dict(orient="records")
    return render_template("dashboard.html", transactions=transactions)

if __name__ == "__main__":
    app.run(debug=True)
