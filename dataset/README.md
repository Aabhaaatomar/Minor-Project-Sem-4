# Dataset — UniPay FraudX

The actual dataset file (`data.xlsx`) is excluded from version control via `.gitignore` to keep the repository lightweight.

## How to Generate the Dataset

Run the model training script from the **project root**:

```bash
python models/model.py
```

This will:
1. Automatically generate a synthetic `data.xlsx` (1,000 transactions) in this folder
2. Train a Random Forest classifier and save `models/fraud_model.pkl`

## Dataset Schema

| Column | Type | Description |
|---|---|---|
| `transaction_id` | string | Unique transaction identifier (TXN00001 …) |
| `amount` | float | Transaction amount in ₹ |
| `hour` | int | Hour of day (0–23) |
| `txn_count_1hr` | int | Number of transactions by this user in the last hour |
| `sender_type` | string | individual / business / government / unknown |
| `receiver_type` | string | individual / business / merchant / unknown |
| `location_type` | string | domestic / international / unknown |
| `is_odd_hour` | int | 1 if hour is 0–5 or 22–23, else 0 |
| `is_high_amount` | int | 1 if amount > ₹10,000, else 0 |
| `is_high_velocity` | int | 1 if txn_count_1hr > 10, else 0 |
| `label` | string | **Normal** or **Suspicious** |

## Notes

- Dataset is **synthetically generated** — no real financial data
- ~47% fraud rate with 10% label noise to simulate real-world imperfection
- Safe for academic and demonstration purposes
