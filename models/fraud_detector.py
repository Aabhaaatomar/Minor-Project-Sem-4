"""
Fraud Detection Service — UniPay FraudX
========================================
Provides a self-contained rule-based + ML hybrid fraud analysis engine.

Features used:
  amount, hour, txn_count_1hr,
  sender_type, receiver_type, location_type,
  is_odd_hour, is_high_amount, is_high_velocity

Model: Random Forest (200 estimators, balanced class weights)

Public API
----------
analyze_transaction(amount, txn_count, hour,
                    sender_type="", receiver_type="", location_type="",
                    model=None)
    → dict with keys: is_fraud, fraud_score, risk_label, triggered_rules,
                       ml_proba, recommendation

get_risk_label(score: float) → str
get_triggered_rules(amount, txn_count, hour, sender_type, receiver_type, location_type) → list[str]
"""

from __future__ import annotations

import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────
HIGH_AMOUNT_THRESHOLD   = 10_000   # ₹
EXTREME_AMOUNT_THRESHOLD = 50_000  # ₹
HIGH_VELOCITY_THRESHOLD  = 10      # transactions per hour
ODD_HOUR_START          = 0
ODD_HOUR_END            = 5        # inclusive — midnight to 5 AM
FRAUD_BLOCK_THRESHOLD   = 70       # risk_score ≥ this → BLOCK
FRAUD_FLAG_THRESHOLD    = 40       # risk_score ≥ this → FLAG_FOR_REVIEW

# Rule weights (must sum to 100 across the most extreme combo)
RULE_WEIGHTS: dict[str, int] = {
    "extreme_amount"        : 45,  # amount > 50 000
    "high_amount"           : 25,  # amount > 10 000
    "high_velocity"         : 25,  # txn_count_1hr > 10
    "odd_hour_high_amount"  : 20,  # 00-05 h AND amount > 4 000
    "unknown_sender"        : 15,  # sender_type == unknown
    "unknown_receiver"      : 15,  # receiver_type == unknown
    "international_high"    : 15,  # international location AND amount > 5 000
}

MODEL_FEATURE_COLS = [
    "amount", "txn_count_1hr", "hour",
    "is_odd_hour", "is_high_amount", "is_high_velocity",
]


# ── Public helpers ────────────────────────────────────────────────────────────

def get_risk_label(score: float) -> str:
    """Map a 0–100 risk score to a human-readable severity label."""
    if score >= 70:
        return "CRITICAL"
    elif score >= 45:
        return "HIGH"
    elif score >= 20:
        return "MEDIUM"
    else:
        return "LOW"


def get_triggered_rules(
    amount: float,
    txn_count: int,
    hour: int,
    sender_type: str = "",
    receiver_type: str = "",
    location_type: str = "",
) -> list[str]:
    """
    Return a list of human-readable rule strings that were violated.
    Each entry can be shown to the user as an explanation.
    """
    rules: list[str] = []

    if amount > EXTREME_AMOUNT_THRESHOLD:
        rules.append(f"Extreme transaction amount (₹{amount:,.0f} > ₹{EXTREME_AMOUNT_THRESHOLD:,})")
    elif amount > HIGH_AMOUNT_THRESHOLD:
        rules.append(f"High transaction amount (₹{amount:,.0f} > ₹{HIGH_AMOUNT_THRESHOLD:,})")

    if txn_count > HIGH_VELOCITY_THRESHOLD:
        rules.append(f"High transaction velocity ({txn_count} txns in last hour > {HIGH_VELOCITY_THRESHOLD})")

    if ODD_HOUR_START <= hour <= ODD_HOUR_END and amount > 4_000:
        rules.append(f"Suspicious time window (hour={hour}, 00:00–05:59) with elevated amount")

    if str(sender_type).lower() == "unknown":
        rules.append("Unknown sender entity type")

    if str(receiver_type).lower() == "unknown":
        rules.append("Unknown receiver entity type")

    if str(location_type).lower() == "international" and amount > 5_000:
        rules.append(f"International transaction with high amount (₹{amount:,.0f})")

    return rules


def _compute_rule_score(
    amount: float,
    txn_count: int,
    hour: int,
    sender_type: str,
    receiver_type: str,
    location_type: str,
) -> int:
    """Compute raw integer risk score (0–100+) from heuristic rules."""
    score = 0

    if amount > EXTREME_AMOUNT_THRESHOLD:
        score += RULE_WEIGHTS["extreme_amount"]
    elif amount > HIGH_AMOUNT_THRESHOLD:
        score += RULE_WEIGHTS["high_amount"]

    if txn_count > HIGH_VELOCITY_THRESHOLD:
        score += RULE_WEIGHTS["high_velocity"]

    if ODD_HOUR_START <= hour <= ODD_HOUR_END and amount > 4_000:
        score += RULE_WEIGHTS["odd_hour_high_amount"]

    if str(sender_type).lower() == "unknown":
        score += RULE_WEIGHTS["unknown_sender"]

    if str(receiver_type).lower() == "unknown":
        score += RULE_WEIGHTS["unknown_receiver"]

    if str(location_type).lower() == "international" and amount > 5_000:
        score += RULE_WEIGHTS["international_high"]

    return min(score, 100)  # cap at 100


def analyze_transaction(
    amount: float,
    txn_count: int,
    hour: int,
    sender_type: str = "",
    receiver_type: str = "",
    location_type: str = "",
    model=None,
) -> dict:
    """
    Hybrid rule-engine + ML fraud analysis.

    Parameters
    ----------
    amount        : Transaction amount in ₹
    txn_count     : Number of transactions by this user in the last hour
    hour          : Hour of day (0–23)
    sender_type   : Entity type of sender (individual/business/government/unknown)
    receiver_type : Entity type of receiver
    location_type : Transaction location type (domestic/international/unknown)
    model         : A loaded sklearn model. If None, falls back to rule-only mode.

    Returns
    -------
    dict with keys:
        is_fraud        : bool
        fraud_score     : int  (0–100)
        risk_label      : str  (LOW / MEDIUM / HIGH / CRITICAL)
        triggered_rules : list[str]
        ml_proba        : float  (0.0–1.0, ML fraud probability)
        recommendation  : str  (APPROVE / FLAG_FOR_REVIEW / BLOCK)
        reason          : str  (human-readable primary reason)
    """
    # ── 1. Rule-based score ───────────────────────────────────────────────────
    rule_score = _compute_rule_score(
        amount, txn_count, hour, sender_type, receiver_type, location_type
    )
    triggered_rules = get_triggered_rules(
        amount, txn_count, hour, sender_type, receiver_type, location_type
    )

    # ── 2. ML inference ───────────────────────────────────────────────────────
    ml_proba    = 0.0
    ml_is_fraud = False

    if model is not None:
        is_odd_hour      = 1 if hour < 6 or hour > 22 else 0
        is_high_amount   = 1 if amount > HIGH_AMOUNT_THRESHOLD else 0
        is_high_velocity = 1 if txn_count > HIGH_VELOCITY_THRESHOLD else 0

        features = pd.DataFrame([{
            "amount"          : amount,
            "txn_count_1hr"   : txn_count,
            "hour"            : hour,
            "is_odd_hour"     : is_odd_hour,
            "is_high_amount"  : is_high_amount,
            "is_high_velocity": is_high_velocity,
        }], columns=MODEL_FEATURE_COLS)

        try:
            proba       = model.predict_proba(features)[0]
            ml_proba    = float(proba[1])          # probability of fraud class
            ml_is_fraud = ml_proba >= 0.5
        except Exception as exc:
            print(f"[fraud_detector] ML inference failed: {exc}")

    # ── 3. Combine scores ─────────────────────────────────────────────────────
    # Blend: 60 % rules, 40 % ML probability (when model available)
    if model is not None:
        fraud_score = int(0.60 * rule_score + 0.40 * ml_proba * 100)
    else:
        fraud_score = rule_score

    fraud_score = max(0, min(100, fraud_score))

    # ── 4. Final decision ─────────────────────────────────────────────────────
    is_fraud = fraud_score >= FRAUD_FLAG_THRESHOLD or ml_is_fraud

    if fraud_score >= FRAUD_BLOCK_THRESHOLD:
        recommendation = "BLOCK"
        reason         = "High-confidence fraud — multiple risk rules triggered."
    elif fraud_score >= FRAUD_FLAG_THRESHOLD:
        recommendation = "FLAG_FOR_REVIEW"
        reason         = "Moderate risk — manual review recommended."
    else:
        recommendation = "APPROVE"
        reason         = "Behavioral patterns within normal bounds."
        if ml_is_fraud and rule_score < FRAUD_FLAG_THRESHOLD:
            # ML says fraud but rules don't — flag for review, don't block
            recommendation = "FLAG_FOR_REVIEW"
            reason         = "ML model detected anomalous pattern."
            is_fraud       = True

    risk_label = get_risk_label(fraud_score)

    return {
        "is_fraud"       : is_fraud,
        "fraud_score"    : fraud_score,
        "risk_label"     : risk_label,
        "triggered_rules": triggered_rules,
        "ml_proba"       : round(ml_proba, 4),
        "recommendation" : recommendation,
        "reason"         : reason,
    }
