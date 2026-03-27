import json
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from datetime import datetime

from app import db
from app.models import Transaction, FraudReport, User
from app.services.fraud_detector import analyze_transaction
from app.models import Blocklist

fraud_bp = Blueprint("fraud", __name__)


def _require_admin():
    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)
    if not user or user.role != "admin":
        return None, (jsonify({"error": "Admin access required"}), 403)
    return user, None


# ─────────────────────────────────────────────
# POST /api/fraud/report
# Report a transaction as fraudulent (manual)
# ─────────────────────────────────────────────
@fraud_bp.route("/report", methods=["POST"])
@jwt_required()
def report_fraud():
    user_id = int(get_jwt_identity())
    data = request.get_json()

    transaction_id = data.get("transaction_id")
    reason = data.get("reason", "")

    tx = Transaction.query.filter_by(transaction_id=transaction_id).first()
    if not tx:
        return jsonify({"error": "Transaction not found"}), 404
    if tx.user_id != user_id:
        return jsonify({"error": "Unauthorized"}), 403

    report = FraudReport(
        transaction_id=tx.id,
        reported_by=user_id,
        reason=reason,
        ml_score=tx.fraud_score,
        status="open",
    )
    tx.is_fraud = True
    tx.status = "flagged"

    db.session.add(report)
    db.session.commit()

    return jsonify({
        "message": "Fraud report submitted",
        "report": report.to_dict(),
    }), 201
