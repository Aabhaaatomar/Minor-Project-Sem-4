from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from app import db, socketio
from app.models import Alert
from flask_socketio import join_room

alerts_bp = Blueprint("alerts", __name__)


# ─────────────────────────────────────────────
# GET /api/alerts/  — get user's alerts
# ─────────────────────────────────────────────
@alerts_bp.route("/", methods=["GET"])
@jwt_required()
def get_alerts():
    user_id = int(get_jwt_identity())
    unread_only = request.args.get("unread", "false").lower() == "true"
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)

    query = Alert.query.filter_by(user_id=user_id)
    if unread_only:
        query = query.filter_by(is_read=False)

    paginated = query.order_by(Alert.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )

    return jsonify({
        "alerts": [a.to_dict() for a in paginated.items],
        "total": paginated.total,
        "unread_count": Alert.query.filter_by(user_id=user_id, is_read=False).count(),
    }), 200



