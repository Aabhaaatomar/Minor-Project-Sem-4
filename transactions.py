import uuid
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from datetime import datetime, timedelta

from app import db
from app.models import Transaction, Blocklist
from app.services.fraud_detector import analyze_transaction
from app.services.alert_service import create_alert

