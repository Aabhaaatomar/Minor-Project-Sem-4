import json
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from datetime import datetime

from app import db
from app.models import Transaction, FraudReport, User
from app.services.fraud_detector import analyze_transaction
from app.models import Blocklist


