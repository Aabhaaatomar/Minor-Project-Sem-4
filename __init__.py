from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager
from flask_migrate import Migrate
from flask_mail import Mail
from flask_socketio import SocketIO
from flask_cors import CORS
from dotenv import load_dotenv
import os

load_dotenv()

db = SQLAlchemy()
jwt = JWTManager()
migrate = Migrate()
mail = Mail()
socketio = SocketIO(cors_allowed_origins="*")


def create_app():
    app = Flask(__name__)
