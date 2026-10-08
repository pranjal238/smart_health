"""
FallGuard AI - Core Application Settings & Configuration
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")

class Settings:
    APP_NAME: str = os.getenv("APP_NAME", "FallGuard AI")
    APP_VERSION: str = os.getenv("APP_VERSION", "1.0.0")
    DEBUG: bool = os.getenv("DEBUG", "True").lower() in ("true", "1", "t")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    
    API_HOST: str = os.getenv("API_HOST", "127.0.0.1")
    API_PORT: int = int(os.getenv("API_PORT", 8000))
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "fallguard-super-secret-key-change-in-production-2026")
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 1440))
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'fall_guard.db'}")
    
    # ML & Inference
    MODEL_PATH: Path = BASE_DIR / os.getenv("MODEL_PATH", "data/models/fall_detection_model.joblib")
    METRICS_PATH: Path = BASE_DIR / os.getenv("METRICS_PATH", "reports/model_metrics.json")
    SAMPLING_RATE: float = float(os.getenv("SAMPLING_RATE", 50.0))
    WINDOW_SIZE_SECONDS: float = float(os.getenv("WINDOW_SIZE_SECONDS", 2.56))
    WINDOW_SAMPLES: int = int(SAMPLING_RATE * WINDOW_SIZE_SECONDS) # 128
    
    # Risk & Fall Thresholds
    FALL_CONFIDENCE_HIGH: float = float(os.getenv("FALL_CONFIDENCE_HIGH", 0.85))
    FALL_CONFIDENCE_MEDIUM: float = float(os.getenv("FALL_CONFIDENCE_MEDIUM", 0.60))
    ALERT_COOLDOWN_SECONDS: int = int(os.getenv("ALERT_COOLDOWN_SECONDS", 5))
    
    # Smartphone Sensor Stream & Bounded Buffer (Strict 5-Second Retention)
    BUFFER_DURATION_SECONDS: float = float(os.getenv("BUFFER_DURATION_SECONDS", 5.0))
    PHONE_SAMPLING_RATE: float = float(os.getenv("PHONE_SAMPLING_RATE", 50.0))
    MAX_BUFFER_SAMPLES: int = int(BUFFER_DURATION_SECONDS * PHONE_SAMPLING_RATE) # 250 samples
    INFERENCE_STRIDE_SAMPLES: int = int(os.getenv("INFERENCE_STRIDE_SAMPLES", 25)) # 0.5s at 50Hz

    # Fall Confirmation & False Alarm Mitigation
    FALL_CONFIRMATION_TIMEOUT_SECONDS: float = float(os.getenv("FALL_CONFIRMATION_TIMEOUT_SECONDS", 15.0))

    # Emergency Alert Provider Settings
    EMERGENCY_PROVIDER: str = os.getenv("EMERGENCY_PROVIDER", "mock").lower() # 'mock', 'twilio'
    TWILIO_ACCOUNT_SID: str = os.getenv("TWILIO_ACCOUNT_SID", "")
    TWILIO_AUTH_TOKEN: str = os.getenv("TWILIO_AUTH_TOKEN", "")
    TWILIO_FROM_NUMBER: str = os.getenv("TWILIO_FROM_NUMBER", "")

    # Directories
    STATIC_DIR: Path = BASE_DIR / "frontend"
    REPORTS_DIR: Path = BASE_DIR / "reports"
    FIGURES_DIR: Path = BASE_DIR / "reports" / "figures"
    DATA_DIR: Path = BASE_DIR / "data"

settings = Settings()
