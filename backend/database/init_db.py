"""
FallGuard AI - Database Initialization & Seed Script
Creates all tables and populates default users, initial alerts, and activity telemetry.
"""

from datetime import datetime, timedelta
import logging
from backend.database.session import engine, SessionLocal
from backend.models.db_models import Base, User, ActivityPrediction, FallEvent, Alert, SystemLog, SensorReading
from backend.core.security import hash_password

logger = logging.getLogger("InitDB")

def init_db():
    """Create all database tables and seed default users."""
    logger.info("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    try:
        # Check if users already exist
        admin_user = db.query(User).filter(User.email == "admin@fallguard.ai").first()
        if not admin_user:
            logger.info("Seeding default ADMIN and CAREGIVER accounts...")
            admin = User(
                name="Clinical Administrator",
                email="admin@fallguard.ai",
                hashed_password=hash_password("admin123"),
                role="ADMIN",
                created_at=datetime.utcnow()
            )
            caregiver = User(
                name="Primary Caregiver",
                email="caregiver@fallguard.ai",
                hashed_password=hash_password("caregiver123"),
                role="CAREGIVER",
                created_at=datetime.utcnow()
            )
            db.add_all([admin, caregiver])
            db.commit()
            
        # Seed initial system log if empty
        if db.query(SystemLog).count() == 0:
            log1 = SystemLog(
                timestamp=datetime.utcnow() - timedelta(hours=2),
                level="INFO",
                event="SYSTEM_STARTUP",
                message="FallGuard AI Central Monitoring Server initialized successfully."
            )
            log2 = SystemLog(
                timestamp=datetime.utcnow() - timedelta(hours=1),
                level="INFO",
                event="MODEL_LOADED",
                message="Wearable Sensor Inference Engine loaded with 50Hz sliding window protocol."
            )
            db.add_all([log1, log2])
            db.commit()
            
        # Seed initial fall event for demo if empty
        if db.query(FallEvent).count() == 0:
            past_fall_time = datetime.utcnow() - timedelta(minutes=45)
            fall_event = FallEvent(
                timestamp=past_fall_time,
                confidence=0.96,
                risk_level="HIGH",
                status="UNACKNOWLEDGED",
                acknowledged=False,
                device_id="PATIENT_ROOM_104",
                acc_peak=3.85,
                gyro_peak=4.12,
                notes="Automated alert triggered: Sudden high-impact deceleration detected from wearable sensor."
            )
            db.add(fall_event)
            db.commit()
            db.refresh(fall_event)
            
            alert = Alert(
                fall_event_id=fall_event.id,
                alert_type="CRITICAL_FALL",
                message="CRITICAL: Severe Fall Detected for Patient in Room 104 (Confidence 96%). Immediate assistance required.",
                created_at=past_fall_time
            )
            db.add(alert)
            db.commit()
            
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_db()
