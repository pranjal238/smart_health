"""
FallGuard AI - SQLAlchemy Database Models
Defines schema for Users, SensorReadings, ActivityPredictions, FallEvents, Alerts, and SystemLogs.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    email = Column(String(150), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(50), default="CAREGIVER", nullable=False) # ADMIN, CAREGIVER, PATIENT
    phone_number = Column(String(50), nullable=True)
    emergency_contact_name = Column(String(100), nullable=True)
    emergency_contact_phone = Column(String(50), nullable=True)
    emergency_contact_email = Column(String(150), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    falls = relationship("FallEvent", back_populates="user")

class SensorReading(Base):
    """
    DEPRECATED TABLE:
    Raw sensor streams are strictly transient in memory (rolling 5-second ring buffer)
    and NEVER continuously accumulated in production databases to safeguard user privacy.
    Retained solely for database migration compatibility.
    """
    __tablename__ = "sensor_readings"
    
    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String(100), default="WEARABLE_DEV_01", index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    acc_x = Column(Float, nullable=False)
    acc_y = Column(Float, nullable=False)
    acc_z = Column(Float, nullable=False)
    gyro_x = Column(Float, nullable=False)
    gyro_y = Column(Float, nullable=False)
    gyro_z = Column(Float, nullable=False)
    acc_magnitude = Column(Float, nullable=True)
    session_id = Column(String(100), nullable=True)

class ActivityPrediction(Base):
    __tablename__ = "activity_predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    activity = Column(String(100), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    risk_level = Column(String(50), default="LOW", nullable=False) # HIGH, MEDIUM, LOW
    is_fall = Column(Boolean, default=False, index=True)
    device_id = Column(String(100), default="WEARABLE_DEV_01")
    model_version = Column(String(50), default="1.0.0")

class FallEvent(Base):
    __tablename__ = "fall_events"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    device_id = Column(String(100), default="WEARABLE_DEV_01", index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    activity = Column(String(100), default="FALL", nullable=False)
    ml_activity = Column(String(100), nullable=True)
    confidence = Column(Float, nullable=False)
    fall_probability = Column(Float, nullable=True)
    safety_override = Column(Boolean, default=False, nullable=False)
    risk_level = Column(String(50), default="HIGH", nullable=False)
    
    # Status lifecycle: PENDING_CONFIRMATION -> CONFIRMED / CANCELLED_BY_USER -> ACKNOWLEDGED / RESOLVED
    status = Column(String(50), default="UNACKNOWLEDGED", nullable=False)
    confirmed = Column(Boolean, default=False, nullable=False)
    alert_sent = Column(Boolean, default=False, nullable=False)
    emergency_contact = Column(String(200), nullable=True)
    
    # Optional emergency GPS coordinates
    location_lat = Column(Float, nullable=True)
    location_lon = Column(Float, nullable=True)
    
    # Caregiver review
    acknowledged = Column(Boolean, default=False, nullable=False)
    acknowledged_at = Column(DateTime, nullable=True)
    acknowledged_by = Column(String(100), nullable=True)
    notes = Column(Text, nullable=True)
    
    # Sensor impact telemetry peaks
    acc_peak = Column(Float, nullable=True)
    gyro_peak = Column(Float, nullable=True)
    model_version = Column(String(50), default="1.0.0")
    
    user = relationship("User", back_populates="falls")
    alerts = relationship("Alert", back_populates="fall_event", cascade="all, delete-orphan")

class Alert(Base):
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    fall_event_id = Column(Integer, ForeignKey("fall_events.id", ondelete="CASCADE"), nullable=True)
    alert_type = Column(String(100), default="CRITICAL_FALL", nullable=False)
    message = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    acknowledged_at = Column(DateTime, nullable=True)
    
    fall_event = relationship("FallEvent", back_populates="alerts")

class SystemLog(Base):
    __tablename__ = "system_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    level = Column(String(50), default="INFO")
    event = Column(String(100), nullable=False)
    message = Column(Text, nullable=False)
