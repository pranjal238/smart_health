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
    role = Column(String(50), default="CAREGIVER", nullable=False) # ADMIN, CAREGIVER
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

class SensorReading(Base):
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
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    confidence = Column(Float, nullable=False)
    risk_level = Column(String(50), default="HIGH", nullable=False)
    status = Column(String(50), default="UNACKNOWLEDGED", nullable=False) # UNACKNOWLEDGED, ACKNOWLEDGED, RESOLVED, FALSE_ALARM
    acknowledged = Column(Boolean, default=False, nullable=False)
    acknowledged_at = Column(DateTime, nullable=True)
    acknowledged_by = Column(String(100), nullable=True)
    notes = Column(Text, nullable=True)
    device_id = Column(String(100), default="WEARABLE_DEV_01")
    acc_peak = Column(Float, nullable=True)
    gyro_peak = Column(Float, nullable=True)
    
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
