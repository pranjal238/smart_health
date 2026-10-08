"""
FallGuard AI - Pydantic Request & Response Schemas
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime

# --- Auth Schemas ---
# --- Auth & Profile Schemas ---
class UserLogin(BaseModel):
    email: str
    password: str

class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    role: str
    phone_number: Optional[str] = None
    emergency_contact_name: Optional[str] = None
    emergency_contact_phone: Optional[str] = None
    emergency_contact_email: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}

class EmergencyContactUpdate(BaseModel):
    emergency_contact_name: str
    emergency_contact_phone: str
    emergency_contact_email: Optional[str] = None
    phone_number: Optional[str] = None

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse

# --- Smartphone & Sensor Schemas ---
class SensorAxisData(BaseModel):
    x: float
    y: float
    z: float

class SmartphoneSensorMessage(BaseModel):
    type: str = "sensor_data"
    version: int = 1
    device_id: str = "phone_001"
    timestamp: float = Field(..., description="Timestamp in epoch seconds or milliseconds")
    unit: str = Field(default="m/s2", description="Accelerometer units: 'm/s2' (Android default) or 'g'")
    accelerometer: SensorAxisData
    gyroscope: SensorAxisData
    location: Optional[Dict[str, float]] = None

class FallConfirmationPayload(BaseModel):
    type: str = "fall_confirmation"
    event_id: int
    action: str = Field(..., description="'I_AM_OK' or 'NEED_HELP'")
    location: Optional[Dict[str, float]] = None

class DeviceStatus(BaseModel):
    device_id: str
    user_id: Optional[int] = None
    connected: bool = False
    last_seen: Optional[datetime] = None
    samples_received: int = 0
    buffer_samples: int = 0
    buffer_seconds: float = 0.0

class SensorReadingCreate(BaseModel):
    device_id: str = "WEARABLE_DEV_01"
    timestamp: Optional[datetime] = None
    acc_x: float
    acc_y: float
    acc_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    session_id: Optional[str] = None

class SensorStreamBatch(BaseModel):
    device_id: str = "WEARABLE_DEV_01"
    readings: List[SensorReadingCreate]

# --- Prediction Schemas ---
class SingleWindowPredictionRequest(BaseModel):
    acc_x: List[float]
    acc_y: List[float]
    acc_z: List[float]
    gyro_x: List[float]
    gyro_y: List[float]
    gyro_z: List[float]
    device_id: Optional[str] = "WEARABLE_DEV_01"
    sampling_rate: Optional[float] = 50.0

class PredictionResponse(BaseModel):
    activity: str
    ml_activity: str = "WALKING"
    ml_fall_probability: float = 0.0
    safety_override: bool = False
    confidence: float
    risk_level: str # HIGH, MEDIUM, LOW
    is_fall: bool
    probabilities: Dict[str, float] = {}
    model_name: str
    model_version: str
    timestamp: str
    features_summary: Optional[Dict[str, Any]] = None

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_processed: int

# --- Fall & Alert Schemas ---
class FallEventResponse(BaseModel):
    id: int
    user_id: Optional[int] = None
    device_id: str
    timestamp: datetime
    activity: str = "FALL"
    ml_activity: Optional[str] = None
    confidence: float
    fall_probability: Optional[float] = None
    safety_override: bool = False
    risk_level: str
    status: str
    confirmed: bool = False
    alert_sent: bool = False
    emergency_contact: Optional[str] = None
    location_lat: Optional[float] = None
    location_lon: Optional[float] = None
    acknowledged: bool
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    notes: Optional[str] = None
    acc_peak: Optional[float] = None
    gyro_peak: Optional[float] = None
    model_version: Optional[str] = "1.0.0"

    model_config = {"from_attributes": True}

class FallAcknowledgeRequest(BaseModel):
    acknowledged_by: str = "Operator"
    notes: Optional[str] = None

class FallNoteRequest(BaseModel):
    notes: str

class AlertResponse(BaseModel):
    id: int
    fall_event_id: Optional[int]
    alert_type: str
    message: str
    created_at: datetime
    acknowledged_at: Optional[datetime]

    model_config = {"from_attributes": True}

# --- Dashboard & Analytics Schemas ---
class ActivityHistoryItem(BaseModel):
    id: int
    timestamp: datetime
    activity: str
    confidence: float
    risk_level: str
    is_fall: bool
    device_id: str

    model_config = {"from_attributes": True}

class DashboardSummary(BaseModel):
    current_activity: str
    current_confidence: float
    current_risk: str
    system_status: str
    sensor_connected: bool
    total_activities_count: int
    total_falls_count: int
    falls_today_count: int
    unacknowledged_falls_count: int
    last_fall_timestamp: Optional[datetime]
    active_alerts: List[AlertResponse]
    recent_activities: List[ActivityHistoryItem]

# --- Simulation Schemas ---
class SimulationControlRequest(BaseModel):
    action: str = Field(..., description="'start', 'stop', 'pause', 'resume', 'trigger_fall'")
    playback_speed: Optional[float] = 1.0 # 1.0 = real-time (50Hz), 2.0 = 2x speed
    scenario: Optional[str] = "mixed_activities_with_fall"

class SimulationStatusResponse(BaseModel):
    is_running: bool
    is_paused: bool
    current_scenario: str
    playback_speed: float
    samples_replayed: int
    total_samples: int
    current_activity: str
    current_risk: str

# --- Model Info Schemas ---
class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    trained_at: Optional[str]
    features_count: int
    classes: List[str]
    window_size_seconds: float
    sampling_rate: float
    metrics: Dict[str, Any]
    feature_importance_top10: List[Dict[str, Any]]
