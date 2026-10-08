"""
FallGuard AI - Central Dashboard Summary API
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from datetime import datetime, date
from typing import Dict, Any

from backend.database.session import get_db
from backend.models.db_models import ActivityPrediction, FallEvent, Alert
from backend.schemas.schemas import DashboardSummary, AlertResponse, ActivityHistoryItem
from backend.services.simulation_service import simulation_service
from backend.services.prediction_service import prediction_service

from backend.services.sensor_stream_manager import sensor_stream_manager

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])

@router.get("/summary", response_model=DashboardSummary)
def get_dashboard_summary(db: Session = Depends(get_db)):
    """Retrieve comprehensive summary statistics for the monitoring dashboard."""
    # Current active state
    latest_pred = db.query(ActivityPrediction).order_by(ActivityPrediction.timestamp.desc()).first()
    
    current_activity = latest_pred.activity if latest_pred else (simulation_service.current_activity or "WALKING")
    current_confidence = latest_pred.confidence if latest_pred else 0.94
    current_risk = latest_pred.risk_level if latest_pred else (simulation_service.current_risk or "LOW")
    
    total_acts = db.query(ActivityPrediction).count()
    total_falls = db.query(FallEvent).count()
    
    today_start = datetime.combine(date.today(), datetime.min.time())
    falls_today = db.query(FallEvent).filter(FallEvent.timestamp >= today_start).count()
    unack_falls = db.query(FallEvent).filter(FallEvent.acknowledged == False).count()
    
    last_fall = db.query(FallEvent).order_by(FallEvent.timestamp.desc()).first()
    
    # Active unacknowledged alerts
    alerts = db.query(Alert).filter(Alert.acknowledged_at == None).order_by(Alert.created_at.desc()).limit(5).all()
    recent_acts = db.query(ActivityPrediction).order_by(ActivityPrediction.timestamp.desc()).limit(15).all()
    
    phone_connected = any(buf.connected for buf in sensor_stream_manager.devices.values())
    is_connected = phone_connected or simulation_service.is_running

    return DashboardSummary(
        current_activity=current_activity,
        current_confidence=round(current_confidence, 4),
        current_risk=current_risk,
        system_status="ONLINE" if is_connected else "READY",
        sensor_connected=is_connected,
        total_activities_count=total_acts,
        total_falls_count=total_falls,
        falls_today_count=falls_today,
        unacknowledged_falls_count=unack_falls,
        last_fall_timestamp=last_fall.timestamp if last_fall else None,
        active_alerts=[AlertResponse.model_validate(a) for a in alerts],
        recent_activities=[ActivityHistoryItem.model_validate(a) for a in recent_acts]
    )
