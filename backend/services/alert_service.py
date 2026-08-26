"""
FallGuard AI - Alert Management & Notification Service
Handles fall event registration, database persistence, caregiver notifications, and acknowledgments.
"""

from datetime import datetime, timedelta, timezone
import logging
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session

from backend.models.db_models import FallEvent, Alert, SystemLog
from backend.services.websocket_service import ws_manager
from backend.core.config import settings

logger = logging.getLogger("AlertService")

class AlertService:
    def __init__(self):
        self.last_alert_time: Optional[datetime] = None

    async def trigger_fall_alert(
        self,
        db: Session,
        confidence: float,
        risk_level: str = "HIGH",
        device_id: str = "WEARABLE_DEV_01",
        acc_peak: Optional[float] = None,
        gyro_peak: Optional[float] = None,
        notes: Optional[str] = None
    ) -> Optional[FallEvent]:
        """
        Record a detected fall event, generate clinical alert, and broadcast over WebSocket.
        Enforces cooldown to prevent alert storms.
        """
        now = datetime.now(timezone.utc)
        if self.last_alert_time:
            cooldown = timedelta(seconds=settings.ALERT_COOLDOWN_SECONDS)
            if now - self.last_alert_time < cooldown:
                logger.info(f"Fall alert ignored due to cooldown ({settings.ALERT_COOLDOWN_SECONDS}s).")
                return None
                
        self.last_alert_time = now
        
        # 1. Create FallEvent
        fall_event = FallEvent(
            timestamp=now,
            confidence=round(confidence, 4),
            risk_level=risk_level,
            status="UNACKNOWLEDGED",
            acknowledged=False,
            device_id=device_id,
            acc_peak=acc_peak,
            gyro_peak=gyro_peak,
            notes=notes or f"High-confidence fall signature ({confidence*100:.1f}%) detected by AI inference engine."
        )
        db.add(fall_event)
        db.commit()
        db.refresh(fall_event)
        
        # 2. Create Alert
        alert_msg = f"[FALL DETECTED] on {device_id}! Confidence: {confidence*100:.1f}%, Risk: {risk_level}"
        alert = Alert(
            fall_event_id=fall_event.id,
            alert_type="CRITICAL_FALL",
            message=alert_msg,
            created_at=now
        )
        db.add(alert)
        
        # 3. Log event
        sys_log = SystemLog(
            timestamp=now,
            level="ALERT",
            event="FALL_DETECTED",
            message=f"Fall event #{fall_event.id} registered for {device_id}. Confidence: {confidence*100:.1f}%"
        )
        db.add(sys_log)
        db.commit()
        db.refresh(alert)
        
        logger.warning(f"[FALL EVENT #{fall_event.id}] {alert_msg}")
        
        # 4. Broadcast via WebSocket
        ws_payload = {
            "type": "FALL_ALERT",
            "event": {
                "id": fall_event.id,
                "timestamp": now.isoformat(),
                "confidence": round(confidence, 4),
                "risk_level": risk_level,
                "status": "UNACKNOWLEDGED",
                "device_id": device_id,
                "acc_peak": acc_peak,
                "gyro_peak": gyro_peak,
                "message": alert_msg
            }
        }
        await ws_manager.broadcast(ws_payload)
        return fall_event

    def acknowledge_fall(
        self,
        db: Session,
        fall_id: int,
        acknowledged_by: str = "Operator",
        notes: Optional[str] = None
    ) -> Optional[FallEvent]:
        """Acknowledge a fall event and update status."""
        fall = db.query(FallEvent).filter(FallEvent.id == fall_id).first()
        if not fall:
            return None
            
        now = datetime.now(timezone.utc)
        fall.acknowledged = True
        fall.acknowledged_at = now
        fall.acknowledged_by = acknowledged_by
        fall.status = "ACKNOWLEDGED"
        if notes:
            fall.notes = (fall.notes + f"\n[Caregiver Note]: {notes}") if fall.notes else notes
            
        # Also update corresponding alert
        alert = db.query(Alert).filter(Alert.fall_event_id == fall_id).first()
        if alert:
            alert.acknowledged_at = now
            
        sys_log = SystemLog(
            timestamp=now,
            level="INFO",
            event="FALL_ACKNOWLEDGED",
            message=f"Fall event #{fall_id} acknowledged by {acknowledged_by}."
        )
        db.add(sys_log)
        db.commit()
        db.refresh(fall)
        return fall

    def add_fall_notes(self, db: Session, fall_id: int, notes: str) -> Optional[FallEvent]:
        """Append clinical investigation notes to a fall event."""
        fall = db.query(FallEvent).filter(FallEvent.id == fall_id).first()
        if not fall:
            return None
            
        now = datetime.now(timezone.utc)
        fall.notes = (fall.notes + f"\n[{now.strftime('%Y-%m-%d %H:%M:%S')}]: {notes}") if fall.notes else notes
        db.commit()
        db.refresh(fall)
        return fall

alert_service = AlertService()
