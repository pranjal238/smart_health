"""
FallGuard AI - Alert Management & Emergency Dispatch Service
Handles fall event registration, database persistence, emergency provider escalation,
caregiver notifications, and acknowledgments.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
import logging
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session

from backend.models.db_models import FallEvent, Alert, SystemLog, User
from backend.services.websocket_service import ws_manager
from backend.core.config import settings

logger = logging.getLogger("AlertService")

# --- Emergency Provider Abstraction ---
class BaseEmergencyProvider(ABC):
    """Abstract interface for external emergency escalation channels (SMS, Phone Call, Push)."""
    @abstractmethod
    async def send_emergency_alert(
        self,
        user_name: str,
        contact_name: str,
        contact_phone: str,
        event: Dict[str, Any]
    ) -> bool:
        pass

class MockEmergencyProvider(BaseEmergencyProvider):
    """Development/Testing emergency provider that logs structured alerts and records dispatched incidents."""
    def __init__(self):
        self.dispatched_alerts: List[Dict[str, Any]] = []

    async def send_emergency_alert(
        self,
        user_name: str,
        contact_name: str,
        contact_phone: str,
        event: Dict[str, Any]
    ) -> bool:
        incident = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_name": user_name,
            "contact_name": contact_name,
            "contact_phone": contact_phone,
            "event": event
        }
        self.dispatched_alerts.append(incident)
        logger.info(
            f"[MOCK EMERGENCY ALERT DISPATCHED] -> To: {contact_name} ({contact_phone}) | "
            f"User: {user_name} | Risk: {event.get('risk_level')} | Confidence: {event.get('confidence')*100:.1f}% | "
            f"Location: {event.get('location')} | Message: {event.get('message')}"
        )
        return True

class TwilioEmergencyProvider(BaseEmergencyProvider):
    """Twilio SMS Emergency Provider using environment credentials."""
    def __init__(self, account_sid: str, auth_token: str, from_number: str):
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_number = from_number

    async def send_emergency_alert(
        self,
        user_name: str,
        contact_name: str,
        contact_phone: str,
        event: Dict[str, Any]
    ) -> bool:
        if not (self.account_sid and self.auth_token and self.from_number):
            logger.warning("Twilio credentials incomplete. Falling back to structured log dispatch.")
            return False

        message_body = (
            f"EMERGENCY ALERT: FallGuard AI detected a confirmed fall for {user_name}.\n"
            f"Risk: {event.get('risk_level')} ({event.get('confidence')*100:.1f}% confidence).\n"
        )
        if event.get("location"):
            lat = event["location"].get("lat")
            lon = event["location"].get("lon")
            if lat and lon:
                message_body += f"Location: https://maps.google.com/?q={lat},{lon}\n"
        message_body += "Please check on them immediately."

        try:
            import httpx
            url = f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Messages.json"
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    url,
                    data={"From": self.from_number, "To": contact_phone, "Body": message_body},
                    auth=(self.account_sid, self.auth_token),
                    timeout=10.0
                )
                if resp.status_code in (200, 201):
                    logger.info(f"Twilio SMS successfully sent to {contact_phone} for user {user_name}.")
                    return True
                else:
                    logger.error(f"Twilio SMS failed with status {resp.status_code}: {resp.text}")
                    return False
        except Exception as e:
            logger.error(f"Error dispatching Twilio SMS: {e}")
            return False


def get_emergency_provider() -> BaseEmergencyProvider:
    if settings.EMERGENCY_PROVIDER == "twilio" and settings.TWILIO_ACCOUNT_SID:
        return TwilioEmergencyProvider(
            account_sid=settings.TWILIO_ACCOUNT_SID,
            auth_token=settings.TWILIO_AUTH_TOKEN,
            from_number=settings.TWILIO_FROM_NUMBER
        )
    return MockEmergencyProvider()


class AlertService:
    def __init__(self):
        self.last_alert_time: Optional[datetime] = None
        self.provider: BaseEmergencyProvider = get_emergency_provider()

    def set_provider(self, provider: BaseEmergencyProvider):
        """Allows swapping provider dynamically (useful for unit tests)."""
        self.provider = provider

    async def dispatch_emergency_alert(
        self,
        db: Session,
        fall_event: FallEvent,
        user: Optional[User] = None,
        location: Optional[Dict[str, float]] = None
    ) -> bool:
        """
        Dispatch confirmed emergency alert to configured emergency contact.
        """
        user_name = user.name if user else "FallGuard User"
        contact_name = (user.emergency_contact_name if user and user.emergency_contact_name else "Designated Caregiver")
        contact_phone = (user.emergency_contact_phone if user and user.emergency_contact_phone else "+15550199")

        alert_msg = f"Fall detected for {user_name}. Please check on them immediately."
        event_data = {
            "fall_id": fall_event.id,
            "timestamp": fall_event.timestamp.isoformat(),
            "risk_level": fall_event.risk_level,
            "confidence": fall_event.confidence,
            "device_id": fall_event.device_id,
            "location": location or ({"lat": fall_event.location_lat, "lon": fall_event.location_lon} if fall_event.location_lat else None),
            "message": alert_msg
        }

        # Send via provider
        success = await self.provider.send_emergency_alert(
            user_name=user_name,
            contact_name=contact_name,
            contact_phone=contact_phone,
            event=event_data
        )

        # Update database record
        fall_event.alert_sent = True
        fall_event.emergency_contact = f"{contact_name} ({contact_phone})"
        db.commit()

        # Create alert record
        alert = Alert(
            fall_event_id=fall_event.id,
            alert_type="CRITICAL_FALL",
            message=f"CRITICAL: Confirmed Fall for {user_name} on {fall_event.device_id}. Emergency contact notified.",
            created_at=datetime.now(timezone.utc)
        )
        db.add(alert)
        db.commit()

        logger.info(f"emergency alert sent for FallEvent #{fall_event.id} to {contact_phone}")
        return success

    async def trigger_fall_alert(
        self,
        db: Session,
        confidence: float,
        risk_level: str = "HIGH",
        device_id: str = "WEARABLE_DEV_01",
        acc_peak: Optional[float] = None,
        gyro_peak: Optional[float] = None,
        notes: Optional[str] = None,
        user_id: Optional[int] = None,
        ml_activity: Optional[str] = None,
        fall_probability: Optional[float] = None,
        safety_override: bool = False,
        confirmed: bool = True
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
            user_id=user_id,
            device_id=device_id,
            timestamp=now,
            activity="FALL",
            ml_activity=ml_activity or "FALL",
            confidence=round(confidence, 4),
            fall_probability=fall_probability or confidence,
            safety_override=safety_override,
            risk_level=risk_level,
            status="UNACKNOWLEDGED",
            confirmed=confirmed,
            acknowledged=False,
            acc_peak=acc_peak,
            gyro_peak=gyro_peak,
            notes=notes or f"High-confidence fall signature ({confidence*100:.1f}%) detected."
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
                "status": fall_event.status,
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
