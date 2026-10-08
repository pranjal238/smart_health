"""
FallGuard AI - Fall Confirmation & False Alarm Mitigation Service
Coordinates the two-tier fall verification lifecycle:
1. Potential fall detected -> PENDING_CONFIRMATION state.
2. High-priority user notification triggered on the smartphone.
3. If user presses "I'm OK" -> Alert cancelled, no emergency contact alerted.
4. If user presses "Need Help" or configurable timeout expires -> Fall confirmed, emergency alert dispatched.
"""

import asyncio
from datetime import datetime, timezone
import logging
from typing import Dict, Optional, Any
from sqlalchemy.orm import Session

from backend.database.session import SessionLocal
from backend.models.db_models import FallEvent, User, SystemLog
from backend.services.alert_service import alert_service
from backend.services.websocket_service import ws_manager
from backend.core.config import settings

logger = logging.getLogger("FallConfirmationService")

class PendingConfirmation:
    def __init__(self, fall_event_id: int, user_id: Optional[int], device_id: str, task: asyncio.Task):
        self.fall_event_id = fall_event_id
        self.user_id = user_id
        self.device_id = device_id
        self.task = task
        self.created_at = datetime.now(timezone.utc)

class FallConfirmationService:
    def __init__(self):
        self.pending_confirmations: Dict[int, PendingConfirmation] = {}

    async def initiate_fall_confirmation(
        self,
        db: Session,
        confidence: float,
        risk_level: str = "HIGH",
        device_id: str = "phone_001",
        user_id: Optional[int] = None,
        ml_activity: str = "WALKING",
        ml_fall_probability: float = 0.0,
        safety_override: bool = False,
        acc_peak: Optional[float] = None,
        gyro_peak: Optional[float] = None,
        location: Optional[Dict[str, float]] = None,
        timeout_seconds: Optional[float] = None
    ) -> FallEvent:
        """
        Create a FallEvent in PENDING_CONFIRMATION status and start countdown timer.
        """
        timeout = timeout_seconds or settings.FALL_CONFIRMATION_TIMEOUT_SECONDS
        now = datetime.now(timezone.utc)

        # 1. Create FallEvent with status = PENDING_CONFIRMATION
        fall_event = FallEvent(
            user_id=user_id,
            device_id=device_id,
            timestamp=now,
            activity="FALL",
            ml_activity=ml_activity,
            confidence=round(confidence, 4),
            fall_probability=round(ml_fall_probability, 4),
            safety_override=safety_override,
            risk_level=risk_level,
            status="PENDING_CONFIRMATION",
            confirmed=False,
            alert_sent=False,
            acknowledged=False,
            acc_peak=acc_peak,
            gyro_peak=gyro_peak,
            location_lat=location.get("lat") if location else None,
            location_lon=location.get("lon") if location else None,
            notes=f"Potential fall detected. Awaiting user response ({timeout}s timeout)."
        )
        db.add(fall_event)
        db.commit()
        db.refresh(fall_event)

        logger.info(f"potential fall detected for device={device_id} user={user_id}. FallEvent #{fall_event.id}")
        logger.info(f"fall confirmation started (timeout={timeout}s)")

        # 2. Broadcast high-priority confirmation request to phone and dashboard
        ws_msg = {
            "type": "FALL_PENDING_CONFIRMATION",
            "event": {
                "id": fall_event.id,
                "event_id": fall_event.id,
                "device_id": device_id,
                "user_id": user_id,
                "timestamp": now.isoformat(),
                "confidence": round(confidence, 4),
                "risk_level": risk_level,
                "timeout_seconds": timeout,
                "safety_override": safety_override,
                "ml_activity": ml_activity,
                "ml_fall_probability": round(ml_fall_probability, 4),
                "message": "Possible fall detected. Are you okay?"
            }
        }
        await ws_manager.broadcast(ws_msg)

        # 3. Schedule async countdown task
        task = asyncio.create_task(self._timeout_handler(fall_event.id, timeout))
        self.pending_confirmations[fall_event.id] = PendingConfirmation(
            fall_event_id=fall_event.id,
            user_id=user_id,
            device_id=device_id,
            task=task
        )
        return fall_event

    async def _timeout_handler(self, fall_event_id: int, timeout_seconds: float):
        """Async worker that waits for timeout and escalates if no response received."""
        try:
            await asyncio.sleep(timeout_seconds)
            logger.info(f"Confirmation timeout ({timeout_seconds}s) expired for FallEvent #{fall_event_id}. Confirming fall.")
            await self.confirm_and_escalate_fall(fall_event_id, reason="TIMEOUT")
        except asyncio.CancelledError:
            pass

    async def handle_user_response(
        self,
        db: Session,
        fall_event_id: int,
        action: str,
        location: Optional[Dict[str, float]] = None
    ) -> Optional[FallEvent]:
        """
        Process smartphone user's button press:
        - "I_AM_OK" -> Cancel emergency escalation.
        - "NEED_HELP" -> Confirm fall immediately.
        """
        action_norm = action.upper().replace(" ", "_").replace("'", "")
        pending = self.pending_confirmations.pop(fall_event_id, None)

        if pending and not pending.task.done():
            pending.task.cancel()

        fall = db.query(FallEvent).filter(FallEvent.id == fall_event_id).first()
        if not fall:
            logger.warning(f"FallEvent #{fall_event_id} not found for response {action}.")
            return None

        if location:
            fall.location_lat = location.get("lat", fall.location_lat)
            fall.location_lon = location.get("lon", fall.location_lon)

        now = datetime.now(timezone.utc)

        if action_norm in ("I_AM_OK", "IM_OK", "OK", "SAFE", "CANCEL"):
            fall.status = "CANCELLED_BY_USER"
            fall.confirmed = False
            fall.acknowledged = True
            fall.acknowledged_at = now
            fall.acknowledged_by = "User (Smartphone)"
            fall.notes = (fall.notes or "") + "\nUser verified safe: Emergency escalation cancelled."
            db.commit()
            db.refresh(fall)

            logger.info(f"user confirmed safe for FallEvent #{fall_event_id}")

            # Notify dashboard and phone that confirmation is resolved safely
            await ws_manager.broadcast({
                "type": "FALL_CONFIRMATION_RESOLVED",
                "data": {
                    "event_id": fall.id,
                    "status": "CANCELLED_BY_USER",
                    "device_id": fall.device_id,
                    "message": "User confirmed safe. Alert cancelled."
                }
            })
            return fall

        elif action_norm in ("NEED_HELP", "HELP", "CONFIRM", "FALL"):
            return await self.confirm_and_escalate_fall(fall_event_id, reason="MANUAL_HELP_REQUEST", db=db)

        return fall

    async def confirm_and_escalate_fall(
        self,
        fall_event_id: int,
        reason: str = "TIMEOUT",
        db: Optional[Session] = None
    ) -> Optional[FallEvent]:
        """
        Escalate a fall to confirmed status and dispatch emergency alert to designated emergency contact.
        """
        should_close = False
        if db is None:
            db = SessionLocal()
            should_close = True

        try:
            self.pending_confirmations.pop(fall_event_id, None)

            fall = db.query(FallEvent).filter(FallEvent.id == fall_event_id).first()
            if not fall:
                return None

            fall.status = "CONFIRMED"
            fall.confirmed = True
            now = datetime.now(timezone.utc)

            user = None
            if fall.user_id:
                user = db.query(User).filter(User.id == fall.user_id).first()

            # Dispatch alert to emergency contact
            location_dict = None
            if fall.location_lat and fall.location_lon:
                location_dict = {"lat": fall.location_lat, "lon": fall.location_lon}

            await alert_service.dispatch_emergency_alert(
                db=db,
                fall_event=fall,
                user=user,
                location=location_dict
            )

            logger.info(f"fall confirmed for FallEvent #{fall.id} (Reason: {reason})")

            # Broadcast confirmed fall to all connected clients
            await ws_manager.broadcast({
                "type": "FALL_CONFIRMATION_RESOLVED",
                "data": {
                    "event_id": fall.id,
                    "status": "CONFIRMED",
                    "device_id": fall.device_id,
                    "reason": reason,
                    "message": f"Fall confirmed ({reason}). Emergency alert dispatched."
                }
            })
            return fall
        finally:
            if should_close:
                db.close()

fall_confirmation_service = FallConfirmationService()
