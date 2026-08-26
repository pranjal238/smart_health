"""
FallGuard AI - Alert Service Lifecycle Tests
"""

import pytest
import asyncio
from datetime import datetime
from backend.database.session import SessionLocal
from backend.services.alert_service import AlertService
from backend.models.db_models import FallEvent, Alert

def test_alert_service_trigger_and_cooldown():
    async def _run():
        service = AlertService()
        db = SessionLocal()
        try:
            # Trigger first alert
            fall1 = await service.trigger_fall_alert(
                db=db,
                confidence=0.95,
                risk_level="HIGH",
                device_id="TEST_DEVICE_ALERT_01",
                acc_peak=4.2,
                gyro_peak=3.8
            )
            assert fall1 is not None
            assert fall1.status == "UNACKNOWLEDGED"
            assert fall1.confidence == 0.95
            
            # Trigger immediate second alert (should be suppressed by cooldown)
            fall2 = await service.trigger_fall_alert(
                db=db,
                confidence=0.92,
                risk_level="HIGH",
                device_id="TEST_DEVICE_ALERT_01"
            )
            assert fall2 is None # Cooldown active
            
            # Acknowledge fall1
            ack = service.acknowledge_fall(db=db, fall_id=fall1.id, acknowledged_by="Caregiver John")
            assert ack is not None
            assert ack.acknowledged is True
            assert ack.status == "ACKNOWLEDGED"
            assert ack.acknowledged_by == "Caregiver John"
        finally:
            db.close()

    asyncio.run(_run())
