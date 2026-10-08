"""
FallGuard AI - Authenticated Smartphone Sensor WebSocket Route Tests
"""

import json
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from backend.main import app
from backend.core.security import create_access_token
from backend.database.session import SessionLocal
from backend.models.db_models import User
from backend.services.sensor_stream_manager import sensor_stream_manager

client = TestClient(app)

def test_websocket_sensor_rejects_unauthenticated():
    """Verify connecting to /ws/sensor without token or with invalid token returns auth error and closes."""
    # Test 1: Invalid token in query param
    with client.websocket_connect("/ws/sensor?token=invalid_jwt_token") as ws:
        msg = json.loads(ws.receive_text())
        assert msg.get("type") == "error"
        assert msg.get("code") == "AUTH_FAILED"

    # Test 2: Missing token without handshake
    with client.websocket_connect("/ws/sensor") as ws:
        # Handshake not sent -> timeout or error
        msg = json.loads(ws.receive_text())
        assert msg.get("type") == "error"
        assert msg.get("code") in ("AUTH_REQUIRED", "AUTH_FAILED")

def test_websocket_sensor_authenticated_streaming():
    """Verify connecting to /ws/sensor with valid JWT succeeds and ingests sensor data."""
    db = SessionLocal()
    try:
        user = db.query(User).first()
        assert user is not None, "At least one user must exist in db"

        token = create_access_token({"sub": str(user.id), "email": user.email, "role": user.role})
        device_id = "test_pytest_phone"

        with client.websocket_connect(f"/ws/sensor?token={token}&device_id={device_id}") as ws:
            # First message must be auth_success
            raw_msg = ws.receive_text()
            msg = json.loads(raw_msg)
            assert msg.get("type") == "auth_success"
            assert msg.get("device_id") == device_id
            assert msg.get("user_id") == user.id

            # Verify buffer was registered in sensor_stream_manager
            buf = sensor_stream_manager.devices.get(f"{user.id}_{device_id}")
            assert buf is not None
            assert buf.connected is True

            # Send sensor_data packet in m/s^2
            sensor_packet = {
                "type": "sensor_data",
                "version": 1,
                "device_id": device_id,
                "timestamp": 12345.67,
                "unit": "m/s2",
                "accelerometer": {"x": 0.1, "y": 0.2, "z": 9.81},
                "gyroscope": {"x": 0.01, "y": 0.01, "z": 0.02}
            }
            ws.send_text(json.dumps(sensor_packet))

            # Send ping
            ws.send_text(json.dumps({"type": "ping"}))
            resp = json.loads(ws.receive_text())
            assert resp.get("type") == "pong"

            # Check buffer size updated
            assert buf.total_samples_received >= 1

    finally:
        db.close()
