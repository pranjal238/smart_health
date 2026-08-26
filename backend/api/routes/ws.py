"""
FallGuard AI - WebSocket Telemetry Streaming Route
"""

import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from backend.services.websocket_service import ws_manager

router = APIRouter(tags=["WebSockets"])
logger = logging.getLogger("WebSocketRoute")

@router.websocket("/ws/monitor")
async def websocket_monitor_endpoint(websocket: WebSocket):
    """
    Real-time streaming telemetry and instant fall alert push endpoint.
    """
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep-alive receive / command ping
            data = await websocket.receive_text()
            # Optional: handle client-side ping/ack messages
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.warning(f"WebSocket connection error: {e}")
        ws_manager.disconnect(websocket)
