"""
FallGuard AI - WebSocket Connection & Telemetry Broadcast Manager
"""

import asyncio
import json
import logging
from typing import List, Dict, Any
from fastapi import WebSocket

logger = logging.getLogger("WebSocketManager")

class ConnectionManager:
    """Manages active dashboard and smartphone WebSocket connections."""
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        # Key: device_id -> WebSocket
        self.device_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Monitor WebSocket client connected. Total active: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"Monitor WebSocket client disconnected. Remaining: {len(self.active_connections)}")

    def register_device(self, device_id: str, websocket: WebSocket):
        self.device_connections[device_id] = websocket
        logger.info(f"Device WebSocket registered for device={device_id}. Total devices: {len(self.device_connections)}")

    def unregister_device(self, device_id: str):
        self.device_connections.pop(device_id, None)
        logger.info(f"Device WebSocket unregistered for device={device_id}. Remaining devices: {len(self.device_connections)}")

    async def send_to_device(self, device_id: str, message: Dict[str, Any]) -> bool:
        ws = self.device_connections.get(device_id)
        if not ws:
            return False
        try:
            await ws.send_text(json.dumps(message, default=str))
            return True
        except Exception as e:
            logger.warning(f"Failed to send targeted message to device {device_id}: {e}")
            self.unregister_device(device_id)
            return False

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast JSON message to all connected monitor and device clients."""
        json_data = json.dumps(message, default=str)
        
        # Broadcast to dashboard monitors
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json_data)
            except Exception as e:
                logger.warning(f"Error broadcasting to monitor client: {e}")
                disconnected.append(connection)
                
        for dead_conn in disconnected:
            self.disconnect(dead_conn)

        # Broadcast to devices if relevant (e.g. alerts or confirmations)
        if message.get("type") in ("FALL_PENDING_CONFIRMATION", "FALL_CONFIRMATION_RESOLVED", "FALL_ALERT"):
            dead_devices = []
            for dev_id, dev_ws in self.device_connections.items():
                try:
                    await dev_ws.send_text(json_data)
                except Exception as e:
                    logger.warning(f"Error sending alert to device {dev_id}: {e}")
                    dead_devices.append(dev_id)
            for dead_dev in dead_devices:
                self.unregister_device(dead_dev)

ws_manager = ConnectionManager()
