"""
FallGuard AI - In-Memory Sensor Stream & Bounded Retention Buffer Manager
Strict 5-Second Transient Ring Buffer Architecture (Max ~250 samples at 50 Hz).
Ensures zero permanent database storage of raw sensor data and multi-device isolation.
"""

from collections import deque
from datetime import datetime, timezone
import time
import math
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
from backend.core.config import settings

logger = logging.getLogger("SensorStreamManager")

STANDARD_GRAVITY = 9.80665

class DeviceSensorBuffer:
    """
    Isolated in-memory bounded ring buffer for a single user/device stream.
    Strictly caps raw sensor data to approximately 5 seconds (~250 samples at 50 Hz).
    Sample 251 drops sample 1 automatically.
    """
    def __init__(self, device_id: str, user_id: Optional[int] = None, maxlen: int = 250):
        self.device_id = device_id
        self.user_id = user_id
        self.maxlen = maxlen
        # Stores tuples: (ts, ax_g, ay_g, az_g, gx_rad, gy_rad, gz_rad)
        self.buffer = deque(maxlen=maxlen)
        self.last_seen: Optional[datetime] = None
        self.connected: bool = False
        self.total_samples_received: int = 0
        self.last_inference_sample_count: int = 0
        self.source_type: str = "REAL_PHONE" # REAL_PHONE or SIMULATOR
        self.last_location: Optional[Dict[str, float]] = None

    def add_reading(
        self,
        timestamp: float,
        ax: float,
        ay: float,
        az: float,
        gx: float,
        gy: float,
        gz: float,
        unit: str = "m/s2",
        location: Optional[Dict[str, float]] = None
    ) -> Tuple[float, float, float, float, float, float, float]:
        """
        Normalize sensor units to g and rad/s and append to the 5-second ring buffer.
        Returns the normalized (ts, ax_g, ay_g, az_g, gx, gy, gz).
        """
        self.last_seen = datetime.now(timezone.utc)
        self.total_samples_received += 1
        if location:
            self.last_location = location

        # Unit Normalization: Convert m/s^2 to g for ML model compatibility
        if unit and unit.lower() in ("m/s2", "m/s^2", "ms2", "ms^-2"):
            ax_g = ax / STANDARD_GRAVITY
            ay_g = ay / STANDARD_GRAVITY
            az_g = az / STANDARD_GRAVITY
        else:
            ax_g = ax
            ay_g = ay
            az_g = az

        gx_rad = gx
        gy_rad = gy
        gz_rad = gz

        reading = (timestamp, ax_g, ay_g, az_g, gx_rad, gy_rad, gz_rad)
        self.buffer.append(reading)
        return reading

    def should_run_inference(self, stride: int = 25, min_samples: int = 64) -> bool:
        """
        Determines if a new sliding window is ready based on sample stride.
        """
        if len(self.buffer) < min_samples:
            return False
        return (self.total_samples_received - self.last_inference_sample_count) >= stride

    def extract_inference_window(self, target_samples: int = 128) -> np.ndarray:
        """
        Extracts the most recent 128-sample window (2.56 seconds at 50 Hz).
        If buffer has between 64 and 127 samples, pads the start to 128.
        Returns array of shape (128, 6): [ax, ay, az, gx, gy, gz].
        """
        self.last_inference_sample_count = self.total_samples_received
        samples = list(self.buffer)
        n = len(samples)

        if n >= target_samples:
            window_slice = samples[-target_samples:]
            data = [[s[1], s[2], s[3], s[4], s[5], s[6]] for s in window_slice]
            return np.array(data, dtype=np.float64)
        else:
            # Pad early readings if buffer has >= 64 samples
            data = [[s[1], s[2], s[3], s[4], s[5], s[6]] for s in samples]
            arr = np.array(data, dtype=np.float64)
            pad_count = target_samples - n
            pad = np.repeat(arr[:1], pad_count, axis=0)
            return np.vstack([pad, arr])

    def get_status(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "user_id": self.user_id,
            "connected": self.connected,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "samples_received": self.total_samples_received,
            "buffer_samples": len(self.buffer),
            "max_buffer_samples": self.maxlen,
            "buffer_seconds": round(len(self.buffer) / settings.PHONE_SAMPLING_RATE, 2),
            "source_type": self.source_type,
            "has_location": self.last_location is not None,
            "last_location": self.last_location
        }

class SensorStreamManager:
    """
    Central manager for isolated per-device rolling sensor buffers.
    Enforces multi-user isolation and memory constraints.
    """
    def __init__(self, max_buffer_samples: int = 250):
        self.max_buffer_samples = max_buffer_samples
        self.devices: Dict[str, DeviceSensorBuffer] = {}

    def get_or_create_buffer(self, device_id: str, user_id: Optional[int] = None, source_type: str = "REAL_PHONE") -> DeviceSensorBuffer:
        key = f"{user_id or 'anon'}_{device_id}"
        if key not in self.devices:
            buf = DeviceSensorBuffer(device_id=device_id, user_id=user_id, maxlen=self.max_buffer_samples)
            buf.source_type = source_type
            self.devices[key] = buf
            logger.info(f"Registered new isolated sensor buffer for user={user_id} device={device_id} (max {self.max_buffer_samples} samples)")
        return self.devices[key]

    def set_device_connection(self, device_id: str, user_id: Optional[int], connected: bool):
        key = f"{user_id or 'anon'}_{device_id}"
        if key in self.devices:
            self.devices[key].connected = connected
            if not connected:
                logger.info(f"Device {device_id} for user {user_id} marked DISCONNECTED.")

    def get_device_status(self, device_id: str, user_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        key = f"{user_id or 'anon'}_{device_id}"
        buf = self.devices.get(key)
        return buf.get_status() if buf else None

    def get_all_device_statuses(self) -> List[Dict[str, Any]]:
        return [buf.get_status() for buf in self.devices.values()]

sensor_stream_manager = SensorStreamManager(max_buffer_samples=settings.MAX_BUFFER_SAMPLES)
