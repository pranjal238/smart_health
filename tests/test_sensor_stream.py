"""
FallGuard AI - Sensor Stream, Unit Conversion & Bounded 5-Second Retention Tests
Proves:
1. Strict 5-second transient buffer retention (max ~250 samples at 50 Hz).
2. Sample 251 drops sample 1 automatically.
3. Unit conversion from Android m/s^2 to trained model g.
4. Multi-user and multi-device buffer isolation.
5. Sliding 128-sample window extraction and padding.
"""

import pytest
import numpy as np
from backend.services.sensor_stream_manager import (
    DeviceSensorBuffer,
    SensorStreamManager,
    STANDARD_GRAVITY
)

def test_unit_conversion_ms2_to_g():
    """Verify Android m/s^2 sensor readings are normalized to g."""
    buf = DeviceSensorBuffer(device_id="phone_test_01", maxlen=250)

    # Stationary phone reporting 9.80665 m/s^2 on Z axis
    reading = buf.add_reading(
        timestamp=100.0,
        ax=0.0,
        ay=0.0,
        az=STANDARD_GRAVITY, # 9.80665 m/s^2
        gx=0.05,
        gy=0.02,
        gz=-0.01,
        unit="m/s2"
    )

    # az should be normalized to 1.0 g
    ts, ax_g, ay_g, az_g, gx, gy, gz = reading
    assert pytest.approx(az_g, rel=1e-4) == 1.0
    assert pytest.approx(ax_g, abs=1e-4) == 0.0
    assert pytest.approx(ay_g, abs=1e-4) == 0.0
    assert pytest.approx(gx, rel=1e-4) == 0.05

def test_unit_passthrough_for_g():
    """Verify readings already in g are not double-scaled."""
    buf = DeviceSensorBuffer(device_id="phone_test_01", maxlen=250)

    reading = buf.add_reading(
        timestamp=100.0,
        ax=0.1,
        ay=0.2,
        az=0.98,
        gx=0.0,
        gy=0.0,
        gz=0.0,
        unit="g"
    )
    ts, ax_g, ay_g, az_g, gx, gy, gz = reading
    assert pytest.approx(az_g, rel=1e-4) == 0.98
    assert pytest.approx(ax_g, rel=1e-4) == 0.1

def test_raw_buffer_never_exceeds_five_seconds():
    """
    CRITICAL PRIVACY TEST:
    Stream 1,000 continuous sensor samples at 50 Hz.
    Strictly assert that the in-memory buffer NEVER exceeds 250 samples (~5.0 seconds).
    Proves that older samples are discarded transiently without continuous accumulation.
    """
    max_samples = 250
    buf = DeviceSensorBuffer(device_id="privacy_test_phone", maxlen=max_samples)

    # Push 1,000 samples (equivalent to 20 seconds of continuous 50Hz sensor data)
    for i in range(1000):
        buf.add_reading(
            timestamp=i * 0.02,
            ax=0.1, ay=0.2, az=9.81,
            gx=0.01, gy=0.01, gz=0.01,
            unit="m/s2"
        )
        # Invariant: buffer size must never exceed maxlen
        assert len(buf.buffer) <= max_samples

    # Final assertion: exactly 250 samples remain
    assert len(buf.buffer) == max_samples
    assert buf.total_samples_received == 1000

    # First sample in buffer should be sample index 750 (1000 - 250)
    first_sample_ts = buf.buffer[0][0]
    expected_first_ts = 750 * 0.02
    assert pytest.approx(first_sample_ts, rel=1e-3) == expected_first_ts

def test_sliding_window_extraction_and_stride():
    """Verify 128-sample window extraction and stride tracking."""
    buf = DeviceSensorBuffer(device_id="window_test_phone", maxlen=250)

    # Less than 64 samples -> should not run inference
    for i in range(50):
        buf.add_reading(i * 0.02, 0.1, 0.2, 9.81, 0.0, 0.0, 0.0)
    assert buf.should_run_inference(stride=25, min_samples=64) is False

    # Add up to 70 samples -> should run inference with padding
    for i in range(50, 70):
        buf.add_reading(i * 0.02, 0.1, 0.2, 9.81, 0.0, 0.0, 0.0)
    assert buf.should_run_inference(stride=25, min_samples=64) is True

    window = buf.extract_inference_window(target_samples=128)
    assert window.shape == (128, 6)

    # Add 25 more samples (stride reached)
    for i in range(70, 95):
        buf.add_reading(i * 0.02, 0.1, 0.2, 9.81, 0.0, 0.0, 0.0)
    assert buf.should_run_inference(stride=25, min_samples=64) is True

def test_multi_user_buffer_isolation():
    """Verify device buffers for different users/devices are completely isolated."""
    manager = SensorStreamManager(max_buffer_samples=250)

    buf_user_a = manager.get_or_create_buffer(device_id="phone_A", user_id=1)
    buf_user_b = manager.get_or_create_buffer(device_id="phone_B", user_id=2)

    # Stream 100 samples into User A
    for i in range(100):
        buf_user_a.add_reading(i * 0.02, 0.5, 0.5, 9.81, 0.1, 0.1, 0.1)

    # Stream 20 samples into User B
    for i in range(20):
        buf_user_b.add_reading(i * 0.02, 0.1, 0.1, 9.81, 0.0, 0.0, 0.0)

    assert len(buf_user_a.buffer) == 100
    assert len(buf_user_b.buffer) == 20
    assert buf_user_a.total_samples_received == 100
    assert buf_user_b.total_samples_received == 20

    # Ensure samples did not cross over
    assert buf_user_a.buffer[0][1] == pytest.approx(0.5 / STANDARD_GRAVITY, rel=1e-3)
    assert buf_user_b.buffer[0][1] == pytest.approx(0.1 / STANDARD_GRAVITY, rel=1e-3)
