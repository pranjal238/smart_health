"""
FallGuard AI - Unit Tests for Sensor Simulator & Alerts
"""

import pytest
from simulation.realtime_simulator import generate_sensor_stream_generator

def test_sensor_simulator_stream():
    gen = generate_sensor_stream_generator(scenario="mixed_activities_with_fall")
    
    # Check first 50 samples
    samples = [next(gen) for _ in range(50)]
    assert len(samples) == 50
    for s in samples:
        assert "acc_x" in s
        assert "acc_y" in s
        assert "acc_z" in s
        assert "gyro_x" in s
        assert "gyro_y" in s
        assert "gyro_z" in s
        assert "ground_truth" in s
        assert "is_fall" in s
        # Acceleration magnitude reasonable (< 15g)
        mag = (s["acc_x"]**2 + s["acc_y"]**2 + s["acc_z"]**2)**0.5
        assert mag < 15.0
