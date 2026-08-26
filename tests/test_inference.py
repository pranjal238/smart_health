"""
FallGuard AI - Unit Tests for Inference & Two-Stage Fall Detection
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from ml.inference.predictor import FallDetector

BASE_DIR = Path(__file__).resolve().parent.parent

def test_inference_normal_activity():
    detector = FallDetector()
    
    # 128 samples of normal walking
    t = np.linspace(0, 2.56, 128)
    ax = 0.2 * np.sin(2 * np.pi * 1.8 * t)
    ay = 0.3 * np.cos(2 * np.pi * 1.8 * t)
    az = 0.98 + 0.2 * np.sin(4 * np.pi * 1.8 * t)
    gx = 0.3 * np.sin(2 * np.pi * 1.8 * t)
    gy = 0.2 * np.cos(2 * np.pi * 1.8 * t)
    gz = 0.1 * np.sin(2 * np.pi * 1.8 * t)
    
    pred = detector.predict_window(ax, ay, az, gx, gy, gz)
    
    assert "activity" in pred
    assert "confidence" in pred
    assert "risk_level" in pred
    assert "is_fall" in pred
    assert pred["is_fall"] is False
    assert pred["risk_level"] in ("LOW", "MEDIUM")

def test_inference_severe_fall():
    detector = FallDetector()
    
    # Check if real dataset exists
    stream_csv = BASE_DIR / "data" / "processed" / "unified_sensor_stream.csv"
    if stream_csv.exists():
        df = pd.read_csv(stream_csv)
        fall_sample = df[df["is_fall"] == 1]
        if not fall_sample.empty:
            w_id = fall_sample["window_id"].iloc[0]
            w_df = fall_sample[fall_sample["window_id"] == w_id].sort_values("step")
            
            pred = detector.predict_window(
                acc_x=w_df["acc_x"].to_numpy(),
                acc_y=w_df["acc_y"].to_numpy(),
                acc_z=w_df["acc_z"].to_numpy(),
                gyro_x=w_df["gyro_x"].to_numpy(),
                gyro_y=w_df["gyro_y"].to_numpy(),
                gyro_z=w_df["gyro_z"].to_numpy()
            )
            assert pred["is_fall"] is True
            assert pred["risk_level"] == "HIGH"
            assert pred["activity"] == "FALL"
            return
            
    # Synthetic kinematic shock pulse test
    ax = np.random.normal(0.05, 0.05, 128)
    ay = np.random.normal(0.10, 0.05, 128)
    az = np.random.normal(0.98, 0.05, 128)
    gx = np.random.normal(0.0, 0.05, 128)
    gy = np.random.normal(0.0, 0.05, 128)
    gz = np.random.normal(0.0, 0.05, 128)
    
    pulse = np.exp(-((np.arange(128) - 60)**2) / 6.0)
    ax += 4.2 * pulse
    ay += 3.1 * pulse
    az += -4.5 * pulse
    gx += 4.8 * pulse
    gy += 3.6 * pulse
    gz += 2.9 * pulse
    
    pred = detector.predict_window(ax, ay, az, gx, gy, gz)
    assert pred["is_fall"] is True
    assert pred["risk_level"] == "HIGH"
    assert pred["activity"] == "FALL"
