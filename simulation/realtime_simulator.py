"""
FallGuard AI - Real-Time Sensor Stream Simulator
Simulates a wearable IoT device (e.g. ESP32 + MPU6050) streaming 6-axis IMU data at 50 Hz.
Replays realistic kinematic activities, stumbles, and high-impact fall events.
"""

import time
import math
import random
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Generator, Dict, Any, Optional

logger = logging.getLogger("SensorSimulator")

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_CSV = BASE_DIR / "data" / "processed" / "unified_sensor_stream.csv"

def generate_sensor_stream_generator(
    scenario: str = "mixed_activities_with_fall",
    sampling_rate: float = 50.0
) -> Generator[Dict[str, Any], None, None]:
    """
    Generator yielding 50Hz sensor readings (ax, ay, az in g, gx, gy, gz in rad/s).
    """
    # Check if real test data exists in unified_sensor_stream.csv
    if PROCESSED_CSV.exists() and scenario == "dataset_replay":
        df = pd.read_csv(PROCESSED_CSV)
        # Select test subject windows
        test_df = df[df["split_origin"] == "test"] if "split_origin" in df.columns else df
        for _, row in test_df.iterrows():
            yield {
                "acc_x": float(row["acc_x"]),
                "acc_y": float(row["acc_y"]),
                "acc_z": float(row["acc_z"]),
                "gyro_x": float(row["gyro_x"]),
                "gyro_y": float(row["gyro_y"]),
                "gyro_z": float(row["gyro_z"]),
                "ground_truth": str(row.get("activity", "UNKNOWN")),
                "is_fall": int(row.get("is_fall", 0))
            }
        return

    # Synthetic realistic physiological & kinematic patterns
    dt = 1.0 / sampling_rate
    t = 0.0
    step = 0
    
    # State machine for realistic sequence
    stages = [
        ("STANDING", 4.0),           # 4 seconds standing
        ("WALKING", 6.0),            # 6 seconds walking
        ("WALKING_UPSTAIRS", 4.0),   # 4 seconds stairs
        ("WALKING", 5.0),            # 5 seconds normal walk
        ("STUMBLE_PRE_FALL", 1.2),   # 1.2s pre-fall balance loss
        ("FALL_IMPACT", 0.5),        # 0.5s high-G shock impact spike
        ("POST_FALL_IMMOBILITY", 8.0), # 8 seconds post-fall on floor
        ("SITTING", 6.0),            # 6 seconds sitting
        ("WALKING", 6.0)             # 6 seconds recovery walk
    ]
    
    while True:
        for stage_name, duration_sec in stages:
            n_samples = int(duration_sec * sampling_rate)
            for i in range(n_samples):
                t += dt
                step += 1
                
                # Base gravity orientation
                if stage_name == "STANDING":
                    ax = random.gauss(0.03, 0.02)
                    ay = random.gauss(0.08, 0.02)
                    az = random.gauss(0.98, 0.02)
                    gx = random.gauss(0.0, 0.01)
                    gy = random.gauss(0.0, 0.01)
                    gz = random.gauss(0.0, 0.01)
                    gt = "STANDING"
                    fall_flag = 0
                    
                elif stage_name == "WALKING":
                    freq = 1.8 # 1.8 Hz cadence
                    ax = 0.25 * math.sin(2 * math.pi * freq * t) + random.gauss(0, 0.06)
                    ay = 0.35 * math.cos(2 * math.pi * freq * t) + random.gauss(0, 0.06)
                    az = 0.98 + 0.30 * math.sin(4 * math.pi * freq * t) + random.gauss(0, 0.08)
                    gx = 0.40 * math.sin(2 * math.pi * freq * t) + random.gauss(0, 0.03)
                    gy = 0.30 * math.cos(2 * math.pi * freq * t) + random.gauss(0, 0.03)
                    gz = 0.20 * math.sin(2 * math.pi * freq * t) + random.gauss(0, 0.03)
                    gt = "WALKING"
                    fall_flag = 0
                    
                elif stage_name == "WALKING_UPSTAIRS":
                    freq = 1.4
                    ax = 0.35 * math.sin(2 * math.pi * freq * t) + random.gauss(0, 0.08)
                    ay = 0.45 * math.cos(2 * math.pi * freq * t) + random.gauss(0, 0.08)
                    az = 1.10 + 0.45 * math.sin(4 * math.pi * freq * t) + random.gauss(0, 0.10)
                    gx = 0.60 * math.sin(2 * math.pi * freq * t) + random.gauss(0, 0.05)
                    gy = 0.50 * math.cos(2 * math.pi * freq * t) + random.gauss(0, 0.05)
                    gz = 0.30 * math.sin(2 * math.pi * freq * t) + random.gauss(0, 0.05)
                    gt = "WALKING_UPSTAIRS"
                    fall_flag = 0
                    
                elif stage_name == "STUMBLE_PRE_FALL":
                    progress = i / n_samples
                    ax = 0.8 * math.sin(progress * math.pi) + random.gauss(0, 0.15)
                    ay = -0.5 * math.sin(progress * math.pi) + random.gauss(0, 0.15)
                    az = 0.85 + 0.6 * math.cos(progress * math.pi) + random.gauss(0, 0.15)
                    gx = 2.4 * math.sin(progress * math.pi) + random.gauss(0, 0.10)
                    gy = -1.8 * math.sin(progress * math.pi) + random.gauss(0, 0.10)
                    gz = 1.5 * math.sin(progress * math.pi) + random.gauss(0, 0.10)
                    gt = "STUMBLE"
                    fall_flag = 0
                    
                elif stage_name == "FALL_IMPACT":
                    # Sudden sharp deceleration spike (> 3.5g peak)
                    progress = i / n_samples
                    pulse = math.exp(-((progress - 0.3)**2) / 0.04)
                    ax = 3.6 * pulse * random.choice([1, -1]) + random.gauss(0, 0.2)
                    ay = 2.8 * pulse * random.choice([1, -1]) + random.gauss(0, 0.2)
                    az = -3.9 * pulse + 0.3 + random.gauss(0, 0.2)
                    gx = 4.2 * pulse * random.choice([1, -1]) + random.gauss(0, 0.1)
                    gy = 3.8 * pulse * random.choice([1, -1]) + random.gauss(0, 0.1)
                    gz = 2.9 * pulse * random.choice([1, -1]) + random.gauss(0, 0.1)
                    gt = "FALL"
                    fall_flag = 1
                    
                elif stage_name == "POST_FALL_IMMOBILITY":
                    # Patient is lying on the floor: gravity vector horizontal, near zero angular velocity
                    ax = random.gauss(0.88, 0.02)
                    ay = random.gauss(0.38, 0.02)
                    az = random.gauss(0.12, 0.02)
                    gx = random.gauss(0.0, 0.01)
                    gy = random.gauss(0.0, 0.01)
                    gz = random.gauss(0.0, 0.01)
                    gt = "LAYING"
                    fall_flag = 0
                    
                elif stage_name == "SITTING":
                    ax = random.gauss(0.06, 0.02)
                    ay = random.gauss(0.82, 0.02)
                    az = random.gauss(0.48, 0.02)
                    gx = random.gauss(0.0, 0.01)
                    gy = random.gauss(0.0, 0.01)
                    gz = random.gauss(0.0, 0.01)
                    gt = "SITTING"
                    fall_flag = 0
                    
                yield {
                    "acc_x": round(ax, 4),
                    "acc_y": round(ay, 4),
                    "acc_z": round(az, 4),
                    "gyro_x": round(gx, 4),
                    "gyro_y": round(gy, 4),
                    "gyro_z": round(gz, 4),
                    "ground_truth": gt,
                    "is_fall": fall_flag
                }

if __name__ == "__main__":
    print("Testing Real-Time Sensor Stream Simulator (10 readings):")
    gen = generate_sensor_stream_generator()
    for _ in range(10):
        print(next(gen))
