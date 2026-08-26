"""
FallGuard AI - Sensor Data Cleaning & Standardization Pipeline
Handles missing/malformed sensor readings, unit conversions, and calibration.
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional

logger = logging.getLogger("Cleaner")

GRAVITY_EARTH = 9.80665 # 1g in m/s^2

def clean_sensor_stream(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw sensor stream:
    - Replace NaNs/Infs using forward fill and linear interpolation
    - Clip extreme outlier spikes caused by sensor hardware glitches (> 15g or > 25 rad/s)
    - Ensure correct numeric data types
    """
    df_clean = df.copy()
    
    sensor_cols = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    for col in sensor_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
            df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
            df_clean[col] = df_clean[col].ffill().bfill()
            
            # Clip physical limits
            if "acc" in col:
                df_clean[col] = df_clean[col].clip(-16.0, 16.0) # ±16g range
            elif "gyro" in col:
                df_clean[col] = df_clean[col].clip(-35.0, 35.0) # ±35 rad/s range
                
    return df_clean

def standardize_units(df: pd.DataFrame, acc_in_ms2: bool = False, gyro_in_deg: bool = False) -> pd.DataFrame:
    """
    Convert acceleration to g and gyroscope to rad/s for universal consistency.
    """
    df_std = df.copy()
    if acc_in_ms2:
        for c in ["acc_x", "acc_y", "acc_z"]:
            if c in df_std.columns:
                df_std[c] = df_std[c] / GRAVITY_EARTH
                
    if gyro_in_deg:
        for c in ["gyro_x", "gyro_y", "gyro_z"]:
            if c in df_std.columns:
                df_std[c] = np.radians(df_std[c])
                
    return df_std
