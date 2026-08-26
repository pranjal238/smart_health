"""
FallGuard AI - Unit Tests for Preprocessing & Zero-Leakage Splitter
"""

import pytest
import numpy as np
import pandas as pd
from ml.preprocessing.cleaner import clean_sensor_stream, standardize_units
from ml.preprocessing.splitter import split_by_subject

def test_clean_sensor_stream():
    data = {
        "acc_x": [0.1, np.nan, 0.3, np.inf, 25.0], # Has NaN, Inf, and extreme spike
        "acc_y": [0.2, 0.2, np.nan, 0.4, 0.5],
        "acc_z": [0.98, 0.98, 0.98, 0.98, 0.98],
        "gyro_x": [0.0, 0.1, 0.0, 50.0, 0.0], # Has extreme gyro spike
        "gyro_y": [0.0, 0.0, 0.0, 0.0, 0.0],
        "gyro_z": [0.0, 0.0, 0.0, 0.0, 0.0]
    }
    df = pd.DataFrame(data)
    df_clean = clean_sensor_stream(df)
    
    assert df_clean.isna().sum().sum() == 0
    assert np.isinf(df_clean.to_numpy()).sum() == 0
    assert df_clean["acc_x"].max() <= 16.0 # Clipped to max physical range
    assert df_clean["gyro_x"].max() <= 35.0

def test_split_by_subject_zero_leakage():
    # 10 subjects, 20 windows each
    records = []
    for s in range(1, 11):
        for w in range(20):
            records.append({
                "subject_id": s,
                "window_id": f"w_{s}_{w}",
                "feat1": np.random.randn(),
                "feat2": np.random.randn(),
                "activity": "WALKING",
                "is_fall": 0
            })
    df_feat = pd.DataFrame(records)
    
    train_df, test_df, train_subs, test_subs = split_by_subject(df_feat, test_subject_ratio=0.30, random_seed=42)
    
    # Verify strict zero overlap between train and test subjects
    overlap = set(train_subs).intersection(set(test_subs))
    assert len(overlap) == 0, f"Data leakage detected! Overlapping subjects: {overlap}"
    
    # Verify non-empty
    assert len(train_df) > 0
    assert len(test_df) > 0
    assert len(train_subs) + len(test_subs) == 10
