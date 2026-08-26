"""
FallGuard AI - Unit Tests for Feature Extraction Engine
"""

import pytest
import numpy as np
from ml.features.feature_engineering import (
    extract_features_from_window,
    extract_features_from_array,
    compute_fft_features
)

def test_compute_fft_features():
    # 2 Hz pure sine wave at 50 Hz sampling rate
    t = np.linspace(0, 2.56, 128)
    signal = np.sin(2 * np.pi * 2.0 * t)
    fft_res = compute_fft_features(signal, sampling_rate=50.0)
    
    assert "dominant_freq" in fft_res
    assert "spectral_energy" in fft_res
    assert "spectral_entropy" in fft_res
    assert abs(fft_res["dominant_freq"] - 2.0) < 0.5 # ~2 Hz detected
    assert fft_res["spectral_energy"] > 0.0

def test_extract_features_from_window():
    n_samples = 128
    ax = np.ones(n_samples) * 0.1
    ay = np.ones(n_samples) * 0.2
    az = np.ones(n_samples) * 0.98
    gx = np.zeros(n_samples)
    gy = np.zeros(n_samples)
    gz = np.zeros(n_samples)
    
    feats = extract_features_from_window(ax, ay, az, gx, gy, gz, sampling_rate=50.0)
    
    assert isinstance(feats, dict)
    assert len(feats) > 50
    # Verify no NaN or Inf values
    for k, v in feats.items():
        assert not np.isnan(v), f"Feature {k} is NaN"
        assert not np.isinf(v), f"Feature {k} is Inf"
        
    assert "acc_mag_mean" in feats
    assert "jerk_max" in feats
    assert "pitch_mean" in feats
    assert "roll_mean" in feats

def test_extract_features_from_array():
    arr = np.random.normal(0, 1, size=(128, 6))
    feats = extract_features_from_array(arr, sampling_rate=50.0)
    assert isinstance(feats, dict)
    assert len(feats) > 50
