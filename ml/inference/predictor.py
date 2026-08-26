"""
FallGuard AI - Real-Time Inference & Two-Stage Fall Detection Engine
Processes raw sensor windows, extracts dynamic features, runs ML inference,
and computes calibrated confidence and risk levels (HIGH / MEDIUM / LOW).
"""

import os
import joblib
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from collections import deque
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

from ml.features.feature_engineering import extract_features_from_window, extract_features_from_array

logger = logging.getLogger("FallPredictor")

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "models" / "fall_detection_model.joblib"

class FallDetector:
    """
    Two-Stage Fall Detection and Activity Recognition Inference Engine.
    """
    def __init__(self, model_path: Optional[Path] = None, window_size: int = 128, sampling_rate: float = 50.0):
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.window_size = window_size
        self.sampling_rate = sampling_rate
        
        self.model_bundle: Optional[Dict[str, Any]] = None
        self.model = None
        self.scaler = None
        self.use_scaled = False
        self.feature_names: List[str] = []
        self.classes: List[str] = []
        self.fall_class_idx: int = -1
        self.model_name = "Heuristic Fall Classifier (Untrained)"
        
        # Real-time streaming sliding buffer
        self.buffer = deque(maxlen=self.window_size)
        self.load_model()

    def load_model(self) -> bool:
        """Load trained model bundle from disk."""
        if not self.model_path.exists():
            logger.warning(f"Model file not found at {self.model_path}. Running with heuristic fallback.")
            return False
            
        try:
            self.model_bundle = joblib.load(self.model_path)
            self.model = self.model_bundle["model"]
            self.scaler = self.model_bundle.get("scaler")
            self.use_scaled = self.model_bundle.get("use_scaled", False)
            self.feature_names = self.model_bundle.get("feature_names", [])
            self.classes = self.model_bundle.get("classes", [])
            self.fall_class_idx = self.model_bundle.get("fall_class_idx", -1)
            self.model_name = self.model_bundle.get("model_name", "Trained Model")
            logger.info(f"Loaded {self.model_name} from {self.model_path} with {len(self.feature_names)} features.")
            return True
        except Exception as e:
            logger.error(f"Failed to load model from {self.model_path}: {e}")
            return False

    def predict_window(
        self,
        acc_x: np.ndarray,
        acc_y: np.ndarray,
        acc_z: np.ndarray,
        gyro_x: np.ndarray,
        gyro_y: np.ndarray,
        gyro_z: np.ndarray
    ) -> Dict[str, Any]:
        """
        Predict activity and fall risk for a 6-axis sensor window.
        """
        now_ts = datetime.now(timezone.utc).isoformat()
        
        # Extract features
        feats = extract_features_from_window(
            acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, sampling_rate=self.sampling_rate
        )
        
        # Fallback heuristic if ML model not loaded
        if self.model is None or not self.feature_names:
            acc_mag_max = feats.get("acc_mag_max", 1.0)
            gyro_mag_max = feats.get("gyro_mag_max", 0.0)
            
            is_fall = acc_mag_max > 2.8 and gyro_mag_max > 2.5
            activity = "FALL" if is_fall else "WALKING"
            confidence = 0.88 if is_fall else 0.75
            risk_level = "HIGH" if is_fall else ("MEDIUM" if acc_mag_max > 2.0 else "LOW")
            
            return {
                "activity": activity,
                "confidence": round(confidence, 4),
                "risk_level": risk_level,
                "is_fall": bool(is_fall),
                "probabilities": {activity: round(confidence, 4)},
                "model_name": self.model_name,
                "model_version": "1.0.0-fallback",
                "timestamp": now_ts,
                "features_summary": {
                    "acc_mag_max": round(acc_mag_max, 3),
                    "gyro_mag_max": round(gyro_mag_max, 3),
                    "jerk_max": round(feats.get("jerk_max", 0.0), 3)
                }
            }

        # Vectorize features in exact training order
        feature_vector = np.array([[feats.get(name, 0.0) for name in self.feature_names]])
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
        
        if self.use_scaled and self.scaler is not None:
            feature_vector = self.scaler.transform(feature_vector)
            
        # Model Prediction
        pred_idx = self.model.predict(feature_vector)[0]
        predicted_activity = self.classes[pred_idx] if pred_idx < len(self.classes) else "UNKNOWN"
        
        prob_dict = {}
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(feature_vector)[0]
            for idx, cls_name in enumerate(self.classes):
                if idx < len(probs):
                    prob_dict[cls_name] = float(round(probs[idx], 4))
            confidence = float(np.max(probs))
            fall_prob = prob_dict.get("FALL", 0.0)
        else:
            confidence = 0.90
            fall_prob = 1.0 if predicted_activity == "FALL" else 0.0
            prob_dict[predicted_activity] = 1.0

        # Two-Stage Fall Detection Logic
        # Stage 1: Fall Screening (High Sensitivity for dangerous impacts + Model Probability)
        is_fall = (predicted_activity == "FALL") or (fall_prob >= 0.50) or (feats.get("acc_mag_max", 0.0) >= 2.8 and feats.get("gyro_mag_max", 0.0) >= 2.2)
        
        # Determine Risk Level: HIGH / MEDIUM / LOW
        if is_fall:
            risk_level = "HIGH"
            predicted_activity = "FALL" # Prioritize fall notification
        elif fall_prob >= 0.25 or (feats.get("acc_mag_max", 0.0) > 2.2 and feats.get("gyro_mag_max", 0.0) > 1.8):
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            "activity": predicted_activity,
            "confidence": round(confidence, 4),
            "risk_level": risk_level,
            "is_fall": bool(is_fall),
            "probabilities": prob_dict,
            "model_name": self.model_name,
            "model_version": "1.0.0",
            "timestamp": now_ts,
            "features_summary": {
                "acc_mag_max": round(feats.get("acc_mag_max", 0.0), 3),
                "gyro_mag_max": round(feats.get("gyro_mag_max", 0.0), 3),
                "jerk_max": round(feats.get("jerk_max", 0.0), 3),
                "pitch_mean": round(feats.get("pitch_mean", 0.0), 2),
                "roll_mean": round(feats.get("roll_mean", 0.0), 2)
            }
        }

    def predict_from_array(self, window_data: np.ndarray) -> Dict[str, Any]:
        """Predict from 2D array of shape (N, 6) [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]."""
        if window_data.shape[1] < 6:
            raise ValueError(f"Expected at least 6 sensor columns, got {window_data.shape[1]}")
        return self.predict_window(
            acc_x=window_data[:, 0],
            acc_y=window_data[:, 1],
            acc_z=window_data[:, 2],
            gyro_x=window_data[:, 3],
            gyro_y=window_data[:, 4],
            gyro_z=window_data[:, 5]
        )

    def add_sensor_reading(
        self, 
        acc_x: float, 
        acc_y: float, 
        acc_z: float, 
        gyro_x: float, 
        gyro_y: float, 
        gyro_z: float
    ) -> Optional[Dict[str, Any]]:
        """
        Stream an individual sensor sample (50 Hz). Returns prediction once window buffer is full.
        """
        self.buffer.append([acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z])
        
        if len(self.buffer) == self.window_size:
            arr = np.array(self.buffer)
            return self.predict_from_array(arr)
        return None
