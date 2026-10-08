"""
FallGuard AI - Central Prediction Service
Coordinates feature extraction, ML inference, and alert triggering.
"""

from datetime import datetime, timezone
import logging
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
import numpy as np

from ml.inference.predictor import FallDetector
from backend.models.db_models import ActivityPrediction
from backend.services.fall_confirmation_service import fall_confirmation_service
from backend.services.alert_service import alert_service
from backend.services.websocket_service import ws_manager
from backend.core.config import settings

logger = logging.getLogger("PredictionService")

class PredictionService:
    def __init__(self):
        self.detector = FallDetector(
            model_path=settings.MODEL_PATH,
            window_size=settings.WINDOW_SAMPLES,
            sampling_rate=settings.SAMPLING_RATE
        )

    def reload_model(self):
        """Reload ML model bundle if new model was trained."""
        return self.detector.load_model()

    async def predict_single_window(
        self,
        db: Session,
        acc_x: List[float],
        acc_y: List[float],
        acc_z: List[float],
        gyro_x: List[float],
        gyro_y: List[float],
        gyro_z: List[float],
        device_id: str = "WEARABLE_DEV_01",
        user_id: Optional[int] = None,
        location: Optional[Dict[str, float]] = None,
        require_confirmation: bool = True
    ) -> Dict[str, Any]:
        """
        Process a full sensor window, perform inference, persist prediction, and initiate fall confirmation if fall.
        """
        pred = self.detector.predict_window(
            acc_x=np.array(acc_x),
            acc_y=np.array(acc_y),
            acc_z=np.array(acc_z),
            gyro_x=np.array(gyro_x),
            gyro_y=np.array(gyro_y),
            gyro_z=np.array(gyro_z)
        )
        
        # Persist prediction summary in DB (audit log only, not raw samples)
        act_record = ActivityPrediction(
            timestamp=datetime.now(timezone.utc),
            activity=pred["activity"],
            confidence=pred["confidence"],
            risk_level=pred["risk_level"],
            is_fall=pred["is_fall"],
            device_id=device_id,
            model_version=pred["model_version"]
        )
        db.add(act_record)
        db.commit()
        
        # Check if fall event detected
        if pred["is_fall"]:
            acc_peak = pred["features_summary"]["acc_mag_max"] if pred.get("features_summary") else None
            gyro_peak = pred["features_summary"]["gyro_mag_max"] if pred.get("features_summary") else None
            
            if require_confirmation:
                await fall_confirmation_service.initiate_fall_confirmation(
                    db=db,
                    confidence=pred["confidence"],
                    risk_level=pred["risk_level"],
                    device_id=device_id,
                    user_id=user_id,
                    ml_activity=pred.get("ml_activity", "WALKING"),
                    ml_fall_probability=pred.get("ml_fall_probability", 0.0),
                    safety_override=pred.get("safety_override", False),
                    acc_peak=acc_peak,
                    gyro_peak=gyro_peak,
                    location=location
                )
            else:
                await alert_service.trigger_fall_alert(
                    db=db,
                    confidence=pred["confidence"],
                    risk_level=pred["risk_level"],
                    device_id=device_id,
                    acc_peak=acc_peak,
                    gyro_peak=gyro_peak,
                    user_id=user_id,
                    ml_activity=pred.get("ml_activity"),
                    fall_probability=pred.get("ml_fall_probability"),
                    safety_override=pred.get("safety_override", False)
                )
            
        # Broadcast live prediction over WebSocket
        ws_payload = {
            "type": "ACTIVITY_UPDATE",
            "data": {
                "timestamp": pred["timestamp"],
                "activity": pred["activity"],
                "ml_activity": pred.get("ml_activity", pred["activity"]),
                "ml_fall_probability": pred.get("ml_fall_probability", 0.0),
                "safety_override": pred.get("safety_override", False),
                "confidence": pred["confidence"],
                "risk_level": pred["risk_level"],
                "is_fall": pred["is_fall"],
                "device_id": device_id,
                "user_id": user_id,
                "probabilities": pred["probabilities"],
                "features_summary": pred.get("features_summary")
            }
        }
        await ws_manager.broadcast(ws_payload)
        return pred

prediction_service = PredictionService()
