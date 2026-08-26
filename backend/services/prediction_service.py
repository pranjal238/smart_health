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
        device_id: str = "WEARABLE_DEV_01"
    ) -> Dict[str, Any]:
        """
        Process a full sensor window, perform inference, persist prediction, and alert if fall.
        """
        pred = self.detector.predict_window(
            acc_x=np.array(acc_x),
            acc_y=np.array(acc_y),
            acc_z=np.array(acc_z),
            gyro_x=np.array(gyro_x),
            gyro_y=np.array(gyro_y),
            gyro_z=np.array(gyro_z)
        )
        
        # Persist prediction in DB
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
        
        # Check if fall event
        if pred["is_fall"]:
            acc_peak = pred["features_summary"]["acc_mag_max"] if pred.get("features_summary") else None
            gyro_peak = pred["features_summary"]["gyro_mag_max"] if pred.get("features_summary") else None
            await alert_service.trigger_fall_alert(
                db=db,
                confidence=pred["confidence"],
                risk_level=pred["risk_level"],
                device_id=device_id,
                acc_peak=acc_peak,
                gyro_peak=gyro_peak
            )
            
        # Broadcast live prediction over WebSocket
        ws_payload = {
            "type": "ACTIVITY_UPDATE",
            "data": {
                "timestamp": pred["timestamp"],
                "activity": pred["activity"],
                "confidence": pred["confidence"],
                "risk_level": pred["risk_level"],
                "is_fall": pred["is_fall"],
                "device_id": device_id,
                "probabilities": pred["probabilities"],
                "features_summary": pred.get("features_summary")
            }
        }
        await ws_manager.broadcast(ws_payload)
        return pred

prediction_service = PredictionService()
