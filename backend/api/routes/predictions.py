"""
FallGuard AI - Sensor Prediction API Endpoints
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from backend.database.session import get_db
from backend.schemas.schemas import SingleWindowPredictionRequest, PredictionResponse, BatchPredictionResponse
from backend.services.prediction_service import prediction_service

router = APIRouter(prefix="/predict", tags=["Predictions"])

@router.post("", response_model=PredictionResponse)
async def predict_window(
    payload: SingleWindowPredictionRequest,
    db: Session = Depends(get_db)
):
    """
    Run two-stage fall detection and activity classification on a 6-axis IMU window.
    """
    n_samples = len(payload.acc_x)
    if not (n_samples == len(payload.acc_y) == len(payload.acc_z) == len(payload.gyro_x) == len(payload.gyro_y) == len(payload.gyro_z)):
        raise HTTPException(
            status_code=400,
            detail="All 6 accelerometer and gyroscope arrays must have identical length."
        )
    if n_samples < 8:
        raise HTTPException(
            status_code=400,
            detail=f"Window requires at least 8 samples, got {n_samples}."
        )
        
    pred = await prediction_service.predict_single_window(
        db=db,
        acc_x=payload.acc_x,
        acc_y=payload.acc_y,
        acc_z=payload.acc_z,
        gyro_x=payload.gyro_x,
        gyro_y=payload.gyro_y,
        gyro_z=payload.gyro_z,
        device_id=payload.device_id or "WEARABLE_DEV_01"
    )
    return pred

@router.post("/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    windows: List[SingleWindowPredictionRequest],
    db: Session = Depends(get_db)
):
    """
    Process multiple sensor windows sequentially.
    """
    preds = []
    for w in windows:
        p = await prediction_service.predict_single_window(
            db=db,
            acc_x=w.acc_x,
            acc_y=w.acc_y,
            acc_z=w.acc_z,
            gyro_x=w.gyro_x,
            gyro_y=w.gyro_y,
            gyro_z=w.gyro_z,
            device_id=w.device_id or "WEARABLE_DEV_01"
        )
        preds.append(p)
    return {
        "predictions": preds,
        "total_processed": len(preds)
    }
