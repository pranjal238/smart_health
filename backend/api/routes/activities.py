"""
FallGuard AI - Activity Telemetry & History API Endpoints
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from backend.database.session import get_db
from backend.models.db_models import ActivityPrediction
from backend.schemas.schemas import ActivityHistoryItem

router = APIRouter(prefix="/activities", tags=["Activities"])

@router.get("/history", response_model=List[ActivityHistoryItem])
def get_activity_history(
    activity: Optional[str] = None,
    risk_level: Optional[str] = None,
    limit: int = Query(50, le=500),
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """Retrieve chronologically ordered time-series predictions."""
    query = db.query(ActivityPrediction)
    if activity:
        query = query.filter(ActivityPrediction.activity == activity.upper())
    if risk_level:
        query = query.filter(ActivityPrediction.risk_level == risk_level.upper())
        
    records = query.order_by(ActivityPrediction.timestamp.desc()).offset(offset).limit(limit).all()
    return records
