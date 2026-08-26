"""
FallGuard AI - Fall Incident & Caregiver Management API Endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional

from backend.database.session import get_db
from backend.models.db_models import FallEvent, Alert
from backend.schemas.schemas import FallEventResponse, FallAcknowledgeRequest, FallNoteRequest
from backend.services.alert_service import alert_service

router = APIRouter(prefix="/falls", tags=["Fall Events"])

@router.get("", response_model=List[FallEventResponse])
def get_fall_events(
    status: Optional[str] = None,
    limit: int = Query(50, le=200),
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """List historical fall incidents sorted newest first."""
    query = db.query(FallEvent)
    if status:
        query = query.filter(FallEvent.status == status.upper())
    falls = query.order_by(FallEvent.timestamp.desc()).offset(offset).limit(limit).all()
    return falls

@router.get("/{fall_id}", response_model=FallEventResponse)
def get_fall_event(fall_id: int, db: Session = Depends(get_db)):
    """Retrieve detailed information for a specific fall incident."""
    fall = db.query(FallEvent).filter(FallEvent.id == fall_id).first()
    if not fall:
        raise HTTPException(status_code=404, detail=f"Fall event #{fall_id} not found.")
    return fall

@router.post("/{fall_id}/acknowledge", response_model=FallEventResponse)
def acknowledge_fall_event(
    fall_id: int,
    payload: FallAcknowledgeRequest,
    db: Session = Depends(get_db)
):
    """Mark a fall event as acknowledged by clinical staff / caregiver."""
    updated = alert_service.acknowledge_fall(
        db=db,
        fall_id=fall_id,
        acknowledged_by=payload.acknowledged_by,
        notes=payload.notes
    )
    if not updated:
        raise HTTPException(status_code=404, detail=f"Fall event #{fall_id} not found.")
    return updated

@router.post("/{fall_id}/notes", response_model=FallEventResponse)
def add_fall_investigation_note(
    fall_id: int,
    payload: FallNoteRequest,
    db: Session = Depends(get_db)
):
    """Add clinical notes to an ongoing or past fall incident."""
    updated = alert_service.add_fall_notes(
        db=db,
        fall_id=fall_id,
        notes=payload.notes
    )
    if not updated:
        raise HTTPException(status_code=404, detail=f"Fall event #{fall_id} not found.")
    return updated
