"""
FallGuard AI - Authentication API Endpoints
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from backend.database.session import get_db
from backend.schemas.schemas import UserLogin, TokenResponse, UserResponse
from backend.services.auth_service import auth_service
from backend.api.dependencies import get_current_user
from backend.models.db_models import User

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/login", response_model=TokenResponse)
def login(login_data: UserLogin, db: Session = Depends(get_db)):
    """Authenticate user with email and password."""
    token_resp = auth_service.authenticate_user(db, login_data)
    if not token_resp:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
    return token_resp

class UserRegister(UserLogin):
    name: str
    role: Optional[str] = "PATIENT"
    phone_number: Optional[str] = None
    emergency_contact_name: Optional[str] = None
    emergency_contact_phone: Optional[str] = None
    emergency_contact_email: Optional[str] = None

from backend.schemas.schemas import EmergencyContactUpdate, DeviceStatus
from backend.services.sensor_stream_manager import sensor_stream_manager
from backend.core.security import hash_password

@router.post("/register", response_model=TokenResponse)
def register(user_data: UserRegister, db: Session = Depends(get_db)):
    """Register a new smartphone user or caregiver account."""
    existing = db.query(User).filter(User.email == user_data.email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email is already registered."
        )
    user = User(
        name=user_data.name,
        email=user_data.email,
        hashed_password=hash_password(user_data.password),
        role=user_data.role or "PATIENT",
        phone_number=user_data.phone_number,
        emergency_contact_name=user_data.emergency_contact_name,
        emergency_contact_phone=user_data.emergency_contact_phone,
        emergency_contact_email=user_data.emergency_contact_email
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    login_data = UserLogin(email=user_data.email, password=user_data.password)
    return auth_service.authenticate_user(db, login_data)

@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    """Retrieve current authenticated user/patient/caregiver profile."""
    return current_user

@router.put("/emergency-contact", response_model=UserResponse)
def update_emergency_contact(
    payload: EmergencyContactUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Configure designated emergency contact details for fall escalation."""
    current_user.emergency_contact_name = payload.emergency_contact_name
    current_user.emergency_contact_phone = payload.emergency_contact_phone
    if payload.emergency_contact_email is not None:
        current_user.emergency_contact_email = payload.emergency_contact_email
    if payload.phone_number is not None:
        current_user.phone_number = payload.phone_number

    db.commit()
    db.refresh(current_user)
    return current_user

@router.get("/devices", response_model=List[Dict[str, Any]])
def get_connected_devices(current_user: User = Depends(get_current_user)):
    """List all connected smartphone devices and in-memory buffer statistics."""
    statuses = sensor_stream_manager.get_all_device_statuses()
    if current_user.role == "ADMIN":
        return statuses
    # Return devices belonging to user or anon
    return [s for s in statuses if s.get("user_id") in (current_user.id, None)]
