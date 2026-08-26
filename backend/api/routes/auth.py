"""
FallGuard AI - Authentication API Endpoints
"""

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

@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    """Retrieve current authenticated caregiver / admin profile."""
    return current_user
