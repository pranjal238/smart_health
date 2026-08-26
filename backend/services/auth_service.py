"""
FallGuard AI - Authentication Service
Handles login validation, user registration, and JWT issuance.
"""

from typing import Optional
from sqlalchemy.orm import Session
from backend.models.db_models import User
from backend.core.security import verify_password, create_access_token, hash_password
from backend.schemas.schemas import UserLogin, TokenResponse, UserResponse

class AuthService:
    def authenticate_user(self, db: Session, login_data: UserLogin) -> Optional[TokenResponse]:
        user = db.query(User).filter(User.email == login_data.email).first()
        if not user:
            return None
            
        if not verify_password(login_data.password, user.hashed_password):
            return None
            
        token_data = {
            "sub": str(user.id),
            "email": user.email,
            "role": user.role,
            "name": user.name
        }
        token = create_access_token(data=token_data)
        
        return TokenResponse(
            access_token=token,
            token_type="bearer",
            user=UserResponse.model_validate(user)
        )

auth_service = AuthService()
