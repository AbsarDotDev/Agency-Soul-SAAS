import os
from datetime import datetime, timedelta, timezone
from typing import Optional, Any, Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import BaseModel, Field

from app.core.config import settings # Assuming your settings are here

# OAuth2PasswordBearer is a utility to get the token from the Authorization header
# tokenUrl is not strictly necessary if you're not using FastAPI's built-in OAuth2 password flow form,
# but it's good practice to specify where the token is obtained.
# For service-to-service or direct token usage, the client just needs to send the token.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth") # Adjusted tokenUrl to /api/auth

class TokenPayload(BaseModel):
    sub: str = Field(..., description="Subject (usually user ID or unique identifier)")
    company_id: int = Field(..., description="Company ID associated with the token")
    role: Optional[str] = Field(None, description="User role, if applicable")
    token_count: Optional[int] = Field(None, description="Available tokens at the time of token creation")
    exp: Optional[int] = Field(None, description="Token expiry timestamp") # Standard JWT claim

def create_access_token(subject: Any, company_id: int, role: Optional[str] = None, token_count: Optional[int] = None, expires_delta: Optional[timedelta] = None) -> str:
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode = {
        "sub": str(subject),
        "company_id": company_id,
        "exp": int(expire.timestamp()) # Ensure exp is an int timestamp
    }
    if role is not None:
        to_encode["role"] = role
    if token_count is not None:
        to_encode["token_count"] = token_count
        
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt

async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]) -> TokenPayload:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload_dict = jwt.decode(
            token, 
            settings.SECRET_KEY, 
            algorithms=[settings.JWT_ALGORITHM],
            options={"verify_aud": False} # No audience claim is being set currently
        )
        
        # Convert timestamp back to datetime for validation if needed, or use directly
        # exp = payload_dict.get("exp")
        # if exp is None or datetime.fromtimestamp(exp, timezone.utc) < datetime.now(timezone.utc):
        #     raise JWTError("Token has expired")
            
        # Validate essential fields before creating TokenPayload
        user_id: Optional[str] = payload_dict.get("sub")
        company_id: Optional[int] = payload_dict.get("company_id")
        
        if user_id is None or company_id is None:
            # Log this specific error for better debugging
            logging.error(f"Token missing required claims: sub or company_id. Payload: {payload_dict}")
            raise credentials_exception
            
        return TokenPayload(**payload_dict)
    
    except JWTError as e:
        logging.error(f"JWTError during token decoding: {e}", exc_info=True)
        raise credentials_exception
    except Exception as e: # Catch any other unexpected errors during parsing
        logging.error(f"Unexpected error during token processing: {e}", exc_info=True)
        raise credentials_exception
