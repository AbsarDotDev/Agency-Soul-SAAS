import os
import secrets
from typing import List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings(BaseModel):
    """Application settings."""
    
    # API settings
    API_V1_STR: str = "/v1"
    SECRET_KEY: str = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
    
    # CORS settings
    BACKEND_CORS_ORIGINS: List[str] = ["*"]
    
    # Database settings
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        f"mysql+mysqlconnector://{os.getenv('DB_USERNAME', 'root')}:{os.getenv('DB_PASSWORD', '')}@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '3306')}/{os.getenv('DB_DATABASE', 'ageny_soul_saas')}"
    )
    
    # Google Generative AI settings
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    
    # OpenAI settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # Token settings
    DEFAULT_TOKEN_ALLOCATION: int = int(os.getenv("DEFAULT_TOKEN_ALLOCATION", "100"))
    
    # Internal Service API Key (for service-to-service authentication of the /auth endpoint itself)
    INTERNAL_SERVICE_API_KEY: Optional[str] = os.getenv("INTERNAL_SERVICE_API_KEY", None)
    
    # JWT settings
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    JWT_ALGORITHM: str = "HS256"
    
    class Config:
        case_sensitive = True

# Create settings object
settings = Settings()
