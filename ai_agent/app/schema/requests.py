from pydantic import BaseModel, Field
from typing import Optional, List, Union, Dict, Any


class AuthRequest(BaseModel):
    """Authentication request model."""
    
    company_id: int = Field(..., description="Company ID")
    user_id: int = Field(..., description="User ID")
    api_key: str = Field(..., description="API key")


class ChatRequest(BaseModel):
    """Chat request model."""
    
    message: str = Field(..., description="User message")
    company_id: int = Field(..., description="Company ID")
    user_id: int = Field(..., description="User ID")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for continuing a conversation")
    
    class Config:
        schema_extra = {
            "example": {
                "message": "How many employees do we have in the HR department?",
                "company_id": 1,
                "user_id": 1,
                "conversation_id": None
            }
        }


class VisualizationRequest(BaseModel):
    """Visualization request model."""
    
    query: str = Field(..., description="Query for visualization")
    company_id: int = Field(..., description="Company ID")
    user_id: int = Field(..., description="User ID")
    visualization_type: Optional[str] = Field(None, description="Type of visualization (bar, line, pie, etc.)")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "Show me the sales trend for the last 6 months",
                "company_id": 1,
                "user_id": 1,
                "visualization_type": "line"
            }
        }


class ActionRequest(BaseModel):
    """Action request model for agent actions."""
    
    action: str = Field(..., description="Action to perform")
    company_id: int = Field(..., description="Company ID")
    user_id: int = Field(..., description="User ID")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Action parameters")
    
    class Config:
        schema_extra = {
            "example": {
                "action": "add_employee",
                "company_id": 1,
                "user_id": 1,
                "parameters": {
                    "name": "John Doe",
                    "email": "john.doe@example.com",
                    "department_id": 2,
                    "designation_id": 3
                }
            }
        }


class TokenCheckRequest(BaseModel):
    """Token check request model."""
    
    company_id: int = Field(..., description="Company ID")
    
    class Config:
        schema_extra = {
            "example": {
                "company_id": 1
            }
        }
