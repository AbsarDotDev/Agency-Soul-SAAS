from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union


class TokenResponse(BaseModel):
    """Token response model."""
    
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field("bearer", description="Token type")
    
    class Config:
        schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer"
            }
        }


class MessageResponse(BaseModel):
    """Response message model."""
    
    text: str = Field(..., description="Response text")
    
    class Config:
        schema_extra = {
            "example": {
                "text": "There are 5 employees in the HR department."
            }
        }


class ChatResponse(BaseModel):
    """Chat response model."""
    
    message: str = Field(..., description="Response message")
    conversation_id: str = Field(..., description="Conversation ID")
    token_usage: int = Field(..., description="Token usage")
    tokens_remaining: int = Field(..., description="Tokens remaining")
    
    class Config:
        schema_extra = {
            "example": {
                "message": "There are 5 employees in the HR department.",
                "conversation_id": "abc123def456",
                "token_usage": 1,
                "tokens_remaining": 99
            }
        }


class VisualizationData(BaseModel):
    """Visualization data model."""
    
    chart_type: str = Field(..., description="Chart type (bar, line, pie, etc.)")
    labels: List[str] = Field(..., description="Chart labels")
    datasets: List[Dict[str, Any]] = Field(..., description="Chart datasets")
    title: str = Field(..., description="Chart title")
    description: Optional[str] = Field(None, description="Chart description")
    
    class Config:
        schema_extra = {
            "example": {
                "chart_type": "line",
                "labels": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
                "datasets": [
                    {
                        "label": "Sales",
                        "data": [12, 19, 3, 5, 2, 3],
                        "borderColor": "rgb(75, 192, 192)",
                        "tension": 0.1
                    }
                ],
                "title": "Sales Trend (Last 6 Months)",
                "description": "This chart shows the sales trend for the last 6 months."
            }
        }


class VisualizationResponse(BaseModel):
    """Visualization response model."""
    
    data: VisualizationData = Field(..., description="Visualization data")
    text_explanation: str = Field(..., description="Text explanation of the visualization")
    token_usage: int = Field(..., description="Token usage")
    tokens_remaining: int = Field(..., description="Tokens remaining")
    
    class Config:
        schema_extra = {
            "example": {
                "data": {
                    "chart_type": "line",
                    "labels": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
                    "datasets": [
                        {
                            "label": "Sales",
                            "data": [12, 19, 3, 5, 2, 3],
                            "borderColor": "rgb(75, 192, 192)",
                            "tension": 0.1
                        }
                    ],
                    "title": "Sales Trend (Last 6 Months)",
                    "description": "This chart shows the sales trend for the last 6 months."
                },
                "text_explanation": "The sales trend shows a peak in February, followed by a decline in March.",
                "token_usage": 2,
                "tokens_remaining": 98
            }
        }


class ActionResponse(BaseModel):
    """Action response model."""
    
    success: bool = Field(..., description="Action success status")
    message: str = Field(..., description="Action result message")
    data: Optional[Dict[str, Any]] = Field(None, description="Action result data")
    token_usage: int = Field(..., description="Token usage")
    tokens_remaining: int = Field(..., description="Tokens remaining")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Employee added successfully",
                "data": {
                    "employee_id": 42,
                    "name": "John Doe",
                    "email": "john.doe@example.com"
                },
                "token_usage": 3,
                "tokens_remaining": 97
            }
        }


class TokenCheckResponse(BaseModel):
    """Token check response model."""
    
    available: bool = Field(..., description="Token availability status")
    count: int = Field(..., description="Available token count")
    
    class Config:
        schema_extra = {
            "example": {
                "available": True,
                "count": 100
            }
        }


class ErrorResponse(BaseModel):
    """Error response model."""
    
    detail: str = Field(..., description="Error detail")
    
    class Config:
        schema_extra = {
            "example": {
                "detail": "Insufficient tokens"
            }
        }
