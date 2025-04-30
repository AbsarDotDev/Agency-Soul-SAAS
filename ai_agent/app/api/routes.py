from fastapi import APIRouter, Depends, HTTPException, status, Body
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from app.agents.sql_agent import SQLAgent
from app.database.connection import DatabaseConnection
from app.core.config import settings

# Create router
router = APIRouter(prefix="/api")

# Create database session dependency
Session = DatabaseConnection.create_session_factory()

def get_db():
    """Database session dependency."""
    db = Session()
    try:
        yield db
    finally:
        db.close()

class MessageRequest(BaseModel):
    """Message request model."""
    message: str = Field(..., description="User message")
    company_id: int = Field(..., description="Company ID for data isolation")
    user_id: str = Field(..., description="User ID")
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID")
    agent_type: Optional[str] = Field("sql", description="Agent type (sql, hrm, etc.)")

class VisualizationRequest(BaseModel):
    """Visualization request model."""
    query: str = Field(..., description="Visualization query")
    company_id: int = Field(..., description="Company ID for data isolation")
    user_id: str = Field(..., description="User ID")
    visualization_type: Optional[str] = Field(None, description="Optional visualization type")
    agent_type: Optional[str] = Field("sql", description="Agent type (sql, hrm, etc.)")

@router.post("/message", response_model=Dict[str, Any], tags=["Agent"])
async def process_message(
    request: MessageRequest,
    db: Session = Depends(get_db)
):
    """Process a message using the appropriate agent."""
    try:
        # Validate required fields
        if not request.message:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Message is required"
            )
        
        if not request.company_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Company ID is required"
            )
        
        if not request.user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User ID is required"
            )
        
        # Select the appropriate agent based on type
        if request.agent_type.lower() == "hrm":
            # For now, we'll use SQLAgent for all requests
            agent = SQLAgent()
        else:
            # Default to SQL agent
            agent = SQLAgent()
        
        # Process the message
        response = await agent.process_message(
            message=request.message,
            company_id=request.company_id,
            user_id=request.user_id,
            conversation_id=request.conversation_id,
            session=db
        )
        
        # Return the response
        return {
            "message": response.message,
            "conversation_id": response.conversation_id,
            "visualization": response.visualization
        }
    except HTTPException as e:
        # Re-raise HTTPExceptions
        raise
    except Exception as e:
        # Log the error
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error processing message: {str(e)}")
        
        # Return a 500 error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing message: {str(e)}"
        )

@router.post("/visualization", response_model=Dict[str, Any], tags=["Agent"])
async def generate_visualization(
    request: VisualizationRequest,
    db: Session = Depends(get_db)
):
    """Generate a visualization using the appropriate agent."""
    try:
        # Validate required fields
        if not request.query:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query is required"
            )
        
        if not request.company_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Company ID is required"
            )
        
        if not request.user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User ID is required"
            )
        
        # Select the appropriate agent based on type
        if request.agent_type.lower() == "hrm":
            # For now, we'll use SQLAgent for all requests
            agent = SQLAgent()
        else:
            # Default to SQL agent
            agent = SQLAgent()
        
        # Generate the visualization
        visualization = await agent.generate_visualization(
            query=request.query,
            company_id=request.company_id,
            user_id=request.user_id,
            visualization_type=request.visualization_type,
            session=db
        )
        
        # Return the visualization
        return {
            "data": visualization.data,
            "explanation": visualization.explanation
        }
    except HTTPException as e:
        # Re-raise HTTPExceptions
        raise
    except Exception as e:
        # Log the error
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error generating visualization: {str(e)}")
        
        # Return a 500 error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating visualization: {str(e)}"
        )

@router.get("/company/{company_id}/tokens", response_model=Dict[str, Any], tags=["Token"])
async def get_token_usage(
    company_id: int,
    db: Session = Depends(get_db)
):
    """Get token usage for a company."""
    try:
        # Get token usage
        result = db.execute("""
            SELECT ai_agent_enabled, ai_agent_tokens_allocated, ai_agent_tokens_used
            FROM users
            WHERE id = :company_id
        """, {
            "company_id": company_id
        }).fetchone()
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Company {company_id} not found"
            )
        
        # Return token usage
        return {
            "enabled": bool(result[0]),
            "allocated": result[1],
            "used": result[2],
            "remaining": result[1] - result[2]
        }
    except HTTPException as e:
        # Re-raise HTTPExceptions
        raise
    except Exception as e:
        # Log the error
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error getting token usage: {str(e)}")
        
        # Return a 500 error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting token usage: {str(e)}"
        )

@router.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"} 