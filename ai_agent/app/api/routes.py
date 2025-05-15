from fastapi import APIRouter, Depends, HTTPException, status, Body
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
import logging

from app.agents.sql_agent import SQLAgent
from app.agents.base_agent import AgentResponse
from app.database.connection import DatabaseConnection
from app.core.config import settings
from app.schema.requests import ChatRequest

# Set up logging for this module
logger = logging.getLogger(__name__)

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

@router.post("/message", response_model=AgentResponse, tags=["Chat"])
async def chat_endpoint(
    request: ChatRequest,
    session: Session = Depends(get_db)
):
    """Handle incoming chat messages."""
    try:
        logger.info(f"Received message request for company {request.company_id}")
        # TODO: Add more sophisticated agent selection logic if needed
        agent = SQLAgent()
        
        # Process the message using the agent
        agent_response_obj: AgentResponse = await agent.process_message(
            message=request.message,
            company_id=request.company_id,
            user_id=request.user_id,
            conversation_id=request.conversation_id,
            session=session
        )
        
        logger.info(f"Agent processed message. Returning response for convo {agent_response_obj.conversation_id}")
        
        # Explicitly return the Pydantic model; FastAPI should serialize it correctly
        # based on the response_model=AgentResponse definition in @router.post
        return agent_response_obj
        
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"An internal error occurred: {str(e)}"
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

class ConversationHistoryResponse(BaseModel):
    messages: List[Dict[str, Any]] = Field(..., description="List of messages in the conversation")

@router.get("/conversation/{conversation_id}", response_model=ConversationHistoryResponse, tags=["Chat"])
async def get_conversation_history_endpoint(
    conversation_id: str,
    company_id: int,  # Assuming company_id is needed for authorization/scoping
    session: Session = Depends(get_db)
):
    """Retrieve the message history for a specific conversation."""
    try:
        logger.info(f"Fetching conversation history for ID: {conversation_id}, Company: {company_id}")
        
        # Use a temporary agent instance to access the base class method
        # In a more complex setup, you might have a dedicated service for this
        temp_agent = SQLAgent() # Or any agent inheriting from BaseAgent
        
        messages = await temp_agent.get_conversation_messages(session, conversation_id, company_id)
        
        if not messages:
             # You might want to distinguish between "not found" and "empty"
             logger.warning(f"No messages found for conversation {conversation_id} or access denied.")
             # Return empty list if no messages found,符合 Pydantic 模型
             # Alternatively, raise HTTPException(status_code=404, detail="Conversation not found")
             
        # Log visualization data if present in any message for debugging
        has_visualization = False
        for msg in messages:
            if 'visualization' in msg and msg['visualization']:
                has_visualization = True
                logger.info(f"Conversation {conversation_id} includes visualization data in response")
                break
        
        if not has_visualization:
            logger.warning(f"No visualization data found in any message for conversation {conversation_id}")
             
        return ConversationHistoryResponse(messages=messages)

    except Exception as e:
        logger.error(f"Error fetching conversation history {conversation_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve conversation history: {str(e)}"
        ) 