from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import Dict, Any, Annotated, Optional, Union, List
import logging
from pydantic import BaseModel, Field
import uuid

from app.core.security import get_current_user, TokenPayload, create_access_token
from app.core.token_manager import TokenManager
from app.database.connection import get_session
from app.schema.requests import AuthRequest, ChatRequest, VisualizationRequest, ActionRequest, TokenCheckRequest
from app.schema.responses import TokenResponse, ChatResponse, VisualizationResponse, ActionResponse, TokenCheckResponse, ErrorResponse
from app.agents.dispatcher import AgentDispatcher
from app.core.dependencies import get_agent_dispatcher
from app.core.config import settings
from app.agents.base_agent import AgentResponse, VisualizationResult, ActionResult

# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter()

# Request models
class MessageRequest(BaseModel):
    """Message request model."""
    message: str = Field(..., description="User message")
    company_id: int = Field(..., description="Company ID")
    user_id: str = Field(..., description="User ID")
    conversation_id: Optional[str] = Field(None, description="Conversation ID")

class VisualizationRequest(BaseModel):
    """Visualization request model."""
    query: str = Field(..., description="Visualization query")
    company_id: int = Field(..., description="Company ID")
    user_id: str = Field(..., description="User ID")
    visualization_type: Optional[str] = Field(None, description="Visualization type (bar, line, pie, etc.)")
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID to associate with this visualization")

class ActionRequest(BaseModel):
    """Action request model."""
    action: str = Field(..., description="Action to perform")
    parameters: Dict[str, Any] = Field(..., description="Action parameters")
    company_id: int = Field(..., description="Company ID")
    user_id: str = Field(..., description="User ID")

class ConversationHistoryRequest(BaseModel):
    """Conversation history request model."""
    company_id: int = Field(..., description="Company ID")
    conversation_id: Optional[str] = Field(None, description="Conversation ID")
    limit: Optional[int] = Field(10, description="Maximum number of conversations to return")
    offset: Optional[int] = Field(0, description="Offset for pagination")

class PastConversationsResponse(BaseModel):
    """Past conversations response model."""
    conversations: List[Dict[str, Any]] = Field(..., description="List of past conversations")
    total: int = Field(..., description="Total number of conversations")


@router.post("/auth", response_model=TokenResponse, responses={401: {"model": ErrorResponse}, 403: {"model": ErrorResponse}})
async def authenticate(auth_data: AuthRequest, session: Session = Depends(get_session)):
    """Authenticate and get access token.
    
    Args:
        auth_data: Authentication data
        session: Database session
        
    Returns:
        JWT token response
        
    Raises:
        HTTPException: If authentication fails
    """
    # Validate the incoming service API key
    if not settings.INTERNAL_SERVICE_API_KEY or auth_data.api_key != settings.INTERNAL_SERVICE_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid service API key"
        )

    # Check if company exists and has AI agent enabled
    try:
        # Execute raw SQL to check company plan and token status
        query = text("""
            SELECT 
                u.id as company_id,
                p.ai_agent_enabled,
                p.ai_agent_default_tokens - u.ai_agent_tokens_used as tokens_available
            FROM users u
            JOIN plans p ON u.plan = p.id
            WHERE u.id = :company_id AND u.type = 'company'
        """)
        
        result = session.execute(
            query,
            {"company_id": auth_data.company_id}
        ).fetchone()
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed: Company not found"
            )
        
        company_id, ai_agent_enabled, tokens_available = result
        
        if not ai_agent_enabled:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="AI Agent is not enabled for this company or its current plan"
            )
        
        # Create access token
        access_token = create_access_token(
            subject=auth_data.user_id,
            company_id=company_id,
            role="user",
            token_count=int(tokens_available) if tokens_available is not None else 0
        )
        
        return TokenResponse(access_token=access_token, token_type="bearer")
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication failed: {str(e)}"
        )


@router.post("/message", response_model=AgentResponse)
async def process_message(
    request: MessageRequest,
    dispatcher=Depends(get_agent_dispatcher),
    session: Session = Depends(get_session),
):
    """Process a user message."""
    
    # --- Direct Visualization Path (Bypass Dispatcher for VIZ) ---
    message_lower = request.message.lower()
    viz_keywords = [
        "visualize", "visualization", "chart", "graph", "plot", "pie chart", 
        "bar chart", "line graph", "show me a chart", "make a graph",
        "create chart", "generate chart", "draw chart", "display chart",
        "employees per department", "department breakdown"
    ]
    is_visualization_request_by_keyword = any(keyword in message_lower for keyword in viz_keywords)

    current_conversation_id = request.conversation_id or str(uuid.uuid4())
    logger.info(f"[ROUTER] Processing message. Input ConvID: {request.conversation_id}, Using ConvID: {current_conversation_id}, Message: {request.message[:60]}")

    if is_visualization_request_by_keyword:
        logger.info(f"[ROUTER-VIZ-BYPASS] Detected visualization request. Bypassing dispatcher. (ConvID: {current_conversation_id})")
        try:
            # Directly instantiate and use SQLAgent for visualization
            from app.agents.sql_agent import SQLAgent # Ensure import
            sql_agent_direct = SQLAgent()
            logger.info(f"[ROUTER-VIZ-BYPASS] SQLAgent instantiated directly.")
            
            direct_viz_response = await sql_agent_direct.generate_visualization(
                query=request.message,
                company_id=request.company_id,
                user_id=request.user_id,
                visualization_type=None, # SQLAgent will infer
                session=session,
                conversation_id=current_conversation_id
            )
            logger.info(f"[ROUTER-VIZ-BYPASS] Response from direct SQLAgent.generate_visualization: {direct_viz_response.dict(exclude_none=True)}")
            direct_viz_response.agent_type = "visualization" # Ensure agent_type
            return direct_viz_response
        except Exception as e_direct_viz:
            logger.error(f"[ROUTER-VIZ-BYPASS] Error in direct visualization call: {str(e_direct_viz)}", exc_info=True)
            return AgentResponse(
                response=f"Error in direct visualization bypass: {str(e_direct_viz)}",
                conversation_id=current_conversation_id,
                conversation_title="Router Viz Bypass Error",
                tokens_used=0, tokens_remaining=None, visualization=None,
                agent_type="error_router_viz_bypass"
            )
    # --- End Direct Visualization Path ---
    
    # --- Normal Dispatcher Path (for non-visualization requests) ---
    logger.info(f"[ROUTER-DISPATCHER-PATH] Not a viz keyword match. Using dispatcher. (ConvID: {current_conversation_id})")
    try:
        # Process message with dispatcher (original logic for non-viz)
        response_from_dispatcher = await dispatcher.process_message(
            message=request.message,
            company_id=request.company_id,
            user_id=request.user_id,
            conversation_id=current_conversation_id, # Use the consistent ID
            session=session
        )
        
        logger.info(f"[ROUTER-DISPATCHER-PATH] Response from dispatcher.process_message: {response_from_dispatcher.dict(exclude_none=True)}")
        
        # Basic validation of dispatcher response
        if not response_from_dispatcher.conversation_id:
             logger.error("[ROUTER-DISPATCHER-PATH] CRITICAL: Dispatcher returned response without conversation_id! Overwriting with current_conversation_id.")
             response_from_dispatcher.conversation_id = current_conversation_id

        # If dispatcher somehow handled a viz request and it has data, ensure agent_type
        if response_from_dispatcher.visualization and response_from_dispatcher.agent_type != "visualization":
            response_from_dispatcher.agent_type = "visualization"
        elif not response_from_dispatcher.agent_type: # Fallback if agent_type is None
            response_from_dispatcher.agent_type = "unknown_from_dispatcher"
            
        return response_from_dispatcher
        
    except Exception as e_dispatcher_path:
        logger.error(f"[ROUTER-DISPATCHER-PATH] Error in dispatcher call: {str(e_dispatcher_path)}", exc_info=True)
        return AgentResponse(
            response=f"Error in router dispatcher path: {str(e_dispatcher_path)}",
            conversation_id=current_conversation_id,
            conversation_title="Router Dispatcher Error",
            tokens_used=0, tokens_remaining=None, visualization=None,
            agent_type="error_router_dispatcher_path"
        )


@router.post("/visualization", response_model=AgentResponse)
async def generate_visualization(
    request: VisualizationRequest,
    dispatcher=Depends(get_agent_dispatcher),
    session: Session = Depends(get_session)
):
    """Generate visualization."""
    try:
        logger.info(f"Generating visualization for query: {request.query[:50]}...")
        
        # Generate visualization with dispatcher
        result = await dispatcher.generate_visualization(
            query=request.query,
            company_id=request.company_id,
            user_id=request.user_id,
            visualization_type=request.visualization_type,
            session=session,
            conversation_id=request.conversation_id
        )
        
        # Debug log the result structure
        logger.info(f"Visualization result type: {type(result).__name__}")
        logger.info(f"Visualization result contains visualization data: {result.visualization is not None}")
        logger.info(f"Visualization agent_type: {result.agent_type}")
        
        # Ensure agent_type is set
        if not result.agent_type:
            result.agent_type = "visualization"
            logger.warning("Setting missing agent_type to 'visualization' in router")
        
        return result
    except Exception as e:
        logger.error(f"Error generating visualization: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate visualization: {str(e)}"
        )


@router.post("/action", response_model=ActionResult)
async def perform_action(
    request: ActionRequest,
    dispatcher=Depends(get_agent_dispatcher),
    session: Session = Depends(get_session)
):
    """Perform action."""
    try:
        logger.info(f"Performing action: {request.action}")
        
        # Perform action with dispatcher
        result = await dispatcher.perform_action(
            action=request.action,
            parameters=request.parameters,
            company_id=request.company_id,
            user_id=request.user_id,
            session=session
        )
        
        return result
    except Exception as e:
        logger.error(f"Error performing action: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to perform action: {str(e)}"
        )


@router.post("/token/check", response_model=TokenCheckResponse)
async def check_tokens(
    token_data: TokenCheckRequest,
    current_user: TokenPayload = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Check available tokens for a company.
    
    Args:
        token_data: Token check request data
        current_user: Current authenticated user
        session: Database session
        
    Returns:
        Token check response
        
    Raises:
        HTTPException: If token check fails or other errors occur
    """
    # Verify company_id matches the token
    if token_data.company_id != current_user.company_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Company ID mismatch"
        )
    
    # Get token count
    token_count = await TokenManager.get_token_count(
        company_id=current_user.company_id,
        session=session
    )
    
    return TokenCheckResponse(
        available=token_count > 0,
        count=token_count
    )


@router.post("/conversations", response_model=PastConversationsResponse)
async def get_conversations(
    request: ConversationHistoryRequest,
    dispatcher=Depends(get_agent_dispatcher),
    session: Session = Depends(get_session)
):
    """Get past conversations."""
    try:
        # Import SQLAgent to access its conversation methods
        from app.agents.sql_agent import SQLAgent
        
        # Create a temporary SQLAgent instance to access its methods
        sql_agent = SQLAgent()
        
        # Get conversations
        if request.conversation_id:
            # Get specific conversation details
            messages = await sql_agent.get_conversation_messages(
                session=session,
                conversation_id=request.conversation_id,
                company_id=request.company_id
            )
            
            return PastConversationsResponse(
                conversations=[{
                    "conversation_id": request.conversation_id,
                    "messages": messages,
                    "title": messages[0]["title"] if messages else "Conversation"
                }],
                total=1
            )
        else:
            # Get list of conversations
            # Execute SQL to get conversation list with pagination
            result = session.execute("""
                SELECT 
                    conversation_id, 
                    MAX(title) as title,
                    MAX(created_at) as last_updated,
                    COUNT(*) as message_count
                FROM agent_conversations
                WHERE company_user_id = :company_id
                GROUP BY conversation_id
                ORDER BY last_updated DESC
                LIMIT :limit OFFSET :offset
            """, {
                "company_id": request.company_id,
                "limit": request.limit,
                "offset": request.offset
            }).fetchall()
            
            # Get total count
            total = session.execute("""
                SELECT COUNT(DISTINCT conversation_id) 
                FROM agent_conversations
                WHERE company_user_id = :company_id
            """, {"company_id": request.company_id}).scalar() or 0
            
            # Format result
            conversations = []
            for row in result:
                conversations.append({
                    "conversation_id": row[0],
                    "title": row[1],
                    "last_updated": row[2].isoformat() if row[2] else None,
                    "message_count": row[3]
                })
                
            return PastConversationsResponse(
                conversations=conversations,
                total=total
            )
            
    except Exception as e:
        logger.error(f"Error getting conversations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get conversations: {str(e)}"
        )
