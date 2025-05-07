from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Dict, Any

from app.core.security import get_current_user, TokenPayload, create_access_token
from app.core.token_manager import TokenManager
from app.database.connection import get_db_session
from app.schema.requests import AuthRequest, ChatRequest, VisualizationRequest, ActionRequest, TokenCheckRequest
from app.schema.responses import TokenResponse, ChatResponse, VisualizationResponse, ActionResponse, TokenCheckResponse, ErrorResponse
from app.agents.dispatcher import AgentDispatcher

router = APIRouter()


@router.post("/auth", response_model=TokenResponse, responses={401: {"model": ErrorResponse}})
async def authenticate(auth_data: AuthRequest, session: Session = Depends(get_db_session)):
    """Authenticate and get access token.
    
    Args:
        auth_data: Authentication data
        session: Database session
        
    Returns:
        JWT token response
        
    Raises:
        HTTPException: If authentication fails
    """
    # In a real implementation, this would validate credentials against the database
    # For now, we'll just issue a token based on the company_id
    
    # Check if company exists and has AI agent enabled
    try:
        # Execute raw SQL to check company API key
        result = session.execute(
            "SELECT id, ai_agent_enabled, tokens_available FROM companies WHERE id = :company_id",
            {"company_id": auth_data.company_id}
        ).fetchone()
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed: Company not found"
            )
        
        company_id, ai_agent_enabled, token_count = result
        
        if not ai_agent_enabled:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="AI Agent is not enabled for this company"
            )
        
        # Create access token
        access_token = create_access_token(
            subject=auth_data.user_id,
            company_id=auth_data.company_id,
            role="user",  # Role could be retrieved from database in a real implementation
            token_count=token_count
        )
        
        return TokenResponse(access_token=access_token, token_type="bearer")
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication failed: {str(e)}"
        )


@router.post("/chat", response_model=ChatResponse, responses={401: {"model": ErrorResponse}, 403: {"model": ErrorResponse}})
async def chat(
    chat_data: ChatRequest,
    current_user: TokenPayload = Depends(get_current_user),
    session: Session = Depends(get_db_session)
):
    """Process chat message and get response.
    
    Args:
        chat_data: Chat request data
        current_user: Current authenticated user
        session: Database session
        
    Returns:
        Chat response
        
    Raises:
        HTTPException: If token check fails or other errors occur
    """
    # Verify company_id matches the token
    if chat_data.company_id != current_user.company_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Company ID mismatch"
        )
    
    # Check token availability
    token_available = await TokenManager.check_token_availability(
        company_id=current_user.company_id,
        session=session
    )
    
    if not token_available:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient tokens"
        )
    
    # Process the message with the agent dispatcher
    dispatcher = AgentDispatcher()
    response = await dispatcher.process_message(
        message=chat_data.message,
        company_id=current_user.company_id,
        user_id=current_user.sub,
        conversation_id=chat_data.conversation_id,
        session=session
    )
    
    # Consume token
    await TokenManager.consume_token(
        company_id=current_user.company_id,
        session=session
    )
    
    # Get remaining tokens
    tokens_remaining = await TokenManager.get_token_count(
        company_id=current_user.company_id,
        session=session
    )
    
    return ChatResponse(
        message=response.message,
        conversation_id=response.conversation_id,
        token_usage=1,  # In a real implementation, this would be the actual token usage
        tokens_remaining=tokens_remaining
    )


@router.post("/visualization", response_model=VisualizationResponse, responses={401: {"model": ErrorResponse}, 403: {"model": ErrorResponse}})
async def generate_visualization(
    viz_data: VisualizationRequest,
    current_user: TokenPayload = Depends(get_current_user),
    session: Session = Depends(get_db_session)
):
    """Generate visualization from query.
    
    Args:
        viz_data: Visualization request data
        current_user: Current authenticated user
        session: Database session
        
    Returns:
        Visualization response
        
    Raises:
        HTTPException: If token check fails or other errors occur
    """
    # Verify company_id matches the token
    if viz_data.company_id != current_user.company_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Company ID mismatch"
        )
    
    # Check token availability
    token_available = await TokenManager.check_token_availability(
        company_id=current_user.company_id,
        session=session
    )
    
    if not token_available:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient tokens"
        )
    
    # Process the visualization request with the agent dispatcher
    dispatcher = AgentDispatcher()
    visualization = await dispatcher.generate_visualization(
        query=viz_data.query,
        company_id=current_user.company_id,
        user_id=current_user.sub,
        visualization_type=viz_data.visualization_type,
        session=session
    )
    
    # Consume tokens for visualization (typically more than a regular chat)
    await TokenManager.consume_token(
        company_id=current_user.company_id,
        session=session,
        amount=2  # Visualizations might consume more tokens
    )
    
    # Get remaining tokens
    tokens_remaining = await TokenManager.get_token_count(
        company_id=current_user.company_id,
        session=session
    )
    
    return VisualizationResponse(
        data=visualization.data,
        text_explanation=visualization.explanation,
        token_usage=2,  # In a real implementation, this would be the actual token usage
        tokens_remaining=tokens_remaining
    )


@router.post("/action", response_model=ActionResponse, responses={401: {"model": ErrorResponse}, 403: {"model": ErrorResponse}})
async def perform_action(
    action_data: ActionRequest,
    current_user: TokenPayload = Depends(get_current_user),
    session: Session = Depends(get_db_session)
):
    """Perform an action through the agent.
    
    Args:
        action_data: Action request data
        current_user: Current authenticated user
        session: Database session
        
    Returns:
        Action response
        
    Raises:
        HTTPException: If token check fails or other errors occur
    """
    # Verify company_id matches the token
    if action_data.company_id != current_user.company_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Company ID mismatch"
        )
    
    # Check token availability
    token_available = await TokenManager.check_token_availability(
        company_id=current_user.company_id,
        session=session
    )
    
    if not token_available:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient tokens"
        )
    
    # Process the action request with the agent dispatcher
    dispatcher = AgentDispatcher()
    action_result = await dispatcher.perform_action(
        action=action_data.action,
        parameters=action_data.parameters,
        company_id=current_user.company_id,
        user_id=current_user.sub,
        session=session
    )
    
    # Consume tokens for action (typically more than a regular chat)
    await TokenManager.consume_token(
        company_id=current_user.company_id,
        session=session,
        amount=3  # Actions might consume more tokens
    )
    
    # Get remaining tokens
    tokens_remaining = await TokenManager.get_token_count(
        company_id=current_user.company_id,
        session=session
    )
    
    return ActionResponse(
        success=action_result.success,
        message=action_result.message,
        data=action_result.data,
        token_usage=3,  # In a real implementation, this would be the actual token usage
        tokens_remaining=tokens_remaining
    )


@router.post("/token/check", response_model=TokenCheckResponse)
async def check_tokens(
    token_data: TokenCheckRequest,
    current_user: TokenPayload = Depends(get_current_user),
    session: Session = Depends(get_db_session)
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
