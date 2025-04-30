from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
import logging
from pydantic import BaseModel, Field
import uuid

from app.core.config import settings

# Set up logging
logger = logging.getLogger(__name__)

class AgentResponse(BaseModel):
    """Agent response model."""
    message: str = Field(..., description="Response message")
    conversation_id: str = Field(..., description="Conversation ID")
    visualization: Optional[Dict[str, Any]] = Field(None, description="Optional visualization data")

class VisualizationResult(BaseModel):
    """Visualization result model."""
    data: Optional[Dict[str, Any]] = Field(None, description="Visualization data")
    explanation: str = Field(..., description="Explanation of the visualization")

class ActionResult(BaseModel):
    """Action result model."""
    success: bool = Field(..., description="Whether the action was successful")
    message: str = Field(..., description="Result message")
    data: Optional[Dict[str, Any]] = Field(None, description="Optional action result data")

class BaseAgent(ABC):
    """Base agent class with common functionality."""
    
    @abstractmethod
    async def process_message(
        self, 
        message: str, 
        company_id: int, 
        user_id: str,
        conversation_id: Optional[str] = None,
        session: Optional[Session] = None
    ) -> AgentResponse:
        """Process a message from a user.
        
        Args:
            message: User message
            company_id: Company ID for data isolation
            user_id: User ID
            conversation_id: Optional conversation ID
            session: Optional database session
            
        Returns:
            Agent response
        """
        pass
    
    @abstractmethod
    async def generate_visualization(
        self, 
        query: str, 
        company_id: int, 
        user_id: str,
        visualization_type: Optional[str] = None,
        session: Optional[Session] = None
    ) -> VisualizationResult:
        """Generate visualization based on query.
        
        Args:
            query: Visualization query
            company_id: Company ID for data isolation
            user_id: User ID
            visualization_type: Optional visualization type (bar, line, pie, etc.)
            session: Optional database session
            
        Returns:
            Visualization result
        """
        pass
    
    def _get_current_date(self) -> str:
        """Get current date string.
        
        Returns:
            Current date string
        """
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d")
    
    async def _save_conversation(
        self,
        session: Session,
        conversation_id: str,
        company_id: int,
        user_id: str,
        message: str,
        response: str,
        agent_type: str,
        tokens_used: int = 1
    ) -> None:
        """Save conversation to database.
        
        Args:
            session: Database session
            conversation_id: Conversation ID
            company_id: Company ID
            user_id: User ID
            message: User message
            response: Agent response
            agent_type: Agent type
            tokens_used: Tokens used
        """
        try:
            # Check if table exists, create if not
            session.execute("""
                CREATE TABLE IF NOT EXISTS agent_conversations (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    conversation_id VARCHAR(36) NOT NULL,
                    company_user_id INT NOT NULL,
                    user_id VARCHAR(255) NOT NULL,
                    message TEXT NOT NULL,
                    response TEXT NOT NULL,
                    agent_type VARCHAR(50) NOT NULL,
                    tokens_used INT DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX (conversation_id),
                    INDEX (company_user_id),
                    INDEX (user_id)
                )
            """)
            
            # Insert conversation
            session.execute("""
                INSERT INTO agent_conversations 
                (conversation_id, company_user_id, user_id, message, response, agent_type, tokens_used)
                VALUES (:conversation_id, :company_user_id, :user_id, :message, :response, :agent_type, :tokens_used)
            """, {
                "conversation_id": conversation_id,
                "company_user_id": company_id,
                "user_id": user_id,
                "message": message,
                "response": response,
                "agent_type": agent_type,
                "tokens_used": tokens_used
            })
            
            session.commit()
        except Exception as e:
            logger.error(f"Error saving conversation: {str(e)}")
            session.rollback()
    
    async def _get_conversation_history(
        self,
        conversation_id: str,
        company_id: int,
        session: Session
    ) -> List[Dict[str, Any]]:
        """Get conversation history for the given conversation ID and company ID.
        
        Args:
            conversation_id: Conversation ID
            company_id: Company ID
            session: Database session
            
        Returns:
            List of conversation messages
        """
        try:
            # First check if the table exists
            table_exists = session.execute("""
                SELECT COUNT(*) 
                FROM information_schema.tables 
                WHERE table_name = 'agent_conversations'
            """).scalar() > 0
            
            if not table_exists:
                return []
            
            # Retrieve conversation history
            result = session.execute("""
                SELECT message, response, created_at
                FROM agent_conversations
                WHERE conversation_id = :conversation_id
                AND company_user_id = :company_id
                ORDER BY created_at ASC
            """, {
                "conversation_id": conversation_id,
                "company_id": company_id
            }).fetchall()
            
            # Convert to list of dictionaries
            conversation_history = []
            for row in result:
                conversation_history.append({
                    "message": row[0],
                    "response": row[1],
                    "created_at": row[2]
                })
            
            return conversation_history
            
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {str(e)}")
            return []
    
    async def _update_token_usage(
        self,
        session: Session,
        company_id: int,
        tokens_used: int
    ) -> bool:
        """Update token usage for the company.
        
        Args:
            session: Database session
            company_id: Company ID
            tokens_used: Tokens used
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Update token usage
            result = session.execute("""
                UPDATE users
                SET ai_agent_tokens_used = ai_agent_tokens_used + :tokens_used
                WHERE id = :company_id
                AND ai_agent_enabled = 1
                AND ai_agent_tokens_allocated > ai_agent_tokens_used + :tokens_used
            """, {
                "company_id": company_id,
                "tokens_used": tokens_used
            })
            
            # Check if update was successful
            if result.rowcount == 0:
                # Check if the company has enough tokens
                company = session.execute("""
                    SELECT ai_agent_enabled, ai_agent_tokens_allocated, ai_agent_tokens_used
                    FROM users
                    WHERE id = :company_id
                """, {
                    "company_id": company_id
                }).fetchone()
                
                if not company:
                    logger.error(f"Company {company_id} not found")
                    return False
                
                if not company[0]:  # ai_agent_enabled
                    logger.error(f"AI agent not enabled for company {company_id}")
                    return False
                
                if company[1] <= company[2] + tokens_used:  # ai_agent_tokens_allocated <= ai_agent_tokens_used + tokens_used
                    logger.error(f"Company {company_id} does not have enough tokens")
                    return False
                
                logger.error(f"Unknown error updating token usage for company {company_id}")
                return False
            
            session.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error updating token usage: {str(e)}")
            session.rollback()
            return False
