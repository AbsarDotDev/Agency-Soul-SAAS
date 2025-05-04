from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
import logging
from pydantic import BaseModel, Field
import uuid
from sqlalchemy import text

from app.core.config import settings
from app.core.llm import get_llm

# Set up logging
logger = logging.getLogger(__name__)

class AgentResponse(BaseModel):
    """Agent response model."""
    response: str = Field(..., description="Agent's response message")
    conversation_id: str = Field(..., description="Conversation ID")
    conversation_title: Optional[str] = Field(None, description="Title generated for the conversation")
    visualization: Optional[Dict[str, Any]] = Field(None, description="Optional visualization data")
    tokens_remaining: Optional[int] = Field(None, description="Tokens remaining after this interaction")
    tokens_used: Optional[int] = Field(None, description="Tokens consumed in this interaction")

class VisualizationResult(BaseModel):
    """Visualization result model."""
    data: Optional[Dict[str, Any]] = Field(None, description="Chart.js compatible data structure")
    explanation: Optional[str] = Field(None, description="Natural language explanation of the visualization")
    tokens_used: int = Field(2, description="Tokens consumed during the visualization generation step - fixed at 2 for simplicity")

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
        tokens_used: int = 1,
        title: Optional[str] = None
    ) -> None:
        """Save conversation to database, generating a title if needed.
        
        Args:
            session: Database session
            conversation_id: Conversation ID
            company_id: Company ID
            user_id: User ID
            message: User message
            response: Agent response
            agent_type: Agent type
            tokens_used: Tokens used
            title: Optional title (if provided, used; otherwise generated if needed)
        """
        try:
            # Check if table exists, create if not (this part is potentially inefficient)
            # Consider moving the CREATE TABLE to an initialization step if possible
            session.execute(text("""
                CREATE TABLE IF NOT EXISTS agent_conversations (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    conversation_id VARCHAR(36) NOT NULL,
                    company_user_id INT NOT NULL,
                    user_id VARCHAR(255) NOT NULL,
                    message TEXT NOT NULL,
                    response TEXT NOT NULL,
                    agent_type VARCHAR(50) NOT NULL,
                    tokens_used INT DEFAULT 1,
                    title VARCHAR(255) NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX (conversation_id),
                    INDEX (company_user_id),
                    INDEX (user_id)
                )
            """))

            final_title = title # Use provided title if available

            # Check if a title needs to be generated (only if no title provided AND it's the first message)
            if final_title is None:
                first_message_check = session.execute(text("""
                    SELECT id FROM agent_conversations 
                    WHERE conversation_id = :conversation_id LIMIT 1
                """), {"conversation_id": conversation_id}).scalar()
                
                if first_message_check is None: # It's the first message
                    try:
                        llm = get_llm() # Get LLM instance
                        prompt = f"Generate a concise title (max 5 words) for a conversation starting with:\nUser: {message}\nAgent: {response}\nTitle:"
                        title_response = await llm.ainvoke(prompt)
                        final_title = title_response.content.strip().strip('"\'')[:255] # Extract, clean, and truncate
                        logger.info(f"Generated title for conversation {conversation_id}: {final_title}")
                    except Exception as title_gen_error:
                        logger.error(f"Failed to generate title for conversation {conversation_id}: {title_gen_error}")
                        final_title = message[:50] + "..." # Fallback title

            # Insert conversation with the final title (either provided, generated, or fallback)
            session.execute(text("""
                INSERT INTO agent_conversations 
                (conversation_id, company_user_id, user_id, message, response, agent_type, tokens_used, title)
                VALUES (:conversation_id, :company_user_id, :user_id, :message, :response, :agent_type, :tokens_used, :title)
            """), {
                "conversation_id": conversation_id,
                "company_user_id": company_id,
                "user_id": user_id,
                "message": message,
                "response": response,
                "agent_type": agent_type,
                "tokens_used": tokens_used,
                "title": final_title # Use the determined title
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
            # Note: information_schema queries might not need text() but wrapping is safe
            table_exists = session.execute(text("""
                SELECT COUNT(*) 
                FROM information_schema.tables 
                WHERE table_name = 'agent_conversations'
            """)).scalar() > 0
            
            if not table_exists:
                return []
            
            # Retrieve conversation history
            result = session.execute(text("""
                SELECT message, response, created_at
                FROM agent_conversations
                WHERE conversation_id = :conversation_id
                AND company_user_id = :company_id
                ORDER BY created_at ASC
            """), {
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

    async def get_conversation_messages(self, session: Session, conversation_id: str, company_id: int) -> List[Dict[str, str]]:
        """Retrieve all messages for a specific conversation."""
        try:
            result = session.execute(text("""
                SELECT message, response, agent_type, created_at
                FROM agent_conversations
                WHERE conversation_id = :conversation_id
                AND company_user_id = :company_id
                ORDER BY created_at ASC
            """), {"conversation_id": conversation_id, "company_id": company_id}).fetchall()
            
            messages = []
            for row in result:
                messages.append({"role": "user", "content": row[0], "timestamp": str(row[3])})
                messages.append({"role": "agent", "content": row[1], "timestamp": str(row[3]), "agent_type": row[2]})

            return messages
        except Exception as e:
            logger.error(f"Error fetching messages for conversation {conversation_id}: {e}")
            return [] # Return empty list on error
    
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
            result = session.execute(text("""
                UPDATE users
                SET ai_agent_tokens_used = ai_agent_tokens_used + :tokens_used
                WHERE id = :company_id
            """), {
                "company_id": company_id,
                "tokens_used": tokens_used
            })
            
            # Check if update affected any rows
            if result.rowcount == 0:
                 # Log a warning if the update didn't change anything (e.g., company ID not found)
                 logger.warning(f"Token usage update did not affect any rows for company_id: {company_id}. User might not exist.")
                 # Potentially check if the user exists separately if this becomes an issue
                 # No need to check token/enabled status again here, assume prior checks passed
                 return False # Indicate update didn't happen as expected

            session.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error updating token usage: {str(e)}")
            session.rollback()
            return False
