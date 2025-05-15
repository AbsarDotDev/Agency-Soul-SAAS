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
    """Standardized response format for agent-based systems."""
    
    response: str = Field(description="Text response from the agent")
    conversation_id: str = Field(description="Conversation ID")
    conversation_title: Optional[str] = Field(None, description="Conversation title")
    tokens_used: int = Field(description="Number of tokens used in this interaction")
    tokens_remaining: Optional[int] = Field(None, description="Number of tokens remaining")
    visualization: Optional[Dict[str, Any]] = Field(None, description="Visualization data if available")
    agent_type: Optional[str] = Field(None, description="Agent type that processed this request")
    
    class Config:
        """Pydantic configuration for AgentResponse."""
        
        json_encoders = {
            # Define custom JSON encoders if needed
        }
        
        # Ensure model fields are properly serialized to JSON
        json_schema_extra = {
            "example": {
                "response": "Here's your visualization of employees per department.",
                "conversation_id": "12345-abcde",
                "conversation_title": "Employee Department Analysis",
                "tokens_used": 10,
                "tokens_remaining": 990,
                "visualization": {
                    "chart_type": "pie",
                    "title": "Employees per Department",
                    "labels": ["HR", "IT", "Marketing"],
                    "datasets": [
                        {
                            "label": "Count",
                            "data": [5, 10, 7],
                            "backgroundColor": ["rgba(255,99,132,0.7)", "rgba(54,162,235,0.7)", "rgba(255,206,86,0.7)"]
                        }
                    ],
                    "options": {}
                },
                "agent_type": "visualization"
            }
        }

class VisualizationResult(BaseModel):
    """Visualization result model."""
    data: Optional[Dict[str, Any]] = Field(None, description="Chart.js compatible data structure")
    explanation: Optional[str] = Field(None, description="Natural language explanation of the visualization")
    query: str = Field(..., description="The original query that generated this visualization")
    chart_type: str = Field("bar", description="The type of chart generated (bar, line, pie, etc.)")
    tokens_used: int = Field(2, description="Tokens consumed during the visualization generation step - fixed at 2 for simplicity")

class ActionResult(BaseModel):
    """Action result model."""
    success: bool = Field(..., description="Whether the action was successful")
    message: str = Field(..., description="Result message")
    data: Optional[Dict[str, Any]] = Field(None, description="Optional action result data")

class BaseAgent(ABC):
    """Base agent class with common functionality."""
    
    def __init__(self):
        """Initialize base agent with LLM."""
        self.llm = get_llm() # Initialize LLM once for all inheriting agents
        # self.embedding_model = get_embedding_model() # Embedding model can be initialized if commonly needed by all base agents
    
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
        session: Optional[Session] = None,
        conversation_id: Optional[str] = None
    ) -> AgentResponse:
        """Generate visualization based on query.
        
        Args:
            query: Visualization query
            company_id: Company ID for data isolation
            user_id: User ID
            visualization_type: Optional visualization type (bar, line, pie, etc.)
            session: Optional database session
            conversation_id: Optional conversation ID
            
        Returns:
            Agent response with visualization data
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
        title: Optional[str] = None,
        visualization: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
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
            visualization: Optional visualization data
            
        Returns:
            The final title determined for the conversation (generated or passed in).
        """
        final_title = title # Initialize with passed-in title
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
                    agent_type VARCHAR(50) NULL,
                    tokens_used INT DEFAULT 1,
                    title VARCHAR(255) NULL,
                    visualization JSON NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX (conversation_id),
                    INDEX (company_user_id)
                )
            """))
            
            # Check if a title needs to be generated (only if no title provided AND it's the first message)
            if final_title is None:
                first_message_check = session.execute(text("""
                    SELECT id FROM agent_conversations 
                    WHERE conversation_id = :conversation_id LIMIT 1
                """), {"conversation_id": conversation_id}).scalar()
                
                if first_message_check is None: # It's the first message
                    try:
                        # llm = get_llm() # Use self.llm
                        prompt = f"""Generate a concise and informative title (around 5-7 words) that summarizes the main topic of the following conversation excerpt.
Focus on the user's primary question or goal. Avoid generic phrases.
If the user's message is very short or a generic greeting, use the agent's response to help infer a topic.

User's first message: "{message}"
Agent's first response: "{response}"

Suggested Title:"""
                        title_response = await self.llm.ainvoke(prompt) # Use self.llm
                        generated_title = title_response.content.strip()
                        
                        # Further cleaning: remove potential quotes, list markers, or "Title:" prefixes
                        generated_title = generated_title.replace('"', '').replace("'", '')
                        if generated_title.lower().startswith("title:"):
                            generated_title = generated_title[len("title:"):].strip()
                        if generated_title.startswith("- "):
                            generated_title = generated_title[2:].strip()
                            
                        final_title = generated_title[:255] # Truncate if necessary
                        
                        logger.info(f"Generated title for conversation {conversation_id}: {final_title}")
                    except Exception as title_gen_error:
                        logger.error(f"Failed to generate title for conversation {conversation_id}: {title_gen_error}")
                        final_title = message[:50] + "..." # Fallback title

            # Prepare parameters
            params = {
                "conversation_id": conversation_id,
                "company_user_id": company_id,
                "user_id": user_id,
                "message": message,
                "response": response,
                "agent_type": agent_type,
                "tokens_used": tokens_used,
                "title": final_title # Use the determined title
            }
            
            # Add visualization if provided
            if visualization:
                # Convert to JSON string if not already
                import json
                if not isinstance(visualization, str):
                    visualization_json = json.dumps(visualization)
                else:
                    visualization_json = visualization
                    
                params["visualization"] = visualization_json
                
                # Insert with visualization
                session.execute(text("""
                    INSERT INTO agent_conversations 
                    (conversation_id, company_user_id, user_id, message, response, agent_type, tokens_used, title, visualization)
                    VALUES (:conversation_id, :company_user_id, :user_id, :message, :response, :agent_type, :tokens_used, :title, :visualization)
                """), params)
                logger.info(f"Saved conversation with visualization data for conversation {conversation_id}")
            else:
                # Insert without visualization
                session.execute(text("""
                    INSERT INTO agent_conversations 
                    (conversation_id, company_user_id, user_id, message, response, agent_type, tokens_used, title)
                    VALUES (:conversation_id, :company_user_id, :user_id, :message, :response, :agent_type, :tokens_used, :title)
                """), params)
            
            session.commit()
            return final_title # Return the determined title
        except Exception as e:
            logger.error(f"Error saving conversation: {str(e)}")
            session.rollback()
            return None # Return None on error or if no title was applicable
    
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
                SELECT message, response, agent_type, created_at, visualization
                FROM agent_conversations
                WHERE conversation_id = :conversation_id
                AND company_user_id = :company_id
                ORDER BY created_at ASC
            """), {"conversation_id": conversation_id, "company_id": company_id}).fetchall()
            
            messages = []
            for row in result:
                # Add user message
                messages.append({"role": "user", "content": row[0], "timestamp": str(row[3])})
                
                # Create agent message
                agent_message = {
                    "role": "agent", 
                    "content": row[1], 
                    "timestamp": str(row[3]), 
                    "agent_type": row[2]
                }
                
                # Add visualization data if present (row[4] is visualization column)
                if row[4]:
                    try:
                        import json
                        # If stored as JSON string, parse it; if already an object, use as is
                        if isinstance(row[4], str):
                            visualization_data = json.loads(row[4])
                        else:
                            visualization_data = row[4]
                        
                        # Ensure options is an object, not an array
                        if 'options' in visualization_data and isinstance(visualization_data['options'], list) and len(visualization_data['options']) == 0:
                            visualization_data['options'] = {}
                            
                        agent_message["visualization"] = visualization_data
                        logger.info(f"Including visualization data for message in conversation {conversation_id}: {visualization_data['chart_type']}")
                    except Exception as viz_error:
                        logger.error(f"Error parsing visualization data: {viz_error}")
                
                messages.append(agent_message)

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

    async def _get_llm_response(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        """Get response from the configured LLM using stored instance."""
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=user_prompt))
        
        try:
            logger.debug(f"Sending messages to LLM: {messages}")
            response = await self.llm.ainvoke(messages)
            logger.debug(f"Received LLM response: {response.content}")
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error getting LLM response: {str(e)}", exc_info=True)
            # Return a generic error message or raise a custom exception
            return "I apologize, but I encountered an error trying to process your request."
