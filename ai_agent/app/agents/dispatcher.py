from typing import Dict, Any, List, Optional, TypedDict, Annotated, Literal
from sqlalchemy.orm import Session
import logging
import uuid
import json
from pydantic import BaseModel, Field
from fastapi import HTTPException, status
from langchain.chat_models import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from app.agents.base_agent import AgentResponse, VisualizationResult, ActionResult
from app.agents.hrm_agent import HRMAgent
from app.agents.sales_agent import SalesAgent
from app.core.config import settings

# Set up logging
logger = logging.getLogger(__name__)


class AgentDispatcher:
    """Agent dispatcher for routing queries to specialized agents."""
    
    def __init__(self):
        """Initialize agent dispatcher."""
        # Initialize agents
        self.hrm_agent = HRMAgent()
        
        # Initialize sales_agent if it exists
        try:
            from app.agents.sales_agent import SalesAgent
            self.sales_agent = SalesAgent()
        except ImportError:
            self.sales_agent = None
            logger.warning("SalesAgent not available")
    
    async def process_message(
        self, 
        message: str, 
        company_id: int, 
        user_id: str,
        conversation_id: Optional[str] = None,
        session: Optional[Session] = None
    ) -> AgentResponse:
        """Process message and route to appropriate agent.
        
        Args:
            message: User message
            company_id: Company ID
            user_id: User ID
            conversation_id: Optional conversation ID
            session: Optional database session
            
        Returns:
            Agent response
        """
        # Route message to appropriate agent based on intent
        agent_type = await self._detect_agent_type(message, company_id, session)
        
        # Get corresponding agent
        agent = self._get_agent(agent_type)
        
        # Process message with selected agent
        response = await agent.process_message(
            message=message,
            company_id=company_id,
            user_id=user_id,
            conversation_id=conversation_id,
            session=session
        )
        
        return response
    
    async def generate_visualization(
        self, 
        query: str, 
        company_id: int, 
        user_id: str,
        visualization_type: Optional[str] = None,
        session: Optional[Session] = None
    ) -> VisualizationResult:
        """Generate visualization.
        
        Args:
            query: Visualization query
            company_id: Company ID
            user_id: User ID
            visualization_type: Optional visualization type
            session: Optional database session
            
        Returns:
            Visualization result
        """
        # Route to appropriate agent based on visualization query
        agent_type = await self._detect_agent_type(query, company_id, session)
        
        # Get corresponding agent
        agent = self._get_agent(agent_type)
        
        # Generate visualization with selected agent
        result = await agent.generate_visualization(
            query=query,
            company_id=company_id,
            user_id=user_id,
            visualization_type=visualization_type,
            session=session
        )
        
        return result
    
    async def perform_action(
        self, 
        action: str, 
        parameters: Dict[str, Any], 
        company_id: int, 
        user_id: str,
        session: Optional[Session] = None
    ) -> ActionResult:
        """Perform action.
        
        Args:
            action: Action to perform
            parameters: Action parameters
            company_id: Company ID
            user_id: User ID
            session: Optional database session
            
        Returns:
            Action result
        """
        # Route to appropriate agent based on action type
        agent_type = self._get_agent_type_for_action(action)
        
        # Get corresponding agent
        agent = self._get_agent(agent_type)
        
        # Perform action with selected agent
        result = await agent.perform_action(
            action=action,
            parameters=parameters,
            company_id=company_id,
            user_id=user_id,
            session=session
        )
        
        return result
    
    def _get_agent(self, agent_type: str):
        """Get agent instance based on type.
        
        Args:
            agent_type: Agent type
            
        Returns:
            Agent instance
            
        Raises:
            HTTPException: If agent type is not supported
        """
        agent_map = {
            "hrm": self.hrm_agent,
            "sales": self.sales_agent
        }
        
        if agent_type not in agent_map or agent_map[agent_type] is None:
            # Default to HRM agent if the requested agent is not available
            logger.warning(f"Agent type {agent_type} not available, defaulting to HRM agent")
            return self.hrm_agent
        
        return agent_map[agent_type]
    
    def _get_agent_type_for_action(self, action: str) -> str:
        """Get agent type for action.
        
        Args:
            action: Action name
            
        Returns:
            Agent type
        """
        # Map of actions to agent types
        action_map = {
            # HRM actions
            "add_employee": "hrm",
            "update_employee": "hrm",
            "delete_employee": "hrm",
            
            # Sales actions
            "add_customer": "sales",
            "update_customer": "sales",
            "delete_customer": "sales",
            "add_invoice": "sales",
            "update_invoice": "sales",
            "delete_invoice": "sales"
        }
        
        return action_map.get(action, "hrm")  # Default to HRM if action not found
    
    async def _detect_agent_type(self, message: str, company_id: int, session: Optional[Session] = None) -> str:
        """Detect agent type based on message content.
        
        Args:
            message: User message
            company_id: Company ID
            session: Optional database session
            
        Returns:
            Agent type (hrm, sales, etc.)
        """
        # Initialize LLM for routing
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=settings.GOOGLE_API_KEY)
        
        # System prompt for agent type detection
        system_prompt = """You are a routing agent for AgenySoul, an ERP/CRM/HRM system.
Your job is to determine which specialized agent should handle a user query.
Based on the user's message, respond with ONLY ONE of the following agent types:
- hrm: For queries related to human resources, employees, hiring, departments, attendance, etc.
- sales: For queries related to sales, revenue, customers, invoices, etc.

Respond with ONLY the agent type, nothing else.
"""
        
        # Get response from LLM
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=message)
            ]
            
            response = llm.invoke(messages)
            agent_type = response.content.lower().strip()
            
            # Ensure the response is one of the valid agent types
            valid_types = ["hrm", "sales"]
            if agent_type not in valid_types:
                logger.warning(f"Invalid agent type detected: {agent_type}, defaulting to hrm")
                agent_type = "hrm"
            
            return agent_type
            
        except Exception as e:
            logger.error(f"Error detecting agent type: {str(e)}")
            return "hrm"  # Default to HRM agent
