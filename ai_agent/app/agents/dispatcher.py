from typing import Dict, Any, List, Optional, TypedDict, Annotated, Literal
from sqlalchemy.orm import Session
import logging
import uuid
import json
import re
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
from app.core.llm import get_llm

# Set up logging
logger = logging.getLogger(__name__)


class AgentDispatcher:
    """Agent dispatcher for routing queries to specialized agents."""
    
    def __init__(self):
        """Initialize agent dispatcher with lazy-loading of agents."""
        # Don't initialize any agents yet - they'll be loaded on demand
        pass
    
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
        # Check if this is an SQL-specific query 
        if self._is_direct_sql_query(message):
            # Direct SQL queries or data exploration should always go to SQL agent
            agent_type = "sql"
            logger.info("Detected direct SQL query, routing to SQL agent")
        else:
            # For other queries, use routing logic
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
        # Check if this is an SQL-specific visualization query
        if self._is_direct_sql_query(query) or "database" in query.lower() or "table" in query.lower():
            # Direct SQL-based visualizations should go to SQL agent
            agent_type = "sql"
        else:
            # For other visualization requests, use standard routing
            agent_type = await self._detect_agent_type(query, company_id, session)
        
        # Log the routing decision
        logger.info(f"Routing visualization request to {agent_type} agent: {query[:50]}...")
        
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
        # Initialize agents on demand to avoid unnecessary loading
        if agent_type == "hrm":
            if not hasattr(self, 'hrm_agent') or self.hrm_agent is None:
                from app.agents.hrm_agent import HRMAgent
                self.hrm_agent = HRMAgent()
            return self.hrm_agent
        
        elif agent_type == "sales":
            if not hasattr(self, 'sales_agent') or self.sales_agent is None:
                from app.agents.sales_agent import SalesAgent
                self.sales_agent = SalesAgent()
            return self.sales_agent
        
        elif agent_type == "finance":
            if not hasattr(self, 'finance_agent') or self.finance_agent is None:
                from app.agents.finance_agent import FinanceAgent
                self.finance_agent = FinanceAgent()
            return self.finance_agent
        
        elif agent_type == "crm":
            if not hasattr(self, 'crm_agent') or self.crm_agent is None:
                from app.agents.crm_agent import CRMAgent
                self.crm_agent = CRMAgent()
            return self.crm_agent
        
        elif agent_type == "sql":
            if not hasattr(self, 'sql_agent') or self.sql_agent is None:
                from app.agents.sql_agent import SQLAgent
                self.sql_agent = SQLAgent()
            return self.sql_agent
        
        else:
            # Default to SQL agent if the requested agent is not available
            logger.warning(f"Agent type {agent_type} not available, defaulting to SQL agent")
            if not hasattr(self, 'sql_agent') or self.sql_agent is None:
                from app.agents.sql_agent import SQLAgent
                self.sql_agent = SQLAgent()
            return self.sql_agent
    
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
            "add_department": "hrm",
            "update_department": "hrm",
            "delete_department": "hrm",
            "add_attendance": "hrm",
            "update_attendance": "hrm",
            
            # Sales actions
            "add_customer": "sales",
            "update_customer": "sales",
            "delete_customer": "sales",
            "add_invoice": "sales",
            "update_invoice": "sales",
            "delete_invoice": "sales",
            "add_lead": "sales",
            "update_lead": "sales",
            "add_deal": "sales",
            "update_deal": "sales",
            
            # Finance actions
            "add_expense": "finance",
            "update_expense": "finance",
            "add_revenue": "finance",
            "update_revenue": "finance",
            "add_budget": "finance",
            "update_budget": "finance",
            
            # CRM actions
            "add_ticket": "crm",
            "update_ticket": "crm",
            "add_contact": "crm",
            "update_contact": "crm",
            
            # SQL/Database actions
            "run_query": "sql",
            "get_table_schema": "sql",
            "list_tables": "sql",
            "generate_report": "sql"
        }
        
        return action_map.get(action, "sql")  # Default to SQL if action not found
    
    def _is_direct_sql_query(self, message: str) -> bool:
        """Check if the message is a direct SQL query or explicit database exploration request.
        
        Args:
            message: User message
            
        Returns:
            True if message is a direct SQL query or database exploration request
        """
        message_lower = message.lower()
        
        # Check for SQL keywords in the message
        sql_keywords = ["select", "from", "where", "group by", "order by", "join", "limit", "having"]
        has_sql_syntax = any(keyword in message_lower for keyword in sql_keywords)
        
        # Check for database exploration intent
        db_exploration_patterns = [
            r"show\s+(?:me\s+)?(?:all\s+)?(?:the\s+)?tables",
            r"list\s+(?:all\s+)?(?:the\s+)?tables",
            r"what\s+tables\s+(?:do\s+you\s+have|are\s+available|exist)",
            r"show\s+(?:me\s+)?(?:the\s+)?schema\s+(?:of|for)",
            r"describe\s+(?:the\s+)?table",
            r"what\s+columns\s+(?:does|do)\s+the\s+table\s+\w+\s+have",
            r"(?:show|what\s+is|tell\s+me)\s+(?:about\s+)?the\s+structure\s+of"
        ]
        
        # Check for database exploration patterns
        has_db_exploration_intent = any(re.search(pattern, message_lower) for pattern in db_exploration_patterns)
        
        return has_sql_syntax or has_db_exploration_intent
    
    async def _detect_agent_type(self, message: str, company_id: int, session: Optional[Session] = None) -> str:
        """Detect agent type based on message content.
        
        Args:
            message: User message
            company_id: Company ID
            session: Optional database session
            
        Returns:
            Agent type (hrm, sales, sql, finance, etc.)
        """
        # Get LLM from the central configuration
        llm = get_llm()
        
        # Check for visualization-related keywords first
        visualization_keywords = [
            "graph", "chart", "plot", "visualize", "visualization", 
            "show me", "display", "diagram", "histogram", "bar chart", 
            "line graph", "pie chart", "scatter plot"
        ]
        
        # Check if this is likely a visualization request
        is_visualization_request = any(keyword in message.lower() for keyword in visualization_keywords)
        
        # System prompt for agent type detection
        system_prompt = """You are a routing agent for AgenySoul, an ERP/CRM/HRM system.
Your job is to determine which specialized agent should handle a user query.
Based on the user's message, respond with ONLY ONE of the following agent types:

- hrm: For queries related to human resources, employees, hiring, departments, attendance, benefits, performance reviews, etc.
- sales: For queries related to sales, revenue forecasting, customers, leads, sales performance, etc.
- finance: For queries related to accounting, financial reports, expenses, profits, budgets, etc.
- crm: For queries related to customer relationships, customer support, satisfaction, engagement, etc.
- sql: For database queries requesting specific aggregate data or complex analysis that doesn't clearly fit other categories.

Respond with ONLY the agent type, nothing else.
"""
        
        # If it's a visualization request, use slightly different prompt
        if is_visualization_request:
            system_prompt = """You are a routing agent for AgenySoul, an ERP/CRM/HRM system.
Your job is to determine which specialized agent should handle a data visualization request.
Based on the user's message, respond with ONLY ONE of the following agent types:

- hrm: For visualizations related to human resources, employees, hiring, departments, attendance, benefits, etc.
- sales: For visualizations related to sales figures, revenue forecasting, customer acquisition, etc.
- finance: For visualizations related to financial data, expenses, profits, budgets, etc.
- crm: For visualizations related to customer relationships, support metrics, satisfaction, etc.
- sql: For more general data visualizations or complex analysis spanning multiple business areas.

Respond with ONLY the agent type, nothing else.
"""
        
        # Get response from LLM
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=message)
            ]
            
            response = await llm.ainvoke(messages)
            agent_type = response.content.lower().strip()
            
            # Ensure the response is one of the valid agent types
            valid_types = ["hrm", "sales", "finance", "crm", "sql"]
            if agent_type not in valid_types:
                logger.warning(f"Invalid agent type detected: {agent_type}, defaulting to sql")
                agent_type = "sql"
            
            logger.info(f"Detected agent type: {agent_type} for message: {message[:50]}...")
            return agent_type
            
        except Exception as e:
            logger.error(f"Error detecting agent type: {str(e)}")
            return "sql"  # Default to SQL agent
