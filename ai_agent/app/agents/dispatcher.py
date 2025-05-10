from typing import Dict, Any, List, Optional, TypedDict, Annotated, Literal
from sqlalchemy.orm import Session
import logging
import uuid
import json
import re
from pydantic import BaseModel, Field
from fastapi import HTTPException, status
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from app.agents.base_agent import AgentResponse, VisualizationResult, ActionResult
from app.agents.hrm_agent import HRMAgent
from app.agents.finance_agent import FinanceAgent
from app.agents.crm_agent import CRMAgent
from app.agents.sql_agent import SQLAgent
from app.agents.project_management_agent import ProjectManagementAgent
from app.agents.product_service_agent import ProductServiceAgent
from app.core.config import settings
from app.core.llm import get_llm
from app.agents.visualization_agent import VisualizationAgent

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
        original_conversation_id: Optional[str] = None, # Renamed to avoid confusion
        session: Optional[Session] = None
    ) -> AgentResponse:
        """Process message and route to appropriate agent."""
        
        # Determine the conversation_id to be used throughout this processing
        current_conversation_id = original_conversation_id or str(uuid.uuid4())
        is_new_conversation_internally = original_conversation_id is None
        
        logger.info(f"[DISPATCHER] Starting process_message. Input ConvID: {original_conversation_id}, Using ConvID: {current_conversation_id}")

        try:
            message_lower = message.lower()
            viz_keywords = [
                "visualize", "visualization", "chart", "graph", "plot", "pie chart", 
                "bar chart", "line graph", "show me a chart", "make a graph",
                "create chart", "generate chart", "draw chart", "display chart",
                "employees per department", "department breakdown"
            ]
            
            is_visualization_request_by_keyword = any(keyword in message_lower for keyword in viz_keywords)
            
            if is_visualization_request_by_keyword:
                logger.info(f"[DISPATCHER-VIZ-PATH] Entered for message: '{message[:50]}...' (Using ConvID: {current_conversation_id})")
                sql_agent_for_viz = self._get_agent("sql")
                
                # Call generate_visualization directly
                # SQLAgent.generate_visualization will handle its own new conversation_id logic if current_conversation_id is None (it shouldn't be here)
                response_from_agent = await sql_agent_for_viz.generate_visualization(
                    query=message, company_id=company_id, user_id=user_id,
                    visualization_type=None, session=session, conversation_id=current_conversation_id
                )
                logger.info(f"[DISPATCHER-VIZ-PATH] Returned from SQLAgent.generate_visualization with ConvID: {response_from_agent.conversation_id}")
                # Ensure the agent_type is set correctly for the viz path
                response_from_agent.agent_type = "visualization"
                logger.info(f"[DISPATCHER-VIZ-PATH] Final response: {response_from_agent.dict(exclude_none=True)}")
                return response_from_agent
            
            # --- Normal Path --- 
            logger.info(f"[DISPATCHER-NORMAL-PATH] Entered for message: '{message[:50]}...' (Using ConvID: {current_conversation_id})")
            agent_to_use_type = "sql" 
            if self._is_direct_sql_query(message):
                logger.info("[DISPATCHER-NORMAL-PATH] Detected direct SQL query.")
            else:
                logger.info("[DISPATCHER-NORMAL-PATH] Detecting agent type...")
                agent_to_use_type = await self._detect_agent_type(message, company_id, session)
            
            actual_agent = self._get_agent(agent_to_use_type)
            logger.info(f"[DISPATCHER-NORMAL-PATH] Using agent: {type(actual_agent).__name__} (type: {agent_to_use_type})")
            
            # Pass the consistently used current_conversation_id
            response_from_agent = await actual_agent.process_message(
                message=message, company_id=company_id, user_id=user_id,
                conversation_id=current_conversation_id, session=session
            )
            logger.info(f"[DISPATCHER-NORMAL-PATH] Returned from {type(actual_agent).__name__}.process_message with ConvID: {response_from_agent.conversation_id}")

            # Ensure agent_type is correctly set based on this path or if viz data is present
            if not response_from_agent.agent_type:
                response_from_agent.agent_type = agent_to_use_type
            if response_from_agent.visualization and response_from_agent.agent_type != "visualization":
                response_from_agent.agent_type = "visualization"
                
            logger.info(f"[DISPATCHER-NORMAL-PATH] Final response: {response_from_agent.dict(exclude_none=True)}")
            return response_from_agent

        except Exception as e_dispatcher_main:
            logger.error(f"[DISPATCHER] CRITICAL ERROR in process_message: {str(e_dispatcher_main)}", exc_info=True)
            # Construct a very specific error response if the main dispatcher logic fails catastrophically
            # Use the consistently defined current_conversation_id for this error response
            # Do not attempt to generate a new UUID here as that might have been part of the problem
            return AgentResponse(
                response=f"Critical Dispatcher Failure: {str(e_dispatcher_main)}",
                conversation_id=current_conversation_id, # Use the ID established at the start
                conversation_title="Dispatcher Main Error",
                tokens_used=0,
                tokens_remaining=None, 
                visualization=None,
                agent_type="error_dispatcher_main"
            )
    
    async def generate_visualization(
        self, 
        query: str, 
        company_id: int, 
        user_id: str,
        visualization_type: Optional[str] = None,
        session: Optional[Session] = None,
        conversation_id: Optional[str] = None
    ) -> AgentResponse:
        """Generate visualization.
        
        Args:
            query: Visualization query
            company_id: Company ID
            user_id: User ID
            visualization_type: Optional visualization type
            session: Optional database session
            conversation_id: Optional conversation ID for continuity
            
        Returns:
            AgentResponse with visualization data
        """
        logger.info(f"Generating visualization for query: {query[:50]}{'...' if len(query) > 50 else ''}")
        
        # Always use visualization agent for this method
        agent = self._get_agent("visualization")
        
        try:
            # Process visualization - now returns AgentResponse
            response = await agent.generate_visualization(
                query=query,
                company_id=company_id,
                user_id=user_id,
                visualization_type=visualization_type,
                session=session,
                conversation_id=conversation_id
            )
            
            # Ensure agent_type is set
            if not response.agent_type:
                response.agent_type = "visualization"
                logger.info("Setting missing agent_type to 'visualization' in dispatcher")
                
            # Log response details for debugging
            if response.visualization:
                viz_info = {
                    "chart_type": response.visualization.get("chart_type", "unknown"),
                    "has_labels": "labels" in response.visualization,
                    "has_datasets": "datasets" in response.visualization
                }
                logger.info(f"Dispatcher returning visualization response: {viz_info}")
            else:
                logger.error("No visualization data in response from visualization agent - this should never happen")
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating visualization: {str(e)}", exc_info=True)
            # Create a proper AgentResponse for error cases
            return AgentResponse(
                response=f"An error occurred while generating the visualization: {str(e)}",
                conversation_id=conversation_id or str(uuid.uuid4()),
                conversation_title=None,
                visualization=None,
                tokens_remaining=None,
                tokens_used=0,
                agent_type="visualization"  # Set correct agent type even for errors
            )
    
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
                self.hrm_agent = HRMAgent()
            return self.hrm_agent
        
        elif agent_type == "sales":
            # Route sales queries to finance agent
            if not hasattr(self, 'finance_agent') or self.finance_agent is None:
                self.finance_agent = FinanceAgent()
            logger.info("Routing sales query to Finance agent (Sales agent has been removed)")
            return self.finance_agent
        
        elif agent_type == "finance":
            if not hasattr(self, 'finance_agent') or self.finance_agent is None:
                self.finance_agent = FinanceAgent()
            return self.finance_agent
        
        elif agent_type == "crm":
            if not hasattr(self, 'crm_agent') or self.crm_agent is None:
                self.crm_agent = CRMAgent()
            return self.crm_agent
        
        elif agent_type == "project":
            if not hasattr(self, 'project_agent') or self.project_agent is None:
                self.project_agent = ProjectManagementAgent()
            return self.project_agent
        
        elif agent_type == "product":
            if not hasattr(self, 'product_agent') or self.product_agent is None:
                self.product_agent = ProductServiceAgent()
            return self.product_agent
        
        elif agent_type == "sql":
            if not hasattr(self, 'sql_agent') or self.sql_agent is None:
                self.sql_agent = SQLAgent()
            return self.sql_agent
        
        elif agent_type == "visualization":
            if not hasattr(self, 'visualization_agent') or self.visualization_agent is None:
                self.visualization_agent = VisualizationAgent()
            return self.visualization_agent
        
        else:
            # Default to SQL agent if the requested agent is not available
            logger.warning(f"Agent type {agent_type} not available, defaulting to SQL agent")
            if not hasattr(self, 'sql_agent') or self.sql_agent is None:
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
            
            # Finance actions (including sales-related actions)
            "add_customer": "finance",
            "update_customer": "finance",
            "delete_customer": "finance",
            "add_invoice": "finance",
            "update_invoice": "finance",
            "delete_invoice": "finance",
            "add_lead": "finance",
            "update_lead": "finance",
            "add_deal": "finance",
            "update_deal": "finance",
            "add_expense": "finance",
            "update_expense": "finance",
            "add_revenue": "finance",
            "update_revenue": "finance",
            "add_budget": "finance",
            "update_budget": "finance",
            
            # CRM actions
            "add_ticket": "crm",
            "update_ticket": "crm",
            "add_contract": "crm",
            "update_contract": "crm",
            
            # Project actions
            "add_project": "project",
            "update_project": "project",
            "add_task": "project",
            "update_task": "project",
            "delete_task": "project",
            "add_milestone": "project",
            "update_milestone": "project",
            
            # Product actions
            "add_product": "product",
            "update_product": "product",
            "delete_product": "product",
            "add_inventory": "product",
            "update_inventory": "product",
            "add_warehouse": "product",
            "update_warehouse": "product",
            
            # SQL actions (general)
            "run_query": "sql",
            "list_tables": "sql",
            "get_schema": "sql",
        }
        
        # Return mapped agent type or default to sql
        return action_map.get(action, "sql")
    
    def _is_direct_sql_query(self, message: str) -> bool:
        """Check if message is a direct SQL query.
        
        Args:
            message: User message
            
        Returns:
            True if message is a direct SQL query
        """
        # Convert to lowercase for case-insensitive matching
        message_lower = message.lower()
        
        # Check for explicit SQL indicators
        sql_keywords = ["select", "from", "where", "join", "order by", "group by", "having", "limit"]
        message_starts_with_sql = any(message_lower.strip().startswith(keyword) for keyword in sql_keywords)
        
        # Check for SQL request keywords
        request_patterns = [
            "run sql", "execute sql", "sql query", "write a query", "using sql", "in sql", "database query",
            "can you query", "query the database", "run a select", "select all", "select * from"
        ]
        has_sql_request = any(pattern in message_lower for pattern in request_patterns)
        
        # Check for table exploration
        exploration_patterns = [
            "show tables", "show all tables", "list tables", "what tables", "describe table", 
            "show columns", "table schema", "table structure", "show me the schema"
        ]
        is_exploration = any(pattern in message_lower for pattern in exploration_patterns)
        
        return message_starts_with_sql or has_sql_request or is_exploration
    
    async def _detect_agent_type(self, message: str, company_id: int, session: Optional[Session] = None) -> str:
        """Detect agent type based on message content.
        
        Args:
            message: User message
            company_id: Company ID
            session: Optional database session
            
        Returns:
            Agent type
        """
        # Use LLM to classify the message
        system_prompt = """You are a message classifier for a business AI assistant.
Your task is to categorize each message into one of the predefined agent types based on its content.

Valid agent types:
1. "hrm" - For queries about employees, departments, attendance, leave, training, etc.
2. "finance" - For queries about expenses, revenues, invoices, bills, budgets, financial reports, sales, deals, customers, etc.
3. "crm" - For queries about customer relationships, support tickets, etc.
4. "project" - For queries about projects, tasks, milestones, time tracking, etc.
5. "product" - For queries about products, services, inventory, warehouses, etc.
6. "sql" - For general database or data analysis queries that don't clearly fit elsewhere

Classification Rules:
- Choose the MOST specific agent type based on message content
- If a message spans multiple domains, pick the PRIMARY focus
- Default to "sql" only if the message doesn't clearly fit others
- NOTE: All sales-related queries (invoices, deals, revenue, customers) should be "finance"

EXAMPLES:
Message: "How many employees are in the Marketing department?"
Classification: "hrm"

Message: "What was our revenue in Q1 vs Q2?"
Classification: "finance"

Message: "Show me open customer support tickets"
Classification: "crm"

Message: "What's the status of the website redesign project?"
Classification: "project"

Message: "How many units of Widget X do we have in stock?"
Classification: "product"

Message: "Show me our sales pipeline for this quarter"
Classification: "finance"

Message: "Can you analyze our data trends over the past year?"
Classification: "sql"

RESPOND ONLY WITH THE AGENT TYPE NAME: "hrm", "finance", "crm", "project", "product", or "sql"
"""
        # Get LLM response with the message to classify
        llm = get_llm()
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Classify this message: {message}")
        ])
        
        # Extract the agent type from the response
        agent_type = response.content.strip().lower()
        
        # Validate the agent type
        valid_types = ["hrm", "finance", "crm", "project", "product", "sql"]
        if agent_type not in valid_types:
            agent_type = "sql"  # Default to SQL if classification is invalid
        
        # Route "sales" to "finance" if it somehow came through
        if agent_type == "sales":
            agent_type = "finance"
            
        # Log the classification
        logger.info(f"Classified message as '{agent_type}': {message[:50]}...")
        
        return agent_type
