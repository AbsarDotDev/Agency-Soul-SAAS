from typing import Dict, Any, List, Optional, TypedDict, Annotated, Literal, Union, Tuple, cast
import logging
import json
import re
from enum import Enum
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
import uuid
import asyncio

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ChatMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from app.agents.base_agent import AgentResponse, VisualizationResult, ActionResult
from app.agents.sql_agent import SQLAgent
from app.agents.hrm_agent import HRMAgent
from app.agents.finance_agent import FinanceAgent
from app.agents.crm_agent import CRMAgent
from app.agents.project_management_agent import ProjectManagementAgent
from app.agents.product_service_agent import ProductServiceAgent
from app.core.llm import get_llm
from app.core.config import settings

# Set up logging
logger = logging.getLogger(__name__)

# Define agent types
class AgentType(str, Enum):
    """Enum for agent types."""
    SQL = "sql"
    HRM = "hrm"
    FINANCE = "finance"
    CRM = "crm"
    SALES = "sales"  # Kept for backward compatibility, but redirects to FINANCE
    PROJECT = "project"  # Added project agent type
    PRODUCT = "product"  # Added product agent type

# Define the state schema
class AgentState(TypedDict):
    """State definition for the LangGraph."""
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]
    user_info: Dict[str, Any]  # Holds user_id, company_id, etc.
    conversation_id: str
    current_agent: Optional[AgentType]
    agent_attempts: Dict[AgentType, int]  # Track attempts per agent type
    final_response: Optional[AgentResponse]
    visualization_result: Optional[VisualizationResult]
    action_result: Optional[ActionResult]
    needs_visualization: bool
    visualization_type: Optional[str]
    is_action: bool
    action_name: Optional[str]
    action_parameters: Optional[Dict[str, Any]]
    tokens_used: int
    next: Optional[Literal["sql", END]]
    next_node: Optional[str]  # Used for routing from router node

# Create agent instances once
_agent_instances = {}

def get_agent(agent_type: AgentType):
    """Get agent instance by type using singleton pattern."""
    # Redirect SALES to FINANCE
    if agent_type == AgentType.SALES:
        logger.info("Sales agent type redirected to Finance agent (Sales agent has been removed)")
        agent_type = AgentType.FINANCE
    
    if agent_type not in _agent_instances:
        if agent_type == AgentType.SQL:
            _agent_instances[agent_type] = SQLAgent()
        elif agent_type == AgentType.HRM:
            _agent_instances[agent_type] = HRMAgent()
        elif agent_type == AgentType.FINANCE:
            _agent_instances[agent_type] = FinanceAgent()
        elif agent_type == AgentType.CRM:
            _agent_instances[agent_type] = CRMAgent()
        elif agent_type == AgentType.PROJECT:
            _agent_instances[agent_type] = ProjectManagementAgent()
        elif agent_type == AgentType.PRODUCT:
            _agent_instances[agent_type] = ProductServiceAgent()
    
    return _agent_instances[agent_type]

# Router logic
async def router_node(state: AgentState) -> AgentState:
    """Route the query to the appropriate agent."""
    messages = state["messages"]
    user_info = state["user_info"]
    
    # Get only user messages for analysis
    user_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]
    if not user_messages:
        state["next_node"] = AgentType.SQL  # Default to SQL if no user messages
        return state
    
    # Get the last user message
    last_user_message = user_messages[-1].content
    
    # Special case handling
    # Check for visualization intent
    if state["needs_visualization"]:
        state["next_node"] = "handle_visualization"
        return state
    
    # Check for action intent
    if state["is_action"]:
        state["next_node"] = "perform_action"
        return state
    
    # Check for direct SQL patterns first
    if is_direct_sql_query(last_user_message):
        state["next_node"] = AgentType.SQL
        return state
    
    # Use LLM to detect the intent for more complex queries
    llm = get_llm()
    
    # Prepare prompt for agent selection
    system_prompt = """You are a routing assistant that determines which specialized agent should handle a user query.

Available agents:
1. HRM Agent: Handles queries about employees, departments, attendance, leave, payroll, assets, appraisals, awards, trainings, etc.
2. Finance Agent: Handles queries about finance, accounting, chart of accounts, transactions, bills, payments, expenses, revenues, budgets, banks, etc.
3. CRM Agent: Handles queries about customers, support tickets, contracts, leads, deals, pipelines, etc.
4. Project Agent: Handles queries about projects, tasks, milestones, project progress, project teams, project timelines, etc.
5. Product Agent: Handles queries about products, product stocks, services, inventory, warehouses, quotations, point of sale (POS), orders, purchases, etc.
6. SQL Agent: Handles general database queries or anything that doesn't clearly fit the above categories.

IMPORTANT ROUTING RULES:
- All product-related queries (products, inventory, warehouses, quotations, POS, orders, stocks, etc.) should be directed to the Product Agent.
- All sales-related queries (deals, leads, sales pipelines, etc.) should be directed to the CRM Agent.
- All project-related queries (projects, tasks, milestones, etc.) should be directed to the Project Agent.
- All customer-focused activities (leads, deals, customer interactions) should go to the CRM Agent.
- All employee-related queries (including training) should go to the HRM Agent.
- If the message mentions SQL, database, tables, or specific data querying actions, choose the SQL Agent.

ALWAYS select the most appropriate agent based on the user's query content and intent.
If the query doesn't clearly match any specialized agent or involves multiple domains, choose the SQL Agent.

Output ONLY the agent type (hrm, finance, crm, project, product, or sql) without any explanation."""
    
    messages_for_routing = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=last_user_message)
    ]
    
    try:
        response = await llm.ainvoke(messages_for_routing)
        agent_type = response.content.strip().lower()
        
        # Validate and map to enum
        if agent_type == "hrm":
            state["next_node"] = AgentType.HRM
        elif agent_type == "finance":
            state["next_node"] = AgentType.FINANCE
        elif agent_type == "crm":
            state["next_node"] = AgentType.CRM
        elif agent_type == "project":
            state["next_node"] = AgentType.PROJECT
        elif agent_type == "product":
            state["next_node"] = AgentType.PRODUCT
        elif agent_type == "sales":
            # Redirect sales to CRM
            logger.info("Sales agent type redirected to CRM agent")
            state["next_node"] = AgentType.CRM
        else:
            # Default to SQL for any invalid or unrecognized response
            state["next_node"] = AgentType.SQL
            
    except Exception as e:
        logger.error(f"Error in router: {str(e)}")
        state["next_node"] = AgentType.SQL  # Default to SQL on error
    
    return state

def is_direct_sql_query(message: str) -> bool:
    """Check if the message contains direct SQL query patterns."""
    # List of SQL-specific keywords and patterns
    sql_patterns = [
        r'\bSELECT\b.*\bFROM\b',
        r'\bINSERT\s+INTO\b',
        r'\bUPDATE\b.*\bSET\b',
        r'\bDELETE\s+FROM\b',
        r'\bJOIN\b',
        r'\bWHERE\b',
        r'\bGROUP\s+BY\b',
        r'\bORDER\s+BY\b',
        r'\bHAVING\b',
        r'\bLIMIT\b',
        r'\bUNION\b',
        r'\bINNER\s+JOIN\b',
        r'\bLEFT\s+JOIN\b',
        r'\bRIGHT\s+JOIN\b',
        r'\bOUTER\s+JOIN\b',
        r'\bVALUES\s*\(',
        r'\bCOUNT\s*\(',
        r'\bSUM\s*\(',
        r'\bAVG\s*\(',
        r'\bMAX\s*\(',
        r'\bMIN\s*\(',
        r'database schema',
        r'SQL query',
        r'table structure',
        r'primary key',
        r'foreign key',
    ]
    
    # Database/data exploration specific keywords
    data_exploration_terms = [
        r'\bquery the database\b',
        r'\bquery database\b',
        r'\breturn all\b.*\bfrom database\b',
        r'\bshow me all\b.*\bin the database\b',
        r'\btables in the database\b',
        r'\bschema of\b',
        r'\bstructure of\b.*\btable\b',
        r'\bcolumns in\b',
        r'\bdata from\b.*\btable\b',
    ]
    
    # Check for direct SQL patterns (case insensitive)
    for pattern in sql_patterns + data_exploration_terms:
        if re.search(pattern, message, re.IGNORECASE):
            return True
    
    return False

# SQL Agent node
async def sql_agent_node(state: AgentState) -> AgentState:
    """Process the message with SQL agent."""
    messages = state["messages"]
    user_info = state["user_info"]
    conversation_id = state["conversation_id"]
    
    # Get only the last user message
    user_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]
    if not user_messages:
        state["final_response"] = AgentResponse(
            response="I couldn't find a query to process.",
            conversation_id=conversation_id,
            agent_type=AgentType.SQL
        )
        return state
    
    last_user_message = user_messages[-1].content
    
    # Get SQL agent and process message
    sql_agent = get_agent(AgentType.SQL)
    
    # Check if the session is in the user_info
    session = user_info.get("session", None)
    
    response = await sql_agent.process_message(
        message=last_user_message,
        company_id=user_info["company_id"],
        user_id=user_info["user_id"],
        conversation_id=conversation_id,
        session=session
    )
    
    # Track tokens used
    state["tokens_used"] += response.tokens_used if response.tokens_used else 0
    
    # Update state with response and ensure agent_type is set
    response.agent_type = AgentType.SQL
    state["final_response"] = response
    state["agent_attempts"][AgentType.SQL] = state["agent_attempts"].get(AgentType.SQL, 0) + 1
    
    # Add agent response to messages
    new_messages = add_messages(messages, [AIMessage(content=response.response)])
    state["messages"] = new_messages
    
    return state

# HRM Agent node
async def hrm_agent_node(state: AgentState) -> AgentState:
    """Process the message with HRM agent."""
    messages = state["messages"]
    user_info = state["user_info"]
    conversation_id = state["conversation_id"]
    
    # Get only the last user message
    user_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]
    if not user_messages:
        state["next"] = AgentType.SQL  # Fall back to SQL
        return state
    
    last_user_message = user_messages[-1].content
    
    # Set current agent
    state["current_agent"] = AgentType.HRM
    
    try:
        # Get HRM agent and process message
        hrm_agent = get_agent(AgentType.HRM)
        
        # Check if the session is in the user_info
        session = user_info.get("session", None)
        
        response = await hrm_agent.process_message(
            message=last_user_message,
            company_id=user_info["company_id"],
            user_id=user_info["user_id"],
            conversation_id=conversation_id,
            session=session
        )
        
        # Track tokens used
        state["tokens_used"] += response.tokens_used if response.tokens_used else 0
        
        # Ensure agent_type is set
        response.agent_type = AgentType.HRM
        
        # Update state with response
        state["final_response"] = response
        state["agent_attempts"][AgentType.HRM] = state["agent_attempts"].get(AgentType.HRM, 0) + 1
        
        # Add agent response to messages
        new_messages = add_messages(messages, [AIMessage(content=response.response)])
        state["messages"] = new_messages
        
        # End this path
        state["next"] = END
        
        return state
        
    except Exception as e:
        logger.error(f"Error in HRM agent: {str(e)}")
        # Fallback to SQL agent on error
        state["current_agent"] = AgentType.SQL
        state["next"] = AgentType.SQL
        return state

# Finance Agent node
async def finance_agent_node(state: AgentState) -> AgentState:
    """Process the message with Finance agent."""
    messages = state["messages"]
    user_info = state["user_info"]
    conversation_id = state["conversation_id"]
    
    # Get only the last user message
    user_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]
    if not user_messages:
        state["next"] = AgentType.SQL  # Fall back to SQL
        return state
    
    last_user_message = user_messages[-1].content
    
    # Set current agent
    state["current_agent"] = AgentType.FINANCE
    
    try:
        # Get Finance agent and process message
        finance_agent = get_agent(AgentType.FINANCE)
        
        # Check if the session is in the user_info
        session = user_info.get("session", None)
        
        response = await finance_agent.process_message(
            message=last_user_message,
            company_id=user_info["company_id"],
            user_id=user_info["user_id"],
            conversation_id=conversation_id,
            session=session
        )
        
        # Track tokens used
        state["tokens_used"] += response.tokens_used if response.tokens_used else 0
        
        # Ensure agent_type is set
        response.agent_type = AgentType.FINANCE
        
        # Update state with response
        state["final_response"] = response
        state["agent_attempts"][AgentType.FINANCE] = state["agent_attempts"].get(AgentType.FINANCE, 0) + 1
        
        # Add agent response to messages
        new_messages = add_messages(messages, [AIMessage(content=response.response)])
        state["messages"] = new_messages
        
        # End this path
        state["next"] = END
        
        return state
        
    except Exception as e:
        logger.error(f"Error in Finance agent: {str(e)}")
        # Fallback to SQL agent on error
        state["current_agent"] = AgentType.SQL
        state["next"] = AgentType.SQL
        return state

# CRM Agent node
async def crm_agent_node(state: AgentState) -> AgentState:
    """Process the message with CRM agent."""
    messages = state["messages"]
    user_info = state["user_info"]
    conversation_id = state["conversation_id"]
    
    # Get only the last user message
    user_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]
    if not user_messages:
        state["next"] = AgentType.SQL  # Fall back to SQL
        return state
    
    last_user_message = user_messages[-1].content
    
    # Set current agent
    state["current_agent"] = AgentType.CRM
    
    try:
        # Get CRM agent and process message
        crm_agent = get_agent(AgentType.CRM)
        
        # Check if the session is in the user_info
        session = user_info.get("session", None)
        
        response = await crm_agent.process_message(
            message=last_user_message,
            company_id=user_info["company_id"],
            user_id=user_info["user_id"],
            conversation_id=conversation_id,
            session=session
        )
        
        # Track tokens used
        state["tokens_used"] += response.tokens_used if response.tokens_used else 0
        
        # Ensure agent_type is set
        response.agent_type = AgentType.CRM
        
        # Update state with response
        state["final_response"] = response
        state["agent_attempts"][AgentType.CRM] = state["agent_attempts"].get(AgentType.CRM, 0) + 1
        
        # Add agent response to messages
        new_messages = add_messages(messages, [AIMessage(content=response.response)])
        state["messages"] = new_messages
        
        # End this path
        state["next"] = END
        
        return state
        
    except Exception as e:
        logger.error(f"Error in CRM agent: {str(e)}")
        # Fallback to SQL agent on error
        state["current_agent"] = AgentType.SQL
        state["next"] = AgentType.SQL
        return state

# Project Agent node
async def project_agent_node(state: AgentState) -> AgentState:
    """Process the message with Project Management agent."""
    messages = state["messages"]
    user_info = state["user_info"]
    conversation_id = state["conversation_id"]
    
    # Get only the last user message
    user_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]
    if not user_messages:
        state["next"] = AgentType.SQL  # Fall back to SQL
        return state
    
    last_user_message = user_messages[-1].content
    
    # Set current agent
    state["current_agent"] = AgentType.PROJECT
    
    try:
        # Get Project agent and process message
        project_agent = get_agent(AgentType.PROJECT)
        
        # Check if the session is in the user_info
        session = user_info.get("session", None)
        
        response = await project_agent.process_message(
            message=last_user_message,
            company_id=user_info["company_id"],
            user_id=user_info["user_id"],
            conversation_id=conversation_id,
            session=session
        )
        
        # Track tokens used
        state["tokens_used"] += response.tokens_used if response.tokens_used else 0
        
        # Ensure agent_type is set
        response.agent_type = AgentType.PROJECT
        
        # Update state with response
        state["final_response"] = response
        state["agent_attempts"][AgentType.PROJECT] = state["agent_attempts"].get(AgentType.PROJECT, 0) + 1
        
        # Add agent response to messages
        new_messages = add_messages(messages, [AIMessage(content=response.response)])
        state["messages"] = new_messages
        
        # End this path
        state["next"] = END
        
        return state
        
    except Exception as e:
        logger.error(f"Error in Project agent: {str(e)}")
        # Fallback to SQL agent on error
        state["current_agent"] = AgentType.SQL
        state["next"] = AgentType.SQL
        return state

# Product Agent node
async def product_agent_node(state: AgentState) -> AgentState:
    """Process the message with Product Service agent."""
    messages = state["messages"]
    user_info = state["user_info"]
    conversation_id = state["conversation_id"]
    
    # Get only the last user message
    user_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]
    if not user_messages:
        state["next"] = AgentType.SQL  # Fall back to SQL
        return state
    
    last_user_message = user_messages[-1].content
    
    # Set current agent
    state["current_agent"] = AgentType.PRODUCT
    
    try:
        # Get Product agent and process message
        product_agent = get_agent(AgentType.PRODUCT)
        
        # Check if the session is in the user_info
        session = user_info.get("session", None)
        
        response = await product_agent.process_message(
            message=last_user_message,
            company_id=user_info["company_id"],
            user_id=user_info["user_id"],
            conversation_id=conversation_id,
            session=session
        )
        
        # Track tokens used
        state["tokens_used"] += response.tokens_used if response.tokens_used else 0
        
        # Ensure agent_type is set
        response.agent_type = AgentType.PRODUCT
        
        # Update state with response
        state["final_response"] = response
        state["agent_attempts"][AgentType.PRODUCT] = state["agent_attempts"].get(AgentType.PRODUCT, 0) + 1
        
        # Add agent response to messages
        new_messages = add_messages(messages, [AIMessage(content=response.response)])
        state["messages"] = new_messages
        
        # End this path
        state["next"] = END
        
        return state
        
    except Exception as e:
        logger.error(f"Error in Product agent: {str(e)}")
        # Fallback to SQL agent on error
        state["current_agent"] = AgentType.SQL
        state["next"] = AgentType.SQL
        return state

# Visualization handler node
async def handle_visualization_node(state: AgentState) -> AgentState:
    """Handle visualization request."""
    messages = state["messages"]
    user_info = state["user_info"]
    visualization_type = state["visualization_type"]
    
    # Get only the last user message
    user_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]
    if not user_messages:
        state["visualization_result"] = VisualizationResult(
            explanation="No query found for visualization."
        )
        return state
    
    last_user_message = user_messages[-1].content
    
    # Determine which agent should handle the visualization
    agent_type = state["current_agent"]
    if not agent_type:
        # Determine appropriate agent via router logic
        message_for_routing = last_user_message
        if "chart" in message_for_routing.lower() or "graph" in message_for_routing.lower() or "plot" in message_for_routing.lower():
            message_for_routing = f"visualization of {message_for_routing}"
        
        # Use the router logic to determine agent
        routing_state = AgentState(
            messages=[HumanMessage(content=message_for_routing)],
            user_info=user_info,
            conversation_id=state["conversation_id"],
            current_agent=None,
            agent_attempts={},
            final_response=None,
            visualization_result=None,
            action_result=None,
            needs_visualization=False,
            visualization_type=None,
            is_action=False,
            action_name=None,
            action_parameters=None,
            tokens_used=0
        )
        
        agent_type_str = await router_node(routing_state)
        if isinstance(agent_type_str, str) and agent_type_str in [e.value for e in AgentType]:
            agent_type = AgentType(agent_type_str)
        else:
            # Default to SQL for visualization
            agent_type = AgentType.SQL
    
    # Get appropriate agent
    agent = get_agent(agent_type)
    
    # Check if the session is in the user_info
    session = user_info.get("session", None)
    
    # Generate visualization
    result = await agent.generate_visualization(
        query=last_user_message,
        company_id=user_info["company_id"],
        user_id=user_info["user_id"],
        visualization_type=visualization_type,
        session=session
    )
    
    # Track tokens used
    state["tokens_used"] += result.tokens_used if result.tokens_used else 0
    
    # Update state with visualization result
    state["visualization_result"] = result
    
    return state

# Action handler node
async def perform_action_node(state: AgentState) -> AgentState:
    """Handle action request."""
    user_info = state["user_info"]
    action_name = state["action_name"]
    action_parameters = state["action_parameters"]
    
    if not action_name:
        state["action_result"] = ActionResult(
            success=False,
            message="No action specified."
        )
        return state
    
    # Use the same route map as the old dispatcher
    action_map = {
        # HRM actions
        "add_employee": AgentType.HRM,
        "update_employee": AgentType.HRM,
        "delete_employee": AgentType.HRM,
        "add_department": AgentType.HRM,
        "update_department": AgentType.HRM,
        "delete_department": AgentType.HRM,
        "add_attendance": AgentType.HRM,
        "update_attendance": AgentType.HRM,
        "add_training": AgentType.HRM,
        "update_training": AgentType.HRM,
        
        # Sales/CRM actions 
        "add_customer": AgentType.CRM,
        "update_customer": AgentType.CRM,
        "delete_customer": AgentType.CRM,
        "add_lead": AgentType.CRM,
        "update_lead": AgentType.CRM,
        "add_deal": AgentType.CRM,
        "update_deal": AgentType.CRM,
        
        # Finance actions
        "add_expense": AgentType.FINANCE,
        "update_expense": AgentType.FINANCE,
        "add_revenue": AgentType.FINANCE,
        "update_revenue": AgentType.FINANCE,
        "add_budget": AgentType.FINANCE,
        "update_budget": AgentType.FINANCE,
        "add_invoice": AgentType.FINANCE,
        "update_invoice": AgentType.FINANCE,
        "delete_invoice": AgentType.FINANCE,
        
        # CRM actions
        "add_ticket": AgentType.CRM,
        "update_ticket": AgentType.CRM,
        
        # Project actions
        "add_project": AgentType.PROJECT,
        "update_project": AgentType.PROJECT,
        "delete_project": AgentType.PROJECT,
        "add_task": AgentType.PROJECT,
        "update_task": AgentType.PROJECT,
        "delete_task": AgentType.PROJECT,
        "add_milestone": AgentType.PROJECT,
        "update_milestone": AgentType.PROJECT,
        "delete_milestone": AgentType.PROJECT,
        
        # Product actions
        "add_product": AgentType.PRODUCT,
        "update_product": AgentType.PRODUCT,
        "delete_product": AgentType.PRODUCT,
        "add_warehouse": AgentType.PRODUCT,
        "update_warehouse": AgentType.PRODUCT,
        "add_stock": AgentType.PRODUCT,
        "update_stock": AgentType.PRODUCT,
        "add_quotation": AgentType.PRODUCT,
        "update_quotation": AgentType.PRODUCT,
        "add_order": AgentType.PRODUCT,
        "update_order": AgentType.PRODUCT,
        "add_purchase": AgentType.PRODUCT,
        "update_purchase": AgentType.PRODUCT,
    }
    
    # Determine which agent should handle the action
    agent_type = action_map.get(action_name, AgentType.SQL)
    
    # Redirect SALES to CRM if it somehow came through
    if agent_type == AgentType.SALES:
        logger.info(f"Sales action '{action_name}' redirected to CRM agent")
        agent_type = AgentType.CRM
    
    # Get appropriate agent
    agent = get_agent(agent_type)
    
    # Check if the session is in the user_info
    session = user_info.get("session", None)
    
    # Execute action
    result = await agent.perform_action(
        action=action_name,
        parameters=action_parameters or {},
        company_id=user_info["company_id"],
        user_id=user_info["user_id"],
        session=session
    )
    
    # Update state with action result
    state["action_result"] = result
    
    return state

# Create the LangGraph
def build_agent_graph():
    """Build the LangGraph for agent routing."""
    workflow = StateGraph(AgentState)
    
    # Add nodes for each agent type and router
    workflow.add_node("router", router_node)
    workflow.add_node(AgentType.SQL, sql_agent_node)
    workflow.add_node(AgentType.HRM, hrm_agent_node)
    workflow.add_node(AgentType.FINANCE, finance_agent_node)
    workflow.add_node(AgentType.CRM, crm_agent_node)
    workflow.add_node(AgentType.PROJECT, project_agent_node)
    workflow.add_node(AgentType.PRODUCT, product_agent_node)
    workflow.add_node("handle_visualization", handle_visualization_node)
    workflow.add_node("perform_action", perform_action_node)
    
    # Set the router as the entry point
    workflow.set_entry_point("router")
    
    # Define routing logic
    def router_conditional_edge(state: AgentState):
        """Determine next node from router."""
        return state["next_node"]
    
    # Define general conditional edge for agent nodes
    def agent_conditional_edge(state: AgentState):
        """Determine next step after an agent node."""
        return state.get("next", END)
    
    # Add edges from router to all agent nodes
    workflow.add_conditional_edges(
        "router",
        router_conditional_edge,
        {
            AgentType.SQL: AgentType.SQL, 
            AgentType.HRM: AgentType.HRM,
            AgentType.FINANCE: AgentType.FINANCE,
            AgentType.CRM: AgentType.CRM,
            AgentType.PROJECT: AgentType.PROJECT,
            AgentType.PRODUCT: AgentType.PRODUCT,
            "handle_visualization": "handle_visualization",
            "perform_action": "perform_action"
        }
    )
    
    # Add conditional edges from each agent to SQL or END
    workflow.add_conditional_edges(AgentType.SQL, agent_conditional_edge)
    workflow.add_conditional_edges(AgentType.HRM, agent_conditional_edge)
    workflow.add_conditional_edges(AgentType.FINANCE, agent_conditional_edge)
    workflow.add_conditional_edges(AgentType.CRM, agent_conditional_edge)
    workflow.add_conditional_edges(AgentType.PROJECT, agent_conditional_edge)
    workflow.add_conditional_edges(AgentType.PRODUCT, agent_conditional_edge)
    
    # End connections
    workflow.add_edge("handle_visualization", END)
    workflow.add_edge("perform_action", END)
    
    return workflow.compile()

# Singleton instance of the compiled graph
_langgraph_instance = None

def get_langgraph():
    """Get the singleton LangGraph instance."""
    global _langgraph_instance
    if _langgraph_instance is None:
        _langgraph_instance = build_agent_graph()
    return _langgraph_instance

# Main class for API interaction
class LangGraphDispatcher:
    """LangGraph-based agent dispatcher for routing queries to specialized agents."""
    
    def __init__(self):
        """Initialize LangGraph dispatcher."""
        self.graph = get_langgraph()
        
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
        # Generate new conversation ID if not provided
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        
        # Create initial state
        initial_state = AgentState(
            messages=[HumanMessage(content=message)],
            user_info={
                "company_id": company_id,
                "user_id": user_id,
                "session": session
            },
            conversation_id=conversation_id,
            current_agent=None,
            agent_attempts={},
            final_response=None,
            visualization_result=None,
            action_result=None,
            needs_visualization=False,
            visualization_type=None,
            is_action=False,
            action_name=None,
            action_parameters=None,
            tokens_used=0
        )
        
        # Check if it's a visualization request
        is_visualization, viz_type = self._check_visualization_intent(message)
        if is_visualization:
            initial_state["needs_visualization"] = True
            initial_state["visualization_type"] = viz_type
        
        # Check if it's an action request (to implement action detection)
        # This would parse the message to see if it's requesting a specific action
        
        # Execute graph
        try:
            # Use ainvoke directly since we're already in an async function
            final_state = await self.graph.ainvoke(initial_state)
            
            # Return the final response
            if final_state["final_response"]:
                return final_state["final_response"]
            else:
                # Create a default response if none was set
                return AgentResponse(
                    response="I'm sorry, I wasn't able to process your request properly.",
                    conversation_id=conversation_id,
                    tokens_used=final_state["tokens_used"]
                )
                
        except Exception as e:
            logger.error(f"Error executing LangGraph: {str(e)}")
            # Return a graceful error response
            return AgentResponse(
                response=f"I encountered an error while processing your request. Please try again later.",
                conversation_id=conversation_id
            )
    
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
        # Create a unique conversation ID for this visualization request
        conversation_id = str(uuid.uuid4())
        
        # Create initial state specifically for visualization
        initial_state = AgentState(
            messages=[HumanMessage(content=query)],
            user_info={
                "company_id": company_id,
                "user_id": user_id,
                "session": session
            },
            conversation_id=conversation_id,
            current_agent=None,
            agent_attempts={},
            final_response=None,
            visualization_result=None,
            action_result=None,
            needs_visualization=True,
            visualization_type=visualization_type,
            is_action=False,
            action_name=None,
            action_parameters=None,
            tokens_used=0
        )
        
        # Execute graph
        try:
            # Use ainvoke directly since we're already in an async function
            final_state = await self.graph.ainvoke(initial_state)
            
            # Return the visualization result
            if final_state["visualization_result"]:
                return final_state["visualization_result"]
            else:
                # Create a default visualization result if none was set
                return VisualizationResult(
                    explanation="I wasn't able to generate a visualization for your query."
                )
                
        except Exception as e:
            logger.error(f"Error executing visualization in LangGraph: {str(e)}")
            # Return a graceful error response
            return VisualizationResult(
                explanation=f"I encountered an error while generating the visualization. Please try again later."
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
        # Create a unique conversation ID for this action request
        conversation_id = str(uuid.uuid4())
        
        # Create initial state specifically for action
        initial_state = AgentState(
            messages=[HumanMessage(content=f"Perform action: {action} with {json.dumps(parameters)}")],
            user_info={
                "company_id": company_id,
                "user_id": user_id,
                "session": session
            },
            conversation_id=conversation_id,
            current_agent=None,
            agent_attempts={},
            final_response=None,
            visualization_result=None,
            action_result=None,
            needs_visualization=False,
            visualization_type=None,
            is_action=True,
            action_name=action,
            action_parameters=parameters,
            tokens_used=0
        )
        
        # Execute graph
        try:
            # Use ainvoke directly since we're already in an async function
            final_state = await self.graph.ainvoke(initial_state)
            
            # Return the action result
            if final_state["action_result"]:
                return final_state["action_result"]
            else:
                # Create a default action result if none was set
                return ActionResult(
                    success=False,
                    message=f"I wasn't able to perform the action '{action}'."
                )
                
        except Exception as e:
            logger.error(f"Error executing action in LangGraph: {str(e)}")
            # Return a graceful error response
            return ActionResult(
                success=False,
                message=f"I encountered an error while performing the action. Please try again later."
            )
    
    def _check_visualization_intent(self, message: str) -> Tuple[bool, Optional[str]]:
        """Check if message indicates visualization intent and detect type.
        
        Args:
            message: User message
            
        Returns:
            Tuple of (is_visualization, visualization_type)
        """
        # Common visualization request phrases
        visualization_phrases = [
            "show me a chart", "create a chart", "generate a chart",
            "show me a graph", "create a graph", "generate a graph",
            "show me a plot", "create a plot", "generate a plot",
            "visualize", "visualization", "chart", "plot", "graph",
            "display data as", "create visualization", "show data as"
        ]
        
        # Common visualization types
        type_patterns = {
            "bar": ["bar chart", "bar graph", "column chart", "histogram"],
            "line": ["line chart", "line graph", "time series", "trend"],
            "pie": ["pie chart", "pie graph", "donut chart", "distribution"],
            "scatter": ["scatter plot", "scatter chart", "point chart", "correlation"],
            "area": ["area chart", "area graph", "stacked area"],
            "heatmap": ["heat map", "heatmap", "density map"],
            "radar": ["radar chart", "spider chart", "web chart"],
            "gantt": ["gantt chart", "timeline", "project chart"],
            "table": ["table visualization", "data table", "tabular"]
        }
        
        # Check for visualization intent
        is_visualization = any(phrase in message.lower() for phrase in visualization_phrases)
        
        # Detect visualization type
        viz_type = None
        if is_visualization:
            for viz_type_name, patterns in type_patterns.items():
                if any(pattern in message.lower() for pattern in patterns):
                    viz_type = viz_type_name
                    break
        
        return is_visualization, viz_type

# Singleton instance for the dispatcher
_dispatcher_instance = None

def get_langgraph_dispatcher():
    """Get the singleton LangGraphDispatcher instance."""
    global _dispatcher_instance
    if _dispatcher_instance is None:
        _dispatcher_instance = LangGraphDispatcher()
    return _dispatcher_instance 