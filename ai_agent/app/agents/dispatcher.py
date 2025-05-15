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
import traceback

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
        original_conversation_id: Optional[str] = None, 
        session: Optional[Session] = None
    ) -> AgentResponse:
        """Process message and route to appropriate agent."""
        
        current_conversation_id = original_conversation_id or str(uuid.uuid4())
        logger.info(f"[DISPATCHER] Start. InConvID: {original_conversation_id}, UseConvID: {current_conversation_id}, Msg: '{message[:60]}...'")
        message_lower = message.lower()
        
        # Log the exact message being checked in lowercase for debugging
        logger.info(f"[DISPATCHER] Checking message_lower: '{message_lower}'")

        try:
            # --- Enhanced Strong Visualization Check --- 
            is_strong_viz_request = False
            
            # Strong visualization keywords (direct matches)
            viz_keywords_strong = [
                "visualize", "visualise", "visualization", "visualisation", "chart", "graph", "plot", "histogram", "diagram",
                "pie chart", "bar chart", "line graph", "scatter plot", "gantt chart",
                "create chart", "generate chart", "draw chart", "display chart", "show me a chart",
                "make a graph", "make a chart", "plot the data", "chart the results",
                "create histogram", "generate histogram", "draw histogram", "show histogram",
                "trend of", "distribution of", "breakdown of", "comparison of", "correlation between"
            ]
            
            # Check for direct strong keyword matches first
            strong_matches = [kw for kw in viz_keywords_strong if kw in message_lower]
            if strong_matches:
                is_strong_viz_request = True
                logger.info(f"[DISPATCHER] Strong viz keyword direct match: {strong_matches}")
            
            # If no strong matches, try the enhanced detection logic
            if not is_strong_viz_request:
                # Soft keywords that might indicate visualization intent
                viz_keywords_soft = [
                    "show me data for", "analyze sales for", "employees per",
                    "showing the", "with respect to", "based on", "display the", "overview of", 
                    "show", "display", "illustrate"
                ]
                
                # Data indicators - things that could be visualized
                data_indicators = [
                    "products", "employees", "customers", "sales", "revenue", "expenses", 
                    "departments", "tasks", "stock", "quantity", "levels", "distribution",
                    "counts", "totals", "status", "progress"
                ]
                
                # Chart terms - explicit mention of visualization type
                chart_terms = [
                    "chart", "graph", "plot", "histogram", "visualisation", "visualization", 
                    "diagram", "bar graph", "pie chart", "line graph", "bar", "pie"
                ]

                # Check for different combinations of keywords
                has_soft_keyword = any(keyword in message_lower for keyword in viz_keywords_soft)
                has_data_indicator = any(indicator in message_lower for indicator in data_indicators)
                has_chart_term = any(term in message_lower for term in chart_terms)
                
                # Log what matched for debugging
                if has_soft_keyword or has_data_indicator or has_chart_term:
                    logger.info(f"[DISPATCHER] Soft viz matches - Keywords: {[kw for kw in viz_keywords_soft if kw in message_lower]}, " +
                              f"Data indicators: {[di for di in data_indicators if di in message_lower]}, " +
                              f"Chart terms: {[ct for ct in chart_terms if ct in message_lower]}")
                
                # Different patterns that indicate visualization intent
                if has_soft_keyword and has_data_indicator and has_chart_term:
                    is_strong_viz_request = True
                    logger.info("[DISPATCHER] Detected viz intent: soft keyword + data indicator + chart term")
                elif has_data_indicator and has_chart_term:
                    is_strong_viz_request = True
                    logger.info("[DISPATCHER] Detected viz intent: data indicator + chart term")
                elif "visuali" in message_lower and has_data_indicator:  
                    # Catches variants like "visualize", "visualise", "visualization" with data
                    is_strong_viz_request = True
                    logger.info("[DISPATCHER] Detected viz intent: visuali* + data indicator")
                elif (len(message_lower.split()) < 10 and 
                      (("by" in message_lower or "of" in message_lower) and has_data_indicator and has_chart_term)):
                    # Short queries with relationship terms + data + chart term
                    is_strong_viz_request = True
                    logger.info("[DISPATCHER] Detected viz intent: short query with relationship + data + chart")
                    
                # Product-specific viz detection (important special case)
                elif ("product" in message_lower or "stock" in message_lower) and ("quantity" in message_lower or "available" in message_lower):
                    is_strong_viz_request = True
                    logger.info("[DISPATCHER] Detected product-specific viz intent")

            # --- Path Decision based on Strong Viz Check --- 
            if is_strong_viz_request:
                logger.info(f"[DISPATCHER-VIZ-BY-KEYWORD-PATH] Strong viz intent detected. Routing to SQLAgent.generate_visualization.")
                sql_agent_for_viz = self._get_agent("sql")
                
                response_from_agent = await sql_agent_for_viz.generate_visualization(
                    query=message, company_id=company_id, user_id=user_id,
                    visualization_type=None, # SQLAgent will infer type if possible
                    session=session, conversation_id=current_conversation_id
                )
                
                # Ensure agent_type is set correctly for direct visualization calls
                if response_from_agent.visualization and response_from_agent.agent_type != "visualization":
                     response_from_agent.agent_type = "visualization"
                     logger.info(f"[DISPATCHER-VIZ-BY-KEYWORD-PATH] Set agent_type to 'visualization' for visualization response")
                elif not response_from_agent.visualization and response_from_agent.agent_type == "sql":
                     # If SQL agent somehow failed to produce viz but was called for it.
                     logger.warning(f"[DISPATCHER-VIZ-BY-KEYWORD-PATH] SQLAgent called for viz but returned no viz data.")

                logger.info(f"[DISPATCHER-VIZ-BY-KEYWORD-PATH] Final response: {response_from_agent.dict(exclude_none=True)}")
                return response_from_agent
            
            # --- Normal Path (no strong viz keywords detected by dispatcher, proceed to LLM classification) --- 
            logger.info(f"[DISPATCHER-NORMAL-PATH] No strong viz keywords determined by initial checks. Proceeding to LLM classification. (UseConvID: {current_conversation_id}) Msg: '{message[:60]}...'")
            agent_to_use_type = "sql" # Default unless LLM classification changes it
            
            # Only use LLM-based agent detection if it's NOT a direct SQL query
            if not self._is_direct_sql_query(message):
                logger.info("[DISPATCHER-NORMAL-PATH] Not direct SQL, detecting agent type via LLM...")
                agent_to_use_type = await self._detect_agent_type(message, company_id, session)
            else:
                logger.info("[DISPATCHER-NORMAL-PATH] Is direct SQL query, will use SQLAgent.")
            
            actual_agent = self._get_agent(agent_to_use_type)
            logger.info(f"[DISPATCHER-NORMAL-PATH] Using agent: {type(actual_agent).__name__} (type: {agent_to_use_type})")
            
            response_from_agent = await actual_agent.process_message(
                message=message, company_id=company_id, user_id=user_id,
                conversation_id=current_conversation_id, session=session
            )
            logger.info(f"[DISPATCHER-NORMAL-PATH] Response from {type(actual_agent).__name__}.process_message (ConvID {response_from_agent.conversation_id}): {response_from_agent.dict(exclude_none=True)}")

            # Ensure agent_type is correctly set or corrected
            # If SQLAgent.process_message decided it IS a visualization, its agent_type will be set.
            if response_from_agent.visualization and response_from_agent.agent_type != "visualization":
                logger.warning(f"[DISPATCHER-NORMAL-PATH] Viz data found. Correcting agent_type from '{response_from_agent.agent_type}' to 'visualization'.")
                response_from_agent.agent_type = "visualization"
            elif not response_from_agent.agent_type: # Fallback if agent_type is still None
                response_from_agent.agent_type = agent_to_use_type
                logger.info(f"[DISPATCHER-NORMAL-PATH] Set agent_type to '{agent_to_use_type}' on response.")
                
            # If it was routed to a non-SQL agent but the user *also* seemed to want a chart for the general query:
            # This is a basic attempt to add a chart if a non-SQL agent didn't provide one.
            if agent_to_use_type != "sql" and not response_from_agent.visualization and any(soft_kw in message_lower for soft_kw in ["show me a chart of", "graph the results for"]):
                logger.info(f"[DISPATCHER-NORMAL-PATH] Agent '{agent_to_use_type}' provided no viz. Attempting supplemental viz with SQLAgent for: '{message[:60]}...'")
                try:
                    sql_agent_for_supplemental_viz = self._get_agent("sql")
                    supplemental_viz_response = await sql_agent_for_supplemental_viz.generate_visualization(
                        query=message, # Use original message for viz query
                        company_id=company_id, user_id=user_id, visualization_type=None,
                        session=session, conversation_id=current_conversation_id
                    )
                    if supplemental_viz_response and supplemental_viz_response.visualization:
                        logger.info(f"[DISPATCHER-NORMAL-PATH] Supplemental viz generated. Merging with primary response.")
                        response_from_agent.visualization = supplemental_viz_response.visualization
                        # The textual response from supplemental_viz_response.response is likely a summary of the chart data.
                        # We could prepend it or just rely on the primary agent's text.
                        # For now, let primary agent's text stand, just add the chart.
                        if response_from_agent.agent_type != "visualization": # Ensure agent type reflects viz presence
                           response_from_agent.agent_type = "visualization"
                    else:
                        logger.info("[DISPATCHER-NORMAL-PATH] Supplemental viz attempt did not yield data.")
                except Exception as e_supp_viz:
                    logger.error(f"[DISPATCHER-NORMAL-PATH] Error during supplemental viz attempt: {str(e_supp_viz)}")

            # --- Supplemental Visualization Logic --- 
            # This runs if the classified agent (e.g. ProductAgent) did not produce a visualization
            # response_from_agent is an AgentResponse object here
            if response_from_agent and response_from_agent.response and not response_from_agent.visualization:
                needs_supplemental_viz = False
                supplemental_viz_keywords = [
                    "chart", "graph", "plot", "histogram", "visualize", "visualise", "visualization", "diagram",
                    "show distribution", "create visualization", "draw a", "display data as", "show stock", "stock levels"
                ]
                if any(phrase in message_lower for phrase in supplemental_viz_keywords):
                    needs_supplemental_viz = True
                    logger.info(f"[DISPATCHER] Supplemental: Direct viz keyword found: '{[kw for kw in supplemental_viz_keywords if kw in message_lower]}' in '{message_lower[:60]}...'")
                
                if not needs_supplemental_viz:
                    data_analysis_patterns = [
                        r"(show|display|visualize|visualise|analyze|overview of|summary of).*(distribution|trend|comparison|breakdown|overview|summary|levels|status|quantities)",
                        r"(compare|correlate|analyze).*(with respect to|by|over time)",
                        r"how many.*(per|by|in each)",
                        r"(count of|number of|total|sum of).*(per|by|group by)",
                        r"(products|sales|revenue|expenses|employees|tasks|stock|inventory).*(levels|status|quantities|figures|numbers|data|information)"
                    ]
                    for pattern in data_analysis_patterns:
                        if re.search(pattern, message_lower): 
                            needs_supplemental_viz = True
                            logger.info(f"[DISPATCHER] Supplemental: Data analysis pattern matched: '{pattern}' in '{message_lower[:60]}...'")
                            break
                            
                if needs_supplemental_viz:
                    logger.info(f"[DISPATCHER] Primary agent '{response_from_agent.agent_type}' provided no viz. Attempting supplemental viz for: '{message[:60]}...'")
                    insights = response_from_agent.response 
                    
                    try:
                        sql_agent_for_supplemental_viz = self._get_agent("sql")
                        supplemental_agent_response = await sql_agent_for_supplemental_viz.generate_visualization(
                            query=message, 
                            company_id=company_id,
                            user_id=user_id,
                            session=session,
                            conversation_id=current_conversation_id, 
                            insights_text=insights
                        )
                        if supplemental_agent_response and supplemental_agent_response.visualization:
                            logger.info(f"[DISPATCHER] Supplemental viz generated. Merging with primary response.")
                            response_from_agent.visualization = supplemental_agent_response.visualization
                            response_from_agent.agent_type = "visualization"
                            # Optionally, append a note to the text response that a chart was added
                            if response_from_agent.response and not response_from_agent.response.endswith((".", "!", "?")):
                                response_from_agent.response += "."
                            response_from_agent.response += " I've also created a chart for this data."
                        else:
                            logger.info("[DISPATCHER] Supplemental viz attempt did not yield data or visualization.")
                    except Exception as e_supp_viz:
                        logger.error(f"[DISPATCHER] Error during supplemental viz attempt: {str(e_supp_viz)}", exc_info=True)

            logger.info(f"[DISPATCHER-NORMAL-PATH] Final response after any supplemental viz: {response_from_agent.dict(exclude_none=True)}")
            return response_from_agent

        except Exception as e_dispatcher_main:
            logger.error(f"[DISPATCHER] CRITICAL ERROR in process_message: {str(e_dispatcher_main)}", exc_info=True)
            return AgentResponse(
                response=f"Critical Dispatcher Failure: {str(e_dispatcher_main)}",
                conversation_id=current_conversation_id, 
                conversation_title="Dispatcher Main Error",
                tokens_used=0, tokens_remaining=None, visualization=None,
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
