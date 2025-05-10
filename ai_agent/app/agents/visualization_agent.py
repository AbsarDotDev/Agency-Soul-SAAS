from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
import logging
import json
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain.tools import Tool
from app.agents.base_agent import BaseAgent, VisualizationResult, AgentResponse
from app.database.connection import get_company_isolated_sql_database
from app.visualizations.generator import generate_visualization_from_data
from app.core.llm import get_llm
from app.core.token_manager import TokenManager

logger = logging.getLogger(__name__)

class VisualizationAgent(BaseAgent):
    """Agent dedicated to generating visualizations from natural language or SQL queries, with company isolation."""

    def __init__(self):
        super().__init__()
        self.type = "visualization"
        self.llm = get_llm()

    async def generate_visualization(
        self,
        query: str,
        company_id: int,
        user_id: str,
        visualization_type: Optional[str] = None,
        session: Optional[Session] = None,
        conversation_id: Optional[str] = None
    ) -> AgentResponse:
        """Generate a visualization from a natural language or SQL query, enforcing company isolation, and return AgentResponse for multimodal chat."""
        try:
            logger.info(f"Generating visualization for '{query}', company_id={company_id}, visualization_type={visualization_type}")
            
            # Create SQL database with company isolation
            sql_database = get_company_isolated_sql_database(
                company_id=company_id,
                sample_rows_in_table_info=3
            )
            
            # Create SQL toolkit
            toolkit = SQLDatabaseToolkit(
                db=sql_database,
                llm=self.llm
            )
            
            # Prompt LLM to generate SQL if needed
            sql_query_prompt = f"""Given the user query: '{query}', if it is not a valid SQL SELECT query, write a SQL SELECT query to retrieve the necessary data. Always include company isolation (WHERE created_by = {company_id} or equivalent) for all tables. Only return the SQL SELECT query itself, no explanation, no markdown."""
            
            agent_executor = toolkit.get_agent()
            response = await agent_executor.ainvoke({"input": sql_query_prompt})
            sql_query = response.get("output", "").strip()
            
            # If the query is not a SELECT, treat the original query as SQL
            if not sql_query.lower().startswith("select"):
                logger.warning(f"Generated query does not start with SELECT: {sql_query}")
                sql_query = query
            
            logger.info(f"SQL query for visualization: {sql_query}")
            
            # Execute the SQL
            data = sql_database.run(sql_query)
            
            # If data is a string, try to parse as JSON
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                    logger.info(f"Parsed string data as JSON")
                except Exception as e:
                    logger.error(f"Failed to parse string data as JSON: {e}")
                    data = []
            
            # Log retrieved data
            logger.info(f"Retrieved data for visualization: {data[:5]}...")
            
            # Generate the visualization
            viz_result = await generate_visualization_from_data(
                data=data,
                query=query,
                visualization_type=visualization_type,
                llm=self.llm
            )
            
            # Log the visualization data
            if viz_result.data:
                logger.info(f"Generated visualization with chart_type={viz_result.data.get('chart_type')}")
                
                # Ensure options is a dictionary, not an array
                if viz_result.data.get('options') is None or (isinstance(viz_result.data.get('options'), list) and len(viz_result.data.get('options')) == 0):
                    viz_result.data['options'] = {}
                    logger.info("Fixed empty options array to be an empty object")
            else:
                logger.warning("No visualization data was generated")
            
            # Update token usage if session is provided
            tokens_remaining = None
            if session:
                # Use fixed tokens (2) as agreed for visualizations
                tokens_used = 2
                token_update_success = await TokenManager.update_tokens_used(
                    session=session,
                    company_id=company_id,
                    tokens_used=tokens_used
                )
                
                if token_update_success:
                    tokens_remaining = await TokenManager.get_token_count(company_id, session)
                    logger.info(f"Tokens updated for visualization. Remaining for company {company_id}: {tokens_remaining}")
                else:
                    logger.error(f"Failed to update token usage for company {company_id}")
                    # Try to get current count anyway
                    tokens_remaining = await TokenManager.get_token_count(company_id, session)
            
            # Create the response with the visualization data
            agent_response = AgentResponse(
                response=viz_result.explanation or "Here is the visualization you requested.",
                conversation_id=conversation_id or "",  # Use provided conversation_id or default to empty
                conversation_title=f"Visualization: {query[:30]}{'...' if len(query) > 30 else ''}",
                visualization=viz_result.data,  # Set the visualization data explicitly
                tokens_remaining=tokens_remaining,
                tokens_used=viz_result.tokens_used,
                agent_type=self.type  # Explicitly set agent_type for frontend
            )
            
            # Log the complete response structure (excluding large data fields)
            visualization_log = "No visualization data" if agent_response.visualization is None else {
                "chart_type": agent_response.visualization.get("chart_type", "unknown"),
                "has_labels": "labels" in agent_response.visualization,
                "has_datasets": "datasets" in agent_response.visualization,
                "options_type": type(agent_response.visualization.get("options", {}))
            }
            
            logger.info(f"Visualization response: conversation_id={agent_response.conversation_id}, agent_type={agent_response.agent_type}, visualization={visualization_log}")
            
            return agent_response
            
        except Exception as e:
            logger.error(f"VisualizationAgent error: {e}", exc_info=True)
            return AgentResponse(
                response=f"Error generating visualization: {str(e)}",
                conversation_id=conversation_id or "",
                conversation_title=None,
                visualization=None,
                tokens_remaining=None,
                tokens_used=0,
                agent_type=self.type  # Still set agent_type even on error
            )

    async def process_message(
        self,
        message: str,
        company_id: int,
        user_id: str,
        conversation_id: Optional[str] = None,
        session: Optional[Session] = None
    ) -> AgentResponse:
        """Process a user message to generate a visualization."""
        logger.info(f"VisualizationAgent processing message: '{message[:50]}...'")
        
        try:
            # Generate visualization
            response = await self.generate_visualization(
                query=message,
                company_id=company_id,
                user_id=user_id,
                conversation_id=conversation_id,
                session=session
            )
            
            # Ensure agent_type is properly set
            response.agent_type = self.type
            
            # Log response details for debugging
            if response.visualization:
                viz_info = {
                    "chart_type": response.visualization.get("chart_type", "unknown"),
                    "has_labels": "labels" in response.visualization,
                    "has_datasets": "datasets" in response.visualization
                }
                logger.info(f"VisualizationAgent created response with visualization: {viz_info}")
            else:
                logger.warning("VisualizationAgent response is missing visualization data")
                
            return response
            
        except Exception as e:
            logger.error(f"VisualizationAgent.process_message error: {e}", exc_info=True)
            return AgentResponse(
                response=f"I encountered an error while creating the visualization: {str(e)}",
                conversation_id=conversation_id or "",
                conversation_title=None,
                visualization=None,
                tokens_remaining=None,
                tokens_used=0,
                agent_type=self.type  # Always set agent_type even on error
            ) 