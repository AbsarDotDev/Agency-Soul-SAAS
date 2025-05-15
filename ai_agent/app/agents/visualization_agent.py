from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
import logging
import json
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain.tools import Tool
from app.agents.base_agent import BaseAgent, VisualizationResult, AgentResponse
from app.database.connection import get_company_isolated_sql_database
from app.visualizations.visualization_agent import VisualizationAgent as LangGraphVisualizationAgent
from app.core.llm import get_llm
from app.core.token_manager import TokenManager

logger = logging.getLogger(__name__)

class VisualizationAgent(BaseAgent):
    """Agent dedicated to generating visualizations from natural language or SQL queries, with company isolation."""

    def __init__(self):
        super().__init__()
        self.type = "visualization"
        self.llm = get_llm()
        self.langgraph_visualization_agent = LangGraphVisualizationAgent()

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
            
            # Use the LangGraph visualization agent
            viz_result = self.langgraph_visualization_agent.generate_visualization(
                query=query,
                company_id=company_id,
                requested_chart_type=visualization_type
            )
            
            # Log the visualization data
            if viz_result.get("chart_data"):
                logger.info(f"Generated visualization with chart_type={viz_result.get('chart_data', {}).get('chart_type')}")
            else:
                logger.warning("No visualization data was generated")
            
            # Update token usage if session is provided
            tokens_remaining = None
            if session:
                # Use fixed tokens (2) as agreed for visualizations
                tokens_used = 2
                try:
                    # Direct token update using engine
                    from app.database.connection import DatabaseConnection
                    from sqlalchemy import text
                    engine = DatabaseConnection.create_engine()
                    with Session(engine) as db_session:
                        stmt_update = text("UPDATE users SET ai_agent_tokens_used = ai_agent_tokens_used + :tokens WHERE id = :company_id")
                        db_session.execute(stmt_update, {"tokens": tokens_used, "company_id": company_id})
                        db_session.commit()
                        stmt_get = text("SELECT p.ai_agent_default_tokens - u.ai_agent_tokens_used FROM users u JOIN plans p ON u.plan = p.id WHERE u.id = :company_id")
                        result = db_session.execute(stmt_get, {"company_id": company_id}).fetchone()
                        tokens_remaining = result[0] if result else None
                    logger.info(f"Tokens updated for visualization. Remaining for company {company_id}: {tokens_remaining}")
                except Exception as e:
                    logger.error(f"Failed to update token usage for company {company_id}: {e}")
                    # Try to get current count anyway
                    tokens_remaining = await TokenManager.get_token_count(company_id, session)
            
            # Create the response with the visualization data
            agent_response = AgentResponse(
                response=viz_result.get("explanation", "Here is the visualization you requested."),
                conversation_id=conversation_id or "",  # Use provided conversation_id or default to empty
                conversation_title=f"Visualization: {query[:30]}{'...' if len(query) > 30 else ''}",
                visualization=viz_result.get("chart_data"),  # Set the visualization data explicitly
                tokens_remaining=tokens_remaining,
                tokens_used=viz_result.get("tokens_used", 2),
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