import logging
from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session
import uuid

from app.agents.base_agent import BaseAgent, AgentResponse, VisualizationResult
from app.core.token_manager import TokenManager
from app.visualizations.langgraph.visualization_agent import VisualizationLangGraphAgent
from app.visualizations.langgraph.workflow import WorkflowManager

# Set up logging
logger = logging.getLogger(__name__)

class VisualizationLangGraphFacade(BaseAgent):
    """Facade for the LangGraph visualization agent, implementing the BaseAgent interface."""
    
    def __init__(self):
        """Initialize the visualization agent facade."""
        super().__init__()
        self.type = "visualization"
        self.visualization_agent = VisualizationLangGraphAgent()
        self.workflow_manager = WorkflowManager(self.visualization_agent.company_id)
    
    async def generate_visualization(
        self,
        query: str,
        company_id: int,
        user_id: str,
        visualization_type: Optional[str] = None,
        session: Optional[Session] = None,
        conversation_id: Optional[str] = None
    ) -> AgentResponse:
        """Generate a visualization from a natural language query using LangGraph workflow."""
        try:
            logger.info(f"Generating visualization for '{query}', company_id={company_id}, visualization_type={visualization_type}")
            
            # Set the company_id in the visualization agent
            self.visualization_agent.company_id = company_id
            
            # Re-initialize the workflow manager with the company ID
            self.workflow_manager = WorkflowManager(company_id)
            
            # Run the visualization workflow
            result = self.visualization_agent.generate_visualization(
                query=query,
                company_id=company_id
            )
            
            # Extract results
            chart_data = result.get('chart_data')
            explanation = result.get('answer', "Here is the visualization you requested.")
            tokens_used = result.get('tokens_used', 0)
            
            # Use fixed tokens (2) as agreed for visualizations
            actual_tokens_used = 2
            
            # Update token usage if session is provided
            tokens_remaining = None
            if session:
                try:
                    # Update token usage directly using SQL
                    from sqlalchemy import text
                    update_query = text("""
                        UPDATE users
                        SET ai_agent_tokens_used = ai_agent_tokens_used + :tokens_to_consume
                        WHERE id = :company_id AND type = 'company'
                    """)
                    session.execute(update_query, {"company_id": company_id, "tokens_to_consume": actual_tokens_used})
                    session.commit()
                    
                    # Get remaining tokens
                    tokens_remaining = await TokenManager.get_token_count(company_id, session)
                    logger.info(f"Tokens updated for visualization. Remaining for company {company_id}: {tokens_remaining}")
                except Exception as e:
                    logger.error(f"Failed to update token usage for company {company_id}: {str(e)}")
                    # Try to get current count anyway
                    tokens_remaining = await TokenManager.get_token_count(company_id, session)
            
            # Create the response with the visualization data
            # Check if we have valid chart data - if not, explain the issue to the user
            if chart_data is None:
                explanation = "I couldn't generate a visualization for your query. Please try again with a different question or more specific details about what you want to see."
            
            logger.info(f"Creating visualization response with chart_data: {chart_data is not None}")
            if chart_data and isinstance(chart_data, dict):
                logger.info(f"Chart type: {chart_data.get('chart_type')}, Has datasets: {'datasets' in chart_data}")
            
            agent_response = AgentResponse(
                response=explanation,
                conversation_id=conversation_id or str(uuid.uuid4()),
                conversation_title=f"Visualization: {query[:30]}{'...' if len(query) > 30 else ''}",
                visualization=chart_data,
                tokens_remaining=tokens_remaining,
                tokens_used=actual_tokens_used,
                agent_type=self.type
            )
            
            # Log the response structure
            visualization_log = "No visualization data" if agent_response.visualization is None else {
                "chart_type": agent_response.visualization.get("chart_type", "unknown"),
                "has_labels": "labels" in agent_response.visualization,
                "has_datasets": "datasets" in agent_response.visualization,
                "options_type": type(agent_response.visualization.get("options", {}))
            }
            
            logger.info(f"Visualization response: conversation_id={agent_response.conversation_id}, agent_type={agent_response.agent_type}, visualization={visualization_log}")
            
            return agent_response
            
        except Exception as e:
            logger.error(f"VisualizationLangGraphFacade error: {e}", exc_info=True)
            return AgentResponse(
                response=f"Error generating visualization: {str(e)}",
                conversation_id=conversation_id or str(uuid.uuid4()),
                conversation_title=None,
                visualization=None,
                tokens_remaining=None,
                tokens_used=0,
                agent_type=self.type
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
        logger.info(f"VisualizationLangGraphFacade processing message: '{message[:50]}...'")
        
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
                logger.info(f"VisualizationLangGraphFacade created response with visualization: {viz_info}")
            else:
                logger.warning("VisualizationLangGraphFacade response is missing visualization data")
                
            return response
            
        except Exception as e:
            logger.error(f"VisualizationLangGraphFacade.process_message error: {e}", exc_info=True)
            return AgentResponse(
                response=f"I encountered an error while creating the visualization: {str(e)}",
                conversation_id=conversation_id or str(uuid.uuid4()),
                conversation_title=None,
                visualization=None,
                tokens_remaining=None,
                tokens_used=0,
                agent_type=self.type
            )
