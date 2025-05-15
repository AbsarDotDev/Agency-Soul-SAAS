"""
Visualization agent for generating charts from SQL data.
This module handles SQL generation, data retrieval, and chart formatting.
"""

import logging
from typing import Dict, Any, Optional, Callable

from app.visualizations.langgraph.visualization_agent import VisualizationLangGraphAgent

# Set up logging with less verbose level
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class VisualizationAgent:
    """Agent for generating data visualizations from SQL queries."""
    
    def __init__(self):
        """Initialize the visualization agent."""
        self.lang_graph_agent = VisualizationLangGraphAgent()
    
    def generate_visualization(self, 
                              query: str, 
                              company_id: int, 
                              requested_chart_type: str = None,
                              streaming: bool = False,
                              event_handler: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Generate a visualization based on the user's query.
        
        Args:
            query: User's natural language query
            company_id: Company ID for data isolation
            requested_chart_type: Optional chart type explicitly requested by user
            streaming: Whether to use streaming mode for real-time updates
            event_handler: Optional callback for streaming events
            
        Returns:
            Dict containing chart data and explanation
        """
        logger.info(f"Using LangGraph-based visualization agent for: '{query}'")
        
        # Delegate to the LangGraph-based implementation
        result = self.lang_graph_agent.generate_visualization(
            query=query,
            company_id=company_id,
            streaming=streaming,
            event_handler=event_handler
        )
        
        # Ensure backward compatibility in the result structure
        return {
            "chart_data": result.get("chart_data"),
            "explanation": result.get("answer", ""),
            "tokens_used": result.get("tokens_used", 0)
        }
