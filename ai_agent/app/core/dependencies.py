from typing import Optional
from sqlalchemy.orm import Session

from app.database.connection import get_session
from app.core.langgraph_agents import get_langgraph_dispatcher
from app.agents.visualization_langgraph_agent import VisualizationLangGraphFacade
from app.visualizations.langgraph.visualization_agent import VisualizationLangGraphAgent

# Singleton instances
_langgraph_dispatcher_instance = None
_visualization_agent_instance = None
_visualization_langgraph_instance = None

def get_agent_dispatcher():
    """Get the agent dispatcher singleton.
    
    Returns:
        Agent dispatcher instance
    """
    global _langgraph_dispatcher_instance
    if _langgraph_dispatcher_instance is None:
        _langgraph_dispatcher_instance = get_langgraph_dispatcher()
    return _langgraph_dispatcher_instance

def get_visualization_agent():
    """Get the visualization agent singleton.
    
    Returns:
        VisualizationLangGraphFacade instance
    """
    global _visualization_agent_instance
    if _visualization_agent_instance is None:
        _visualization_agent_instance = VisualizationLangGraphFacade()
    return _visualization_agent_instance 

def get_visualization_langgraph_agent():
    """Get the direct LangGraph visualization agent singleton.
    
    Returns:
        VisualizationLangGraphAgent instance
    """
    global _visualization_langgraph_instance
    if _visualization_langgraph_instance is None:
        _visualization_langgraph_instance = VisualizationLangGraphAgent()
    return _visualization_langgraph_instance 