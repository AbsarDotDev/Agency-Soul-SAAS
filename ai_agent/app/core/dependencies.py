from typing import Optional
from sqlalchemy.orm import Session

from app.database.connection import get_session
from app.core.langgraph_agents import get_langgraph_dispatcher

# Singleton instance of the dispatcher
_langgraph_dispatcher_instance = None

def get_agent_dispatcher():
    """Get the agent dispatcher singleton.
    
    Returns:
        Agent dispatcher instance
    """
    global _langgraph_dispatcher_instance
    if _langgraph_dispatcher_instance is None:
        _langgraph_dispatcher_instance = get_langgraph_dispatcher()
    return _langgraph_dispatcher_instance 