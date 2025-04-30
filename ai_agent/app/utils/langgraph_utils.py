from typing import Dict, Any, List, Optional, TypedDict, Annotated, Literal, Callable, Union
from langgraph.graph import StateGraph, END, START
from langchain.chat_models import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph.message import add_messages
import logging
import uuid
import json

from app.core.config import settings

# Set up logging
logger = logging.getLogger(__name__)


class ChatState(TypedDict):
    """Chat state for LangGraph."""
    
    messages: Annotated[List[Union[HumanMessage, SystemMessage, AIMessage]], add_messages]
    company_id: int
    user_id: str
    conversation_id: str


def create_llm(model_name: str = "gemini-1.5-pro"):
    """Create LangChain LLM instance.
    
    Args:
        model_name: Model name
        
    Returns:
        LLM instance
    """
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=0.7,
    )


def create_conversational_graph(
    system_prompt: str,
    nodes: Dict[str, Callable],
    conditional_edges: Optional[Dict[str, Callable]] = None
) -> StateGraph:
    """Create a conversational graph with LangGraph.
    
    Args:
        system_prompt: System prompt
        nodes: Node functions
        conditional_edges: Optional conditional edge functions
        
    Returns:
        StateGraph instance
    """
    # Create state graph
    graph = StateGraph(ChatState)
    
    # Add system message to messages
    graph.set_entry_point("start_node")
    
    # Add start node to prepare the state
    def start_node(state: ChatState) -> Dict[str, Any]:
        """Start node to initialize state."""
        return {
            "messages": [SystemMessage(content=system_prompt)]
        }
    
    # Add nodes
    graph.add_node("start_node", start_node)
    for name, node_func in nodes.items():
        graph.add_node(name, node_func)
    
    # Add edges
    if conditional_edges:
        for node_name, edge_func in conditional_edges.items():
            graph.add_conditional_edges(node_name, edge_func)
    
    # Add default edge from start to first node
    first_node = list(nodes.keys())[0]
    graph.add_edge("start_node", first_node)
    
    # Compile graph
    return graph.compile()


def create_agent_node(llm, node_name: str) -> Callable:
    """Create an agent node for LangGraph.
    
    Args:
        llm: LLM instance
        node_name: Node name
        
    Returns:
        Node function
    """
    def agent_node(state: ChatState) -> Dict[str, Any]:
        """Agent node function."""
        # Get response from LLM
        response = llm.invoke(state["messages"])
        return {"messages": [response]}
    
    return agent_node


def create_routing_node(llm) -> Callable:
    """Create a routing node for LangGraph.
    
    Args:
        llm: LLM instance
        
    Returns:
        Routing function
    """
    def routing_function(state: ChatState) -> str:
        """Routing function."""
        # Get last message
        last_message = state["messages"][-1]
        
        # Send the last message to the routing LLM to decide the next step
        routing_prompt = SystemMessage(content="""
You are a routing agent. Your job is to determine the next step in the conversation.
Based on the last message, decide whether to:
1. continue - Continue the conversation
2. finish - End the conversation
3. retrieve - Retrieve information
4. visualize - Generate a visualization

Respond with ONLY one of these options, nothing else.
        """)
        
        messages = [routing_prompt, last_message]
        response = llm.invoke(messages)
        
        # Parse response
        decision = response.content.lower().strip()
        
        # Map decisions to nodes
        decision_map = {
            "continue": "chat_node",
            "finish": END,
            "retrieve": "retrieval_node",
            "visualize": "visualization_node"
        }
        
        return decision_map.get(decision, "chat_node")
    
    return routing_function


def create_retrieval_node(llm, database_query_function) -> Callable:
    """Create a retrieval node for LangGraph.
    
    Args:
        llm: LLM instance
        database_query_function: Function to query database
        
    Returns:
        Retrieval node function
    """
    def retrieval_node(state: ChatState) -> Dict[str, Any]:
        """Retrieval node function."""
        # Get last message
        last_message = state["messages"][-1].content
        
        try:
            # Get data from database
            data = database_query_function(
                query=last_message,
                company_id=state["company_id"]
            )
            
            # Format data as message
            response_content = f"I found the following information:\n\n{json.dumps(data, indent=2)}"
            
            # Create AI message
            response = AIMessage(content=response_content)
            
            return {"messages": [response]}
        except Exception as e:
            logger.error(f"Error in retrieval node: {str(e)}")
            error_response = AIMessage(
                content="I'm sorry, I couldn't retrieve that information. " + 
                        "There was an error accessing the database."
            )
            return {"messages": [error_response]}
    
    return retrieval_node


def create_visualization_node(llm, visualization_function) -> Callable:
    """Create a visualization node for LangGraph.
    
    Args:
        llm: LLM instance
        visualization_function: Function to generate visualization
        
    Returns:
        Visualization node function
    """
    def visualization_node(state: ChatState) -> Dict[str, Any]:
        """Visualization node function."""
        # Get last message
        last_message = state["messages"][-1].content
        
        try:
            # Generate visualization
            visualization = visualization_function(
                query=last_message,
                company_id=state["company_id"]
            )
            
            # Create response with visualization data
            response_content = f"Here's the visualization you requested:\n\n" + \
                             f"Title: {visualization.get('title', 'Chart')}\n" + \
                             f"Description: {visualization.get('description', '')}\n\n" + \
                             f"[Visualization data: {json.dumps(visualization, indent=2)}]"
            
            # Create AI message
            response = AIMessage(content=response_content)
            
            return {"messages": [response]}
        except Exception as e:
            logger.error(f"Error in visualization node: {str(e)}")
            error_response = AIMessage(
                content="I'm sorry, I couldn't generate that visualization. " + 
                        "There was an error processing your request."
            )
            return {"messages": [error_response]}
    
    return visualization_node 