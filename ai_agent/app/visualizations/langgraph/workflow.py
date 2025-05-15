from typing import Dict, Any, Optional
import logging
import uuid

from langgraph.graph import StateGraph, END
from langchain_core.output_parsers import JsonOutputParser

from app.visualizations.langgraph.state import State
from app.visualizations.langgraph.sql_agent import SQLAgent
from app.visualizations.langgraph.chart_agent import ChartAgent

# Set up logging with less verbose level
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class WorkflowManager:
    """
    Manager for the visualization workflow graph.
    Creates, configures, and runs the LangGraph workflow.
    """
    
    def __init__(self, company_id: Optional[int] = None):
        """
        Initialize the workflow manager.
        
        Args:
            company_id: Optional company ID for token tracking
        """
        self.company_id = company_id
        self.sql_agent = SQLAgent(company_id)
        self.chart_agent = ChartAgent(company_id)
    
    def create_workflow(self) -> StateGraph:
        """
        Create and configure the workflow graph.
        
        Returns:
            Configured StateGraph for the visualization workflow
        """
        # Create the graph with our state schema
        workflow = StateGraph(State)
        
        # Add nodes for each step in the workflow
        workflow.add_node("is_question_relevant", self.sql_agent.is_question_relevant)
        workflow.add_node("generate_query", self.sql_agent.generate_query)
        workflow.add_node("execute_query", self.sql_agent.execute_query)
        workflow.add_node("choose_chart_type", self.chart_agent.choose_chart_type)
        workflow.add_node("format_chart_data", self.chart_agent.format_chart_data)
        
        # Define the edges (the flow between nodes)
        workflow.add_edge("is_question_relevant", "generate_query")
        workflow.add_edge("generate_query", "execute_query")
        workflow.add_edge("execute_query", "choose_chart_type")
        workflow.add_edge("choose_chart_type", "format_chart_data")
        workflow.add_edge("format_chart_data", END)
        
        # Set the entry point
        workflow.set_entry_point("is_question_relevant")
        
        return workflow
    
    def get_compiled_graph(self):
        """
        Get the compiled workflow graph for direct execution or streaming.
        
        Returns:
            Compiled LangGraph workflow
        """
        return self.create_workflow().compile()
    
    def run(self, question: str) -> Dict[str, Any]:
        """
        Run the visualization workflow with a question.
        
        Args:
            question: User's natural language question
            
        Returns:
            Dictionary with workflow results (answer, chart data, etc.)
        """
        # Create a unique ID for this run
        run_id = str(uuid.uuid4())
        logger.info(f"Starting visualization workflow {run_id} for question: {question}")
        
        # Compile the workflow
        app = self.create_workflow().compile()
        
        try:
            # Prepare the initial state
            initial_state = {
                "question": question,
                "company_id": self.company_id or 0
            }
            
            # Run the workflow
            result = app.invoke(initial_state)
            
            logger.info(f"Visualization workflow {run_id} completed successfully")
            
            # Return a cleaned result
            return {
                "chart_type": result.get("chart_type", ""),
                "chart_reason": result.get("chart_reason", ""),
                "chart_data": result.get("chart_data", {}),
                "sql_query": result.get("sql_query", ""),
                "relevance_reasoning": result.get("relevance_reasoning", ""),
                "sql_generation_reason": result.get("sql_generation_reason", ""),
                "execution_error": result.get("execution_error", "")
            }
        except Exception as e:
            logger.error(f"Error running visualization workflow {run_id}: {str(e)}")
            return {
                "chart_type": "none",
                "chart_reason": f"Error: {str(e)}",
                "chart_data": None,
                "error": str(e)
            }
    
    def run_with_streaming(self, question: str, event_handler=None):
        """
        Run the workflow with streaming updates.
        
        Args:
            question: User's natural language question
            event_handler: Optional callback function for streaming events
            
        Returns:
            Final result dictionary
        """
        # Create a unique ID for this run
        run_id = str(uuid.uuid4())
        logger.info(f"Starting streaming visualization workflow {run_id} for question: {question}")
        
        # Compile the workflow
        app = self.create_workflow().compile()
        
        try:
            # Prepare the initial state
            initial_state = {
                "question": question,
                "company_id": self.company_id or 0
            }
            
            # If no event handler is provided, use a simple logger
            if event_handler is None:
                def default_handler(event):
                    logger.debug(f"Workflow event: {event['status']}")
                event_handler = default_handler
            
            # Run with streaming
            for event in app.stream(initial_state):
                # Process event
                event_handler(event)
                
                # Check if final result
                if event["status"] == "done":
                    result = event["result"]
                    logger.info(f"Streaming visualization workflow {run_id} completed successfully")
                    
                    # Return a cleaned result
                    return {
                        "chart_type": result.get("chart_type", ""),
                        "chart_reason": result.get("chart_reason", ""),
                        "chart_data": result.get("chart_data", {}),
                        "sql_query": result.get("sql_query", ""),
                        "relevance_reasoning": result.get("relevance_reasoning", ""),
                        "sql_generation_reason": result.get("sql_generation_reason", ""),
                        "execution_error": result.get("execution_error", "")
                    }
            
            # Should not reach here with normal execution
            return {
                "chart_type": "none",
                "chart_data": None,
                "error": "No result from workflow"
            }
        except Exception as e:
            logger.error(f"Error running streaming visualization workflow {run_id}: {str(e)}")
            return {
                "chart_type": "none",
                "chart_reason": f"Error: {str(e)}",
                "chart_data": None,
                "error": str(e)
            } 