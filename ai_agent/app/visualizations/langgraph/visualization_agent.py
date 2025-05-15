import logging
import json
import re
import pandas as pd
from typing import Dict, Any, List, Optional, Callable
import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from app.database.connection import get_company_isolated_sql_database
from app.visualizations.langgraph.workflow import WorkflowManager
from app.visualizations.chart_utils import extract_chart_type_from_query
from app.core.token_manager import TokenManager
from app.visualizations.langgraph.sql_agent import SQLAgent
from app.visualizations.langgraph.chart_agent import ChartAgent

# Set up logging with less verbose level
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class VisualizationLangGraphAgent:
    """
    Main visualization agent class that serves as the entry point for visualization functionality.
    Uses LangGraph-based workflow to dynamically generate SQL, retrieve data, and create visualizations.
    """
    
    def __init__(self):
        """Initialize the visualization agent."""
        self.tokens_used = 0
        self.sql_agent = None
        self.chart_agent = None
        self.company_id = None
    
    def generate_visualization(self, 
                              query: str, 
                              company_id: int,
                              streaming: bool = False,
                              event_handler: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Generate a visualization based on natural language query.
        
        Args:
            query: Natural language query asking for visualization
            company_id: Company ID for data isolation
            streaming: Whether to use streaming mode for real-time updates
            event_handler: Optional callback for streaming events
            
        Returns:
            Dictionary containing visualization data and explanation
        """
        logger.info(f"Generating visualization for: '{query}' (company_id: {company_id})")
        start_time = time.time()
        
        try:
            # Store company ID
            self.company_id = company_id
            
            # Create workflow manager with company ID for token tracking
            workflow_manager = WorkflowManager(company_id)
            
            # Save references to agents for adapter methods
            self.sql_agent = workflow_manager.sql_agent
            self.chart_agent = workflow_manager.chart_agent
            
            # Run workflow with or without streaming
            if streaming and event_handler:
                result = workflow_manager.run_with_streaming(query, event_handler)
            else:
                result = workflow_manager.run(query)
            
            # Track tokens used
            self.tokens_used = getattr(workflow_manager.sql_agent.llm_manager, 'tokens_used', 0)
            self.tokens_used += getattr(workflow_manager.chart_agent.llm_manager, 'tokens_used', 0)
            
            # Create the explanation from available information
            explanation = ""
            if result.get("chart_reason"):
                explanation += f"Chart selection: {result.get('chart_reason')}\n\n"
            if result.get("relevance_reasoning"):
                explanation += f"SQL relevance: {result.get('relevance_reasoning')}\n\n"
            if result.get("sql_generation_reason"):
                explanation += f"Query generation: {result.get('sql_generation_reason')}\n\n"
            if result.get("execution_error"):
                explanation += f"Error: {result.get('execution_error')}\n\n"
            
            # Ensure all keys are present
            final_result = {
                "chart_data": result.get("chart_data"),
                "chart_type": result.get("chart_type", ""),
                "answer": explanation.strip(),
                "sql_query": result.get("sql_query", ""),
                "error": result.get("execution_error", ""),
                "tokens_used": self.tokens_used,
                "execution_time": round(time.time() - start_time, 2)
            }
            
            logger.info(f"Visualization generation completed in {final_result['execution_time']}s using {self.tokens_used} tokens")
            return final_result
        
        except Exception as e:
            logger.error(f"Error generating visualization: {str(e)}")
            execution_time = round(time.time() - start_time, 2)
            
            return {
                "chart_data": None,
                "chart_type": "none",
                "answer": f"Error generating visualization: {str(e)}",
                "sql_query": "",
                "error": str(e),
                "tokens_used": self.tokens_used,
                "execution_time": execution_time
            }
    
    # Adapter methods for workflow_manager.py compatibility
    
    def generate_visualization_direct(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Direct visualization generation method for workflow_manager."""
        query = state.get("query", "")
        company_id = state.get("company_id", 0)
        
        result = self.generate_visualization(query, company_id)
        
        # Convert format to match what workflow_manager expects
        return {
            "chart_data": result.get("chart_data"),
            "explanation": result.get("answer", ""),
            "tokens_used": result.get("tokens_used", 0)
        }
    
    def is_question_relevant(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Adapter for SQL agent's is_question_relevant method."""
        if self.sql_agent is None:
            self.sql_agent = SQLAgent(state.get("company_id", 0))
        return self.sql_agent.is_question_relevant(state)
    
    def generate_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Adapter for SQL agent's generate_query method."""
        if self.sql_agent is None:
            self.sql_agent = SQLAgent(state.get("company_id", 0))
        return self.sql_agent.generate_query(state)
    
    def execute_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Adapter for SQL agent's execute_query method."""
        if self.sql_agent is None:
            self.sql_agent = SQLAgent(state.get("company_id", 0))
        return self.sql_agent.execute_query(state)
    
    def choose_chart_type(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Adapter for chart agent's choose_chart_type method."""
        if self.chart_agent is None:
            self.chart_agent = ChartAgent(state.get("company_id", 0))
        return self.chart_agent.choose_chart_type(state)
    
    def format_chart_data(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Adapter for chart agent's format_chart_data method."""
        if self.chart_agent is None:
            self.chart_agent = ChartAgent(state.get("company_id", 0))
        return self.chart_agent.format_chart_data(state)
