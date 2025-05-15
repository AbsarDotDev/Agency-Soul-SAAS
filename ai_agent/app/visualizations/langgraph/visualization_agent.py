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
            # Ensure llm_manager objects exist before trying to access tokens_used
            sql_tokens = 0
            if hasattr(workflow_manager, 'sql_agent') and hasattr(workflow_manager.sql_agent, 'llm_manager'):
                sql_tokens = getattr(workflow_manager.sql_agent.llm_manager, 'tokens_used', 0)
            
            chart_tokens = 0
            if hasattr(workflow_manager, 'chart_agent') and hasattr(workflow_manager.chart_agent, 'llm_manager'):
                chart_tokens = getattr(workflow_manager.chart_agent.llm_manager, 'tokens_used', 0)
            
            self.tokens_used = sql_tokens + chart_tokens
            
            # Generate a user-friendly summary of the visualization
            visualization_summary = self._generate_visualization_summary(
                question=query,
                chart_type=result.get("chart_type"),
                chart_data=result.get("chart_data"),
                sql_query=result.get("sql_query")
            )
            # Add summary generation tokens to total
            # Assuming _generate_visualization_summary uses self.sql_agent.llm_manager or similar
            if hasattr(self, 'sql_agent') and hasattr(self.sql_agent, 'llm_manager'):
                 self.tokens_used += getattr(self.sql_agent.llm_manager, 'tokens_used_last_call', 0)


            # Ensure all keys are present
            final_result = {
                "chart_data": result.get("chart_data"),
                "chart_type": result.get("chart_type", ""),
                "answer": visualization_summary, # Use the new summary
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

    def _generate_visualization_summary(self, question: str, chart_type: Optional[str], chart_data: Optional[Dict[str, Any]], sql_query: Optional[str]) -> str:
        """
        Generate a user-friendly summary for the visualization using an LLM.
        """
        if not chart_type or chart_type == "none" or not chart_data:
            return "I couldn't generate a visualization for your query. Please try again or rephrase your question."

        # Prepare a sample of the chart data for the prompt
        data_sample_str = "Data not available for summary."
        if chart_data and chart_data.get("datasets"):
            try:
                # Try to get a few labels and data points
                labels_sample = chart_data.get("labels", [])[:3]
                datasets_sample = []
                for ds in chart_data.get("datasets", [])[:1]: # Sample from first dataset
                    dataset_summary = {"label": ds.get("label", "Data")}
                    if isinstance(ds.get("data"), list) and ds["data"]:
                        if isinstance(ds["data"][0], dict) and "y" in ds["data"][0]: # scatter/bubble
                            dataset_summary["points"] = ds["data"][:3]
                        else: # bar/line/pie
                            dataset_summary["values"] = ds["data"][:3]
                    datasets_sample.append(dataset_summary)
                
                if labels_sample or datasets_sample:
                    data_sample_str = f"Labels sample: {labels_sample}, Datasets sample: {datasets_sample}"
                else:
                    data_sample_str = "Chart data structure available but specific sample points are not easily summarized."

            except Exception as e:
                logger.error(f"Error creating data sample for summary: {e}")
                data_sample_str = "Error processing data for summary."
        elif chart_data and chart_data.get("data") and isinstance(chart_data.get("data"), list): # For table
             data_sample_str = f"Table data sample: {chart_data.get('data')[:2]}"


        prompt_template = ChatPromptTemplate.from_messages([
            ("system", 
             "You are an AI assistant tasked with creating a concise, user-friendly summary for a data visualization. "
             "The user asked a question, and a chart was generated. Your goal is to explain what the chart shows in simple terms. "
             "Do NOT mention the SQL query or technical details of how the chart was made. Focus on the data insights."
             "Keep the summary to 1-3 sentences."),
            ("human", 
             "Original question: {question}\n"
             "Chart type generated: {chart_type}\n"
             "Data sample: {data_sample}\n\n"
             "Please provide a brief, user-friendly summary of this visualization:")
        ])

        try:
            # Ensure SQL agent and its LLM manager are initialized
            if self.sql_agent is None or self.sql_agent.llm_manager is None:
                # Initialize if they were not created, e.g. if workflow had an early error
                # This assumes company_id is available in self.
                current_company_id = self.company_id if hasattr(self, 'company_id') else 0
                if self.sql_agent is None:
                    self.sql_agent = SQLAgent(company_id=current_company_id)
                if self.sql_agent.llm_manager is None: # Should be created by SQLAgent constructor
                    logger.error("LLM Manager in SQLAgent is None even after SQLAgent init for summary.")
                    return "There was an issue generating the summary."


            summary = self.sql_agent.llm_manager.invoke(
                prompt_template,
                question=question,
                chart_type=chart_type,
                data_sample=data_sample_str
            )
            if summary and isinstance(summary, str):
                return summary.strip()
            else:
                logger.warning(f"LLM returned an empty or non-string summary: {summary}")
                return "Here is the visualization you requested."
        except Exception as e:
            logger.error(f"Error generating visualization summary with LLM: {e}")
            return "I've generated the visualization. If you need a summary, please ask!"
