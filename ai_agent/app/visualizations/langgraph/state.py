from typing import List, Any, Annotated, Dict, Optional, Union
from typing_extensions import TypedDict
import operator

class State(TypedDict):
    """State for the visualization workflow."""
    # Input
    question: str                                      # The user's question
    company_id: int                                    # For company data isolation
    
    # SQL Agent Fields
    is_sql_relevant: bool                              # Whether the question can be answered with SQL
    relevant_tables: List[str]                         # Tables identified as relevant to the query
    relevance_reasoning: str                           # Reasoning for SQL relevance decision
    sql_query: str                                     # Generated SQL query
    sql_generation_reason: str                         # Reasoning for SQL query generation
    results: List[Dict[str, Any]]                      # Results from executing the SQL query
    execution_error: str                               # Error message from SQL execution (if any)
    
    # Chart Agent Fields
    chart_type: Annotated[str, operator.add]           # Type of chart to display
    chart_reason: Annotated[str, operator.add]         # Reasoning for the chart type selection
    chart_data: Dict[str, Any]                         # Formatted data for visualization

# Additional state classes for workflow_manager.py
class VisualizationInputState(TypedDict):
    """Input state for the visualization workflow."""
    query: str                   # The user's query
    company_id: int              # Company ID for data isolation
    chart_type: Optional[str]    # Optional requested chart type

class VisualizationOutputState(TypedDict):
    """Output state for the visualization workflow."""
    chart_data: Dict[str, Any]   # Formatted chart data for visualization
    explanation: str             # Text explanation of the visualization
    tokens_used: int             # Number of tokens used in the process
