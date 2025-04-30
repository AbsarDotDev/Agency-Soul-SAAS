import json
import logging
from typing import Dict, Any, List, Optional, Union

from app.agents.base_agent import VisualizationResult
from app.core.llm import get_llm

# Set up logging
logger = logging.getLogger(__name__)

async def generate_visualization_from_data(
    data: Union[List[Dict[str, Any]], Dict[str, Any]],
    query: str,
    visualization_type: Optional[str] = None,
    llm=None
) -> VisualizationResult:
    """Generate visualization data from SQL or structured data.
    
    Args:
        data: Data to visualize (list of dictionaries or dictionary)
        query: User query that prompted this visualization
        visualization_type: Optional visualization type (bar, line, pie, etc.)
        llm: Optional LLM instance (will be created if not provided)
        
    Returns:
        VisualizationResult with chart data and explanation
    """
    if llm is None:
        llm = get_llm()
    
    # Ensure we have data to visualize
    if not data:
        return VisualizationResult(
            data=None,
            explanation="No data available for visualization."
        )
    
    # Convert data to JSON string for prompt
    data_json = json.dumps(data, default=str)
    
    # Create a prompt for visualization data generation
    system_prompt = f"""
    You are an expert data visualizer. Based on the provided data and the user's question,
    generate a visualization specification that best presents the data to answer the question.
    
    The data is: {data_json}
    
    The user wants to visualize: {query}
    
    Create a chart specification in JSON format following this structure:
    {{
      "chart_type": "bar|line|pie|scatter|radar|bubble",
      "title": "Chart title based on query",
      "labels": ["Label1", "Label2", ...],
      "datasets": [
        {{
          "label": "Dataset Label",
          "data": [value1, value2, ...],
          "backgroundColor": "color code(s)"
        }}
      ],
      "options": {{
        "scales": {{
          "y": {{
            "title": "Y-axis label"
          }},
          "x": {{
            "title": "X-axis label"
          }}
        }}
      }}
    }}
    
    Also include a separate explanation of what the visualization shows and interesting insights from the data.
    
    Return a valid JSON object with these keys:
    {{
      "chart": Chart specification as above,
      "explanation": "Text explanation of what the visualization shows"
    }}
    """
    
    if visualization_type:
        system_prompt += f"\nThe user has specifically requested a {visualization_type} chart."
    
    # Generate visualization data using the LLM
    try:
        response = await llm.ainvoke(system_prompt)
        
        # Extract the JSON from the response
        response_text = response.content if hasattr(response, 'content') else response
        
        # Parse the JSON response
        try:
            result = json.loads(response_text)
            
            # Extract the visualization data and explanation
            chart_data = result.get("chart", {})
            explanation = result.get("explanation", "No explanation provided.")
            
            return VisualizationResult(
                data=chart_data,
                explanation=explanation
            )
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing LLM response as JSON: {str(e)}")
            
            # Attempt to extract JSON if the response contains other text
            try:
                # Try to find JSON-like content within the response
                start_idx = response_text.find("{")
                end_idx = response_text.rfind("}")
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = response_text[start_idx:end_idx + 1]
                    result = json.loads(json_str)
                    
                    # Extract the visualization data and explanation
                    chart_data = result.get("chart", {})
                    explanation = result.get("explanation", "No explanation provided.")
                    
                    return VisualizationResult(
                        data=chart_data,
                        explanation=explanation
                    )
            except:
                pass
            
            # Fallback to a default visualization
            return _create_fallback_visualization(data, query, visualization_type)
    
    except Exception as e:
        logger.error(f"Error generating visualization: {str(e)}")
        return _create_fallback_visualization(data, query, visualization_type)

def _create_fallback_visualization(
    data: Union[List[Dict[str, Any]], Dict[str, Any]],
    query: str,
    visualization_type: Optional[str] = None
) -> VisualizationResult:
    """Create a fallback visualization when the LLM fails.
    
    Args:
        data: Data to visualize
        query: User query that prompted this visualization
        visualization_type: Optional visualization type
        
    Returns:
        VisualizationResult with basic chart data
    """
    # Create a simple bar chart from the first 10 data points if data is a list
    if isinstance(data, list) and len(data) > 0:
        # Extract keys from the first item
        first_item = data[0]
        keys = list(first_item.keys())
        
        if len(keys) >= 2:
            # Use the first two columns for a basic chart
            labels = [str(item.get(keys[0], "")) for item in data[:10]]
            values = [_convert_to_number(item.get(keys[1], 0)) for item in data[:10]]
            
            chart_type = visualization_type or "bar"
            
            return VisualizationResult(
                data={
                    "chart_type": chart_type,
                    "title": f"Data visualization for: {query}",
                    "labels": labels,
                    "datasets": [
                        {
                            "label": keys[1],
                            "data": values,
                            "backgroundColor": "rgba(75, 192, 192, 0.6)"
                        }
                    ]
                },
                explanation=f"This is a basic {chart_type} chart showing {keys[1]} for each {keys[0]}. This is a fallback visualization as the custom generation failed."
            )
    
    # For dictionaries or empty lists, return a message
    return VisualizationResult(
        data=None,
        explanation="Could not generate a visualization for this data. The data may not be suitable for visualization, or the visualization generation may have failed."
    )

def _convert_to_number(value) -> float:
    """Convert a value to a number, defaulting to 0 if conversion fails.
    
    Args:
        value: Value to convert
        
    Returns:
        Converted number or 0
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0 