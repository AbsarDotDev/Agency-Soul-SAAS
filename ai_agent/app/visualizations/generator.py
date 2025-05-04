import json
import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import base64
from io import BytesIO
from typing import Dict, Any, List, Optional, Union, Tuple
import re

from app.agents.base_agent import VisualizationResult
from app.core.llm import get_llm, get_embedding_model
from app.core.token_manager import TokenManager

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
        VisualizationResult including chart data, explanation, and tokens used.
    """
    if llm is None:
        llm = get_llm()
    
    # Ensure we have data to visualize
    if not data:
        return VisualizationResult(
            data=None,
            explanation="No data available for visualization.",
            tokens_used=0
        )
    
    try:
        # Convert data to DataFrame for easier manipulation
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            # Handle dictionary with lists as values
            if all(isinstance(v, list) for v in data.values()):
                df = pd.DataFrame(data)
            else:
                # Single row dict
                df = pd.DataFrame([data])
        else:
            # Unknown format, try to convert to DataFrame
            df = pd.DataFrame(data)
        
        # Ensure DataFrame is not empty
        if df.empty:
            return VisualizationResult(
                data=None,
                explanation="No data available for visualization.",
                tokens_used=0
            )
        
        # Try LLM-based visualization generation first
        visualization_tokens = 0
        try:
            llm_result = await _generate_visualization_with_llm(df, query, visualization_type, llm)
            if llm_result and llm_result.data and llm_result.tokens_used > 0:
                return llm_result
            elif llm_result:
                logger.warning(f"LLM visualization call succeeded but returned no data or tokens. Falling back.")
                visualization_tokens = llm_result.tokens_used
            else:
                logger.warning(f"LLM visualization call succeeded but returned no data or tokens. Falling back.")
                visualization_tokens = 0
        except Exception as e:
            logger.warning(f"LLM visualization generation failed: {str(e)}. Falling back to direct generation.")
            pass
        
        # If LLM-based generation fails or doesn't return data, use direct visualization generation
        logger.info("Falling back to direct visualization generation.")
        direct_result = _generate_visualization_directly(df, query, visualization_type)
        direct_result.tokens_used += visualization_tokens
        return direct_result
    
    except Exception as e:
        logger.error(f"Error in visualization generation: {str(e)}")
        # Always return a visualization, even if it's basic
        return _create_emergency_fallback_visualization(data, query, visualization_type)

async def _generate_visualization_with_llm(
    df: pd.DataFrame,
    query: str,
    visualization_type: Optional[str] = None,
    llm=None
) -> VisualizationResult:
    """Generate visualization using LLM and track tokens.
    
    Args:
        df: DataFrame with data to visualize
        query: User query
        visualization_type: Optional visualization type
        llm: LLM instance
        
    Returns:
        VisualizationResult including tokens used by this LLM call.
    """
    data_json = df.head(20).to_json(orient='records')
    
    # Simplified prompt focusing *only* on the JSON output structure
    system_prompt = f"""
CONTEXT:
User query: "{query}"
Data (first 20 rows JSON): {data_json}
Requested chart type (if any): {visualization_type or 'auto-detect'}

TASK:
Generate ONLY the JSON specification for a Chart.js chart based on the context. 
Adhere STRICTLY to this JSON structure:
{{
  "chart_type": "bar|line|pie|scatter|radar|bubble",
  "title": "Concise chart title reflecting query",
  "labels": ["Label1", "Label2", ...],
  "datasets": [
    {{
      "label": "Dataset Label", 
      "data": [value1, value2, ...],
      "backgroundColor": ["rgba(R,G,B,0.7)", ...]
    }}
  ],
  "options": {{ 
    "scales": {{ "y": {{ "title": "Y-axis label" }}, "x": {{ "title": "X-axis label" }} }}
  }}
}}

IMPORTANT: The "options" field must ALWAYS be an OBJECT ({{ }}), NEVER an array ([ ]). 
For pie charts, you can use empty options: "options": {{ }} but never "options": []

OUTPUT:
Return ONLY the valid JSON object. Do not include explanations or any text outside the JSON structure.
"""
    
    # Use fixed token count (2) for visualization as requested
    tokens_used = 2
    
    try:
        # We still count tokens for debugging, but we'll return fixed value
        prompt_tokens = TokenManager.count_tokens(system_prompt)
        
        logger.info("Invoking LLM for visualization refinement...")
        response = await llm.ainvoke(system_prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        logger.info(f"LLM visualization response raw: {response_text[:500]}...")
        
        completion_tokens = TokenManager.count_tokens(response_text)
        # Log actual usage for debugging but return fixed value
        logger.info(f"LLM visualization call token usage: prompt={prompt_tokens}, completion={completion_tokens}, total={prompt_tokens + completion_tokens} (fixed value: {tokens_used})")
        
        # --- More Robust JSON Parsing --- 
        chart_data = None
        explanation = "Visualization generated based on data."
        parsed_json = None
        
        # Clean potential markdown fences first
        response_text_cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", response_text.strip(), flags=re.IGNORECASE)

        try:
            # Attempt 1: Parse the cleaned response directly
            parsed_json = json.loads(response_text_cleaned)
            if isinstance(parsed_json, dict) and "chart_type" in parsed_json:
                chart_data = parsed_json
                
                # Ensure options is always an object, never an array
                if "options" in chart_data:
                    if chart_data["options"] is None or (isinstance(chart_data["options"], list) and len(chart_data["options"]) == 0):
                        chart_data["options"] = {}
                else:
                    chart_data["options"] = {}
                
                logger.info("Successfully parsed cleaned LLM response as chart JSON.")
            else:
                logger.warning("Parsed JSON from LLM response doesn't match expected chart structure.")
                # Handle potential {chart:..., explanation:...} structure if prompt ignored
                if isinstance(parsed_json, dict) and "chart" in parsed_json:
                    chart_data = parsed_json.get("chart")
                    
                    # Ensure options is always an object here too
                    if "options" in chart_data:
                        if chart_data["options"] is None or (isinstance(chart_data["options"], list) and len(chart_data["options"]) == 0):
                            chart_data["options"] = {}
                    else:
                        chart_data["options"] = {}
                        
                    explanation = parsed_json.get("explanation", explanation)
                    logger.info("Extracted chart/explanation from unexpected LLM response structure.")
        
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse cleaned LLM response as JSON: {e}. Original response may have had issues.")
            # No further extraction attempts needed as markdown was pre-cleaned

        # --- Final check and return --- 
        if chart_data and isinstance(chart_data, dict) and 'chart_type' in chart_data and 'labels' in chart_data and 'datasets' in chart_data:
            # Validate essential keys further if needed
            return VisualizationResult(
                data=chart_data,
                explanation=explanation,
                tokens_used=tokens_used
            )
        else:
            logger.error("Failed to obtain valid chart JSON from LLM response after parsing attempts.")
            # Return tokens used even on failure, but no chart data
            return VisualizationResult(data=None, explanation="Could not generate visualization details.", tokens_used=tokens_used)

    except Exception as e:
        logger.error(f"Exception during LLM visualization invoke/parsing: {str(e)}", exc_info=True)
        # Return fixed token count even on error
        return VisualizationResult(data=None, explanation=f"Error generating visualization details: {str(e)}", tokens_used=tokens_used)

def _generate_visualization_directly(
    df: pd.DataFrame,
    query: str,
    visualization_type: Optional[str] = None
) -> VisualizationResult:
    """Generate visualization directly from DataFrame (0 tokens used)."""
    # Determine best visualization type if not specified
    if not visualization_type:
        visualization_type = _determine_best_visualization_type(df, query)
    
    # Generate visualization based on type
    if visualization_type == "pie":
        result = _generate_pie_chart(df, query)
    elif visualization_type == "line":
        result = _generate_line_chart(df, query)
    elif visualization_type == "scatter":
        result = _generate_scatter_chart(df, query)
    else:  # Default to bar chart
        result = _generate_bar_chart(df, query)
    
    result.tokens_used = 0 # Explicitly set tokens to 0 for direct generation
    return result

def _determine_best_visualization_type(df: pd.DataFrame, query: str) -> str:
    """Determine the best visualization type based on data and query.
    
    Args:
        df: DataFrame with data to visualize
        query: User query
        
    Returns:
        Visualization type
    """
    query_lower = query.lower()
    
    # Check for keywords in query
    if any(term in query_lower for term in ["pie", "distribution", "breakdown", "composition"]):
        return "pie"
    elif any(term in query_lower for term in ["line", "trend", "time", "over time", "growth"]):
        return "line"
    elif any(term in query_lower for term in ["scatter", "correlation", "relationship"]):
        return "scatter"
    
    # If query doesn't give clear indication, check data characteristics
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    # Check for date/time columns
    date_columns = [col for col in df.columns if df[col].dtype == 'datetime64[ns]' 
                   or (isinstance(df[col].dtype, object) and 'date' in col.lower())]
    
    # If we have date columns and numeric columns, line chart is good
    if date_columns and numeric_columns:
        return "line"
    
    # If we have exactly two numeric columns, scatter might be appropriate
    if len(numeric_columns) == 2:
        return "scatter"
    
    # If we have one categorical column and one numeric column, pie chart might work
    if len(categorical_columns) == 1 and len(numeric_columns) == 1 and len(df) <= 10:
        return "pie"
    
    # Default to bar chart
    return "bar"

def _generate_bar_chart(df: pd.DataFrame, query: str) -> VisualizationResult:
    """Generate a bar chart visualization.
    
    Args:
        df: DataFrame with data to visualize
        query: User query
        
    Returns:
        VisualizationResult
    """
    # Find columns for x and y axes
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    if not numeric_columns:
        # If no numeric columns, count occurrences of categorical values
        if categorical_columns:
            x_column = categorical_columns[0]
            # Count occurrences and convert to DataFrame
            value_counts = df[x_column].value_counts()
            df_counts = pd.DataFrame({
                x_column: value_counts.index,
                'count': value_counts.values
            })
            x_column = x_column
            y_column = 'count'
            df = df_counts
        else:
            # No suitable columns
            return _create_fallback_visualization(df.to_dict('records'), query, "bar")
    else:
        # Use first categorical column for x axis and first numeric column for y axis
        if categorical_columns:
            x_column = categorical_columns[0]
            y_column = numeric_columns[0]
            
            # Aggregate data if there are multiple values for each category
            if df[x_column].nunique() < len(df):
                df = df.groupby(x_column)[y_column].sum().reset_index()
        else:
            # Use first two numeric columns if no categorical columns
            x_column = numeric_columns[0]
            y_column = numeric_columns[1] if len(numeric_columns) > 1 else numeric_columns[0]
    
    # Limit to top 15 items
    if len(df) > 15:
        df = df.sort_values(by=y_column, ascending=False).head(15)
    
    # Create chart data
    labels = df[x_column].astype(str).tolist()
    data = df[y_column].tolist()
    
    # Generate a title based on the query
    title = f"Bar Chart: {y_column} by {x_column}"
    
    # Create chart specification
    chart_data = {
        "chart_type": "bar",
        "title": title,
        "labels": labels,
        "datasets": [
            {
                "label": y_column,
                "data": data,
                "backgroundColor": "rgba(75, 192, 192, 0.6)"
            }
        ],
        "options": {
            "scales": {
                "y": {
                    "title": y_column
                },
                "x": {
                    "title": x_column
                }
            }
        }
    }
    
    # Generate explanation
    explanation = f"This bar chart shows {y_column} for each {x_column}. "
    
    if len(df) > 0:
        max_idx = df[y_column].idxmax()
        min_idx = df[y_column].idxmin()
        max_value = df.loc[max_idx, x_column]
        min_value = df.loc[min_idx, x_column]
        explanation += f"The highest value is for {max_value} and the lowest is for {min_value}."
    
    return VisualizationResult(
        data=chart_data,
        explanation=explanation,
        tokens_used=0
    )

def _generate_line_chart(df: pd.DataFrame, query: str) -> VisualizationResult:
    """Generate a line chart visualization.
    
    Args:
        df: DataFrame with data to visualize
        query: User query
        
    Returns:
        VisualizationResult
    """
    # Check for date/time columns
    date_columns = [col for col in df.columns if df[col].dtype == 'datetime64[ns]' 
                   or (isinstance(df[col].dtype, object) and ('date' in col.lower() or 'time' in col.lower()))]
    
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    
    if not numeric_columns:
        # No numeric columns, fallback to bar chart
        return _generate_bar_chart(df, query)
    
    # If we have date columns, use the first one for x axis
    if date_columns:
        x_column = date_columns[0]
        
        # Convert to datetime if it's not already
        if df[x_column].dtype != 'datetime64[ns]':
            try:
                df[x_column] = pd.to_datetime(df[x_column])
            except:
                # If conversion fails, use as is
                pass
        
        # Sort by date
        df = df.sort_values(by=x_column)
        
        # Format dates for labels
        labels = df[x_column].dt.strftime('%Y-%m-%d').tolist()
    else:
        # If no date columns, use the first categorical column or first numeric column
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        if categorical_columns:
            x_column = categorical_columns[0]
        else:
            x_column = numeric_columns[0]
            numeric_columns.remove(x_column)
        
        # Sort by x column
        df = df.sort_values(by=x_column)
        
        # Convert to string for labels
        labels = df[x_column].astype(str).tolist()
    
    # Use all remaining numeric columns for datasets
    datasets = []
    
    for y_column in numeric_columns[:3]:  # Limit to first 3 numeric columns
        data = df[y_column].tolist()
        
        datasets.append({
            "label": y_column,
            "data": data,
            "borderColor": _get_random_color(),
            "fill": False
        })
    
    # Generate a title based on the query
    title = f"Line Chart: {', '.join([d['label'] for d in datasets])} over {x_column}"
    
    # Create chart specification
    chart_data = {
        "chart_type": "line",
        "title": title,
        "labels": labels,
        "datasets": datasets,
        "options": {
            "scales": {
                "y": {
                    "title": "Value"
                },
                "x": {
                    "title": x_column
                }
            }
        }
    }
    
    # Generate explanation
    explanation = f"This line chart shows how {', '.join([d['label'] for d in datasets])} changes over {x_column}. "
    
    if len(datasets) > 0 and len(df) > 1:
        first_dataset = datasets[0]
        first_value = first_dataset['data'][0]
        last_value = first_dataset['data'][-1]
        
        if last_value > first_value:
            explanation += f"Overall, there is an upward trend in {first_dataset['label']}."
        elif last_value < first_value:
            explanation += f"Overall, there is a downward trend in {first_dataset['label']}."
        else:
            explanation += f"Overall, {first_dataset['label']} remained relatively stable."
    
    return VisualizationResult(
        data=chart_data,
        explanation=explanation,
        tokens_used=0
    )

def _generate_pie_chart(df: pd.DataFrame, query: str) -> VisualizationResult:
    """Generate a pie chart visualization.
    
    Args:
        df: DataFrame with data to visualize
        query: User query
        
    Returns:
        VisualizationResult
    """
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    if not numeric_columns or not categorical_columns:
        # Need both numeric and categorical columns for pie chart
        if not categorical_columns and len(numeric_columns) >= 2:
            # If no categorical but multiple numeric, use first numeric as categories
            category_column = numeric_columns[0]
            value_column = numeric_columns[1]
        elif len(categorical_columns) >= 1 and not numeric_columns:
            # If only categorical columns, count occurrences
            category_column = categorical_columns[0]
            value_counts = df[category_column].value_counts()
            df = pd.DataFrame({
                category_column: value_counts.index,
                'count': value_counts.values
            })
            value_column = 'count'
        else:
            # Fallback to bar chart
            return _generate_bar_chart(df, query)
    else:
        # Use first categorical column and first numeric column
        category_column = categorical_columns[0]
        value_column = numeric_columns[0]
        
        # Aggregate data if needed
        if df[category_column].nunique() < len(df):
            df = df.groupby(category_column)[value_column].sum().reset_index()
    
    # Limit to top 10 categories
    if len(df) > 10:
        df = df.sort_values(by=value_column, ascending=False).head(10)
    
    # Create chart data
    labels = df[category_column].astype(str).tolist()
    data = df[value_column].tolist()
    
    # Generate colors
    colors = [_get_random_color(0.7) for _ in range(len(labels))]
    
    # Generate a title based on the query
    title = f"Pie Chart: Distribution of {value_column} by {category_column}"
    
    # Create chart specification
    chart_data = {
        "chart_type": "pie",
        "title": title,
        "labels": labels,
        "datasets": [
            {
                "data": data,
                "backgroundColor": colors
            }
        ]
    }
    
    # Generate explanation
    explanation = f"This pie chart shows the distribution of {value_column} across different {category_column} categories. "
    
    if len(df) > 0:
        top_category = df.loc[df[value_column].idxmax(), category_column]
        top_percentage = (df[value_column].max() / df[value_column].sum()) * 100
        explanation += f"The largest segment is {top_category}, which accounts for approximately {top_percentage:.1f}% of the total."
    
    return VisualizationResult(
        data=chart_data,
        explanation=explanation,
        tokens_used=0
    )

def _generate_scatter_chart(df: pd.DataFrame, query: str) -> VisualizationResult:
    """Generate a scatter chart visualization.
    
    Args:
        df: DataFrame with data to visualize
        query: User query
        
    Returns:
        VisualizationResult
    """
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    
    if len(numeric_columns) < 2:
        # Need at least two numeric columns for scatter plot
        return _generate_bar_chart(df, query)
    
    # Use first two numeric columns
    x_column = numeric_columns[0]
    y_column = numeric_columns[1]
    
    # Create data points
    data = []
    for _, row in df.iterrows():
        data.append({
            "x": float(row[x_column]),
            "y": float(row[y_column])
        })
    
    # Generate a title based on the query
    title = f"Scatter Plot: {y_column} vs {x_column}"
    
    # Create chart specification
    chart_data = {
        "chart_type": "scatter",
        "title": title,
        "datasets": [
            {
                "label": f"{y_column} vs {x_column}",
                "data": data,
                "backgroundColor": "rgba(75, 192, 192, 0.6)"
            }
        ],
        "options": {
            "scales": {
                "y": {
                    "title": y_column
                },
                "x": {
                    "title": x_column
                }
            }
        }
    }
    
    # Generate explanation
    explanation = f"This scatter plot shows the relationship between {x_column} and {y_column}. "
    
    # Calculate correlation
    try:
        correlation = df[x_column].corr(df[y_column])
        if correlation > 0.7:
            explanation += f"There appears to be a strong positive correlation ({correlation:.2f}) between the two variables."
        elif correlation > 0.3:
            explanation += f"There appears to be a moderate positive correlation ({correlation:.2f}) between the two variables."
        elif correlation > -0.3:
            explanation += f"There appears to be little to no correlation ({correlation:.2f}) between the two variables."
        elif correlation > -0.7:
            explanation += f"There appears to be a moderate negative correlation ({correlation:.2f}) between the two variables."
        else:
            explanation += f"There appears to be a strong negative correlation ({correlation:.2f}) between the two variables."
    except:
        explanation += "The relationship between these variables could not be determined statistically."
    
    return VisualizationResult(
        data=chart_data,
        explanation=explanation,
        tokens_used=0
    )

def _create_fallback_visualization(
    data: Union[List[Dict[str, Any]], Dict[str, Any]],
    query: str,
    visualization_type: Optional[str] = None
) -> VisualizationResult:
    """Create a fallback visualization when specific chart generation fails.
    
    Args:
        data: Data to visualize
        query: User query
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
                explanation=f"This is a basic {chart_type} chart showing {keys[1]} for each {keys[0]}. This is a fallback visualization as the custom generation failed.",
                tokens_used=0
            )
    
    # For dictionaries or empty lists, create minimal visualization
    chart_type = visualization_type or "bar"
    
    return VisualizationResult(
        data={
            "chart_type": chart_type,
            "title": f"Data visualization for: {query}",
            "labels": ["No data"],
            "datasets": [
                {
                    "label": "No data available",
                    "data": [0],
                    "backgroundColor": "rgba(200, 200, 200, 0.6)"
                }
            ]
        },
        explanation="Could not generate a detailed visualization for this data. The data structure may not be suitable for the requested visualization type.",
        tokens_used=0
    )

def _create_emergency_fallback_visualization(
    data: Any,
    query: str,
    visualization_type: Optional[str] = None
) -> VisualizationResult:
    """Create an emergency fallback visualization when all else fails.
    
    Args:
        data: Original data (may be in any format)
        query: User query
        visualization_type: Optional visualization type
        
    Returns:
        VisualizationResult with minimal chart data
    """
    chart_type = visualization_type or "bar"
    
    return VisualizationResult(
        data={
            "chart_type": chart_type,
            "title": f"Visualization for: {query}",
            "labels": ["Data point 1", "Data point 2"],
            "datasets": [
                {
                    "label": "Sample data",
                    "data": [1, 2],
                    "backgroundColor": "rgba(75, 192, 192, 0.6)"
                }
            ]
        },
        explanation="I encountered an error while generating a visualization for your data. This is a placeholder chart. You may want to try a different query or visualization type.",
        tokens_used=0
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

def _get_random_color(opacity: float = 0.6) -> str:
    """Generate a random color with the given opacity.
    
    Args:
        opacity: Opacity value (0-1)
        
    Returns:
        Color string in rgba format
    """
    import random
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    
    return f"rgba({r}, {g}, {b}, {opacity})" 