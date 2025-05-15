from typing import Dict, Any, List, Optional
import logging
import json
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from app.visualizations.langgraph.llm_manager import LLMManager
from app.visualizations.chart_utils import create_chart_colors, get_chart_color

# Set up logging with less verbose level
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Chart type map for standardization
CHART_TYPE_MAP = {
    "bar": "bar",
    "horizontal_bar": "horizontalBar",
    "line": "line",
    "pie": "pie",
    "doughnut": "doughnut",
    "radar": "radar",
    "scatter": "scatter",
    "bubble": "bubble",
    "polarArea": "polarArea",
    "table": "table",
    "number": "number"
}

def _adjust_color_alpha(color_str: str, alpha: float) -> str:
    """Helper to add or adjust alpha in a color string."""
    if color_str.startswith('rgba'):
        return re.sub(r"rgba\(([^,]+,[^,]+,[^,]+),[^\)]+\)", f"rgba(\1,{alpha})", color_str)
    elif color_str.startswith('rgb'):
        return color_str.replace('rgb', 'rgba').replace(')', f',{alpha})')
    elif color_str.startswith('#') and len(color_str) == 7: # #RRGGBB
        r = int(color_str[1:3], 16)
        g = int(color_str[3:5], 16)
        b = int(color_str[5:7], 16)
        return f"rgba({r},{g},{b},{alpha})"
    elif color_str.startswith('#') and len(color_str) == 4: # #RGB
        r = int(color_str[1] * 2, 16)
        g = int(color_str[2] * 2, 16)
        b = int(color_str[3] * 2, 16)
        return f"rgba({r},{g},{b},{alpha})"
    return color_str # Fallback if format is unknown

class ChartAgent:
    """
    Agent responsible for visualization-related tasks in the workflow.
    Determines chart types, formats data, and generates visualizations.
    """
    
    def __init__(self, company_id: Optional[int] = None):
        """
        Initialize the chart agent.
        
        Args:
            company_id: Optional company ID for token tracking
        """
        self.llm_manager = LLMManager(company_id)
    
    def choose_chart_type(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Choose the most appropriate chart type for the data.
        If user explicitly requests a chart type, try to honor it.
        """
        question = state['question']
        results = state.get('results', [])
        sql_query = state.get('sql_query', '')
        data_type = state.get('data_type', 'unknown')
        columns = state.get('columns', [])
        
        # If there's an error in SQL execution or no results, return no chart
        if 'error' in state or not results:
            return {
                "chart_type": "none",
                "chart_reason": "No data available for visualization due to an error or empty results."
            }

        # Attempt to honor explicit user request first
        requested_chart_type = None
        lower_question = question.lower()
        for chart_key, chart_js_type in CHART_TYPE_MAP.items():
            if chart_key.replace("_", " ") in lower_question: # e.g. "horizontal bar"
                requested_chart_type = chart_js_type
                break
            elif chart_js_type.lower() in lower_question: # e.g. "horizontalBar" (less likely from user)
                requested_chart_type = chart_js_type
                break
        
        if requested_chart_type:
            # Basic validation if the requested chart type is somewhat suitable
            # This can be expanded, for now, we assume if user requests it, we try it.
            # except for table/number which are more fallbacks.
            if requested_chart_type not in ["table", "number"]:
                 logger.info(f"User explicitly requested chart type: {requested_chart_type}. Attempting to use it.")
                 # We might still need a reason, but it's primarily user-driven
                 state['chart_type'] = requested_chart_type
                 state['chart_reason'] = f"User requested a {requested_chart_type} chart."
                 # We directly format based on user's choice if possible, LLM choice is a fallback.
                 # The workflow should decide if it calls choose_chart_type and then format, or if format_chart_data itself can use state['chart_type']
                 # For now, returning the user's choice here for the workflow to use.
                 return {
                    "chart_type": requested_chart_type,
                    "chart_reason": f"User requested a {requested_chart_type} chart."
                }

        # Analyze the data to determine the chart type via LLM
        try:
            # Create prompt for chart type selection using simple strings
            system_prompt = """You are an AI assistant specialized in data visualization. Your task is to recommend the most appropriate chart type based on the user's question, SQL query, and the resulting data. Return your recommendation as valid JSON.

Available chart types:
- bar: For comparing categories, especially with a small number of categories.
- horizontalBar: Similar to bar, but horizontal. Good for long labels.
- line: For showing trends over time or continuous data.
- pie: For showing parts of a whole, useful when there are few categories (ideally < 7).
- doughnut: Similar to pie, but with a hole in the center.
- scatter: For showing relationship between two (or three for bubble) numerical variables.
- bubble: A variation of scatter, where a third dimension is shown by the size of the bubbles. Requires 3 data points per item (x, y, size).
- radar: For comparing multiple quantitative variables for one or more series.
- polarArea: Similar to a pie chart, but compares data by the angle and distance from the center.
- table: For raw data when a chart visualization isn't suitable or if data is complex.
- number: For a single, important value or KPI.

Consider the data structure:
1. Number of categories/points (e.g., < 10 is good for pie charts).
2. Whether time is involved (line charts work well).
3. Number of variables being compared (e.g., 2 for scatter, 3 for bubble).
4. The specific user question and any explicit chart type mentions.
5. If the data seems unsuitable for any chart, recommend 'table'.

Return a JSON object with:
1. chart_type: One of the available types (e.g., "bar", "line", "bubble", "table").
2. chart_reason: Brief explanation of your choice.
"""

            human_prompt = f"""
User Question: {question}
SQL Query Used: {sql_query}
Resulting Data Structure Type: {data_type}
Resulting Data Columns: {columns}
Data Sample (first 5 rows): {str(results[:5])}

Based on all this information, recommend the most appropriate chart type. If the user's question explicitly asked for a chart type that seems reasonable for the data, prioritize that. Otherwise, choose the best fit.
"""

            # Set up prompt template
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", human_prompt)
            ])

            # Get chart recommendation
            response_content = self.llm_manager.invoke(prompt)
            
            # Extract JSON from the response using regex
            chart_json = self._extract_json(response_content, default_type='table')
            
            # Ensure chart_type is one of the known types
            if chart_json.get('chart_type') not in CHART_TYPE_MAP.values() and chart_json.get('chart_type') not in CHART_TYPE_MAP.keys() :
                logger.warning(f"LLM recommended an unknown chart type: {chart_json.get('chart_type')}. Defaulting to table.")
                chart_json['chart_type'] = 'table'
                chart_json['chart_reason'] = "LLM recommended an unknown chart type, defaulted to table."
            elif chart_json.get('chart_type') in CHART_TYPE_MAP.keys(): # if LLM returns 'bar' instead of 'bar'
                chart_json['chart_type'] = CHART_TYPE_MAP[chart_json.get('chart_type')]

            logger.info(f"LLM recommended chart type: {chart_json.get('chart_type')}, Reason: {chart_json.get('chart_reason')}")
            return chart_json
        except Exception as e:
            logger.error(f"Error selecting chart type via LLM: {str(e)}")
            return {
                "chart_type": "table", # Fallback to table on error
                "chart_reason": f"Error selecting chart type: {str(e)}. Displaying data as a table."
            }
            
    def format_chart_data(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format the data according to the chosen chart type.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with formatted chart data
        """
        chart_type = state.get('chart_type', 'none')
        results = state.get('results', [])
        columns = state.get('columns', [])
        question = state.get('question', '')
        data_type = state.get('data_type', 'unknown')
        
        # If there's no chart type or results, return no chart data
        if chart_type == 'none' or not results:
            state['chart_data'] = None
            return state
        
        try:
            # Check for SQL string results that weren't properly processed
            if len(results) == 1 and 'result' in results[0] and isinstance(results[0]['result'], str):
                result_str = results[0]['result']
                if result_str.startswith('[') and ')' in result_str:
                    try:
                        # Try to parse tuples from string representation
                        import ast
                        parsed_tuples = ast.literal_eval(result_str)
                        
                        # Generate label/value pairs
                        if all(isinstance(item, tuple) and len(item) >= 2 for item in parsed_tuples):
                            results = []
                            for tup in parsed_tuples:
                                results.append({
                                    "label": str(tup[0]),
                                    "value": float(tup[1]) if isinstance(tup[1], (int, float)) else 0
                                })
                            data_type = 'key_value'
                            columns = ['label', 'value']
                            logger.info(f"Parsed tuple string into {len(results)} label/value pairs")
                    except Exception as e:
                        logger.error(f"Failed to parse tuple string: {str(e)}")
            
            # Process data based on chart type
            if chart_type in ['bar', 'horizontalBar']:
                formatted_data = self._format_bar_data(results, question, data_type, columns, state.get('sql_query',''), horizontal=(chart_type == 'horizontalBar'))
            elif chart_type == 'line':
                formatted_data = self._format_line_data(results, question, data_type, columns, state.get('sql_query',''))
            elif chart_type in ['pie', 'doughnut']:
                formatted_data = self._format_pie_data(results, question, data_type, columns, chart_type, state.get('sql_query',''))
            elif chart_type == 'scatter':
                formatted_data = self._format_scatter_data(results, question, data_type, columns, state.get('sql_query',''))
            elif chart_type == 'bubble':
                formatted_data = self._format_bubble_data(results, question, data_type, columns, state.get('sql_query',''))
            elif chart_type == 'radar':
                formatted_data = self._format_radar_data(results, question, data_type, columns, state.get('sql_query',''))
            else:
                # Default to table for unsupported or 'table'/'none' chart types
                logger.info(f"Chart type '{chart_type}' not specifically handled or is 'table'/'none'. Formatting as table.")
                formatted_data = {
                    'chart_type': 'table',
                    'title': self._generate_title(question),
                    'data': results,
                    'columns': columns,
                    'headers': columns
                }
            
            logger.info(f"Formatted chart data for type {chart_type}: {formatted_data.keys()}")
            
            # Add to state
            state['chart_data'] = formatted_data
            return state
            
        except Exception as e:
            logger.error(f"Error formatting chart data: {str(e)}")
            # Fallback to table format
            state['chart_data'] = {
                'chart_type': 'table',
                'title': self._generate_title(question),
                'data': results,
                'columns': columns,
                'headers': columns
            }
            state['error'] = f"Error formatting chart data: {str(e)}"
            return state
    
    def _extract_json(self, text: str, default_type: str = 'bar') -> Dict[str, Any]:
        """
        Extract JSON from text, with fallback to default values.
        Tries to find chart_type and chart_reason.
        """
        try:
            # Try direct parsing first
            try:
                data = json.loads(text)
                if 'chart_type' in data and 'chart_reason' in data:
                    return data
            except json.JSONDecodeError:
                pass
                
            # Try to extract JSON from code blocks
            json_pattern = r'```(?:json)?\\s*([\\s\\S]*?)\\s*```'
            matches = re.findall(json_pattern, text, re.DOTALL)
            
            if matches:
                for match in matches:
                    try:
                        data = json.loads(match)
                        if 'chart_type' in data and 'chart_reason' in data:
                            return data
                    except json.JSONDecodeError:
                        continue
            
            # Try to extract JSON between curly braces (most common LLM mistake)
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                try:
                    json_str = text[start_idx:end_idx]
                    data = json.loads(json_str)
                    if 'chart_type' in data and 'chart_reason' in data:
                         return data
                    # If only chart_type is present, try to find a reason
                    if 'chart_type' in data and 'chart_reason' not in data:
                        data['chart_reason'] = "LLM selected this chart type as most appropriate."
                        return data
                except json.JSONDecodeError:
                    pass
            
            # Fallback: Look for chart type mentions in text if direct JSON fails
            chart_type_keys = list(CHART_TYPE_MAP.keys()) 
            chart_type_values = list(CHART_TYPE_MAP.values())
            all_possible_types = chart_type_keys + chart_type_values

            found_type = default_type 
            lower_text = text.lower()

            for type_mention in all_possible_types:
                # e.g. "chart_type": "line" or "The best chart is line."
                if f'"{type_mention}"' in lower_text or f"'{type_mention}'" in lower_text or f" {type_mention} " in lower_text:
                    found_type = CHART_TYPE_MAP.get(type_mention, type_mention) # Ensure we get the Chart.js type
                    break
            
            reason_match = re.search(r'(?:reason|explanation)[:\\s]*(.*?)(?:\\n|\\.|$)', text, re.IGNORECASE | re.DOTALL)
            reason = reason_match.group(1).strip() if reason_match else f"Selected {found_type} based on data characteristics."
            
            logger.warning(f"Could not parse full JSON for chart type selection. Fallback: type='{found_type}', reason='{reason}'")
            return {
                "chart_type": found_type,
                "chart_reason": reason
            }
            
        except Exception as e:
            logger.error(f"Error extracting JSON for chart type: {str(e)}")
            return {
                "chart_type": default_type,
                "chart_reason": f"Default chart ({default_type}) selected due to parsing error: {e}"
            }
    
    def _format_bar_data(self, results: List[Dict[str, Any]], question: str, data_type: str, columns: List[str], sql_query: Optional[str], horizontal: bool = False) -> Dict[str, Any]:
        """Format data for bar or horizontal bar charts."""
        if not results:
            return self._empty_chart_data("bar" if not horizontal else "horizontalBar", question, "No data to display.")

        labels = []
        values = []

        if data_type == 'key_value' and all('label' in item and 'value' in item for item in results):
            labels = [str(item['label']) for item in results]
            try:
                values = [float(item['value']) if item['value'] is not None else 0.0 for item in results]
            except (ValueError, TypeError) as e:
                logger.error(f"Error converting bar data values to float: {e}. Using 0.0 for unparseable values.")
                values = [0.0] * len(results) # Fallback, consider how to handle
        elif columns and len(columns) >= 2 and all(isinstance(item, dict) for item in results):
            label_col, value_col = columns[0], columns[1]
            labels = [str(item.get(label_col, '')) for item in results]
            try:
                values = [float(item.get(value_col)) if item.get(value_col) is not None else 0.0 for item in results]
            except (ValueError, TypeError) as e:
                logger.error(f"Error converting bar data values from columns to float: {e}. Using 0.0.")
                values = [0.0] * len(results)
        else:
            logger.warning(f"Bar chart data is not in expected 'key_value' or multi-column format. Attempting LLM format or table fallback.")
            # Attempt to use the first column as label and second as value if possible, otherwise LLM
            if results and isinstance(results[0], dict) and len(results[0].keys()) >=2:
                 keys = list(results[0].keys())
                 labels = [str(r.get(keys[0],'')) for r in results]
                 try:
                    values = [float(r.get(keys[1])) if r.get(keys[1]) is not None else 0.0 for r in results]
                 except (ValueError, TypeError):
                    return self._format_with_llm(results, question, "bar" if not horizontal else "horizontalBar", sql_query)
            else: # Fallback to LLM if data structure is unclear
                return self._format_with_llm(results, question, "bar" if not horizontal else "horizontalBar", sql_query)

        if not labels or not values: # If after processing, still no valid labels/values
             return self._empty_chart_data("bar" if not horizontal else "horizontalBar", question, "Could not extract labels or values for the chart.")

        # Generate a label for the dataset
        data_label_text = "Value" # Default
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a data labeling expert. Given a question, provide a concise label for the data series (e.g., 'Count', 'Amount'). Max 3 words."),
                ("human", "Question: {question}\nData Sample: {data_sample}\n\nProvide a concise label for what the numbers represent.")
            ])
            data_sample_str = f"Labels: {labels[:3]}, Values: {values[:3]}"
            llm_response = self.llm_manager.invoke(prompt, question=question, data_sample=data_sample_str)
            if llm_response and isinstance(llm_response, str): # Check if LLM returned a string response
                data_label_text = llm_response.strip()
        except Exception as e:
            logger.error(f"LLM failed to generate data label for bar chart: {e}")
            
        return {
            "chart_type": "horizontalBar" if horizontal else "bar",
            "title": self._generate_title(question),
            "labels": labels,
            "datasets": [{
                "label": data_label_text,
                "data": values,
                "backgroundColor": create_chart_colors(len(values)) # Use multiple colors for bar chart
            }],
            "options": {
                "indexAxis": 'y' if horizontal else 'x',
                "scales": {
                    "y": {"beginAtZero": True, "title": {"display": True, "text": "Value" if horizontal else "Category"}},
                    "x": {"beginAtZero": True, "title": {"display": True, "text": "Category" if horizontal else "Value"}}
                }
            }
        }

    def _format_line_data(self, results: List[Dict[str, Any]], question: str, data_type:str, columns: List[str], sql_query: Optional[str]) -> Dict[str, Any]:
        """Format data for line charts."""
        if not results:
            return self._empty_chart_data("line", question, "No data to display.")

        labels = []
        values = []

        if data_type == 'key_value' and all('label' in item and 'value' in item for item in results):
            labels = [str(item['label']) for item in results]
            try:
                values = [float(item['value']) if item['value'] is not None else 0.0 for item in results]
            except (ValueError, TypeError) as e:
                logger.error(f"Error converting line data values to float: {e}. Using 0.0.")
                values = [0.0] * len(results)
        elif columns and len(columns) >= 2 and all(isinstance(item, dict) for item in results):
            label_col, value_col = columns[0], columns[1]
            labels = [str(item.get(label_col, '')) for item in results]
            try:
                values = [float(item.get(value_col)) if item.get(value_col) is not None else 0.0 for item in results]
            except (ValueError, TypeError) as e:
                logger.error(f"Error converting line data values from columns to float: {e}. Using 0.0.")
                values = [0.0] * len(results)
        else:
            logger.warning("Line chart data is not in expected 'key_value' or multi-column format. Attempting LLM format or table fallback.")
            if results and isinstance(results[0], dict) and len(results[0].keys()) >=2:
                 keys = list(results[0].keys())
                 labels = [str(r.get(keys[0],'')) for r in results]
                 try:
                    values = [float(r.get(keys[1])) if r.get(keys[1]) is not None else 0.0 for r in results]
                 except (ValueError, TypeError):
                    return self._format_with_llm(results, question, "line", sql_query)
            else: # Fallback to LLM
                return self._format_with_llm(results, question, "line", sql_query)
        
        if not labels or not values:
             return self._empty_chart_data("line", question, "Could not extract labels or values for the chart.")

        y_label_text = "Value" # Default
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a data labeling expert. Given a question, provide a concise label for the Y-axis of a line chart (e.g., 'Revenue', 'Count'). Max 3 words."),
                ("human", "Question: {question}\nData Sample (Y-values): {y_sample}\n\nProvide a concise Y-axis label.")
            ])
            y_sample_str = str(values[:3])
            llm_response = self.llm_manager.invoke(prompt, question=question, y_sample=y_sample_str)
            if llm_response and isinstance(llm_response, str): # Check if LLM returned a string response
                y_label_text = llm_response.strip()
        except Exception as e:
            logger.error(f"LLM failed to generate Y-axis label for line chart: {e}")

        return {
            "chart_type": "line",
            "title": self._generate_title(question),
            "labels": labels, # X-axis labels
            "datasets": [{
                "label": y_label_text, # Legend label for the line
                "data": values,    # Y-axis values
                "borderColor": get_chart_color(),
                "backgroundColor": _adjust_color_alpha(get_chart_color(), 0.2), # Use helper for alpha
                "fill": True, # Changed to true for area under line
                "tension": 0.1
            }],
            "options": {
                "scales": {
                    "y": {"beginAtZero": True, "title": {"display": True, "text": y_label_text}},
                    "x": {"title": {"display": True, "text": "Category / Time"}} # Generic X-axis title
                }
            }
        }

    def _format_pie_data(self, results: List[Dict[str, Any]], question: str, data_type:str, columns:List[str], chart_type: str, sql_query: Optional[str]) -> Dict[str, Any]:
        """Format data for pie or doughnut charts."""
        if not results:
            return self._empty_chart_data(chart_type, question, "No data to display.")

        labels = []
        values = []

        if data_type == 'key_value' and all('label' in item and 'value' in item for item in results):
            labels = [str(item['label']) for item in results]
            try:
                values = [float(item['value']) if item['value'] is not None else 0.0 for item in results]
            except (ValueError, TypeError) as e:
                logger.error(f"Error converting pie data values to float: {e}. Using 0.0.")
                values = [0.0] * len(results)
        elif columns and len(columns) >= 2 and all(isinstance(item, dict) for item in results):
            label_col, value_col = columns[0], columns[1]
            labels = [str(item.get(label_col, '')) for item in results]
            try:
                values = [float(item.get(value_col)) if item.get(value_col) is not None else 0.0 for item in results]
            except (ValueError, TypeError) as e:
                logger.error(f"Error converting pie data values from columns to float: {e}. Using 0.0.")
                values = [0.0] * len(results)

        else:
            logger.warning("Pie/Doughnut chart data is not in expected 'key_value' or multi-column format. Attempting LLM format or table fallback.")
            if results and isinstance(results[0], dict) and len(results[0].keys()) >=2:
                 keys = list(results[0].keys())
                 labels = [str(r.get(keys[0],'')) for r in results]
                 try:
                    values = [float(r.get(keys[1])) if r.get(keys[1]) is not None else 0.0 for r in results]
                 except (ValueError, TypeError):
                    return self._format_with_llm(results, question, chart_type, sql_query)
            else: # Fallback to LLM
                return self._format_with_llm(results, question, chart_type, sql_query)

        if not labels or not values or sum(values) == 0: # Check if sum is zero, pie chart would be empty
             return self._empty_chart_data(chart_type, question, "Could not extract valid data for the chart or all values are zero.")

        return {
            "chart_type": chart_type,
            "title": self._generate_title(question),
            "labels": labels,
            "datasets": [{
                "label": "Distribution", # Generic label for pie/doughnut
                "data": values,
                "backgroundColor": create_chart_colors(len(labels))
            }],
            "options": {
                 "responsive": True,
                 "plugins": {
                    "legend": {"position": "top"},
                    "tooltip": {
                        "callbacks": {
                             # Ensure this is a string that Chart.js can evaluate
                            "label": '''function(tooltipItem) { 
                                let label = tooltipItem.label || ''; 
                                let value = tooltipItem.raw || 0; 
                                let sum = tooltipItem.dataset.data.reduce((a, b) => a + b, 0);
                                let percentage = sum > 0 ? (value / sum * 100).toFixed(2) + '%' : '0.00%';
                                return label + ': ' + value + ' (' + percentage + ')'; 
                            }'''
                        }
                    }
                }
            }
        }

    def _format_scatter_data(self, results: List[Dict[str, Any]], question: str, data_type:str, columns:List[str], sql_query: Optional[str]) -> Dict[str, Any]:
        """Format data for scatter plots."""
        if not results:
            return self._empty_chart_data("scatter", question, "No data to display.")

        data_points = []
        # Scatter plots need x/y coordinate pairs. Expects dicts with 'x', 'y' or uses first two columns.
        if data_type == 'xyz_data' and all('x' in item and 'y' in item for item in results): # xyz_data could be from a previous step
            try:
                data_points = [
                    {"x": float(item['x']) if item['x'] is not None else 0.0, 
                     "y": float(item['y']) if item['y'] is not None else 0.0}
                    for item in results
                ]
            except (ValueError, TypeError) as e:
                 logger.error(f"Error converting scatter x,y data to float: {e}. Using (0,0).")
                 data_points = [{"x": 0.0, "y": 0.0}] * len(results)

        elif columns and len(columns) >= 2 and all(isinstance(item, dict) for item in results):
            x_col, y_col = columns[0], columns[1]
            try:
                data_points = [
                    {"x": float(item.get(x_col)) if item.get(x_col) is not None else 0.0, 
                     "y": float(item.get(y_col)) if item.get(y_col) is not None else 0.0}
                    for item in results
                ]
            except (ValueError, TypeError) as e:
                logger.error(f"Error converting scatter data from columns to float: {e}. Using (0,0).")
                data_points = [{"x": 0.0, "y": 0.0}] * len(results)
        else:
            logger.warning("Scatter chart data not in expected format. Attempting LLM or table fallback.")
            # Try a generic conversion if results is list of dicts with at least 2 keys
            if results and isinstance(results[0], dict) and len(results[0].keys()) >= 2:
                keys = list(results[0].keys())
                try:
                    data_points = [
                        {"x": float(r.get(keys[0])) if r.get(keys[0]) is not None else 0.0,
                         "y": float(r.get(keys[1])) if r.get(keys[1]) is not None else 0.0}
                        for r in results
                    ]
                except (ValueError, TypeError):
                     return self._format_with_llm(results, question, "scatter", sql_query) # Fallback to LLM
            else:
                return self._format_with_llm(results, question, "scatter", sql_query)

        if not data_points:
            return self._empty_chart_data("scatter", question, "Could not extract valid data points.")

        # Generate axis labels
        x_label_text, y_label_text = "X-Axis", "Y-Axis" # Defaults
        try:
            axis_prompt_system = "You are a data labeling expert. Given a question, provide concise x-axis and y-axis labels for a scatter plot. Return as JSON: {\"x_label\": \"...\", \"y_label\": \"...\"}. Max 3 words per label."
            axis_prompt_human = "Question: {question}\nData Sample (first 3 x,y pairs): {data_sample}\n\nProvide concise x-axis and y-axis labels as JSON."
            sample_for_prompt = [{"x": dp["x"], "y": dp["y"]} for dp in data_points[:3]]
            
            llm_response = self.llm_manager.invoke(
                ChatPromptTemplate.from_messages([("system", axis_prompt_system), ("human", axis_prompt_human)]),
                question=question, data_sample=json.dumps(sample_for_prompt)
            )
            if llm_response: # Check if LLM returned a response
                try:
                    # Try to extract JSON from the response first
                    json_str = self._extract_json_from_response(llm_response)
                    if json_str: # Check if a JSON string was extracted
                        axis_labels = json.loads(json_str)
                        x_label_text = axis_labels.get("x_label", "X-Axis")
                        y_label_text = axis_labels.get("y_label", "Y-Axis")
                    else: # Fallback if no JSON extracted
                        logger.warning(f"LLM response for scatter axis labels was not valid JSON or empty: {llm_response}. Using defaults.")
                except json.JSONDecodeError: # Fallback if JSON parsing fails
                    logger.error(f"LLM response for scatter axis labels could not be parsed as JSON: {llm_response}. Using defaults.")
            # If llm_response is None or parsing fails, x_label_text and y_label_text will retain their default values
        except Exception as e:
            logger.error(f"LLM failed to generate axis labels for scatter plot: {e}")
            
        return {
            "chart_type": "scatter",
            "title": self._generate_title(question),
            "datasets": [{
                "label": "Data Points", # Can be improved with LLM if needed
                "data": data_points,
                "backgroundColor": get_chart_color(), 
                "pointRadius": 5 
            }],
            "options": {
                "scales": {
                    "y": {"beginAtZero": True, "title": {"display": True, "text": y_label_text}},
                    "x": {"beginAtZero": True, "title": {"display": True, "text": x_label_text}}
                }
            }
        }

    def _format_bubble_data(self, results: List[Dict[str, Any]], question: str, data_type:str, columns:List[str], sql_query: Optional[str]) -> Dict[str, Any]:
        """Format data for bubble charts. Expects data with x, y, and r (radius/size)."""
        if not results:
            return self._empty_chart_data("bubble", question, "No data to display.")

        data_bubbles = []
        # Bubble charts need x, y, r coordinate/size.
        # Expects dicts with 'x', 'y', 'size' or uses first three columns.
        if data_type == 'xyz_data' and all('x' in item and 'y' in item and 'size' in item for item in results):
            try:
                data_bubbles = [
                    {"x": float(item['x']) if item['x'] is not None else 0.0, 
                     "y": float(item['y']) if item['y'] is not None else 0.0,
                     "r": abs(float(item['size'])) if item['size'] is not None else 1.0 } # Radius must be non-negative
                    for item in results
                ]
            except (ValueError, TypeError) as e:
                logger.error(f"Error converting bubble x,y,size data to float: {e}. Using (0,0,1).")
                data_bubbles = [{"x": 0.0, "y": 0.0, "r":1.0}] * len(results)

        elif columns and len(columns) >= 3 and all(isinstance(item, dict) for item in results):
            x_col, y_col, r_col = columns[0], columns[1], columns[2]
            try:
                data_bubbles = [
                    {"x": float(item.get(x_col)) if item.get(x_col) is not None else 0.0, 
                     "y": float(item.get(y_col)) if item.get(y_col) is not None else 0.0,
                     "r": abs(float(item.get(r_col))) if item.get(r_col) is not None else 1.0}
                    for item in results
                ]
            except (ValueError, TypeError) as e:
                logger.error(f"Error converting bubble data from columns to float: {e}. Using (0,0,1).")
                data_bubbles = [{"x": 0.0, "y": 0.0, "r":1.0}] * len(results)
        else:
            logger.warning("Bubble chart data not in expected format. Attempting LLM or table fallback.")
            if results and isinstance(results[0], dict) and len(results[0].keys()) >= 3:
                keys = list(results[0].keys())
                try:
                    data_bubbles = [
                        {"x": float(r.get(keys[0])) if r.get(keys[0]) is not None else 0.0,
                         "y": float(r.get(keys[1])) if r.get(keys[1]) is not None else 0.0,
                         "r": abs(float(r.get(keys[2]))) if r.get(keys[2]) is not None else 1.0}
                        for r in results
                    ]
                except (ValueError, TypeError):
                    return self._format_with_llm(results, question, "bubble", sql_query) # Fallback to LLM
            else:
                return self._format_with_llm(results, question, "bubble", sql_query)


        if not data_bubbles:
            return self._empty_chart_data("bubble", question, "Could not extract valid data points for bubble chart.")

        # Generate axis labels and dataset label
        x_label_text, y_label_text, dataset_label_text = "X-Axis", "Y-Axis", "Bubbles" # Defaults
        try:
            prompt_system = "You are a data labeling expert. Given a question, provide concise x-axis, y-axis, and dataset labels for a bubble chart. Return as JSON: {\"x_label\": \"...\", \"y_label\": \"...\", \"dataset_label\": \"...\"}. Max 3 words per label."
            prompt_human = "Question: {question}\nData Sample (first 3 x,y,r points): {data_sample}\n\nProvide labels as JSON."
            sample_for_prompt = [{"x": dp["x"], "y": dp["y"], "r": dp["r"]} for dp in data_bubbles[:3]]
            
            llm_response = self.llm_manager.invoke(
                ChatPromptTemplate.from_messages([("system", prompt_system), ("human", prompt_human)]),
                question=question, data_sample=json.dumps(sample_for_prompt)
            )
            if llm_response: # Check if LLM returned a response
                try:
                    json_str = self._extract_json_from_response(llm_response)
                    if json_str: # Check if a JSON string was extracted
                        labels_json = json.loads(json_str)
                        x_label_text = labels_json.get("x_label", "X-Axis")
                        y_label_text = labels_json.get("y_label", "Y-Axis")
                        dataset_label_text = labels_json.get("dataset_label", "Bubbles")
                    else: # Fallback if no JSON extracted
                        logger.warning(f"LLM response for bubble labels was not valid JSON or empty: {llm_response}. Using defaults.")
                except json.JSONDecodeError: # Fallback if JSON parsing fails
                     logger.error(f"LLM response for bubble labels was not valid JSON: {llm_response}. Using defaults.")
            # If llm_response is None or parsing fails, labels retain default values
        except Exception as e:
            logger.error(f"LLM failed to generate labels for bubble chart: {e}")
            
        return {
            "chart_type": "bubble",
            "title": self._generate_title(question),
            "datasets": [{
                "label": dataset_label_text,
                "data": data_bubbles,
                "backgroundColor": _adjust_color_alpha(get_chart_color(), 0.5), # Use helper for alpha
                "borderColor": get_chart_color() 
            }],
            "options": {
                "scales": {
                    "y": {"beginAtZero": True, "title": {"display": True, "text": y_label_text}},
                    "x": {"beginAtZero": True, "title": {"display": True, "text": x_label_text}}
                },
                "plugins": {
                    "tooltip": {
                        "callbacks": { # Ensure this is a string for Chart.js
                            "label": '''function(tooltipItem) {
                                let label = tooltipItem.dataset.label || '';
                                let dataPoint = tooltipItem.raw;
                                return label + ': (x: ' + dataPoint.x + ', y: ' + dataPoint.y + ', size: ' + dataPoint.r + ')';
                            }'''
                        }
                    }
                }
            }
        }

    def _format_radar_data(self, results: List[Dict[str, Any]], question: str, data_type: str, columns: List[str], sql_query: Optional[str]) -> Dict[str, Any]:
        """Format data for radar charts."""
        if not results:
            return self._empty_chart_data("radar", question, "No data to display.")

        labels = []
        datasets_data = [] # This will be a list of datasets for Chart.js

        # Radar usually compares multiple metrics (columns) for one or more items (rows)
        # Or one item across multiple categories (labels as rows, values in one column)
        
        # Case 1: Each row is an item, first column is item label, subsequent columns are metrics
        if columns and len(columns) > 1 and all(isinstance(item, dict) for item in results):
            metric_labels = columns[1:] # Metrics are column headers (excluding the first label column)
            labels = metric_labels # These become the points on the radar
            
            item_label_column = columns[0]
            
            for i, row_item in enumerate(results):
                item_label = str(row_item.get(item_label_column, f"Item {i+1}"))
                values = []
                for metric_col in metric_labels:
                    try:
                        val = float(row_item.get(metric_col)) if row_item.get(metric_col) is not None else 0.0
                        values.append(val)
                    except (ValueError, TypeError):
                        logger.warning(f"Could not convert radar data '{row_item.get(metric_col)}' to float for metric '{metric_col}'. Using 0.0.")
                        values.append(0.0)
                
                base_color = get_chart_color(i)
                datasets_data.append({
                    "label": item_label,
                    "data": values,
                    "backgroundColor": _adjust_color_alpha(base_color, 0.2),  # Use helper for alpha
                    "borderColor": base_color,
                    "pointBackgroundColor": base_color
                })

        # Case 2: Data is already structured as label/value (one item, multiple metrics)
        # where 'label' is metric name, 'value' is the score for that metric.
        elif data_type == 'key_value' and all('label' in item and 'value' in item for item in results):
            labels = [str(item['label']) for item in results] # Labels are the metric names
            try:
                values = [float(item['value']) if item['value'] is not None else 0.0 for item in results]
            except (ValueError, TypeError) as e:
                logger.error(f"Error converting radar key_value data to float: {e}. Using 0.0.")
                values = [0.0] * len(results)

            if values: # Only create a dataset if there are values
                color = get_chart_color(0)
                datasets_data.append({
                    "label": "Metrics", # Generic label, could be improved
                    "data": values,
                    "backgroundColor": _adjust_color_alpha(color, 0.2),
                    "borderColor": color,
                    "pointBackgroundColor": color
                })
        else:
            logger.warning("Radar chart data is not in expected multi-column or 'key_value' format. Attempting LLM format or table fallback.")
            return self._format_with_llm(results, question, "radar", sql_query)

        if not labels or not datasets_data or not any(ds['data'] for ds in datasets_data):
             return self._empty_chart_data("radar", question, "Could not extract valid data for the radar chart.")
        
        return {
            "chart_type": "radar",
            "title": self._generate_title(question),
            "labels": labels, # These are the points/axes of the radar
            "datasets": datasets_data,
            "options": {
                "responsive": True,
                "scales": {
                    "r": { # Radial axis configuration
                        "angleLines": {"display": True},
                        "suggestedMin": 0 # Often good to start radar from 0
                    }
                },
                "plugins": {"legend": {"position": "top"}}
            }
        }
    
    def _format_with_llm(self, results: List[Dict[str, Any]], question: str, chart_type: str, sql_query: Optional[str]) -> Dict[str, Any]:
        """Format data for any chart type using LLM when standard formatting fails or for complex types."""
        if not results: # Handle empty results before calling LLM
            return self._empty_chart_data(chart_type, question, "No data to format.")

        # Create a prompt for the LLM to format the data using simple strings
        system_prompt = f"""You are a data formatting expert for Chart.js. Your task is to format SQL query results for a '{chart_type}' chart.
Return ONLY a valid JSON object that can be parsed directly by Chart.js.
The JSON should include 'chart_type', 'title', 'labels' (if applicable), 'datasets', and 'options'.
For 'datasets', ensure 'data' is an array of numbers (or objects like {{x,y,r}} for bubble/scatter).
If the data is unsuitable for '{chart_type}', you can suggest 'table' as chart_type with appropriate table data structure.
DO NOT include any explanations, markdown formatting, or code blocks - just the raw JSON.
Example for a bar chart:
{{
  "chart_type": "bar",
  "title": "Employee Counts",
  "labels": ["HR", "Sales", "IT"],
  "datasets": [{{ "label": "Count", "data": [5, 12, 7], "backgroundColor": ["rgba(75,192,192,0.6)"] }}],
  "options": {{ "scales": {{ "y": {{ "beginAtZero": True }} }} }}
}}
"""

        # Prepare a sample of results for the prompt to avoid overly long prompts
        results_sample = results[:5] # Max 5 rows for the prompt
        try:
            results_sample_json = json.dumps(results_sample)
        except TypeError: # Handle non-serializable data
            results_sample_json = str(results_sample)


        human_prompt = f"""Question: {question}
SQL Query: {sql_query if sql_query else "Not available"}
Results Sample: {results_sample_json}

Format ALL results (not just the sample) for a '{chart_type}' chart according to Chart.js structure.
If data is unsuitable, make 'chart_type': 'table' and structure 'data' as an array of objects, and 'columns' as an array of strings.
"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
        
        try:
            # Invoke LLM to format the data
            llm_response_content = self.llm_manager.invoke(
                prompt, 
                question=question, 
                sql_query=sql_query if sql_query else "Not available", 
                results=results_sample_json, # Pass sample to LLM
                chart_type=chart_type
            )
            
            # Try to parse the response as JSON directly
            try:
                formatted_data = json.loads(self._extract_json_from_response(llm_response_content))
            except json.JSONDecodeError as e:
                logger.error(f"LLM response for {chart_type} formatting was not valid JSON: {llm_response_content}. Error: {e}")
                # Fallback to a simple table if LLM fails to produce valid JSON
                return self._create_fallback_table(results, question, columns if columns else (list(results[0].keys()) if results and isinstance(results[0], dict) else []))

            logger.info(f"LLM-formatted data for {chart_type} chart. Keys: {formatted_data.keys()}")
            
            # Basic validation and enrichment
            formatted_data["chart_type"] = formatted_data.get("chart_type", chart_type) # Ensure chart type is present
            if "title" not in formatted_data or not formatted_data["title"]:
                formatted_data["title"] = self._generate_title(question)
            
            # Ensure datasets is a list
            if "datasets" in formatted_data and not isinstance(formatted_data["datasets"], list):
                logger.warning(f"LLM returned 'datasets' not as a list for {chart_type}. Wrapping it.")
                formatted_data["datasets"] = [formatted_data["datasets"]]

            # If LLM switched to table, ensure columns and data are present
            if formatted_data["chart_type"] == "table":
                if "data" not in formatted_data: formatted_data["data"] = results
                if "columns" not in formatted_data:
                    formatted_data["columns"] = columns if columns else (list(results[0].keys()) if results and isinstance(results[0], dict) else [])
                if "headers" not in formatted_data: formatted_data["headers"] = formatted_data["columns"]
            
            # For non-table charts, ensure essential keys are present
            elif formatted_data["chart_type"] not in ["scatter", "bubble"] and ("datasets" not in formatted_data or "labels" not in formatted_data):
                 logger.warning(f"LLM formatted data for {chart_type} is missing 'datasets' or 'labels'. Attempting to build defaults.")
                 # Attempt to build minimal structure if LLM failed badly
                 current_columns = formatted_data.get("columns", []) # Get columns from formatted_data if available
                 if not results: return self._empty_chart_data(chart_type, question, "No data provided to LLM formatter.")

                 if isinstance(results[0], dict) and len(results[0].keys()) >= 1:
                     keys = list(results[0].keys())
                     if "labels" not in formatted_data:
                          formatted_data["labels"] = [str(row.get(keys[0], "")) for row in results]
                     if len(keys) >= 2 and "datasets" not in formatted_data:
                         try:
                             values = [float(row.get(keys[1])) if row.get(keys[1]) is not None else 0.0 for row in results]
                             formatted_data["datasets"] = [{"label": "Data", "data": values, "backgroundColor": get_chart_color()}]
                         except (ValueError, TypeError):
                              logger.error("Could not convert secondary data to float for LLM fallback dataset.")
                              return self._create_fallback_table(results, question, current_columns if current_columns else (list(results[0].keys()) if results and isinstance(results[0], dict) else []))
                     elif "datasets" not in formatted_data: # single column data, less ideal for most charts
                          formatted_data["datasets"] = [{"label": "Data", "data": [1]*len(results), "backgroundColor": get_chart_color()}]
            elif formatted_data["chart_type"] in ["scatter", "bubble"] and "datasets" not in formatted_data:
                 logger.warning(f"LLM formatted data for {chart_type} is missing 'datasets'. Attempting to build defaults.")
                 # Simplified default for scatter/bubble if datasets are missing
                 if not results: return self._empty_chart_data(chart_type, question, "No data provided to LLM formatter.")
                 data_points = []
                 if isinstance(results[0], dict) and len(results[0].keys()) >= (3 if chart_type == 'bubble' else 2):
                    keys = list(results[0].keys())
                    try:
                        if chart_type == 'bubble':
                            data_points = [{"x": float(r.get(keys[0],0)), "y": float(r.get(keys[1],0)), "r": float(r.get(keys[2],1))} for r in results]
                        else: # scatter
                            data_points = [{"x": float(r.get(keys[0],0)), "y": float(r.get(keys[1],0))} for r in results]
                        formatted_data["datasets"] = [{"label": "Data", "data": data_points, "backgroundColor": get_chart_color()}]
                    except (ValueError, TypeError):
                        logger.error(f"Could not convert data for {chart_type} LLM fallback dataset.")
                        return self._create_fallback_table(results, question, current_columns if current_columns else (list(results[0].keys()) if results and isinstance(results[0], dict) else []))
                 else:
                    formatted_data["datasets"] = [{"label": "Data", "data": [], "backgroundColor": get_chart_color()}]

            return formatted_data
        except Exception as e:
            logger.error(f"Error formatting data with LLM for {chart_type}: {str(e)}")
            # Fallback to a simple table on any other error during LLM formatting
            # Try to get columns from the state if available, else derive from results
            current_columns = state.get("columns", []) if "state" in locals() else (list(results[0].keys()) if results and isinstance(results[0], dict) else [])
            return self._create_fallback_table(results, question, current_columns)

    def _create_fallback_table(self, results: List[Dict[str, Any]], question: str, column_names: List[str]) -> Dict[str, Any]:
        logger.info(f"Creating fallback table for question: {question}")
        return {
            'chart_type': 'table',
            'title': self._generate_title(question) + " (Data Table)",
            'data': results,
            'columns': column_names,
            'headers': column_names
        }

    def _empty_chart_data(self, chart_type: str, question: str, reason: str) -> Dict[str, Any]:
        logger.warning(f"Generating empty chart data for {chart_type} due to: {reason}")
        return {
            "chart_type": chart_type, # Keep original intended type
            "title": self._generate_title(question),
            "labels": [],
            "datasets": [{"label": "No Data", "data": []}],
            "options": {},
            "error_message": reason # Add an error message for frontend to display
        }

    def _extract_json_from_response(self, response: str) -> str:
        """
        Extract valid JSON from a response string that may contain additional text.
        
        Args:
            response: The raw response from the LLM
            
        Returns:
            Cleaned JSON string
        """
        # First, try to parse the entire response as JSON
        try:
            json.loads(response)
            return response  # If it parses successfully, return as is
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON between triple backticks
        json_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        matches = re.findall(json_pattern, response, re.DOTALL)
        if matches:
            for match in matches:
                try:
                    json.loads(match)
                    return match  # Return the first valid JSON
                except json.JSONDecodeError:
                    continue
        
        # Try to extract JSON between curly braces
        try:
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                json.loads(json_str)  # Test if valid
                return json_str
        except json.JSONDecodeError:
            pass
        
        # Create a default valid JSON as a fallback
        return '{"chart_type": "bar", "reasoning": "Failed to parse response"}'
    
    def _generate_title(self, question: str) -> str:
        """Generate a concise chart title from the user's question."""
        try:
            # Use LLM to generate a more natural title
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert at creating concise and descriptive chart titles. Based on the user's question, generate a suitable chart title. Max 10 words."),
                ("human", "User question: {question}\n\nGenerate a chart title:")
            ])
            llm_title = self.llm_manager.invoke(prompt, question=question)
            if llm_title and isinstance(llm_title, str) and len(llm_title) > 5 and not llm_title.lower().startswith("error"):
                # Basic cleaning of LLM title
                clean_title = llm_title.strip().replace('"', '').replace("'", "")
                # Capitalize first letter
                if clean_title:
                    clean_title = clean_title[0].upper() + clean_title[1:]
                return clean_title[:80] # Limit length
        except Exception as e:
            logger.error(f"LLM title generation failed: {e}. Using rule-based fallback.")

        # Fallback to rule-based title generation
        title = question.replace("?", "").replace(".", "")
        title = re.sub(r'^(visualize|show|display|create|generate|give|provide|plot|chart|graph|make)\\s+(me|a|an|the)?\\s*', '', title, flags=re.IGNORECASE)
        title = re.sub(r'\\s*(pie|bar|line|scatter|bubble|radar|chart|graph|visualization|for me|please)\\b', '', title, flags=re.IGNORECASE)
        
        title = title.strip()
        if title:
            title = title[0].upper() + title[1:]
            return title[:80] # Limit length
        else:
            return "Data Visualization" 