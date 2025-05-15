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
    "polarArea": "polarArea"
}

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
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state with chosen chart type and reasoning
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
                "chart_reason": "No data available for visualization"
            }
            
        # Analyze the data to determine the chart type
        try:
            # Create prompt for chart type selection using simple strings
            system_prompt = """You are an AI assistant specialized in data visualization. Your task is to recommend the most appropriate chart type based on the user's question, SQL query, and the resulting data. Return your recommendation as valid JSON.

Available chart types:
- bar: For comparing categories, especially with a small number of categories
- line: For showing trends over time or continuous data
- pie: For showing parts of a whole, useful when there are few categories
- scatter: For showing relationship between two variables
- table: For raw data when visualization isn't helpful
- number: For a single value or KPI

Consider the data structure:
1. Number of categories/points (< 10 is good for pie charts)
2. Whether time is involved (line charts work well)
3. Number of variables being compared
4. The specific user question

Return a JSON object with:
1. chart_type: One of the available types
2. chart_reason: Brief explanation of your choice
"""

            human_prompt = f"""
Question: {question}
SQL Query: {sql_query}
Data Structure: {data_type}
Columns: {columns}
Data (sample): {str(results[:5])}

Recommend the most appropriate chart type.
"""

            # Set up prompt template
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", human_prompt)
            ])

            # Get chart recommendation - use self.llm_manager.llm directly
            response = self.llm_manager.invoke(prompt)
            
            # Extract JSON from the response using regex
            chart_json = self._extract_json(response, default_type='table')
            
            return chart_json
        except Exception as e:
            logger.error(f"Error selecting chart type: {str(e)}")
            return {
                "chart_type": "table",
                "chart_reason": f"Error selecting chart: {str(e)}"
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
                # Bar chart formatting
                if data_type == 'key_value' or (len(results) > 0 and all(isinstance(item, dict) and 'label' in item and 'value' in item for item in results)):
                    # Extract labels and values for chart.js
                    labels = [item['label'] for item in results]
                    values = [float(item['value']) if isinstance(item['value'], (int, float)) else 0 for item in results]
                    
                    formatted_data = {
                        'chart_type': chart_type,
                        'title': self._generate_title(question),
                        'labels': labels,
                        'datasets': [{
                            'label': 'Value',
                            'data': values,
                            'backgroundColor': create_chart_colors(len(values))
                        }],
                        'options': {
                            'scales': {
                                'y': {'beginAtZero': True}
                            }
                        }
                    }
                elif len(columns) >= 2 and results and all(isinstance(item, dict) for item in results):
                    # Try to use the first two columns as labels and values
                    labels = [str(item[columns[0]]) for item in results]
                    values = [float(item[columns[1]]) if isinstance(item[columns[1]], (int, float)) else 0 for item in results]
                    
                    formatted_data = {
                        'chart_type': chart_type,
                        'title': self._generate_title(question),
                        'labels': labels,
                        'datasets': [{
                            'label': columns[1],
                            'data': values,
                            'backgroundColor': create_chart_colors(len(values))
                        }],
                        'options': {
                            'scales': {
                                'y': {'beginAtZero': True}
                            }
                        }
                    }
                else:
                    # Default formatting for incompatible data
                    formatted_data = {
                        'chart_type': 'table',
                        'title': self._generate_title(question),
                        'data': results,
                        'columns': columns,
                        'headers': columns
                    }
                    
            elif chart_type in ['pie', 'doughnut']:
                # Pie/doughnut chart formatting
                if data_type == 'key_value' or (len(results) > 0 and all(isinstance(item, dict) and 'label' in item and 'value' in item for item in results)):
                    # Extract labels and values for chart.js
                    labels = [item['label'] for item in results]
                    values = [float(item['value']) if isinstance(item['value'], (int, float)) else 0 for item in results]
                    
                    formatted_data = {
                        'chart_type': chart_type,
                        'title': self._generate_title(question),
                        'labels': labels,
                        'datasets': [{
                            'data': values,
                            'backgroundColor': create_chart_colors(len(values))
                        }],
                        'options': {}
                    }
                elif len(columns) >= 2 and results:
                    # Try to use the first two columns as labels and values
                    labels = [str(item[columns[0]]) for item in results]
                    values = [float(item[columns[1]]) if isinstance(item[columns[1]], (int, float)) else 0 for item in results]
                    
                    formatted_data = {
                        'chart_type': chart_type,
                        'title': self._generate_title(question),
                        'labels': labels,
                        'datasets': [{
                            'data': values,
                            'backgroundColor': create_chart_colors(len(values))
                        }],
                        'options': {}
                    }
                else:
                    # Default formatting for incompatible data
                    formatted_data = {
                        'chart_type': 'table',
                        'title': self._generate_title(question),
                        'data': results,
                        'columns': columns,
                        'headers': columns
                    }
                    
            elif chart_type == 'line':
                # Line chart formatting
                if data_type == 'key_value' or (len(results) > 0 and all(isinstance(item, dict) and 'label' in item and 'value' in item for item in results)):
                    # Extract labels and values for chart.js
                    labels = [item['label'] for item in results]
                    values = [float(item['value']) if isinstance(item['value'], (int, float)) else 0 for item in results]
                    
                    formatted_data = {
                        'chart_type': 'line',
                        'title': self._generate_title(question),
                        'labels': labels,
                        'datasets': [{
                            'label': 'Value',
                            'data': values,
                            'borderColor': get_chart_color(),
                            'backgroundColor': get_chart_color() + '20',
                            'fill': False,
                            'tension': 0.1
                        }],
                        'options': {
                            'scales': {
                                'y': {'beginAtZero': True}
                            }
                        }
                    }
                elif len(columns) >= 2 and results and isinstance(results[0], dict):
                    # Try to use the first two columns as labels and values
                    labels = [str(item[columns[0]]) for item in results]
                    values = [float(item[columns[1]]) if isinstance(item[columns[1]], (int, float)) else 0 for item in results]
                    
                    formatted_data = {
                        'chart_type': 'line',
                        'title': self._generate_title(question),
                        'labels': labels,
                        'datasets': [{
                            'label': columns[1],
                            'data': values,
                            'borderColor': get_chart_color(),
                            'backgroundColor': get_chart_color() + '20',
                            'fill': False,
                            'tension': 0.1
                        }],
                        'options': {
                            'scales': {
                                'y': {'beginAtZero': True}
                            }
                        }
                    }
                else:
                    # Default formatting for incompatible data
                    formatted_data = {
                        'chart_type': 'table',
                        'title': self._generate_title(question),
                        'data': results,
                        'columns': columns,
                        'headers': columns
                    }
            else:
                # Default to table format for unsupported chart types
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
        
        Args:
            text: Text potentially containing JSON
            default_type: Default chart type to use if extraction fails
            
        Returns:
            Dictionary with chart type and reason
        """
        try:
            # Try direct parsing first
            try:
                data = json.loads(text)
                if 'chart_type' in data:
                    return data
            except json.JSONDecodeError:
                pass
                
            # Try to extract JSON from code blocks
            json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
            matches = re.findall(json_pattern, text, re.DOTALL)
            
            if matches:
                for match in matches:
                    try:
                        data = json.loads(match)
                        if 'chart_type' in data:
                            return data
                    except json.JSONDecodeError:
                        continue
            
            # Try to extract JSON between curly braces
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                try:
                    json_str = text[start_idx:end_idx]
                    data = json.loads(json_str)
                    if 'chart_type' in data:
                        return data
                except json.JSONDecodeError:
                    pass
            
            # Look for chart type mentions in text
            chart_types = ['bar', 'line', 'pie', 'table', 'number', 'scatter']
            found_type = default_type
            
            for chart_type in chart_types:
                if f"'{chart_type}'" in text or f'"{chart_type}"' in text or f" {chart_type} " in text.lower():
                    found_type = chart_type
                    break
                    
            # Extract reasoning if possible
            reason_match = re.search(r'because\s+(.*?)(?:\.|$)', text, re.IGNORECASE | re.DOTALL)
            reason = reason_match.group(1).strip() if reason_match else "This chart type best fits the data"
            
            return {
                "chart_type": found_type,
                "chart_reason": reason
            }
            
        except Exception as e:
            logger.error(f"Error extracting JSON: {str(e)}")
            return {
                "chart_type": default_type,
                "chart_reason": "Default chart selected due to parsing error"
            }
    
    def _format_bar_data(self, results: List[Dict[str, Any]], question: str, horizontal: bool = False) -> Dict[str, Any]:
        """Format data for bar or horizontal bar charts."""
        # Convert to list of lists if in dict format for consistency
        if results and isinstance(results[0], dict):
            # Extract keys and create a consistent order
            keys = list(results[0].keys())
            data_list = [[row.get(k) for k in keys] for row in results]
        else:
            data_list = results
            
        # Simple case: 2 columns (category and value)
        if len(data_list[0]) == 2:
            labels = [str(row[0]) for row in data_list]
            values = [float(row[1]) if row[1] is not None else 0 for row in data_list]
            
            # Generate a label for the dataset based on the question
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a data labeling expert. Given a question and data, provide a concise label for the data series."),
                ("human", "Question: {question}\nData: {data}\n\nProvide a concise label describing what the numbers represent (e.g., 'Sales', 'Count', 'Revenue'). Keep it to 1-3 words.")
            ])
            
            data_label = self.llm_manager.invoke(prompt, question=question, data=str(data_list[:3]))
            
            return {
                "chart_type": "horizontalBar" if horizontal else "bar",
                "title": self._generate_title(question),
                "labels": labels,
                "datasets": [{
                    "label": data_label.strip(),
                    "data": values,
                    "backgroundColor": get_chart_color()
                }],
                "options": {
                    "indexAxis": 'y' if horizontal else 'x',
                    "scales": {
                        "y": {"title": {"display": True, "text": "Categories" if horizontal else data_label.strip()}},
                        "x": {"title": {"display": True, "text": data_label.strip() if horizontal else "Categories"}}
                    }
                }
            }
        
        # Multiple series case (3+ columns)
        elif len(data_list[0]) >= 3:
            # Assume first column is category, the rest are different series
            labels = [str(row[0]) for row in data_list]
            
            # Get unique second column values if they exist (for grouping)
            if len(data_list[0]) >= 3:
                series = {}
                for row in data_list:
                    series_name = str(row[1])
                    if series_name not in series:
                        series[series_name] = []
                    series[series_name].append(float(row[2]) if row[2] is not None else 0)
                
                datasets = []
                colors = create_chart_colors(len(series))
                
                for i, (name, values) in enumerate(series.items()):
                    datasets.append({
                        "label": name,
                        "data": values,
                        "backgroundColor": colors[i]
                    })
            else:
                # Multiple metrics for each category
                datasets = []
                metric_count = len(data_list[0]) - 1
                colors = create_chart_colors(metric_count)
                
                for i in range(1, len(data_list[0])):
                    values = [float(row[i]) if row[i] is not None else 0 for row in data_list]
                    datasets.append({
                        "label": f"Metric {i}",
                        "data": values,
                        "backgroundColor": colors[i-1]
                    })
            
            return {
                "chart_type": "horizontalBar" if horizontal else "bar",
                "title": self._generate_title(question),
                "labels": labels,
                "datasets": datasets,
                "options": {
                    "indexAxis": 'y' if horizontal else 'x',
                    "scales": {
                        "y": {"stacked": False},
                        "x": {"stacked": False}
                    }
                }
            }
        
        # Fallback for unexpected formats
        else:
            return self._format_with_llm(results, question, "bar", "")
    
    def _format_line_data(self, results: List[Dict[str, Any]], question: str) -> Dict[str, Any]:
        """Format data for line charts."""
        # Convert to list of lists if in dict format for consistency
        if results and isinstance(results[0], dict):
            # Extract keys and create a consistent order
            keys = list(results[0].keys())
            data_list = [[row.get(k) for k in keys] for row in results]
        else:
            data_list = results
            
        # Simple case: 2 columns (x and y values)
        if len(data_list[0]) == 2:
            x_values = [str(row[0]) for row in data_list]
            y_values = [float(row[1]) if row[1] is not None else 0 for row in data_list]
            
            # Generate a label for the y-axis based on the question
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a data labeling expert. Given a question and data, provide a concise label for the y-axis."),
                ("human", "Question: {question}\nData: {data}\n\nProvide a concise y-axis label describing what the numbers represent (e.g., 'Sales', 'Temperature', 'Revenue'). Keep it to 1-3 words.")
            ])
            
            y_label = self.llm_manager.invoke(prompt, question=question, data=str(data_list[:3]))
            
            return {
                "chart_type": "line",
                "title": self._generate_title(question),
                "labels": x_values,
                "datasets": [{
                    "label": y_label.strip(),
                    "data": y_values,
                    "borderColor": get_chart_color(),
                    "fill": False,
                    "tension": 0.4
                }],
                "options": {
                    "scales": {
                        "y": {"title": {"display": True, "text": y_label.strip()}},
                        "x": {"title": {"display": True, "text": "Time Period"}}
                    }
                }
            }
        
        # Multiple series case
        elif len(data_list[0]) >= 3:
            # Try to identify which columns represent what based on data types
            x_values = [str(row[0]) for row in data_list]
            
            if len(data_list[0]) == 3:
                # Check if second column is a category or a value
                if all(isinstance(row[1], (str)) for row in data_list):
                    # Second column is category, third is value
                    series = {}
                    for row in data_list:
                        series_name = str(row[1])
                        if series_name not in series:
                            series[series_name] = {}
                        x_val = str(row[0])
                        series[series_name][x_val] = float(row[2]) if row[2] is not None else 0
                    
                    # Ensure all series have values for all x values
                    datasets = []
                    colors = create_chart_colors(len(series))
                    
                    for i, (name, values) in enumerate(series.items()):
                        data_points = [values.get(x, None) for x in x_values]
                        datasets.append({
                            "label": name,
                            "data": data_points,
                            "borderColor": colors[i],
                            "fill": False,
                            "tension": 0.4
                        })
                else:
                    # Second and third columns are both values (two metrics)
                    datasets = [
                        {
                            "label": "Metric 1",
                            "data": [float(row[1]) if row[1] is not None else 0 for row in data_list],
                            "borderColor": get_chart_color(0),
                            "fill": False,
                            "tension": 0.4
                        },
                        {
                            "label": "Metric 2",
                            "data": [float(row[2]) if row[2] is not None else 0 for row in data_list],
                            "borderColor": get_chart_color(1),
                            "fill": False,
                            "tension": 0.4
                        }
                    ]
            else:
                # Multiple metrics for each x value
                datasets = []
                for i in range(1, len(data_list[0])):
                    datasets.append({
                        "label": f"Metric {i}",
                        "data": [float(row[i]) if row[i] is not None else 0 for row in data_list],
                        "borderColor": get_chart_color(i-1),
                        "fill": False,
                        "tension": 0.4
                    })
            
            return {
                "chart_type": "line",
                "title": self._generate_title(question),
                "labels": x_values,
                "datasets": datasets,
                "options": {
                    "scales": {
                        "y": {"title": {"display": True, "text": "Value"}},
                        "x": {"title": {"display": True, "text": "Time Period"}}
                    }
                }
            }
        
        # Fallback for unexpected formats
        else:
            return self._format_with_llm(results, question, "line", "")
    
    def _format_pie_data(self, results: List[Dict[str, Any]], question: str, chart_type: str) -> Dict[str, Any]:
        """Format data for pie or doughnut charts."""
        # Convert to list of lists if in dict format for consistency
        if results and isinstance(results[0], dict):
            # Extract keys and create a consistent order
            keys = list(results[0].keys())
            data_list = [[row.get(k) for k in keys] for row in results]
        else:
            data_list = results
        
        # Pie charts need category/value pairs
        if len(data_list[0]) >= 2:
            labels = [str(row[0]) for row in data_list]
            values = [float(row[1]) if row[1] is not None else 0 for row in data_list]
            
            # Generate a label for the dataset based on the question
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a data labeling expert. Given a question and data, provide a concise label for what the data represents."),
                ("human", "Question: {question}\nData: {data}\n\nProvide a concise label describing what the pie chart segments represent (e.g., 'Sales Distribution', 'Market Share'). Keep it to 1-4 words.")
            ])
            
            data_label = self.llm_manager.invoke(prompt, question=question, data=str(data_list[:3]))
            
            return {
                "chart_type": chart_type,
                "title": self._generate_title(question),
                "labels": labels,
                "datasets": [{
                    "label": data_label.strip(),
                    "data": values,
                    "backgroundColor": create_chart_colors(len(labels))
                }],
                "options": {
                    "plugins": {
                        "tooltip": {
                            "callbacks": {
                                "label": "function(tooltipItem) { return tooltipItem.label + ': ' + tooltipItem.parsed + '%'; }"
                            }
                        }
                    }
                }
            }
        
        # Fallback for unexpected formats
        else:
            return self._format_with_llm(results, question, chart_type, "")
    
    def _format_scatter_data(self, results: List[Dict[str, Any]], question: str) -> Dict[str, Any]:
        """Format data for scatter plots."""
        # Convert to list of lists if in dict format for consistency
        if results and isinstance(results[0], dict):
            # Extract keys and create a consistent order
            keys = list(results[0].keys())
            data_list = [[row.get(k) for k in keys] for row in results]
        else:
            data_list = results
        
        # Scatter plots need x/y coordinate pairs
        if len(data_list[0]) >= 2:
            # Simple case: 2 columns (x and y)
            if len(data_list[0]) == 2:
                data_points = [
                    {"x": float(row[0]) if row[0] is not None else 0, 
                     "y": float(row[1]) if row[1] is not None else 0}
                    for row in data_list
                ]
                
                # Generate axis labels
                axis_prompt_system = "You are a data labeling expert. Given a question and data, provide concise axis labels."
                axis_prompt_human = "Question: {question}\nData: {data}\n\nProvide concise x-axis and y-axis labels as a JSON: {\"x_label\": \"...\", \"y_label\": \"...\"}"
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", axis_prompt_system),
                    ("human", axis_prompt_human)
                ])
                
                labels_response = self.llm_manager.invoke(prompt, question=question, data=str(data_list[:3]))
                
                try:
                    axis_labels = json.loads(labels_response)
                    x_label = axis_labels.get("x_label", "X Axis")
                    y_label = axis_labels.get("y_label", "Y Axis")
                except:
                    x_label = "X Axis"
                    y_label = "Y Axis"
                
                return {
                    "chart_type": "scatter",
                    "title": self._generate_title(question),
                    "datasets": [{
                        "label": "Data Points",
                        "data": data_points,
                        "backgroundColor": get_chart_color(),
                        "pointRadius": 6
                    }],
                    "options": {
                        "scales": {
                            "y": {"title": {"display": True, "text": y_label}},
                            "x": {"title": {"display": True, "text": x_label}}
                        }
                    }
                }
            
            # Case with categories (3 columns)
            elif len(data_list[0]) >= 3:
                # Group by category (assuming 3rd column is category)
                categories = {}
                for row in data_list:
                    category = str(row[2]) if len(row) > 2 else "Default"
                    if category not in categories:
                        categories[category] = []
                    categories[category].append({
                        "x": float(row[0]) if row[0] is not None else 0,
                        "y": float(row[1]) if row[1] is not None else 0
                    })
                
                # Create datasets for each category
                datasets = []
                colors = create_chart_colors(len(categories))
                
                for i, (category, points) in enumerate(categories.items()):
                    datasets.append({
                        "label": category,
                        "data": points,
                        "backgroundColor": colors[i],
                        "pointRadius": 6
                    })
                
                # Generate axis labels
                axis_prompt_system = "You are a data labeling expert. Given a question and data, provide concise axis labels."
                axis_prompt_human = "Question: {question}\nData: {data}\n\nProvide concise x-axis and y-axis labels as a JSON: {\"x_label\": \"...\", \"y_label\": \"...\"}"
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", axis_prompt_system),
                    ("human", axis_prompt_human)
                ])
                
                labels_response = self.llm_manager.invoke(prompt, question=question, data=str(data_list[:3]))
                
                try:
                    axis_labels = json.loads(labels_response)
                    x_label = axis_labels.get("x_label", "X Axis")
                    y_label = axis_labels.get("y_label", "Y Axis")
                except:
                    x_label = "X Axis"
                    y_label = "Y Axis"
                
                return {
                    "chart_type": "scatter",
                    "title": self._generate_title(question),
                    "datasets": datasets,
                    "options": {
                        "scales": {
                            "y": {"title": {"display": True, "text": y_label}},
                            "x": {"title": {"display": True, "text": x_label}}
                        }
                    }
                }
        
        # Fallback for unexpected formats
        return self._format_with_llm(results, question, "scatter", "")
    
    def _format_radar_data(self, results: List[Dict[str, Any]], question: str) -> Dict[str, Any]:
        """Format data for radar charts."""
        # Convert to list of lists if in dict format for consistency
        if results and isinstance(results[0], dict):
            # Extract keys and create a consistent order
            keys = list(results[0].keys())
            data_list = [[row.get(k) for k in keys] for row in results]
        else:
            data_list = results
        
        # For radar charts, categories are the axes
        if len(data_list[0]) >= 2:
            # Case with one entity measured across multiple categories
            if all(isinstance(row[0], str) for row in data_list):
                labels = [str(row[0]) for row in data_list]
                values = [float(row[1]) if row[1] is not None else 0 for row in data_list]
                
                return {
                    "chart_type": "radar",
                    "title": self._generate_title(question),
                    "labels": labels,
                    "datasets": [{
                        "label": "Entity Performance",
                        "data": values,
                        "backgroundColor": get_chart_color(0) + "40",  # Add transparency
                        "borderColor": get_chart_color(0),
                        "pointBackgroundColor": get_chart_color(0)
                    }],
                    "options": {}
                }
            
            # Case with multiple entities measured across same categories (3+ columns)
            elif len(data_list[0]) >= 3:
                # Assume first column contains category names
                labels = [str(row[0]) for row in data_list]
                
                datasets = []
                for i in range(1, len(data_list[0])):
                    color = get_chart_color(i-1)
                    datasets.append({
                        "label": f"Entity {i}",
                        "data": [float(row[i]) if row[i] is not None else 0 for row in data_list],
                        "backgroundColor": color + "40",  # Add transparency
                        "borderColor": color,
                        "pointBackgroundColor": color
                    })
                
                return {
                    "chart_type": "radar",
                    "title": self._generate_title(question),
                    "labels": labels,
                    "datasets": datasets,
                    "options": {}
                }
        
        # Fallback for unexpected formats
        return self._format_with_llm(results, question, "radar", "")
    
    def _format_with_llm(self, results: List[Dict[str, Any]], question: str, chart_type: str, sql_query: str) -> Dict[str, Any]:
        """Format data for any chart type using LLM when standard formatting fails."""
        # Create a prompt for the LLM to format the data using simple strings
        system_prompt = """You are a data formatting expert. Your task is to format SQL query results for a {chart_type} chart.

For a {chart_type} chart, format the data into a structure with these components:
- chart_type: The type of chart ("{chart_type}")
- title: A concise, descriptive title for the chart
- labels: An array of labels for the chart (x-axis labels for bar/line, segment labels for pie)
- datasets: An array of dataset objects containing the data points
- options: Configuration options for the chart

Return ONLY a valid JSON object with this structure that can be parsed directly by Chart.js.
DO NOT include any explanations, markdown formatting, or code blocks - just the raw JSON."""

        human_prompt = """Question: {question}

SQL Query: {sql_query}

Results: {results}

Format the data for a {chart_type} chart:"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
        
        try:
            # Invoke LLM to format the data
            response = self.llm_manager.invoke(
                prompt, 
                question=question, 
                sql_query=sql_query, 
                results=json.dumps(results[:10] if len(results) > 10 else results),
                chart_type=chart_type
            )
            
            # Try to parse the response as JSON directly
            try:
                formatted_data = json.loads(response)
            except json.JSONDecodeError:
                # If that fails, try to extract valid JSON
                clean_json = self._extract_json_from_response(response)
                formatted_data = json.loads(clean_json)
                
            logger.info(f"LLM-formatted data for {chart_type} chart")
            
            # Validate and enforce minimum chart structure
            if "chart_type" not in formatted_data:
                formatted_data["chart_type"] = chart_type
                
            if "title" not in formatted_data:
                formatted_data["title"] = self._generate_title(question)
                
            if "datasets" not in formatted_data:
                # Build a minimal dataset if missing
                if isinstance(results[0], dict):
                    keys = list(results[0].keys())
                    values = [float(row.get(keys[1], 0)) if row.get(keys[1]) is not None else 0 for row in results]
                else:
                    values = [float(row[1]) if row[1] is not None else 0 for row in results]
                
                formatted_data["datasets"] = [{
                    "label": "Data",
                    "data": values,
                    "backgroundColor": get_chart_color()
                }]
                
            if "labels" not in formatted_data:
                # Build labels if missing
                if isinstance(results[0], dict):
                    keys = list(results[0].keys())
                    formatted_data["labels"] = [str(row.get(keys[0], "")) for row in results]
                else:
                    formatted_data["labels"] = [str(row[0]) for row in results]
            
            return formatted_data
        except Exception as e:
            logger.error(f"Error formatting data with LLM: {str(e)}")
            
            # Build a minimal fallback structure
            if isinstance(results[0], dict):
                keys = list(results[0].keys())
                labels = [str(row.get(keys[0], "")) for row in results]
                values = [float(row.get(keys[1], 0)) if row.get(keys[1]) is not None else 0 for row in results]
            else:
                labels = [str(row[0]) for row in results]
                values = [float(row[1]) if row[1] is not None else 0 for row in results]
            
            return {
                "chart_type": chart_type,
                "title": self._generate_title(question),
                "labels": labels,
                "datasets": [{
                    "label": "Data",
                    "data": values,
                    "backgroundColor": create_chart_colors(len(labels)) if chart_type in ['pie', 'doughnut'] else get_chart_color()
                }],
                "options": {}
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
        # Remove question marks and common phrases
        title = question.replace("?", "").replace(".", "")
        title = re.sub(r'^(show|display|create|generate|give|provide|plot|chart|graph)\s+', '', title, flags=re.IGNORECASE)
        title = re.sub(r'^(a|an|the)\s+', '', title, flags=re.IGNORECASE)
        title = re.sub(r'\b(pie|bar|line|scatter|radar|chart|graph|visualization|for me|please)\b', '', title, flags=re.IGNORECASE)
        
        # Capitalize first letter and limit length
        title = title.strip()
        if title:
            title = title[0].upper() + title[1:]
            if len(title) > 60:
                title = title[:57] + "..."
        else:
            title = "Data Visualization"
        
        return title 