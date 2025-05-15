"""
Chart utilities for data visualization.
This module provides helpers for formatting and preparing data for Chart.js.
"""

import logging
import json
import re
import random
from typing import Dict, Any, List, Optional

import pandas as pd

# Set up logging
logger = logging.getLogger(__name__)

# Standard chart colors
CHART_COLORS = [
    "rgba(75, 192, 192, 0.6)",    # teal
    "rgba(255, 99, 132, 0.6)",    # red
    "rgba(255, 205, 86, 0.6)",    # yellow
    "rgba(54, 162, 235, 0.6)",    # blue
    "rgba(153, 102, 255, 0.6)",   # purple
    "rgba(255, 159, 64, 0.6)",    # orange
    "rgba(201, 203, 207, 0.6)",   # grey
    "rgba(0, 204, 150, 0.6)",     # green
    "rgba(255, 69, 0, 0.6)",      # orangered
    "rgba(147, 112, 219, 0.6)"    # mediumpurple
]

def get_chart_color(index: int = None) -> str:
    """Get a chart color by index or a random color if index is None."""
    if index is None:
        return random.choice(CHART_COLORS)
    return CHART_COLORS[index % len(CHART_COLORS)]

def create_chart_colors(count: int) -> List[str]:
    """Create a list of chart colors."""
    return [get_chart_color(i) for i in range(count)]

def format_chart_data(
    chart_type: str, 
    data: List[Dict[str, Any]], 
    title: str = None, 
    x_axis: str = None, 
    y_axis: str = None
) -> Dict[str, Any]:
    """
    Format data for Chart.js based on chart type.
    
    Args:
        chart_type: Type of chart (bar, line, pie, etc.)
        data: List of data rows
        title: Chart title
        x_axis: X-axis label
        y_axis: Y-axis label
        
    Returns:
        Formatted chart data for Chart.js
    """
    if not data:
        return None
        
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(data)
    
    # Ensure we have at least 2 columns
    if len(df.columns) < 2:
        logger.warning("Not enough columns for visualization")
        if len(df.columns) == 1:
            # Create a count column
            col_name = df.columns[0]
            df = df.groupby(col_name).size().reset_index(name='count')
    
    # Get column names
    columns = df.columns.tolist()
    
    # Determine which columns to use for labels and data
    label_col = columns[0]  # First column for labels
    data_col = columns[1] if len(columns) > 1 else None  # Second column for data values
    
    # Create chart data structure based on chart type
    if chart_type in ["pie", "doughnut", "polarArea"]:
        # Get unique labels and values
        labels = df[label_col].astype(str).tolist()
        values = df[data_col].tolist() if data_col else [1] * len(labels)
        
        return {
            "chart_type": chart_type,
            "title": title or f"Distribution of {data_col or 'items'} by {label_col}",
            "labels": labels,
            "datasets": [{
                "label": data_col or "Value",
                "data": values,
                "backgroundColor": create_chart_colors(len(labels))
            }],
            "options": {}
        }
    
    elif chart_type in ["bar", "line"]:
        # Get unique labels and values
        labels = df[label_col].astype(str).tolist()
        values = df[data_col].tolist() if data_col else [1] * len(labels)
        
        return {
            "chart_type": chart_type,
            "title": title or f"{data_col or 'Count'} by {label_col}",
            "labels": labels,
            "datasets": [{
                "label": data_col or "Count",
                "data": values,
                "backgroundColor": get_chart_color() if chart_type == "bar" else "rgba(75, 192, 192, 0.6)",
                "borderColor": "rgba(75, 192, 192, 1)" if chart_type == "line" else None
            }],
            "options": {
                "scales": {
                    "y": {"title": {"display": True, "text": y_axis or data_col or "Count"}},
                    "x": {"title": {"display": True, "text": x_axis or label_col}}
                }
            }
        }
    
    elif chart_type == "scatter":
        # Create scatter data points
        data_points = []
        for _, row in df.iterrows():
            try:
                data_points.append({
                    "x": float(row[columns[0]]) if columns[0] else 0,
                    "y": float(row[columns[1]]) if len(columns) > 1 else 0
                })
            except (ValueError, TypeError):
                # Skip points that can't be converted to float
                continue
                
        return {
            "chart_type": "scatter",
            "title": title or f"Relationship between {columns[0]} and {columns[1]}",
            "datasets": [{
                "label": f"{columns[0]} vs {columns[1]}",
                "data": data_points,
                "backgroundColor": "rgba(75, 192, 192, 0.6)"
            }],
            "options": {
                "scales": {
                    "y": {"title": {"display": True, "text": y_axis or columns[1]}},
                    "x": {"title": {"display": True, "text": x_axis or columns[0]}}
                }
            }
        }
    
    else:
        # Default to bar chart for unknown types
        labels = df[label_col].astype(str).tolist()
        values = df[data_col].tolist() if data_col else [1] * len(labels)
        
        return {
            "chart_type": "bar",
            "title": title or "Data Visualization",
            "labels": labels,
            "datasets": [{
                "label": data_col or "Value",
                "data": values,
                "backgroundColor": "rgba(75, 192, 192, 0.6)"
            }],
            "options": {
                "scales": {
                    "y": {"title": {"display": True, "text": y_axis or data_col or "Value"}},
                    "x": {"title": {"display": True, "text": x_axis or label_col}}
                }
            }
        }

def extract_chart_type_from_query(query: str) -> Optional[str]:
    """Extract chart type from user query."""
    query = query.lower()
    
    if "bar chart" in query or "bar graph" in query:
        return "bar"
    elif "pie chart" in query:
        return "pie"
    elif "line chart" in query or "line graph" in query or "trend" in query:
        return "line"
    elif "scatter plot" in query or "scatter chart" in query:
        return "scatter"
    elif "doughnut chart" in query or "donut chart" in query:
        return "doughnut"
    elif "radar chart" in query or "radar graph" in query:
        return "radar"
    elif "polar chart" in query or "polar area" in query:
        return "polarArea"
    
    return None

def create_dummy_data(query: str, chart_type: str) -> List[Dict[str, Any]]:
    """Create dummy data when SQL returns no results."""
    # Extract key terms for creating relevant dummy data
    keywords = re.findall(r'\b\w+\b', query.lower())
    
    if "employee" in keywords and "department" in keywords:
        # Example: employees per department
        return [
            {"department": "IT", "employees": 12},
            {"department": "HR", "employees": 8},
            {"department": "Finance", "employees": 6},
            {"department": "Marketing", "employees": 10},
            {"department": "Sales", "employees": 15}
        ]
    elif "revenue" in keywords or "sales" in keywords:
        # Example: revenue by month
        return [
            {"month": "Jan", "revenue": 12500},
            {"month": "Feb", "revenue": 14200},
            {"month": "Mar", "revenue": 16800},
            {"month": "Apr", "revenue": 15300},
            {"month": "May", "revenue": 17600}
        ]
    elif "project" in keywords:
        # Example: projects by status
        return [
            {"status": "Completed", "count": 24},
            {"status": "In Progress", "count": 18},
            {"status": "Not Started", "count": 7},
            {"status": "On Hold", "count": 5},
            {"status": "Cancelled", "count": 2}
        ]
    elif "expense" in keywords or "cost" in keywords:
        # Example: expenses by category
        return [
            {"category": "Salaries", "amount": 45000},
            {"category": "Rent", "amount": 12000},
            {"category": "Utilities", "amount": 5500},
            {"category": "Marketing", "amount": 8500},
            {"category": "Office Supplies", "amount": 3200}
        ]
    else:
        # Generic dummy data
        return [
            {"category": "Category A", "value": 45},
            {"category": "Category B", "value": 32},
            {"category": "Category C", "value": 68},
            {"category": "Category D", "value": 27},
            {"category": "Category E", "value": 55}
        ]
