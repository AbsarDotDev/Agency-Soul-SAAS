import matplotlib.pyplot as plt
import io
import base64
import json
from typing import Dict, Any, List, Optional, Tuple


class VisualizationGenerator:
    """Utility class for generating visualizations."""
    
    @staticmethod
    def generate_chart_data(
        chart_type: str,
        labels: List[str],
        data: List[float],
        title: str,
        dataset_label: str = "Data",
        colors: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate chart data for frontend rendering.
        
        Args:
            chart_type: Chart type (bar, line, pie, etc.)
            labels: Chart labels
            data: Chart data values
            title: Chart title
            dataset_label: Label for the dataset
            colors: Optional list of colors
            
        Returns:
            Chart data dictionary
        """
        # Default colors if not provided
        if not colors:
            if chart_type == "pie":
                # Generate colors for pie chart
                colors = [f"rgba({(i * 50) % 256}, {(i * 100) % 256}, {(i * 150) % 256}, 0.7)" for i in range(len(data))]
            else:
                # Default color for bar/line
                colors = ["rgba(54, 162, 235, 0.5)"]
        
        # Create chart data based on type
        if chart_type == "pie":
            return {
                "chart_type": chart_type,
                "labels": labels,
                "datasets": [{
                    "data": data,
                    "backgroundColor": colors
                }],
                "title": title,
                "description": f"Distribution of {dataset_label}"
            }
        elif chart_type == "bar":
            return {
                "chart_type": chart_type,
                "labels": labels,
                "datasets": [{
                    "label": dataset_label,
                    "data": data,
                    "backgroundColor": colors[0] if len(colors) == 1 else colors
                }],
                "title": title,
                "description": f"Bar chart of {dataset_label}"
            }
        elif chart_type == "line":
            return {
                "chart_type": chart_type,
                "labels": labels,
                "datasets": [{
                    "label": dataset_label,
                    "data": data,
                    "borderColor": colors[0].replace("0.5", "1") if len(colors) == 1 else colors[0],
                    "backgroundColor": colors[0],
                    "tension": 0.1
                }],
                "title": title,
                "description": f"Line chart of {dataset_label} over time"
            }
        else:
            # Default to bar chart
            return {
                "chart_type": "bar",
                "labels": labels,
                "datasets": [{
                    "label": dataset_label,
                    "data": data,
                    "backgroundColor": colors[0] if len(colors) == 1 else colors
                }],
                "title": title,
                "description": f"Chart of {dataset_label}"
            }
    
    @staticmethod
    def generate_image(
        chart_data: Dict[str, Any],
        width: int = 10,
        height: int = 6,
        dpi: int = 100
    ) -> str:
        """Generate chart image and return as base64 string.
        
        Args:
            chart_data: Chart data dictionary
            width: Chart width in inches
            height: Chart height in inches
            dpi: Chart resolution (dots per inch)
            
        Returns:
            Base64 encoded image string
        """
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
        
        # Get chart type and data
        chart_type = chart_data.get("chart_type", "bar")
        labels = chart_data.get("labels", [])
        title = chart_data.get("title", "Chart")
        
        # Get dataset
        datasets = chart_data.get("datasets", [])
        if not datasets:
            # No datasets, create empty chart
            ax.set_title(title)
            ax.set_xlabel("No Data")
            ax.set_ylabel("No Data")
            
            # Save to in-memory buffer
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png")
            buffer.seek(0)
            
            # Encode to base64
            image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
            plt.close(fig)
            
            return image_base64
        
        # Get first dataset
        dataset = datasets[0]
        data = dataset.get("data", [])
        
        # Generate chart based on type
        if chart_type == "pie":
            # Create pie chart
            wedges, texts, autotexts = ax.pie(
                data, 
                labels=labels, 
                autopct="%1.1f%%",
                startangle=90,
                colors=dataset.get("backgroundColor", None)
            )
            ax.set_title(title)
            ax.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle
            
        elif chart_type == "bar":
            # Create bar chart
            x = range(len(labels))
            bars = ax.bar(
                x, 
                data, 
                color=dataset.get("backgroundColor", "skyblue"),
                label=dataset.get("label", "Data")
            )
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_title(title)
            ax.legend()
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f"{height:,.0f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom"
                )
                
        elif chart_type == "line":
            # Create line chart
            x = range(len(labels))
            line = ax.plot(
                x, 
                data, 
                color=dataset.get("borderColor", "skyblue"),
                label=dataset.get("label", "Data"),
                marker="o"
            )
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_title(title)
            ax.legend()
            
            # Add grid
            ax.grid(True, linestyle="--", alpha=0.7)
            
            # Add value labels on line points
            for i, value in enumerate(data):
                ax.annotate(f"{value:,.0f}",
                    xy=(i, value),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center", va="bottom"
                )
        
        else:
            # Unsupported chart type, create text
            ax.text(0.5, 0.5, f"Unsupported chart type: {chart_type}",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes
            )
            ax.set_title(title)
        
        # Save to in-memory buffer
        buffer = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buffer, format="png")
        buffer.seek(0)
        
        # Encode to base64
        image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        plt.close(fig)
        
        return image_base64 