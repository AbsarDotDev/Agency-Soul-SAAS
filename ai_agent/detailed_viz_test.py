"""
Test script for the visualization agent using the workflow directly
"""
import asyncio
import logging
import json
from app.visualizations.langgraph.visualization_agent import VisualizationLangGraphAgent
from app.visualizations.langgraph.workflow import WorkflowManager

# Configure logging to show detailed info
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def print_separator():
    print("\n" + "="*80 + "\n")

def pretty_print_result(result):
    """Print the visualization result in a readable format"""
    print("\nRESULT:")
    chart_data = result.get('chart_data')
    if chart_data:
        print(f"✅ Chart generated successfully!")
        print(f"  Type: {chart_data.get('chart_type', 'unknown')}")
        print(f"  Title: {chart_data.get('title', 'No title')}")
        
        datasets = chart_data.get('datasets', [])
        print(f"  Datasets: {len(datasets)}")
        
        labels = chart_data.get('labels', [])
        print(f"  Labels: {len(labels)} items")
        if labels:
            print(f"  First few labels: {labels[:5]}")
        
        if datasets:
            data = datasets[0].get('data', [])
            print(f"  Data points: {len(data)}")
            if data:
                print(f"  Sample data: {data[:5]}")
    else:
        print("❌ No chart data generated")
    
    explanation = result.get('answer', '')
    if explanation:
        print(f"\nExplanation: {explanation}")
    
    error = result.get('error', '')
    if error:
        print(f"\n⚠️ Error: {error}")

def test_visualization(query, company_id=1):
    """Test the visualization workflow with a specific query"""
    print_separator()
    print(f"TESTING: '{query}'")
    
    # Create visualization agent
    agent = VisualizationLangGraphAgent()
    
    # Directly use the agent to generate visualization
    result = agent.generate_visualization(query=query, company_id=company_id)
    
    # Print the results
    pretty_print_result(result)
    return result

def main():
    print("Starting visualization tests...")
    
    # Test with different types of queries
    test_visualization("Show me invoice totals by month")
    test_visualization("How many customers do we have by country?")
    test_visualization("Show me revenue by product category")
    test_visualization("Display sales performance over the last 6 months") 

if __name__ == "__main__":
    main()
