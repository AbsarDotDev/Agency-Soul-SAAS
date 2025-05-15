import asyncio
import logging
from app.agents.visualization_langgraph_agent import VisualizationLangGraphFacade

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def test_visualization(query, company_id=1, user_id='test_user', chart_type=None):
    """
    Test the visualization workflow with detailed logging.
    
    Args:
        query: The natural language query to visualize
        company_id: The company ID for data isolation
        user_id: The user ID making the request
        chart_type: Optional specific chart type to use
    """
    # Create the agent
    agent = VisualizationLangGraphFacade()
    
    print(f"\n\n{'=' * 80}")
    print(f"TESTING VISUALIZATION QUERY: '{query}'")
    print(f"{'=' * 80}\n")
    
    # Call the agent
    try:
        result = await agent.generate_visualization(query, company_id, user_id, chart_type)
        
        # Check results
        print(f"VISUALIZATION RESULT:")
        print(f"- Has chart data: {hasattr(result, 'chart_data') and result.chart_data is not None}")
        if hasattr(result, 'message'):
            print(f"- Message: {result.message}")
        elif hasattr(result, 'content'):
            print(f"- Content: {result.content}")
        else:
            print(f"- Result: {result}")
        
        # Print chart data summary if available
        if result.chart_data:
            chart_data = result.chart_data
            chart_type = chart_data.get('chart_type', 'unknown')
            title = chart_data.get('title', 'No title')
            datasets = chart_data.get('datasets', [])
            labels = chart_data.get('labels', [])
            
            print(f"\nCHART DETAILS:")
            print(f"- Type: {chart_type}")
            print(f"- Title: {title}")
            print(f"- Number of datasets: {len(datasets)}")
            print(f"- Number of labels: {len(labels)}")
            print(f"- First few labels: {labels[:5]}")
            
            if datasets:
                print(f"- First dataset label: {datasets[0].get('label', 'No label')}")
                data = datasets[0].get('data', [])
                print(f"- First dataset data points: {len(data)}")
                print(f"- Sample data: {data[:5]}")
        else:
            print("\nNo chart data was generated.")
    
    except Exception as e:
        print(f"ERROR: {str(e)}")

async def main():
    """Run multiple visualization tests."""
    # Test with various queries
    await test_visualization("Show me sales by month")
    await test_visualization("How many invoices were created per month?")
    await test_visualization("Show me the distribution of invoice amounts")
    
    # Test with specific chart type
    await test_visualization("Show me the top customers by revenue", chart_type="bar")

if __name__ == "__main__":
    asyncio.run(main())
