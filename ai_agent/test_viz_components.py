from app.visualizations.langgraph.visualization_agent import VisualizationLangGraphAgent

def test_visualization_agent():
    agent = VisualizationLangGraphAgent()
    state = {
        'query': 'Show me sales by month',
        'company_id': 1,
        'chart_type': 'bar'
    }
    
    # Test execute_sql method
    try:
        sql_result = agent.execute_sql(state)
        print(f"SQL execution result: {sql_result}")
        
        # Add SQL result to state
        state.update(sql_result)
        
        # Test format_data_for_visualization method
        if sql_result.get('is_valid', False):
            viz_result = agent.format_data_for_visualization(state)
            print(f"Visualization result: {viz_result.get('chart_data') is not None}")
            print(f"Explanation: {viz_result.get('explanation', 'No explanation')}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_visualization_agent()
