import asyncio
from app.agents.visualization_langgraph_agent import VisualizationLangGraphFacade

async def test():
    agent = VisualizationLangGraphFacade()
    result = await agent.generate_visualization('Show me sales by month', 1, 'test_user')
    print(f'Got visualization: {result.chart_data is not None}')

if __name__ == "__main__":
    asyncio.run(test())
