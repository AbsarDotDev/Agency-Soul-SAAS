"""
Test script for visualization functionality.
To run: cd ai_agent && poetry run python -m app.api.test_visualization
"""
import asyncio
import logging
import json

from app.agents.sql_agent import SQLAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_visualization():
    """Test the visualization generation functionality."""
    # Create SQL agent
    agent = SQLAgent()
    
    # Test visualization request
    message = "Create pie chart showing employees per department"
    company_id = 6
    user_id = "admin"
    
    logger.info(f"Testing visualization with message: {message}")
    
    # Call the generate_visualization method directly
    response = await agent.generate_visualization(
        query=message,
        company_id=company_id,
        user_id=user_id,
        conversation_id=None
    )
    
    # Print the response
    logger.info(f"Response agent_type: {response.agent_type}")
    logger.info(f"Response message: {response.response}")
    
    if response.visualization:
        logger.info(f"Visualization data available: {json.dumps(response.visualization, indent=2)}")
    else:
        logger.error("NO VISUALIZATION DATA RETURNED!")
    
    return response

if __name__ == "__main__":
    asyncio.run(test_visualization()) 