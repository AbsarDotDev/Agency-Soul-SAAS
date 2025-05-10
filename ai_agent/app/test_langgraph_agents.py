import asyncio
import logging
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the LangGraph agents
from app.core.langgraph_agents import (
    AgentType, 
    get_agent, 
    build_agent_graph,
    get_langgraph_dispatcher
)
from app.agents.base_agent import AgentResponse

async def test_agents():
    """Test the specialized agents implementation."""
    # Test each agent type
    agent_types = [
        AgentType.SQL,
        AgentType.HRM,
        AgentType.FINANCE,
        AgentType.CRM,
        AgentType.PROJECT,
        AgentType.PRODUCT
    ]
    
    # You'll need to set these for testing
    company_id = 1  # Company ID for testing
    user_id = "test_user"  # User ID for testing
    
    for agent_type in agent_types:
        print(f"\n----- Testing {agent_type.value} agent -----")
        
        # Create agent instance
        agent = get_agent(agent_type)
        
        # Test query
        test_query = ""
        if agent_type == AgentType.SQL:
            test_query = "How many employees do we have?"
        elif agent_type == AgentType.HRM:
            test_query = "List all departments and how many employees are in each."
        elif agent_type == AgentType.FINANCE:
            test_query = "What's our total revenue this month?"
        elif agent_type == AgentType.CRM:
            test_query = "Show me our top 5 customers by revenue."
        elif agent_type == AgentType.PROJECT:
            test_query = "What projects are currently in progress?"
        elif agent_type == AgentType.PRODUCT:
            test_query = "Show me our top selling products."
        
        # Process message with agent
        try:
            response = await agent.process_message(
                message=test_query,
                company_id=company_id,
                user_id=user_id
            )
            print(f"Query: {test_query}")
            print(f"Response: {response.response}\n")
        except Exception as e:
            print(f"Error processing message: {str(e)}")
            
    print("\n----- Testing Dispatcher -----")
    dispatcher = get_langgraph_dispatcher()
    response = await dispatcher.process_message(
        message="How many employees do we have?",
        company_id=company_id,
        user_id=user_id
    )
    print(f"Dispatcher response: {response.response}")
    print(f"Agent type: {response.agent_type}")
    print(f"Tokens used: {response.tokens_used}")

if __name__ == "__main__":
    asyncio.run(test_agents()) 