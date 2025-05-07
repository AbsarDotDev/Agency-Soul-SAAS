#!/usr/bin/env python3
"""
Test script to demonstrate the usage of MCP MySQL integration with our agent.
This script can be used to test the SQL agent's ability to dynamically 
determine company isolation columns and execute proper company-scoped queries.
"""

import asyncio
import os
import sys
import json
from sqlalchemy.orm import Session

# Add parent directory to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.agents.sql_agent import SQLAgent, MCPMySQLTool
from app.database.connection import DatabaseConnection, get_company_isolation_column, build_table_isolation_info


async def test_mcp_mysql_integration():
    """Test MCP MySQL integration with SQL agent."""
    # Create session
    engine = DatabaseConnection.create_engine()
    session = Session(engine)
    
    # Test company ID (replace with valid company ID)
    company_id = 1  # This should be a valid company ID in your system
    
    try:
        print("=== Testing MCP MySQL Integration ===")
        
        # First, test listing tables
        print("\n1. Testing table listing:")
        mcp_tool = MCPMySQLTool()
        tables = await mcp_tool.list_tables(company_id)
        print(f"Found {len(tables)} tables")
        print(f"First 10 tables: {tables[:10]}")
        
        # Next, test getting schema for a specific table
        print("\n2. Testing schema retrieval:")
        test_table = "employees"  # Replace with a valid table name
        schema_info = await mcp_tool.get_table_schema(test_table, company_id)
        print(f"Schema for {test_table}:")
        print(json.dumps(schema_info, indent=2))
        
        # Test company isolation column detection
        print("\n3. Testing company isolation column detection:")
        isolation_col = get_company_isolation_column(test_table, engine)
        print(f"Isolation column for {test_table}: {isolation_col}")
        
        # Build and display isolation mapping for all tables
        print("\n4. Testing complete isolation mapping generation:")
        isolation_map = build_table_isolation_info(engine)
        print(f"Found isolation columns for {len(isolation_map)} tables")
        print("Sample isolation mapping:")
        sample_entries = list(isolation_map.items())[:5]
        for table, info in sample_entries:
            print(f"  - {table}: {info['column']}")
        
        # Test executing a query with company isolation
        print("\n5. Testing query execution with company isolation:")
        query = "SELECT * FROM employees LIMIT 5"
        result = await mcp_tool.execute_query(query, company_id)
        if result["status"] == "success":
            print("Query executed successfully")
            print(f"Results: {json.dumps(result['results'], indent=2)}")
        else:
            print(f"Query execution failed: {result['error']}")
        
        # Test the SQL agent directly
        print("\n6. Testing SQL agent with company isolation:")
        sql_agent = SQLAgent()
        response = await sql_agent.process_message(
            message="Show me the total number of employees per department",
            company_id=company_id,
            user_id="test_user",
            session=session
        )
        print(f"Agent response: {response.message}")
        
        # Test visualization generation
        print("\n7. Testing visualization generation:")
        viz_response = await sql_agent.generate_visualization(
            query="Create a bar chart showing the number of employees per department",
            company_id=company_id,
            user_id="test_user",
            session=session
        )
        if viz_response.data:
            print("Visualization generated successfully")
            print(f"Explanation: {viz_response.explanation}")
        else:
            print(f"Visualization generation failed: {viz_response.explanation}")
    
    except Exception as e:
        print(f"Error during test: {str(e)}")
    finally:
        session.close()


if __name__ == "__main__":
    # Run the async test function
    asyncio.run(test_mcp_mysql_integration()) 