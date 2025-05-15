#!/usr/bin/env python

"""
Test script to verify LangGraph imports.
Run this with the nomessos conda environment activated.
"""

import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

print("\nTesting imports:")
try:
    from langgraph.prebuilt import ToolNode, tools_condition
    print("✅ Successfully imported ToolNode and tools_condition from langgraph.prebuilt")
    
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    print("✅ Successfully imported message types from langchain_core.messages")
    
    from langgraph.graph import END, StateGraph
    print("✅ Successfully imported StateGraph from langgraph.graph")
    
    from app.core.langgraph_agents import get_langgraph_dispatcher
    print("✅ Successfully imported get_langgraph_dispatcher from app.core.langgraph_agents")

    print("\nAll imports successful! Your environment is correctly set up.")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nPlease ensure you're running this script with the 'nomessos' conda environment active:")
    print("    conda activate nomessos")
    print("    python test_imports.py") 