"""
Test script for the LLM manager with problematic JSON templates
"""

import sys
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the ai_agent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ai_agent'))

# Import the LLM manager
from app.visualizations.langgraph.llm_manager import LLMManager
from langchain_core.prompts import ChatPromptTemplate

def test_basic_json():
    # Create an instance of the LLM manager
    llm_manager = LLMManager()
    
    # Test prompt with JSON template containing "is_relevant" field
    system_prompt = r"""You are a data analyst with expertise in SQL and database schema analysis.
Given a user's question and database schema, identify the relevant tables and columns needed to answer the question.

If the question cannot be answered using the provided schema, set is_relevant to false.

Respond with a JSON object in the following format:
{
    "is_relevant": boolean,
    "relevant_tables": [
        {
            "table_name": string,
            "columns": [string],
            "entity_columns": [string]
        }
    ],
    "reasoning": string
}"""

    human_prompt = r"""===Database schema:
{schema}

===User question:
{question}

Identify relevant tables and columns:"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt)
    ])
    
    # Sample schema and question
    schema = """
    employees (id, name, department_id, salary, hire_date)
    departments (id, name, manager_id)
    """
    
    question = "How many employees are in each department?"
    
    # Invoke the LLM manager
    try:
        response = llm_manager.invoke(prompt, schema=schema, question=question)
        logger.info(f"Test 1 Success! LLM response: {response[:100]}...")
    except Exception as e:
        logger.error(f"Test 1 Error invoking LLM: {str(e)}")

def test_complex_json():
    # Create an instance of the LLM manager
    llm_manager = LLMManager()
    
    # Test prompt with complex nested JSON template
    system_prompt = r"""Generate a chart configuration for the given data.

The configuration should follow this JSON format:
{
    "chart_type": string,
    "title": string,
    "datasets": [
        {
            "label": string,
            "data": [number],
            "backgroundColor": [string],
            "options": {
                "is_stacked": boolean,
                "show_legend": boolean,
                "data_labels": {
                    "position": string,
                    "format": string
                }
            }
        }
    ],
    "labels": [string],
    "meta": {
        "created_at": string,
        "user_question": string,
        "explanation": string
    }
}"""

    human_prompt = r"""Question: {question}

Data: {data}

Generate chart configuration:"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt)
    ])
    
    # Sample data
    data = """
    [
        {"department": "Sales", "count": 42}, 
        {"department": "Engineering", "count": 37}, 
        {"department": "Marketing", "count": 18}
    ]
    """
    
    question = "Show me a bar chart of employees by department"
    
    # Invoke the LLM manager
    try:
        response = llm_manager.invoke(prompt, data=data, question=question)
        logger.info(f"Test 2 Success! LLM response: {response[:100]}...")
    except Exception as e:
        logger.error(f"Test 2 Error invoking LLM: {str(e)}")
        
if __name__ == "__main__":
    test_basic_json()
    test_complex_json() 