import re

# Path to the file
file_path = '/Applications/XAMPP/xamppfiles/htdocs/nomessos/ai_agent/app/visualizations/langgraph/visualization_agent.py'

# Read the file
with open(file_path, 'r') as file:
    content = file.read()

# Add debugging log to generate_sql
pattern = r'def generate_sql\(self, state: Dict\[str, Any\]\) -> Dict\[str, Any\]:.*?query = state\[\'query\'\]'
replacement = '''def generate_sql(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate SQL query based on parsed question and unique nouns."""
        query = state['query']
        parsed_question = state.get('parsed_question', {})
        unique_nouns = state.get('unique_nouns', [            
        # Generate a default SQL query in case LLM fails
        default_sql = self._create_default_sql_query(parsed_question, company_id)
        
        # If we have a default SQL, log it
        if default_sql:
            logger.info(f"Created default SQL as fallback: {default_sql[:50]}...")
        query = state['query']'''

modified_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

# Fix the validation handling for SQL validation 
pattern2 = r'def validate_and_fix_sql\(self, state: Dict\[str, Any\]\) -> Dict\[str, Any\]:.*?if not sql_query or not is_valid:'
replacement2 = '''def validate_and_fix_sql(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and fix the generated SQL query to ensure company isolation and SQL correctness."""
        sql_query = state.get('sql_query', '')
        company_id = state['company_id']
        query = state['query']
        is_valid = state.get('is_valid', False)
        
        # Debug output to track flow
        logger.info(f"Validating SQL: query_length={len(sql_query)}, is_valid={is_valid}")
        
        # If SQL generation already marked the query as invalid, pass that through
        if not sql_query or not is_valid:'''

modified_content = re.sub(pattern2, replacement2, modified_content, flags=re.DOTALL)

# Write the changes back to the file
with open(file_path, 'w') as file:
    file.write(modified_content)

print("Successfully added debug info and fixed generate_sql method!")
