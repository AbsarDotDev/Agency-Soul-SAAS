"""
Quick script to fix the visualization agent by patching key issues
"""
import os
import re

# Path to the visualization agent file
file_path = "/Applications/XAMPP/xamppfiles/htdocs/nomessos/ai_agent/app/visualizations/langgraph/visualization_agent.py"

# Read the file
with open(file_path, 'r') as f:
    content = f.read()

# Fix 1: In generate_sql method, ensure is_relevant logic works correctly
content = re.sub(
    r'if not parsed_question.get\(\'is_relevant\', False\):\s+logger\.error\("Question was determined not relevant for SQL generation"\)',
    'if not parsed_question.get(\'is_relevant\', False):\n            logger.warning("Question was determined not relevant for SQL generation")',
    content
)

# Fix 2: Make sure we return valid SQL even when LLM generation fails
pattern = re.compile(r'(# Make sure we have a proper SQL query.*?default_sql = self\._create_default_sql_query\(parsed_question, company_id\).*?if default_sql:.*?logger\.info\(f"Using default SQL: \{default_sql\}"\).*?return \{"sql_query": default_sql, "is_valid": True\}.*?else:.*?return \{"sql_query": "", "is_valid": False, "error_message": "Could not generate a valid SQL query"\})', re.DOTALL)

replacement = '''# Make sure we have a proper SQL query with a minimum structure
            if not sql_query or len(sql_query) < 10 or "SELECT" not in sql_query.upper():
                logger.warning("Generated SQL query is empty or invalid, using default SQL")
                if default_sql:
                    logger.info(f"Using default SQL: {default_sql}")
                    return {"sql_query": default_sql, "is_valid": True}
                else:
                    # Final fallback - create a very simple query
                    tables = parsed_question.get('relevant_tables', [])
                    if tables and tables[0].get('table_name'):
                        table_name = tables[0].get('table_name')
                        isolation_col = tables[0].get('isolation_column', 'created_by')
                        fallback_sql = f"SELECT * FROM {table_name} WHERE {isolation_col} = {company_id} LIMIT 10"
                        logger.info(f"Using emergency fallback SQL: {fallback_sql}")
                        return {"sql_query": fallback_sql, "is_valid": True}
                    else:
                        return {"sql_query": "", "is_valid": False, "error_message": "Could not generate a valid SQL query"}'''

# Apply the second fix
content = pattern.sub(replacement, content)

# Fix 3: Update the _create_default_sql_query method to be more robust
# We'll look for date columns more aggressively and ensure proper isolation
pattern_default_sql = re.compile(r'def _create_default_sql_query.*?try:.*?if not parsed_question.*?return "".*?return ""', re.DOTALL)

replacement_default_sql = '''def _create_default_sql_query(self, parsed_question: Dict[str, Any], company_id: int) -> str:
        """Create a default SQL query when LLM fails to generate one."""
        try:
            if not parsed_question.get('is_relevant', False):
                logger.warning("Cannot create default SQL: question not relevant")
                return ""
                
            # Get the first relevant table
            tables = parsed_question.get('relevant_tables', [])
            if not tables:
                logger.warning("Cannot create default SQL: no relevant tables")
                return ""
                
            # Use the first table as primary table
            primary_table = tables[0]
            table_name = primary_table.get('table_name')
            isolation_column = primary_table.get('isolation_column') or 'created_by'  # Default to created_by
            
            if not table_name:
                logger.warning("Cannot create default SQL: missing table name")
                return ""
                
            # Get columns to select
            columns = primary_table.get('columns', [])
            if not columns:
                columns = ['*']  # Fallback to all columns
                
            # Create default SQL for common scenarios
            # Common data visualization request patterns:
            
            # 1. Any column with "date", "time", "month", "created" in name - time series
            date_column = None
            for col in columns:
                if 'date' in col.lower() or 'time' in col.lower() or 'month' in col.lower() or 'created_at' == col:
                    date_column = col
                    break
            
            # 2. Identify metrics columns
            numeric_col = None
            for col in columns:
                if col in ['amount', 'price', 'total', 'sum', 'value', 'quantity', 'count', 'revenue']:
                    numeric_col = col
                    break
            
            # 3. Identify category columns
            category_col = None
            for col in columns:
                if col in ['name', 'category', 'type', 'status', 'department', 'group']:
                    category_col = col
                    break
            
            # If we found date & numeric columns - do a time series
            if date_column and numeric_col:
                sql = f"""SELECT 
                    DATE_FORMAT({date_column}, '%Y-%m') AS month,
                    SUM({numeric_col}) AS total_{numeric_col}
                FROM {table_name}
                WHERE {table_name}.{isolation_column} = {company_id}
                GROUP BY month
                ORDER BY month DESC
                LIMIT 12"""
                return sql
            
            # If we found date column - do a count by date
            elif date_column:
                sql = f"""SELECT 
                    DATE_FORMAT({date_column}, '%Y-%m') AS month,
                    COUNT(*) AS count
                FROM {table_name}
                WHERE {table_name}.{isolation_column} = {company_id}
                GROUP BY month
                ORDER BY month DESC
                LIMIT 12"""
                return sql
            
            # If we found category & numeric columns - do category comparison
            elif category_col and numeric_col:
                sql = f"""SELECT 
                    {category_col},
                    SUM({numeric_col}) AS total_{numeric_col}
                FROM {table_name}
                WHERE {table_name}.{isolation_column} = {company_id}
                GROUP BY {category_col}
                ORDER BY total_{numeric_col} DESC
                LIMIT 10"""
                return sql
            
            # If we found category column - do count by category
            elif category_col:
                sql = f"""SELECT 
                    {category_col},
                    COUNT(*) AS count
                FROM {table_name}
                WHERE {table_name}.{isolation_column} = {company_id}
                GROUP BY {category_col}
                ORDER BY count DESC
                LIMIT 10"""
                return sql
            
            # If we found numeric column - do a top records by that metric
            elif numeric_col:
                other_col = None
                for col in columns:
                    if col != numeric_col and col != 'id':
                        other_col = col
                        break
                        
                select_col = other_col if other_col else 'id'
                
                sql = f"""SELECT 
                    {select_col},
                    {numeric_col}
                FROM {table_name}
                WHERE {table_name}.{isolation_column} = {company_id}
                ORDER BY {numeric_col} DESC
                LIMIT 10"""
                return sql
            
            # Ultimate fallback - return everything with a limit
            else:
                sql = f"""SELECT * FROM {table_name}
                WHERE {table_name}.{isolation_column} = {company_id}
                LIMIT 10"""
                return sql
                
        except Exception as e:
            logger.error(f"Error creating default SQL query: {str(e)}")
            return ""'''

# Apply the third fix (more robust default SQL)
if '_create_default_sql_query' in content:
    content = re.sub(pattern_default_sql, replacement_default_sql, content)

# Write the updated content back to file
with open(file_path, 'w') as f:
    f.write(content)

print("Successfully patched the visualization agent!")
