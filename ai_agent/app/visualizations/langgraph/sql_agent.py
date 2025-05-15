from typing import Dict, Any, List, Optional
import logging
import json
import re
import datetime
import decimal
import pandas as pd
import ast
from decimal import Decimal, InvalidOperation

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase

from app.visualizations.langgraph.database_manager import DatabaseManager
from app.visualizations.langgraph.llm_manager import LLMManager

# Set up logging with less verbose level
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SQLAgent:
    """
    Agent responsible for SQL-related tasks in the visualization workflow.
    Parses questions, generates queries, and ensures company data isolation.
    Uses langchain's SQLDatabaseToolkit for schema-aware query generation.
    """
    
    def __init__(self, company_id: Optional[int] = None):
        """
        Initialize the SQL agent.
        
        Args:
            company_id: Company ID for data isolation
        """
        self.company_id = company_id
        self.db_manager = DatabaseManager()
        self.llm_manager = LLMManager()
        if company_id is not None:
            self.llm_manager.set_company_id(company_id)
    
    def is_question_relevant(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine if the question is relevant for SQL processing.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state with relevance flag and reasoning
        """
        question = state.get('question', '')
        
        # Get database schema - this could be large and need special handling
        schema = self.db_manager.get_schema(self.company_id)
        
        # Create the relevance prompt
        system_prompt = "You are an AI assistant specialized in determining if a question can be answered using SQL queries. Your task is to evaluate if the provided question is relevant for SQL processing. Return your response as valid JSON."
        
        human_prompt = "Question: {question}\n\nDatabase Schema:\n{schema}\n\nDetermine if this question can be answered by querying a database. Return a JSON with 'is_relevant' (boolean), 'relevant_tables' (list of table names that would be used), and 'reasoning' (your explanation).\n\nYour response MUST be in valid JSON format."
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
        
        try:
            # Invoke the LLM to check relevance
            response = self.llm_manager.invoke(
                prompt, 
                question=question,
                schema=schema
            )
            
            # Try to parse the response as JSON
            try:
                # First try direct parsing
                relevance_data = json.loads(response)
            except json.JSONDecodeError:
                # If that fails, try to extract JSON from the response
                logger.warning("Failed to parse SQL relevance response as JSON. Attempting extraction.")
                json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
                matches = re.findall(json_pattern, response, re.DOTALL)
                
                if matches:
                    for match in matches:
                        try:
                            relevance_data = json.loads(match)
                            break
                        except json.JSONDecodeError:
                            continue
                else:
                    # Try to find anything that looks like JSON between curly braces
                    try:
                        start_idx = response.find('{')
                        end_idx = response.rfind('}') + 1
                        if start_idx >= 0 and end_idx > start_idx:
                            json_str = response[start_idx:end_idx]
                            relevance_data = json.loads(json_str)
                        else:
                            raise json.JSONDecodeError("No JSON found", response, 0)
                    except json.JSONDecodeError:
                        # Fallback with default values
                        relevance_data = {
                            "is_relevant": True,  # Default to true to attempt query generation
                            "relevant_tables": [],
                            "reasoning": "Error parsing response, assuming SQL is relevant"
                        }
            
            # Update the state with relevance information
            state['is_sql_relevant'] = relevance_data.get('is_relevant', True)  # Default to True
            state['relevant_tables'] = relevance_data.get('relevant_tables', [])
            state['relevance_reasoning'] = relevance_data.get('reasoning', '')
            
        except Exception as e:
            logger.error(f"Error determining SQL relevance: {str(e)}")
            # Default to True to attempt query generation anyway
            state['is_sql_relevant'] = True
            state['relevant_tables'] = []
            state['relevance_reasoning'] = f"Error in relevance check: {str(e)}. Proceeding with SQL generation."
        
        return state
    
    def generate_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a SQL query based on the user question using SQLDatabaseToolkit.
        Falls back to direct LLM query generation if the toolkit fails.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state with SQL query
        """
        # Only proceed if the question is relevant for SQL
        if not state.get('is_sql_relevant', False):
            state['sql_query'] = ''
            state['sql_generation_reason'] = 'Question not relevant for SQL processing'
            return state
        
        question = state.get('question', '')
        
        try:
            # Get a company-isolated SQLDatabase instance
            db = self.db_manager.get_db(self.company_id)
            
            # Use direct LLM approach - this is more reliable for our use case
            # than trying to use the SQLDatabaseToolkit in this particular setup
            sql_query = self._generate_query_via_llm(question, db)
            
            # Verify company isolation in the query
            if sql_query and self.company_id is not None:
                # Add company isolation if missing
                if "created_by" not in sql_query and str(self.company_id) not in sql_query:
                    # Get tables in the query
                    table_pattern = r"FROM\s+`?(\w+)`?|JOIN\s+`?(\w+)`?"
                    tables = []
                    for match in re.finditer(table_pattern, sql_query, re.IGNORECASE):
                        table = match.group(1) or match.group(2)
                        if table:
                            tables.append(table)
                    
                    # Add isolation for the first table if possible
                    if tables:
                        from app.database.connection import get_company_isolation_column
                        isolation_col = get_company_isolation_column(tables[0])
                        if isolation_col:
                            where_pos = sql_query.upper().find("WHERE")
                            if where_pos > 0:
                                sql_query = f"{sql_query[:where_pos + 5]} {isolation_col} = {self.company_id} AND {sql_query[where_pos + 5:]}"
                            else:
                                # Add WHERE clause for isolation
                                for clause in ["GROUP BY", "ORDER BY", "LIMIT", "HAVING"]:
                                    clause_pos = sql_query.upper().find(clause)
                                    if clause_pos > 0:
                                        sql_query = f"{sql_query[:clause_pos]} WHERE {isolation_col} = {self.company_id} {sql_query[clause_pos:]}"
                                        break
                                else:
                                    # If no clauses found, add WHERE at the end
                                    sql_query = f"{sql_query} WHERE {isolation_col} = {self.company_id}"
            
            # Update the state
            state['sql_query'] = sql_query
            state['sql_generation_reason'] = f"Generated SQL query for database using schema awareness."
            
        except Exception as e:
            logger.error(f"Error generating SQL query: {str(e)}")
            try:
                # Attempt to fall back to direct LLM query generation
                logger.info("Falling back to direct LLM query generation after error")
                db = self.db_manager.get_db(self.company_id)
                sql_query = self._generate_query_via_llm(question, db)
                state['sql_query'] = sql_query
                state['sql_generation_reason'] = f"Generated SQL query with direct LLM approach after toolkit error: {str(e)}"
            except Exception as fallback_error:
                logger.error(f"Error in fallback query generation: {str(fallback_error)}")
                state['sql_query'] = ''
                state['sql_generation_reason'] = f"Error in query generation: {str(e)}, fallback also failed: {str(fallback_error)}"
        
        return state
    
    def _generate_query_via_llm(self, question: str, db: SQLDatabase) -> str:
        """
        Directly generate a SQL query using the LLM and schema information.
        Used as a fallback when SQLDatabaseToolkit fails.
        
        Args:
            question: User question
            db: SQLDatabase instance
            
        Returns:
            Generated SQL query
        """
        # Create a prompt with the schema and question
        schema_info = db.get_table_info()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert SQL query generator. Your task is to create a well-formed SQL query based on the question and database schema provided.

IMPORTANT GUIDELINES:
1. Use the EXACT table and column names from the schema
2. ALWAYS include company isolation by adding 'created_by = {company_id}' in the WHERE clause (or the appropriate company isolation column for the table if not 'created_by').
3. Write complete queries with all necessary JOINs, WHERE conditions, and columns
4. Format dates and perform date calculations using MySQL/MariaDB-compatible syntax (e.g., DATEDIFF(), TIMESTAMPDIFF(), DATE_FORMAT()). Avoid SQLite-specific functions like JULIANDAY().
5. Do not use placeholder text like 'your_columns' or '...'
6. Limit results to a reasonable number (e.g., LIMIT 10 for large result sets by default, unless the question implies otherwise like "all" or "top N")
7. Do not include any explanations, just return a valid SQL query. Ensure the query is directly executable.

DATABASE SCHEMA:
{schema}

EXAMPLE QUERY (for department-employee count):
SELECT d.name AS department_name, COUNT(e.id) AS employee_count
FROM departments d
JOIN employees e ON d.id = e.department_id
WHERE d.created_by = {company_id}
GROUP BY d.name
ORDER BY employee_count DESC;"""),
            ("human", "Create a SQL query to answer this question: {question}")
        ])
        
        # Invoke the LLM
        result = self.llm_manager.invoke(
            prompt,
            schema=schema_info,
            company_id=self.company_id,
            question=question
        )
        
        # Extract the SQL query - strip markdown code blocks if present
        sql_pattern = r"```sql\s*(.*?)\s*```"
        sql_matches = re.findall(sql_pattern, result, re.DOTALL)
        
        if sql_matches:
            return sql_matches[0].strip()
        else:
            # If no code blocks, return the whole response if it looks like SQL
            if "SELECT" in result and "FROM" in result:
                return result.strip()
            else:
                logger.warning("LLM response doesn't look like SQL. Attempting to extract anyway.")
                # Try to find anything that looks like a SQL query
                sql_pattern = r"SELECT\s+.*?FROM.*?(?:WHERE|GROUP BY|ORDER BY|LIMIT|;|$)"
                sql_match = re.search(sql_pattern, result, re.DOTALL | re.IGNORECASE)
                if sql_match:
                    return sql_match.group(0).strip()
                else:
                    raise ValueError(f"Could not extract SQL query from LLM response: {result}")
    
    def _parse_string_tuple_result(self, result_str: str) -> List[Dict[str, Any]]:
        """
        Robustly parse a string representation of a list of tuples,
        e.g., "[('label1', Decimal('10.5')), ('label2', 20)]"
        into a list of dicts, e.g., [{'label': 'label1', 'value': 10.5}, ...]
        Handles Decimal objects by converting them to floats.
        """
        logger.info(f"Attempting to parse string tuple result: {result_str}")
        parsed_results = []
        
        temp_str = result_str
        # Order of replacement matters. First, more specific (quoted) then general.
        # Replace Decimal('number_string') or Decimal("number_string") with 'number_string'
        temp_str = re.sub(r"Decimal\((['\"])((?:\\\\\1|.)*?)\1\)", r"'\2'", temp_str)
        # Replace Decimal(possibly_unquoted_number_or_string_literal) with 'content_as_string'
        # This is a bit more aggressive if the content isn't already a string literal.
        # It aims to capture the inner content and wrap it in single quotes for ast.literal_eval.
        temp_str = re.sub(r"Decimal\(([^\)]+)\)", r"'\1'", temp_str)

        logger.debug(f"String after Decimal replacement for ast.literal_eval: {temp_str}")

        try:
            # Ensure that any remaining non-string literals that should be strings are quoted
            # e.g. if a label itself was %Y-%m without quotes, ast.literal_eval would fail.
            # This is complex to do safely with regex for arbitrary structures.
            # The primary goal here is to handle the Decimal values correctly.
            
            evaluated_data = ast.literal_eval(temp_str)
            if isinstance(evaluated_data, list):
                for item in evaluated_data:
                    if isinstance(item, tuple) and len(item) >= 2:
                        label = str(item[0])
                        value_candidate = item[1]
                        try:
                            # The value might already be a number if original was not Decimal
                            # or it's a string like '5000.00' after replacement.
                            value = float(value_candidate)
                        except (ValueError, TypeError, InvalidOperation):
                            logger.warning(f"AST path: Could not convert value '{value_candidate}' to float for label '{label}'. Using 0.0.")
                            value = 0.0
                        parsed_results.append({"label": label, "value": value})
                    elif isinstance(item, dict) and 'label' in item and 'value' in item: # If it's already dict
                        parsed_results.append(item) # Assume already correctly typed

                if parsed_results:
                    logger.info(f"Successfully parsed tuple string using ast.literal_eval after Decimal replacement into {len(parsed_results)} items.")
                    return parsed_results
        except Exception as e_ast:
            logger.warning(f"ast.literal_eval failed after Decimal replacement: {e_ast}. Original: '{result_str}', Processed: '{temp_str}'. Falling back to regex.")
            parsed_results = [] # Reset for regex attempt

        # Attempt 2: Regex-based parsing as a more robust fallback
        # This regex tries to capture ('label_string', value_part)
        # value_part can be Decimal('number_string'), Decimal(number_literal), or just a number_literal
        # The previous regex was good, let's ensure it can handle labels like '%Y-%m'
        tuple_pattern = re.compile(
            r"\\(\\s*"                                      # Start of tuple
            r"(['\"])((?:\\\\\\2|.)*?)\\2\\s*,\\s*"          # Group 2&3: Label string (e.g., '%Y-%m', handles escaped quotes within)
            r"(?:"                                      # Non-capturing group for value alternatives
            r"Decimal\\((['\"])((?:\\\\\\4|.)*?)\\4\\)"       # Alt 1: Decimal('number_string') - Group 5 is number_string
            r"|"
            r"Decimal\\(([^\'\")][^\\)]*)\\)"           # Alt 2: Decimal(number_literal_no_quotes) - Group 6 is number_literal
            r"|"
            r"(['\"]?)((?:[+-]?\\d*\\.?\\d+)|(?:(?:\\\\\\7|.)*?))\\7?" # Alt 3: A direct number or a general string, possibly quoted - Group 8 is value, Group 7 is quote
            r")\\s*\\)"                                 # Closing parenthesis of the tuple
        )

        matches = tuple_pattern.findall(result_str) # Use original result_str for regex, not temp_str
        logger.debug(f"Regex matches found: {len(matches)}")

        for i, match in enumerate(matches):
            logger.debug(f"Regex match {i}: {match}")
            label = match[2]  # Content of the first quoted string (label)
            
            # Extract value based on which alternative matched
            val_decimal_quoted = match[4]
            val_decimal_unquoted = match[5]
            val_direct = match[7] # This is group 8 from regex, adjusted for 0-indexed `match` tuple
            
            final_value_str = None
            if val_decimal_quoted:
                final_value_str = val_decimal_quoted
            elif val_decimal_unquoted:
                final_value_str = val_decimal_unquoted
            elif val_direct:
                final_value_str = val_direct

            if label is not None and final_value_str is not None:
                try:
                    value = float(final_value_str)
                    parsed_results.append({"label": str(label), "value": value})
                except (ValueError, TypeError, InvalidOperation) as e:
                    logger.warning(f"Regex parsing: Could not convert value '{final_value_str}' to float for label '{label}'. Error: {e}. Using 0.0.")
                    parsed_results.append({"label": str(label), "value": 0.0})
            else:
                logger.warning(f"Regex parsing: Skipped a segment, label or value_str was None. Label: {label}, ValStr: {final_value_str}, Match: {match}")


        if parsed_results:
            logger.info(f"Successfully parsed tuple string using regex into {len(parsed_results)} items.")
            return parsed_results
        else:
            logger.error(f"Failed to parse tuple string with all methods: {result_str}")
            # To aid chart_agent, return an empty list rather than error dict if truly unparseable to prevent downstream errors
            return [] # Ensure it's an empty list

    def execute_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the SQL query and retrieve results.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state with query results
        """
        # Only proceed if we have a SQL query
        sql_query = state.get('sql_query', '')
        if not sql_query:
            state['results'] = []
            state['error'] = 'No SQL query to execute'
            return state
            
        try:
            # Get the company-isolated database
            db = self.db_manager.get_db(self.company_id)
            
            # Log the query
            logger.info(f"Executing SQL query: {sql_query}")
            
            # Execute the query with company isolation
            result = db.run(sql_query)
            
            if isinstance(result, str):
                logger.warning(f"SQL query returned a string result: {result}")
                # Check if it's the specific problematic string format
                if result.strip().startswith("[(") and result.strip().endswith(")]"):
                    parsed_data = self._parse_string_tuple_result(result)
                    if parsed_data and not (len(parsed_data) == 1 and "error" in parsed_data[0]):
                        logger.info(f"Successfully parsed tuple string into {len(parsed_data)} items using _parse_string_tuple_result.")
                        state['results'] = parsed_data
                        state['columns'] = ['label', 'value'] if parsed_data else []
                        state['data_type'] = 'key_value' if parsed_data else 'unknown'
                        return state
                    else:
                        logger.error(f"Failed to parse string tuple result or error in parsing: {parsed_data}")
                        # Fallback or error state
                        state['results'] = [{"error": "Failed to parse SQL string result", "details": result}]
                        state['columns'] = []
                        state['data_type'] = 'error'
                        state['error'] = "Failed to parse SQL string result."
                        return state
                else: # Generic string result, treat as single value or error
                    logger.warning(f"SQL query returned a generic string, not a parseable tuple list: {result}")
                    state['results'] = [{'result': result}]
                    state['columns'] = ['result']
                    state['data_type'] = 'string_value' # or 'error'
                    # state['error'] = "SQL query returned an unhandled string." # Optional
                    return state
            elif isinstance(result, pd.DataFrame):
                results = result.to_dict(orient='records')
            elif isinstance(result, list):
                results = result
            else:
                # Try to handle other return types
                try:
                    if hasattr(result, '__iter__'):
                        results = list(result)
                    else:
                        results = [{'result': result}]
                except:
                    results = [{'result': str(result)}]
            
            # Process results into visualization-friendly format
            processed_results = self.process_sql_results(results)
            
            # Update state
            state['results'] = processed_results['data']
            state['data_type'] = processed_results['type']
            state['columns'] = processed_results.get('columns', [])
            
            # Remove error if present
            if 'error' in state:
                del state['error']
                
            return state
            
        except Exception as e:
            error_msg = f"SQL execution error: {str(e)}"
            logger.error(error_msg)
            state['error'] = error_msg
            state['results'] = []
            return state
    
    def process_sql_results(self, results: List[Dict]) -> Dict:
        """
        Process SQL results into a format suitable for visualization.
        
        Args:
            results: List of result dictionaries from SQL query
            
        Returns:
            Processed data for visualization
        """
        try:
            # Check if results is None or empty
            if not results:
                logger.warning("No results returned from SQL query")
                return {
                    'data': [],
                    'type': 'empty',
                    'error': 'No data found'
                }
            
            # Handle string result format which might be a tuple representation
            if isinstance(results, str):
                logger.info(f"Converting string result to structured data: {results}")
                # If it looks like a list of tuples: [('IT', 2), ('Audit', 2)]
                if results.startswith("[") and ")]" in results:
                    try:
                        # First try to parse it using literal_eval
                        import ast
                        parsed_tuples = ast.literal_eval(results)
                        
                        # Convert tuples to dictionaries
                        processed_results = []
                        for tup in parsed_tuples:
                            if len(tup) == 2:
                                # Assume first item is label, second is value
                                processed_results.append({
                                    "label": str(tup[0]),
                                    "value": float(tup[1]) if isinstance(tup[1], (int, float)) else 0
                                })
                        
                        return {
                            'data': processed_results,
                            'type': 'key_value',
                            'columns': ['label', 'value']
                        }
                    except (SyntaxError, ValueError) as e:
                        logger.error(f"Failed to parse tuple string: {e}")
                
                # Return an error state
                return {
                    'data': [{"error": results}],
                    'type': 'error',
                    'error': f"Received string result: {results}"
                }
            
            # Convert special types to strings for JSON serialization
            processed_results = []
            
            # Handle if results is a tuple or list of tuples
            if isinstance(results, tuple) or (isinstance(results, list) and len(results) > 0 and isinstance(results[0], tuple)):
                logger.info(f"Processing tuple result: {results}")
                
                # Convert tuples to dictionaries
                for item in results:
                    if isinstance(item, tuple) and len(item) == 2:
                        # Assume first item is label, second is value for visualization
                        processed_results.append({
                            "label": str(item[0]),
                            "value": float(item[1]) if isinstance(item[1], (int, float)) else 0
                        })
            
                return {
                    'data': processed_results,
                    'type': 'key_value',
                    'columns': ['label', 'value']
                }
            
            # Normal dictionary processing
            for row in results:
                processed_row = {}
                
                # Handle if row is not a dictionary (could be a tuple from a direct SQL result)
                if not isinstance(row, dict):
                    # Try to convert a tuple or list to a dict if possible
                    if isinstance(row, (tuple, list)) and len(row) > 0:
                        # If the first item looks like a collection of column names, use those
                        if hasattr(row[0], 'keys') and callable(getattr(row[0], 'keys')):
                            column_names = row[0].keys()
                            processed_row = {column_names[i]: value for i, value in enumerate(row) if i < len(column_names)}
                        else:
                            # Otherwise, just use index as key
                            if len(row) == 2:
                                # Use meaningful names for pairs (likely label/value pairs)
                                processed_row = {"label": row[0], "value": row[1]}
                            else:
                                processed_row = {f"column_{i}": value for i, value in enumerate(row)}
                    else:
                        # Fall back to a simple representation
                        processed_row = {"value": str(row)}
                else:
                    # Normal dictionary processing
                    for key, value in row.items():
                        # Convert dates, decimals, etc. to strings
                        if isinstance(value, (datetime.date, datetime.datetime)):
                            processed_row[key] = value.isoformat()
                        elif isinstance(value, decimal.Decimal):
                            processed_row[key] = float(value)
                        else:
                            processed_row[key] = value
                
                processed_results.append(processed_row)
            
            # Determine the type of data for visualization
            if len(processed_results) == 0:
                data_type = 'empty'
            elif len(processed_results) == 1:
                data_type = 'single'
            elif len(processed_results[0]) == 2 and 'label' in processed_results[0] and 'value' in processed_results[0]:
                # Two columns with label and value is ideal for pie/bar charts
                data_type = 'key_value'
            else:
                data_type = 'table'
            
            return {
                'data': processed_results,
                'type': data_type,
                'columns': list(processed_results[0].keys()) if processed_results else []
            }
        except Exception as e:
            logger.error(f"Error processing SQL results: {str(e)}")
            return {
                'data': [],
                'type': 'error',
                'error': f"Error processing results: {str(e)}"
            } 