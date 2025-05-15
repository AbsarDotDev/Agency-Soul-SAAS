from typing import List, Dict, Any, Optional, Tuple
import logging
import pandas as pd
import json
import sqlalchemy as sa

from app.database.connection import get_company_isolated_sql_database
from app.database.connection import _verify_company_filter

# Set up logging
logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Manager for database interactions, ensuring company data isolation.
    Handles schema retrieval, query execution, and data formatting.
    """
    
    def __init__(self):
        """Initialize the database manager."""
        self.schema_cache = {}  # Cache for database schemas
        self.table_columns_cache = {}  # Cache for table columns
    
    def get_schema(self, company_id: int, use_cache: bool = True) -> str:
        """
        Get the database schema with appropriate company isolation.
        
        Args:
            company_id: Company ID for data isolation
            use_cache: Whether to use cached schema if available
            
        Returns:
            String representation of the database schema
        """
        cache_key = f"schema_{company_id}"
        
        # Return cached schema if available and requested
        if use_cache and cache_key in self.schema_cache:
            return self.schema_cache[cache_key]
        
        try:
            # Get isolated database for this company
            sql_database = get_company_isolated_sql_database(
                company_id=company_id,
                sample_rows_in_table_info=3
            )
            
            # Get the schema from langchain
            schema = sql_database.get_table_info()
            
            # Add detailed schema information
            schema += "\n\n-- DETAILED TABLE SCHEMA --\n"
            schema += self._get_detailed_schema_info(company_id)
            
            # Append table column information for common joins
            schema += "\n\n-- COMMON TABLE COLUMNS AND RELATIONSHIPS --\n"
            schema += self._get_key_tables_info(company_id)
            
            # Append example queries
            schema += "\n\n-- EXAMPLE QUERIES --\n"
            schema += self._get_example_queries(company_id)
            
            # Cache the schema
            self.schema_cache[cache_key] = schema
            
            return schema
        except Exception as e:
            logger.error(f"Error retrieving database schema for company {company_id}: {str(e)}")
            raise Exception(f"Error retrieving database schema: {str(e)}")
    
    def _get_detailed_schema_info(self, company_id: int) -> str:
        """
        Get detailed schema information using SQLAlchemy metadata.
        
        Args:
            company_id: Company ID
            
        Returns:
            String with detailed schema information
        """
        try:
            # Get isolated database for this company
            sql_database = get_company_isolated_sql_database(company_id)
            
            # Get detailed schema info
            schema_info = sql_database.get_detailed_schema_info()
            
            # Format the schema info as a string
            result = []
            
            # Focus on key tables first
            key_tables = [
                'departments', 'employees', 'users', 'customers', 'projects',
                'invoices', 'invoice_products', 'product_services', 'product_service_categories'
            ]
            
            # Process key tables first
            for table_name in key_tables:
                if table_name in schema_info:
                    table_info = schema_info[table_name]
                    result.append(f"Table: {table_name}")
                    
                    # Add table isolation column info
                    isolation_col = self._get_isolation_column_for_table(table_name)
                    if isolation_col:
                        result.append(f"  Isolation Column: {isolation_col}")
                        
                    # Add column info
                    result.append("  Columns:")
                    for col in table_info:
                        col_str = f"    - {col['name']} ({col['type']})"
                        if col.get('primary_key'):
                            col_str += " [PRIMARY KEY]"
                        if 'foreign_key' in col:
                            fk = col['foreign_key']
                            col_str += f" [REFERENCES {fk['references_table']}.{fk['references_column']}]"
                        result.append(col_str)
                    
                    # Add empty line for readability
                    result.append("")
            
            return "\n".join(result)
            
        except Exception as e:
            logger.error(f"Error getting detailed schema info: {str(e)}")
            return "Error retrieving detailed schema info."
    
    def _get_key_tables_info(self, company_id: int) -> str:
        """
        Get information about key tables, their columns and relationships.
        
        Args:
            company_id: Company ID 
            
        Returns:
            String with key tables information
        """
        try:
            # Get isolated database for this company
            sql_database = get_company_isolated_sql_database(company_id)
            
            # Get core tables metadata
            engine = sql_database._engine
            metadata = sa.MetaData()
            metadata.reflect(bind=engine, only=['employees', 'departments', 'users', 'projects', 'customers'])
            
            info = []
            for table_name in metadata.tables:
                table = metadata.tables[table_name]
                columns_info = ", ".join([f"{col.name} ({col.type})" for col in table.columns])
                info.append(f"Table '{table_name}' columns: {columns_info}")
                
                # Add primary key info
                if table.primary_key:
                    pk_cols = ", ".join([col.name for col in table.primary_key])
                    info.append(f"Table '{table_name}' primary key: {pk_cols}")
                
                # Add isolation column info
                isolation_col = self._get_isolation_column_for_table(table_name)
                if isolation_col:
                    info.append(f"Table '{table_name}' company isolation column: {isolation_col}")
                
                info.append("")  # Empty line for readability
                
            return "\n".join(info)
            
        except Exception as e:
            logger.error(f"Error getting key tables info: {str(e)}")
            return "Error retrieving key tables info."
    
    def _get_example_queries(self, company_id: int) -> str:
        """
        Get example queries for common tables.
        
        Args:
            company_id: Company ID
            
        Returns:
            String with example queries
        """
        return f"""
-- Example employee count by department query
SELECT d.name as department_name, COUNT(e.id) as employee_count 
FROM departments d 
JOIN employees e ON d.id = e.department_id 
WHERE d.created_by = {company_id} 
GROUP BY d.name 
ORDER BY employee_count DESC;

-- Example project count by user/employee query
SELECT u.name as employee_name, COUNT(p.id) as project_count 
FROM users u 
JOIN project_users pu ON u.id = pu.user_id 
JOIN projects p ON pu.project_id = p.id 
WHERE p.created_by = {company_id} 
GROUP BY u.name 
ORDER BY project_count DESC;

-- Example revenue by month query (using MySQL date functions)
SELECT DATE_FORMAT(created_at, '%Y-%m') as month, SUM(amount) as total_revenue 
FROM invoices 
WHERE created_by = {company_id} 
GROUP BY month 
ORDER BY month;

-- Example customer count by country query
SELECT billing_country as country, COUNT(id) as customer_count 
FROM customers 
WHERE created_by = {company_id} 
GROUP BY billing_country 
ORDER BY customer_count DESC;

-- Example sales performance over the last 6 months
SELECT DATE_FORMAT(invoice_date, '%Y-%m') as month, SUM(total_amount) as total_sales 
FROM invoices 
WHERE created_by = {company_id} 
AND invoice_date >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH) 
GROUP BY month 
ORDER BY month;

-- Example revenue by product category
SELECT psc.name as category_name, SUM(ip.quantity * ip.price) as total_revenue 
FROM product_service_categories psc 
JOIN product_services ps ON psc.id = ps.category_id 
JOIN invoice_products ip ON ps.id = ip.product_id 
JOIN invoices i ON ip.invoice_id = i.id 
WHERE psc.created_by = {company_id} 
AND i.created_by = {company_id} 
GROUP BY psc.name 
ORDER BY total_revenue DESC;
"""
    
    def get_table_sample(self, company_id: int, table_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get a sample of rows from a specific table with company isolation.
        
        Args:
            company_id: Company ID for data isolation
            table_name: Name of the table to sample
            limit: Maximum number of rows to return
            
        Returns:
            List of dictionary rows from the table
        """
        try:
            # Get isolated database for this company
            sql_database = get_company_isolated_sql_database(
                company_id=company_id
            )
            
            # Create safe query with company isolation
            isolation_column = self._get_isolation_column_for_table(table_name)
            
            if isolation_column:
                query = f"SELECT * FROM `{table_name}` WHERE `{isolation_column}` = {company_id} LIMIT {limit}"
            else:
                # If no isolation column, log warning and add safeguards
                logger.warning(f"No isolation column found for table {table_name} - using limited query")
                query = f"SELECT * FROM `{table_name}` LIMIT {limit}"
            
            # Verify the query has proper company isolation
            is_safe, _ = _verify_company_filter(query, company_id)
            
            if not is_safe:
                logger.warning(f"Query lacks proper company isolation: {query}")
                return []
            
            # Execute the query
            result = sql_database.run(query)
            
            # If we have results, add table structure information
            if result and isinstance(result, list):
                # Get table column information
                engine = sql_database._engine
                metadata = sa.MetaData()
                metadata.reflect(bind=engine, only=[table_name])
                
                # If we have table metadata, add it to the first result as a special field
                if table_name in metadata.tables:
                    column_info = {
                        "table_name": table_name,
                        "columns": [
                            {"name": col.name, "type": str(col.type)}
                            for col in metadata.tables[table_name].columns
                        ],
                        "isolation_column": isolation_column,
                        "example_where_clause": f"WHERE `{isolation_column}` = {company_id}" if isolation_column else ""
                    }
                    
                    # Add table structure information as first item
                    info_item = {"__table_info__": column_info}
                    result.insert(0, info_item)
            
            return result if isinstance(result, list) else []
            
        except Exception as e:
            logger.error(f"Error getting sample from table {table_name}: {str(e)}")
            return []
    
    def get_unique_values(self, company_id: int, table_name: str, column_name: str, 
                         limit: int = 100) -> List[str]:
        """
        Get unique values from a specific column with company isolation.
        
        Args:
            company_id: Company ID for data isolation
            table_name: Name of the table
            column_name: Name of the column
            limit: Maximum number of unique values to return
            
        Returns:
            List of unique values from the column
        """
        try:
            # Get isolated database for this company
            sql_database = get_company_isolated_sql_database(
                company_id=company_id
            )
            
            # Create safe query with company isolation
            isolation_column = self._get_isolation_column_for_table(table_name)
            
            if isolation_column:
                query = f"""
                SELECT DISTINCT `{column_name}` 
                FROM `{table_name}` 
                WHERE `{isolation_column}` = {company_id}
                AND `{column_name}` IS NOT NULL
                AND `{column_name}` != ''
                LIMIT {limit}
                """
            else:
                # If no isolation column, log warning and add safeguards
                logger.warning(f"No isolation column found for table {table_name} - using limited query")
                query = f"""
                SELECT DISTINCT `{column_name}` 
                FROM `{table_name}`
                WHERE `{column_name}` IS NOT NULL
                AND `{column_name}` != ''
                LIMIT {limit}
                """
            
            # Verify the query has proper company isolation
            is_safe, _ = _verify_company_filter(query, company_id)
            
            if not is_safe:
                logger.warning(f"Query lacks proper company isolation: {query}")
                return []
            
            # Execute the query
            result = sql_database.run(query)
            
            if isinstance(result, list):
                # Extract the values from the result
                values = [str(row[column_name]) for row in result if column_name in row]
                return values
            else:
                return []
            
        except Exception as e:
            logger.error(f"Error getting unique values from {table_name}.{column_name}: {str(e)}")
            return []
    
    def execute_query(self, company_id: int, query: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Execute a SQL query with company isolation verification.
        
        Args:
            company_id: Company ID for data isolation
            query: SQL query to execute
            
        Returns:
            Tuple of (results, error_message)
        """
        try:
            # Get isolated database for this company
            sql_database = get_company_isolated_sql_database(
                company_id=company_id
            )
            
            # Verify the query has proper company isolation
            is_safe, missing_tables = _verify_company_filter(query, company_id)
            
            if not is_safe:
                error_msg = f"Query lacks proper company isolation for tables: {', '.join(missing_tables)}"
                logger.warning(error_msg)
                return [], error_msg
            
            # Execute the query
            result = sql_database.run(query)
            
            # Handle different result types
            if isinstance(result, str):
                # Handle error response or string representations
                if "Error" in result:
                    return [], f"SQL error: {result}"
                
                # Handle tuple representations like "[('IT', 2), ('Audit', 2)]"
                if result.startswith("[") and ")" in result and "," in result:
                    try:
                        # Try to convert to structured data
                        import ast
                        parsed_result = ast.literal_eval(result)
                        
                        # Convert tuples to dictionaries if possible
                        if all(isinstance(item, tuple) and len(item) >= 2 for item in parsed_result):
                            dict_results = []
                            for tup in parsed_result:
                                if len(tup) == 2:
                                    dict_results.append({
                                        "label": str(tup[0]),
                                        "value": tup[1]
                                    })
                                else:
                                    # Create a dictionary from tuple items
                                    dict_result = {}
                                    for i, val in enumerate(tup):
                                        dict_result[f"column_{i}"] = val
                                    dict_results.append(dict_result)
                            return dict_results, None
                    except (SyntaxError, ValueError) as e:
                        logger.error(f"Error parsing tuple string: {str(e)}")
                
                try:
                    # Try to parse as JSON
                    parsed_result = json.loads(result)
                    return parsed_result if isinstance(parsed_result, list) else [], None
                except:
                    return [], f"Invalid result format: {result}"
            
            # Handle tuple or list of tuples
            if isinstance(result, tuple) or (isinstance(result, list) and len(result) > 0 and isinstance(result[0], tuple)):
                logger.info(f"Converting tuple results to dictionaries: {result}")
                dict_results = []
                
                # Convert tuples to dictionaries
                for item in result:
                    if isinstance(item, tuple):
                        if len(item) == 2:
                            # For 2-item tuples, use label/value format for visualization
                            dict_results.append({
                                "label": str(item[0]),
                                "value": item[1]
                            })
                        else:
                            # Create a dictionary from tuple items
                            dict_result = {}
                            for i, val in enumerate(item):
                                dict_result[f"column_{i}"] = val
                            dict_results.append(dict_result)
                    else:
                        # If not a tuple, just add as is
                        dict_results.append({"value": item})
                
                return dict_results, None
            
            # Return the result if it's already a list
            return result if isinstance(result, list) else [], None
            
        except Exception as e:
            error_msg = f"Error executing query: {str(e)}"
            logger.error(error_msg)
            return [], error_msg
    
    def _get_isolation_column_for_table(self, table_name: str) -> Optional[str]:
        """
        Get the company isolation column for a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Name of the isolation column or None if not found
        """
        from app.database.connection import get_company_isolation_column
        
        return get_company_isolation_column(table_name)

    def get_db(self, company_id: int) -> Any:
        """
        Get the SQLDatabase object for the given company.
        
        Args:
            company_id: Company ID for data isolation
            
        Returns:
            LangChain SQLDatabase object
        """
        try:
            # Get an isolated SQLDatabase for this company
            # This gets a SQLDatabase that enforces company isolation in SQL queries
            sql_database = get_company_isolated_sql_database(
                company_id=company_id,
                sample_rows_in_table_info=3
            )
            
            # Return the SQLDatabase instance
            return sql_database
        except Exception as e:
            logger.error(f"Error getting SQLDatabase for company {company_id}: {str(e)}")
            raise Exception(f"Error getting database connection: {str(e)}") 