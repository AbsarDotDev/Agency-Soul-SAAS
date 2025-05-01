from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Text, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from datetime import datetime
from sqlalchemy import text
import os
import logging
import re
import sqlparse
from typing import Dict, Any, Optional, Callable, List, Tuple
from langchain_community.utilities.sql_database import SQLDatabase

from app.core.config import settings

# Set up logging
logger = logging.getLogger(__name__)

# Create SQLAlchemy engine
engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create scoped session to ensure thread safety
db_session = scoped_session(SessionLocal)

# Create base class for models
Base = declarative_base()

def get_db_session():
    """Get database session.
    
    Yields:
        Database session
        
    Note:
        This function is meant to be used as a dependency in FastAPI endpoints.
    """
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def get_company_scoped_query(session, model, company_id):
    """Get query scoped to a specific company.
    
    Args:
        session: Database session
        model: SQLAlchemy model
        company_id: Company ID to scope the query
        
    Returns:
        Query object scoped to the specified company
        
    Note:
        This function ensures company data isolation by filtering all queries
        with the company_id.
    """
    if hasattr(model, 'company_id'):
        return session.query(model).filter(model.company_id == company_id)
    else:
        raise ValueError(f"Model {model.__name__} does not have company_id column")

class DatabaseConnection:
    """Database connection utilities for MySQL."""
    
    @staticmethod
    def get_connection_string() -> str:
        """Get database connection string from environment variables."""
        db_host = os.getenv("DB_HOST", "localhost")
        db_port = os.getenv("DB_PORT", "3306")
        db_name = os.getenv("DB_DATABASE", "")
        db_user = os.getenv("DB_USERNAME", "")
        db_password = os.getenv("DB_PASSWORD", "")
        
        return f"mysql+mysqlconnector://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    
    @staticmethod
    def create_engine():
        """Create SQLAlchemy engine for database connection."""
        try:
            connection_string = DatabaseConnection.get_connection_string()
            engine = create_engine(connection_string)
            return engine
        except Exception as e:
            logger.error(f"Failed to create database engine: {str(e)}")
            raise
    
    @staticmethod
    def create_session_factory():
        """Create SQLAlchemy session factory."""
        try:
            engine = DatabaseConnection.create_engine()
            Session = sessionmaker(bind=engine)
            return Session
        except Exception as e:
            logger.error(f"Failed to create session factory: {str(e)}")
            raise
    
    @staticmethod
    def get_sql_database(include_tables=None, exclude_tables=None, sample_rows_in_table_info=3) -> SQLDatabase:
        """Create a LangChain SQLDatabase instance for use with SQL toolkit.
        
        Args:
            include_tables: Optional list of tables to include
            exclude_tables: Optional list of tables to exclude
            sample_rows_in_table_info: Number of sample rows to include in table info
            
        Returns:
            SQLDatabase instance
        """
        try:
            engine = DatabaseConnection.create_engine()
            
            # Default tables to exclude (system tables, etc.)
            default_exclude = ["failed_jobs", "migrations", "password_resets", "personal_access_tokens"]
            
            if exclude_tables:
                exclude_tables = list(set(exclude_tables + default_exclude))
            else:
                exclude_tables = default_exclude
            
            return SQLDatabase(
                engine=engine,
                include_tables=include_tables,
                exclude_tables=exclude_tables,
                sample_rows_in_table_info=sample_rows_in_table_info,
            )
        except Exception as e:
            logger.error(f"Failed to create SQLDatabase: {str(e)}")
            raise

# Mapping of tables to their company identifier columns
COMPANY_ISOLATION_MAPPING = {
    # Standard company identifier mappings
    "users": {"column": "created_by", "alias": ["u", "user", "users"]},
    "employees": {"column": "created_by", "alias": ["e", "employee", "employees"]},
    "departments": {"column": "created_by", "alias": ["d", "dept", "department", "departments"]},
    "branches": {"column": "created_by", "alias": ["b", "branch", "branches"]},
    "customers": {"column": "created_by", "alias": ["c", "customer", "customers"]},
    "invoices": {"column": "created_by", "alias": ["i", "invoice", "invoices"]},
    "products": {"column": "created_by", "alias": ["p", "product", "products"]},
    "orders": {"column": "created_by", "alias": ["o", "order", "orders"]},
    # Add more tables as needed
}

def get_company_scoped_query(sql_query: str, company_id: int) -> str:
    """Modify SQL query to ensure it's scoped to the given company ID.
    
    This function adds WHERE clauses to ensure data isolation between companies.
    It analyzes SQL query structure and adds appropriate company_id or created_by filters.
    
    Args:
        sql_query: Original SQL query
        company_id: Company ID to scope the query to
        
    Returns:
        Modified SQL query with company scope
    """
    try:
        # Parse the SQL query using sqlparse
        parsed = sqlparse.parse(sql_query)
        if not parsed:
            logger.warning(f"Failed to parse SQL query: {sql_query}")
            return _add_company_id_comment(sql_query, company_id)
        
        # Get the first statement (we only handle one statement at a time)
        stmt = parsed[0]
        
        # Check if this is a SELECT query
        if stmt.get_type() != 'SELECT':
            logger.warning(f"Non-SELECT query detected, not applying company isolation: {sql_query}")
            return _add_company_id_comment(sql_query, company_id)
        
        # Extract table names from the query
        tables = _extract_tables_from_query(sql_query)
        if not tables:
            logger.warning(f"No tables found in query, not applying company isolation: {sql_query}")
            return _add_company_id_comment(sql_query, company_id)
        
        # Modify the query to add company isolation
        modified_query = _add_company_filters(sql_query, tables, company_id)
        
        return modified_query
    
    except Exception as e:
        logger.error(f"Error applying company isolation to query: {str(e)}")
        # If any error occurs, return the original query with a warning comment
        return _add_company_id_comment(sql_query, company_id)

def _add_company_id_comment(sql_query: str, company_id: int) -> str:
    """Add a comment with company ID to the SQL query.
    
    Args:
        sql_query: Original SQL query
        company_id: Company ID
        
    Returns:
        SQL query with company ID comment
    """
    return f"""
    /* Company ID: {company_id} */
    /* WARNING: Company data isolation could not be automatically applied to this query */
    {sql_query}
    """

def _extract_tables_from_query(sql_query: str) -> List[Tuple[str, str]]:
    """Extract table names and their aliases from a SQL query.
    
    Args:
        sql_query: SQL query
        
    Returns:
        List of tuples (table_name, alias)
    """
    # Simple regex to find tables in FROM and JOIN clauses
    # This is a basic implementation and might not catch all cases
    tables = []
    
    # Look for tables in FROM clause
    from_regex = r"FROM\s+([a-zA-Z0-9_]+)(?:\s+(?:as\s+)?([a-zA-Z0-9_]+))?"
    for match in re.finditer(from_regex, sql_query, re.IGNORECASE):
        table = match.group(1)
        alias = match.group(2) if match.group(2) else table
        tables.append((table, alias))
    
    # Look for tables in JOIN clauses
    join_regex = r"JOIN\s+([a-zA-Z0-9_]+)(?:\s+(?:as\s+)?([a-zA-Z0-9_]+))?"
    for match in re.finditer(join_regex, sql_query, re.IGNORECASE):
        table = match.group(1)
        alias = match.group(2) if match.group(2) else table
        tables.append((table, alias))
    
    return tables

def _add_company_filters(sql_query: str, tables: List[Tuple[str, str]], company_id: int) -> str:
    """Add company filters to a SQL query.
    
    Args:
        sql_query: Original SQL query
        tables: List of tuples (table_name, alias)
        company_id: Company ID to scope the query to
        
    Returns:
        Modified SQL query with company filters
    """
    # Check if this query already has a WHERE clause
    has_where = re.search(r"\bWHERE\b", sql_query, re.IGNORECASE) is not None
    # Check if this query already has a GROUP BY clause
    has_group_by = re.search(r"\bGROUP\s+BY\b", sql_query, re.IGNORECASE) is not None
    # Check if this query already has a ORDER BY clause
    has_order_by = re.search(r"\bORDER\s+BY\b", sql_query, re.IGNORECASE) is not None
    # Check if this query already has a LIMIT clause
    has_limit = re.search(r"\bLIMIT\b", sql_query, re.IGNORECASE) is not None

    where_conditions = []
    
    # Iterate over tables found in the query
    for table, alias in tables:
        table_lower = table.lower()
        if table_lower in COMPANY_ISOLATION_MAPPING:
            isolation_col = COMPANY_ISOLATION_MAPPING[table_lower]["column"]
            # Use alias if available, otherwise table name
            table_ref = alias if alias != table else table
            where_conditions.append(f"{table_ref}.{isolation_col} = {company_id}")
    
    # Combine conditions with AND
    if not where_conditions:
        # No isolatable tables found, return original query with comment
        return _add_company_id_comment(sql_query, company_id)
    
    company_filter = " AND ".join(where_conditions)

    # Find the position to insert the WHERE/AND clause
    insert_pos = len(sql_query)
    if has_limit:
        insert_pos = min(insert_pos, re.search(r"\bLIMIT\b", sql_query, re.IGNORECASE).start())
    if has_order_by:
        insert_pos = min(insert_pos, re.search(r"\bORDER\s+BY\b", sql_query, re.IGNORECASE).start())
    if has_group_by:
        insert_pos = min(insert_pos, re.search(r"\bGROUP\s+BY\b", sql_query, re.IGNORECASE).start())
    if has_where:
        insert_pos = min(insert_pos, re.search(r"\bWHERE\b", sql_query, re.IGNORECASE).end())
    
    # Build the modified query
    if has_where:
        # Insert conditions after existing WHERE clause
        modified_query = sql_query[:insert_pos] + f" AND {company_filter} " + sql_query[insert_pos:]
    else:
        # Insert new WHERE clause before GROUP BY/ORDER BY/LIMIT
        modified_query = sql_query[:insert_pos] + f" WHERE {company_filter} " + sql_query[insert_pos:]
        
    return f"/* Company ID: {company_id} */\n{modified_query}"

class CompanyIsolationSQLDatabase(SQLDatabase):
    """ Custom SQLDatabase class to automatically apply company isolation."""
    def __init__(self, company_id: int, **kwargs):
        self.company_id = company_id
        # Filter out unsupported arguments before calling super
        supported_args = ['engine', 'schema', 'metadata', 'ignore_tables', 'include_tables', 
                          'sample_rows_in_table_info', 'indexes_in_table_info', 
                          'custom_table_info', 'view_support', 'max_string_length']
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in supported_args}
        super().__init__(**filtered_kwargs)

    def run(self, command: str, fetch: str = "all", **kwargs) -> Any:
        """ Execute SQL query after applying company isolation."""
        modified_command = get_company_scoped_query(command, self.company_id)
        logger.debug(f"Executing modified SQL: {modified_command}")
        # Execute the command using the parent class method
        try:
            # Pass through any extra kwargs received (like 'parameters')
            return super().run(modified_command, fetch=fetch, **kwargs) 
        except Exception as e:
            # Log the modified command along with the error
            logger.error(f"Error running modified SQL query: {modified_command}\nError: {e}")
            # Re-raise the exception to be handled by the agent
            raise

def get_company_isolated_sql_database(company_id: int, **kwargs) -> SQLDatabase:
    """ Get a company-isolated SQLDatabase instance."""
    try:
        engine = DatabaseConnection.create_engine()
        
        # Default tables to exclude (system tables, etc.)
        default_exclude = ["failed_jobs", "migrations", "password_resets", "personal_access_tokens"]
        
        # Combine provided exclude_tables with defaults
        exclude_tables = list(set((kwargs.get('exclude_tables', []) or []) + default_exclude))
        kwargs['exclude_tables'] = exclude_tables # Update kwargs
        
        # Remove exclude_tables from kwargs before passing to CompanyIsolationSQLDatabase
        # as it doesn't directly support it in its __init__ signature
        if 'exclude_tables' in kwargs:
             del kwargs['exclude_tables'] # This was likely causing the error

        return CompanyIsolationSQLDatabase(
            engine=engine,
            company_id=company_id,
            # Pass other supported kwargs like include_tables, sample_rows_in_table_info
            **kwargs
        )
    except Exception as e:
        logger.error(f"Failed to create CompanyIsolatedSQLDatabase: {str(e)}")
        raise
