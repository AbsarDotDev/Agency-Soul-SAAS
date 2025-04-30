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
    
    # Build the company filters
    company_filters = []
    for table, alias in tables:
        if table.lower() in COMPANY_ISOLATION_MAPPING:
            col = COMPANY_ISOLATION_MAPPING[table.lower()]["column"]
            company_filters.append(f"{alias}.{col} = {company_id}")
    
    if not company_filters:
        # If no applicable tables found for filtering, return with a warning comment
        return _add_company_id_comment(sql_query, company_id)
    
    # Join the company filters with AND
    company_filter_str = " AND ".join(company_filters)
    
    # Add the company filter to the query
    if has_where:
        # Add to existing WHERE clause
        modified_query = re.sub(
            r"WHERE",
            f"WHERE ({company_filter_str}) AND ",
            sql_query, 
            flags=re.IGNORECASE,
            count=1
        )
    else:
        # Check if there's a GROUP BY, ORDER BY, or LIMIT clause
        order_by_match = re.search(r"\bORDER\s+BY\b", sql_query, re.IGNORECASE)
        group_by_match = re.search(r"\bGROUP\s+BY\b", sql_query, re.IGNORECASE)
        limit_match = re.search(r"\bLIMIT\b", sql_query, re.IGNORECASE)
        
        # Find the position to insert the WHERE clause
        if group_by_match:
            position = group_by_match.start()
            modified_query = f"{sql_query[:position]} WHERE {company_filter_str} {sql_query[position:]}"
        elif order_by_match:
            position = order_by_match.start()
            modified_query = f"{sql_query[:position]} WHERE {company_filter_str} {sql_query[position:]}"
        elif limit_match:
            position = limit_match.start()
            modified_query = f"{sql_query[:position]} WHERE {company_filter_str} {sql_query[position:]}"
        else:
            # Add at the end of the query
            modified_query = f"{sql_query} WHERE {company_filter_str}"
    
    # Add a comment to indicate company isolation was applied
    modified_query = f"""
    /* Company ID: {company_id} */
    /* Company isolation applied automatically */
    {modified_query}
    """
    
    return modified_query

class CompanyIsolationSQLDatabase(SQLDatabase):
    """Extended SQLDatabase with company isolation built in."""
    
    def __init__(self, company_id: int, **kwargs):
        """Initialize with company ID for data isolation.
        
        Args:
            company_id: Company ID for data isolation
            **kwargs: Additional arguments for SQLDatabase
        """
        super().__init__(**kwargs)
        self.company_id = company_id
    
    def run(self, command: str, fetch: str = "all") -> Any:
        """Run a SQL command with company isolation applied.
        
        Args:
            command: SQL command to run
            fetch: Fetch strategy ('all', 'one', 'many', etc.)
            
        Returns:
            Query result
        """
        # Apply company isolation to the command
        scoped_command = get_company_scoped_query(command, self.company_id)
        logger.info(f"Running company-isolated query: {scoped_command}")
        
        # Use the parent class's run method with the scoped command
        return super().run(scoped_command, fetch)

def get_company_isolated_sql_database(company_id: int, **kwargs) -> SQLDatabase:
    """Create a SQLDatabase instance with company isolation.
    
    Args:
        company_id: Company ID for data isolation
        **kwargs: Additional arguments for SQLDatabase
        
    Returns:
        SQLDatabase instance with company isolation
    """
    # Create base SQLDatabase instance
    engine = DatabaseConnection.create_engine()
    
    # Default tables to exclude (system tables, etc.)
    default_exclude = ["failed_jobs", "migrations", "password_resets", "personal_access_tokens"]
    
    exclude_tables = kwargs.pop("exclude_tables", None)
    if exclude_tables:
        exclude_tables = list(set(exclude_tables + default_exclude))
    else:
        exclude_tables = default_exclude
    
    # Create company-isolated instance
    return CompanyIsolationSQLDatabase(
        company_id=company_id,
        engine=engine,
        exclude_tables=exclude_tables,
        **kwargs
    )
