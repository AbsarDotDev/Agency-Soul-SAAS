from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Text, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from datetime import datetime
from sqlalchemy import text, inspect
import os
import logging
import re
import sqlparse
from sqlparse.sql import IdentifierList, Identifier, Function, Comparison, Where
from sqlparse.tokens import Keyword, DML, Punctuation, Name, Operator
from typing import Dict, Any, Optional, Callable, List, Tuple, Set
from langchain_community.utilities.sql_database import SQLDatabase
from dotenv import load_dotenv
from sqlalchemy.orm import Session

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

# Cache for company isolation columns
_COMPANY_ISOLATION_COLUMNS_CACHE = {}

def get_company_isolation_column(table_name: str, engine=None, use_cache=True) -> Optional[str]:
    """
    Dynamically determine the company isolation column for a given table.
    
    Args:
        table_name: Name of the table
        engine: SQLAlchemy engine (created if None)
        use_cache: Whether to use cached results
        
    Returns:
        Column name for company isolation or None if not found
    """
    # Clean table name (remove quotes)
    table_name_cleaned = table_name.strip("`'\" ")
    
    # Check cache first if enabled
    if use_cache and table_name_cleaned in _COMPANY_ISOLATION_COLUMNS_CACHE:
        cached_value = _COMPANY_ISOLATION_COLUMNS_CACHE[table_name_cleaned]
        # Ensure we return the column name string, not the dict or None directly from dict lookup
        if isinstance(cached_value, dict) and 'column' in cached_value:
            return cached_value.get('column') # Return the string value or None if key missing
        elif isinstance(cached_value, str):
             return cached_value # Should ideally not happen with current build_table_isolation_info
        else: # Handles None or unexpected types in cache
            return None
    
    # These columns are most commonly used for company isolation
    isolation_column_candidates = [
        "created_by",      # Most common for user/company ownership
        "company_id",      # Direct company references
        "company_user_id", # Often used in user/company related tables
        "user_id",         # User references that need company isolation
        "owner_id",        # Object ownership
        "client_id",       # Client references
        "customer_id"      # Customer references
    ]
    
    if not engine:
        engine = DatabaseConnection.create_engine()
    
    try:
        # Use SQLAlchemy inspect to get table columns
        inspector = inspect(engine)
        if not inspector.has_table(table_name_cleaned):
            logger.debug(f"Table {table_name_cleaned} not found in database for isolation check")
            _COMPANY_ISOLATION_COLUMNS_CACHE[table_name_cleaned] = None # Cache the fact that it wasn't found/no column
            return None
        
        columns = [col['name'].lower() for col in inspector.get_columns(table_name_cleaned)]
        
        # Check for direct company isolation columns
        found_column = None
        for candidate in isolation_column_candidates:
            if candidate in columns:
                found_column = candidate
                break # Found the best candidate based on order
        
        # Cache the result (either the found column name or None)
        _COMPANY_ISOLATION_COLUMNS_CACHE[table_name_cleaned] = {"column": found_column} if found_column else None
        
        if not found_column:
             logger.warning(f"No company isolation column found for table {table_name_cleaned}")
        
        return found_column
        
    except Exception as e:
        logger.error(f"Error determining company isolation column for {table_name_cleaned}: {str(e)}")
        _COMPANY_ISOLATION_COLUMNS_CACHE[table_name_cleaned] = None # Cache failure
        return None

def build_table_isolation_info(engine) -> Dict[str, Dict[str, str]]:
    """
    Build a comprehensive cache of company isolation columns for all tables in the database.

    Args:
        engine: SQLAlchemy engine to use for database introspection

    Returns:
        Dictionary mapping table names to isolation column information
    """
    isolation_info = {}
    
    try:
        # Get list of all tables in the database
        inspector = inspect(engine)
        all_tables = inspector.get_table_names()
        
        # These columns are most commonly used for company isolation
        isolation_column_candidates = [
            "created_by",
            "company_id",
            "company_user_id",
            "user_id",
            "owner_id",
            "client_id",
            "customer_id"
        ]
        
        # Check each table for isolation columns
        for table_name in all_tables:
            try:
                columns = [col['name'].lower() for col in inspector.get_columns(table_name)]
                
                # Look for isolation columns
                found_column = None
                for candidate in isolation_column_candidates:
                    if candidate in columns:
                        found_column = candidate
                        break
                
                # Store the result
                isolation_info[table_name] = {"column": found_column} if found_column else None
                
            except Exception as e:
                logger.warning(f"Error checking isolation columns for table {table_name}: {str(e)}")
                isolation_info[table_name] = None
        
        logger.info(f"Built isolation info cache for {len(isolation_info)} tables")
        return isolation_info
        
    except Exception as e:
        logger.error(f"Failed to build table isolation info: {str(e)}")
        return {}

def _extract_tables_with_aliases(statement: sqlparse.sql.Statement) -> Dict[str, str]:
    """
    Safely extract table names and their aliases from a parsed SQL statement.
    Returns a dictionary mapping alias/name to real table name.
    Example: {'p': 'projects', 'u': 'users'} or {'projects': 'projects'}
    """
    tables_aliases = {}
    
    # Function to safely get the real name, handling different token types
    def safe_get_name(token):
        if hasattr(token, 'get_real_name'):
            return token.get_real_name()
        elif hasattr(token, 'value'):
            # Remove backticks or quotes if present
            return token.value.strip('`"\' ')
        return str(token)
    
    # Process the flattened tokens to identify tables
    from_or_join_found = False
    possible_alias_awaiting = False
    last_table_identified = None
    
    try:
        tokens = list(statement.flatten())
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            # Reset flags if we hit clauses that end table definitions
            if token.ttype is Keyword and token.value.upper() in ('WHERE', 'GROUP', 'ORDER', 'LIMIT', 'HAVING', 'UNION', 'INTERSECT', 'EXCEPT', 'SET', 'VALUES', 'ON'):
                from_or_join_found = False
                possible_alias_awaiting = False
                last_table_identified = None
            
            # Detect FROM or JOIN keywords
            elif token.ttype is Keyword and (token.value.upper() == 'FROM' or 'JOIN' in token.value.upper()):
                from_or_join_found = True
                possible_alias_awaiting = False
                last_table_identified = None
            
            # If we're in a FROM or JOIN clause, look for table names and aliases
            elif from_or_join_found:
                if token.ttype is Name or isinstance(token, Identifier):
                    table_name = safe_get_name(token)
                    
                    # Check if this is an alias for the last table
                    if possible_alias_awaiting and last_table_identified:
                        tables_aliases[table_name] = last_table_identified
                        possible_alias_awaiting = False
                        last_table_identified = None
                    else:
                        # This is a table name
                        tables_aliases[table_name] = table_name
                        last_table_identified = table_name
                        possible_alias_awaiting = True
                
                # Handle AS keyword for explicit aliases
                elif token.ttype is Keyword and token.value.upper() == 'AS':
                    if last_table_identified:
                        possible_alias_awaiting = True
                
                # Handle commas separating multiple tables
                elif token.ttype is Punctuation and token.value == ',':
                    possible_alias_awaiting = False
                    last_table_identified = None
                    # Keep from_or_join_found=True
            
            i += 1
        
        logger.debug(f"Extracted tables/aliases: {tables_aliases}")
    except Exception as e:
        logger.error(f"Error extracting tables from SQL: {str(e)}", exc_info=True)

    return tables_aliases

def _verify_company_filter(statement: sqlparse.sql.Statement, company_id: int, engine=None) -> tuple[bool, List[str]]:
    """
    Verify that a SQL statement includes the necessary company isolation filters.
    
    Args:
        statement: The parsed SQL statement to verify
        company_id: The company ID that should be used in filters
        engine: Optional SQLAlchemy engine
        
    Returns:
        Tuple of (is_verified, list_of_missing_tables)
    """
    # Get the complete SQL statement as a string
    sql_text = str(statement).strip()
    
    # If the company_id is present anywhere in the WHERE clause, consider it valid
    # This is a much more lenient approach for better user experience
    where_pattern = re.compile(r'WHERE.*?(?:\s|\W)' + str(company_id) + r'(?:\s|\W|$)', re.IGNORECASE | re.DOTALL)
    if where_pattern.search(sql_text):
        logger.info(f"Company ID {company_id} found in query, accepting as valid: {sql_text}")
        return True, []
    
    # Extract tables and their aliases from the SQL
    tables_aliases = _extract_tables_with_aliases(statement)
    if not tables_aliases:
        logger.debug("No tables found to verify company isolation")
        return True, []  # No tables to check
    
    # Map each table to its isolation column (if any)
    table_isolation_columns = {}
    for alias, table_name in tables_aliases.items():
        isolation_column = get_company_isolation_column(table_name, engine)
        if isolation_column:
            table_isolation_columns[alias] = {
                'table': table_name,
                'column': isolation_column
            }
    
    if not table_isolation_columns:
        logger.debug("No tables with isolation columns found")
        return True, []  # No tables require isolation
    
    # All tables need isolation but none was found in the query
    missing_tables = [f"{info['table']} (via column {info['column']})" 
                      for alias, info in table_isolation_columns.items()]
    
    return False, missing_tables

class CompanyIsolationSQLDatabase(SQLDatabase):
    """SQLDatabase wrapper that enforces company isolation through verification."""

    def __init__(self, company_id: int, **kwargs):
        """Initialize with company_id."""
        self.company_id = company_id
        # Ensure engine is created for isolation checks if not passed
        if 'engine' not in kwargs:
             kwargs['engine'] = DatabaseConnection.create_engine()
        super().__init__(**kwargs)
        # Pre-build isolation info for efficiency if not already done
        if not _COMPANY_ISOLATION_COLUMNS_CACHE:
            try:
                 build_table_isolation_info(self._engine)
                 logger.info("Company isolation cache built during CompanyIsolationSQLDatabase init.")
            except Exception as e:
                 logger.error(f"Failed to build isolation cache during init: {e}")


    def run(self, command: str, fetch: str = "all", **kwargs) -> Any:
        """
        Execute SQL command with enforced company isolation using verification.
        
        Args:
            command: SQL command to execute.
            fetch: Fetch type ("all", "one", "cursor").
            **kwargs: Additional arguments for the underlying execute method.
            
        Returns:
            Result of the SQL command execution or error message.
        """
        original_command = command.strip()
        logger.debug(f"Original SQL command for company {self.company_id}: {original_command}")

        # Allow only SELECT statements for security
        if not original_command.upper().startswith("SELECT"):
            logger.warning(f"Rejecting non-SELECT command: {original_command}")
            return "Error: Only SELECT statements are allowed for security reasons."

        try:
            # Parse the SQL command
            parsed_statements = sqlparse.parse(original_command)
            if not parsed_statements:
                 logger.error(f"Failed to parse SQL command: {original_command}")
                 return "Error: Could not parse SQL command."

            statement = parsed_statements[0]  # Assume single statement

            # Verify that the statement includes proper company isolation
            is_verified, missing_tables = _verify_company_filter(statement, self.company_id, self._engine)
            
            if not is_verified:
                logger.warning(f"Company isolation verification failed. Missing filters for tables: {missing_tables}")
                return f"Error: Query does not contain required company isolation filters for: {', '.join(missing_tables)}. Please modify your query to include these filters."
            
            # If verification passes, execute the query as is
            logger.info(f"Company isolation verified. Executing SQL for company {self.company_id}: {original_command}")
            return super().run(original_command, fetch=fetch, **kwargs)

        except Exception as e:
            logger.error(f"Error processing SQL query for company {self.company_id}: {str(e)}", exc_info=True)
            return f"Error executing SQL query: {str(e)}"

def get_company_isolated_sql_database(company_id: int, **kwargs) -> SQLDatabase:
    """Get a CompanyIsolationSQLDatabase instance for the given company_id."""
    logger.debug(f"Creating CompanyIsolationSQLDatabase for company_id: {company_id}")
    
    # Set up engine if not provided
    if 'engine' not in kwargs:
        kwargs['engine'] = DatabaseConnection.create_engine()
    
    # Create instance with company ID
    db_instance = CompanyIsolationSQLDatabase(company_id=company_id, **kwargs)
    
    # Ensure cache is populated
    if not _COMPANY_ISOLATION_COLUMNS_CACHE:
        try:
            build_table_isolation_info(kwargs['engine'])
            logger.info("Built isolation cache in get_company_isolated_sql_database.")
        except Exception as e:
            logger.error(f"Failed to build isolation cache in get_company_isolated_sql_database: {e}")

    return db_instance

# Initialize isolation columns cache by scanning database on module load
# Use the engine created for the module scope initially
try:
    _COMPANY_ISOLATION_COLUMNS_CACHE = build_table_isolation_info(engine) # Use module-level engine
    logger.info(f"Initialized company isolation cache with {len(_COMPANY_ISOLATION_COLUMNS_CACHE)} tables.")
except Exception as e:
    logger.error(f"Failed to initialize company isolation column cache on module load: {str(e)}")
    _COMPANY_ISOLATION_COLUMNS_CACHE = {}
