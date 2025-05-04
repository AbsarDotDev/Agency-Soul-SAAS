# Dynamic Company Data Isolation Solution

## Overview

This document explains the implementation of dynamic company data isolation in the AI Agent system. The system now automatically detects and applies appropriate company isolation filters to SQL queries without requiring manual table mapping configuration.

## Problem

The previous approach required maintaining a manual mapping of tables to their company isolation columns:

```python
COMPANY_ISOLATION_MAPPING = {
    "users": {"column": "created_by"},
    "employees": {"column": "created_by"},
    # ... 200+ other tables
}
```

This approach was:
1. Error-prone with ~200 tables in the database
2. Difficult to maintain as schema evolved
3. Not adaptable to new tables without code changes
4. Missed tables not listed in the mapping

## Solution

The new solution takes a dynamic approach:

1. **Automatic Column Detection**: The system automatically inspects table schemas to find company isolation columns
2. **Schema Caching**: Found columns are cached for performance
3. **Dynamic Query Modification**: SQL queries are modified at runtime to include appropriate company filters
4. **Database Integration**: MCP MySQL server integration provides schema information 
5. **Agent Enhancement**: SQL agent is enhanced with schema awareness tools

## Key Components

### 1. Dynamic Company Isolation Column Detection

```python
def get_company_isolation_column(table_name: str, engine=None, use_cache=True) -> Optional[str]:
    """Dynamically determine the company isolation column for a given table."""
    # Common isolation column names to check
    isolation_column_candidates = [
        "created_by",      # Most common for user/company ownership
        "company_id",      # Direct company references
        "user_id",         # User references that need company isolation
        "owner_id",        # Object ownership
        "client_id",       # Client references
        "customer_id"      # Customer references
    ]
    
    # Check table schema for these columns
    # First candidates found is used
```

### 2. Schema Caching

The system initializes a cache of company isolation columns on startup:

```python
# Initialize isolation columns cache by scanning database on module load
try:
    _COMPANY_ISOLATION_COLUMNS_CACHE = build_table_isolation_info()
    logger.info(f"Initialized company isolation cache with {len(_COMPANY_ISOLATION_COLUMNS_CACHE)} tables")
except Exception as e:
    logger.error(f"Failed to initialize company isolation column cache: {str(e)}")
    _COMPANY_ISOLATION_COLUMNS_CACHE = {}
```

### 3. SQL Query Modification

SQL queries are dynamically modified to include company isolation filters:

```python
def _add_company_filter_to_statement(statement: sqlparse.sql.Statement, company_id: int):
    """Add company filter to SQL statement to ensure data isolation."""
    # Extract tables and their aliases
    # For each table, get its isolation column
    # Add WHERE clauses with the appropriate company ID filters
```

### 4. MCP MySQL Integration

The MCPMySQLTool provides direct database inspection capabilities:

```python
class MCPMySQLTool:
    """Tool for interacting with the MCP MySQL server."""
    
    @staticmethod
    async def list_tables(company_id: int) -> List[str]:
        """List all tables in the database."""
        
    @staticmethod
    async def get_table_schema(table_name: str, company_id: int) -> List[Dict[str, Any]]:
        """Get schema information for a table."""
        
    @staticmethod
    async def execute_query(query: str, company_id: int) -> Dict[str, Any]:
        """Execute a SQL query with company isolation."""
```

## Usage Example

The SQL agent now uses this infrastructure automatically. When a user sends a query:

1. The system dynamically determines which tables are involved
2. It identifies the appropriate company isolation columns
3. It modifies the query to include company isolation filters
4. The query executes with proper data isolation

Example:

```
User query: "Show me all employees"

Original SQL: SELECT * FROM employees

Modified SQL: SELECT * FROM employees WHERE employees.created_by = 12345
```

## Testing

You can test the dynamic company isolation functionality using the provided test script:

```
python ai_agent/app/api/test_mcp_integration.py
```

This script demonstrates:
1. Table listing via MCP
2. Schema retrieval
3. Company isolation column detection
4. Isolation mapping generation
5. Query execution with company isolation
6. SQL agent usage with company isolation
7. Visualization generation

## Benefits

1. **Maintainability**: No need to manually update a mapping as schema changes
2. **Scalability**: Automatically works with new tables 
3. **Reliability**: Reduces risk of data leakage between companies
4. **Flexibility**: Adapts to different column naming conventions
5. **Performance**: Uses caching for efficient operation

## Conclusion

The dynamic company isolation approach solves the maintenance challenge of the previous static mapping. By automatically detecting and applying appropriate company filters, the system ensures proper data isolation while remaining flexible and easy to maintain. 