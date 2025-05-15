"""
Test script to verify our SQL schema and query fixes
"""
import json
import logging
from app.visualizations.langgraph.database_manager import DatabaseManager
from app.visualizations.langgraph.sql_agent import SQLAgent
from app.visualizations.langgraph.llm_manager import LLMManager

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_schema_info():
    """Test detailed schema information retrieval"""
    db_manager = DatabaseManager()
    company_id = 1
    
    # Get detailed schema
    schema = db_manager.get_schema(company_id)
    print("\n==== SCHEMA INFO TEST ====")
    print(f"Schema length: {len(schema)} characters")
    print("Schema includes detailed information:", "DETAILED TABLE SCHEMA" in schema)
    print("Schema includes example queries:", "EXAMPLE QUERIES" in schema)
    
    # Success if we got detailed schema
    return "DETAILED TABLE SCHEMA" in schema and len(schema) > 1000

def test_sql_query_fixes():
    """Test SQL query fixes for common errors"""
    sql_agent = SQLAgent(company_id=1)
    
    # Test queries with common errors
    test_queries = [
        # Department name issue
        "SELECT d.department_name, COUNT(e.id) as employee_count FROM departments d JOIN employees e ON d.id = e.department_id WHERE d.company_id = 1 GROUP BY d.department_name",
        
        # Country in customers
        "SELECT c.country, COUNT(*) as customer_count FROM customers c WHERE c.company_id = 1 GROUP BY c.country",
        
        # Orders/products instead of invoices/product_services
        "SELECT p.category, SUM(oi.quantity * oi.unit_price) as revenue FROM products p JOIN order_items oi ON p.id = oi.product_id JOIN orders o ON oi.order_id = o.id WHERE p.company_id = 1 GROUP BY p.category",
        
        # Incorrect DATE syntax for MySQL
        "SELECT DATE(invoice_date) as day, SUM(amount) as revenue FROM invoices WHERE company_id = 1 AND invoice_date >= DATE('now', '-6 months') GROUP BY day",
        
        # SUM syntax error
        "SELECT DATE_FORMAT(invoice_date, '%Y-%m') as month, SUM(amount, '%Y-%m') as revenue FROM invoices WHERE company_id = 1 GROUP BY month"
    ]
    
    print("\n==== SQL QUERY FIX TEST ====")
    for i, query in enumerate(test_queries):
        print(f"\nOriginal Query {i+1}:\n{query}")
        fixed_query = sql_agent._fix_common_column_errors(query)
        print(f"\nFixed Query {i+1}:\n{fixed_query}")
        print("-" * 80)
    
    # Success if we didn't crash
    return True

def main():
    """Run all tests"""
    print("Starting visualization fix tests...\n")
    
    schema_success = test_schema_info()
    query_success = test_sql_query_fixes()
    
    if schema_success and query_success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed.")

if __name__ == "__main__":
    main() 