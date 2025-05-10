from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
import logging
import uuid
import json
import re
from fastapi import HTTPException
# Import message types from langchain directly
from langchain_core.messages import SystemMessage, HumanMessage
from sqlalchemy import text

from app.agents.base_agent import BaseAgent, AgentResponse, VisualizationResult, ActionResult
from app.database.queries import DatabaseQueries
from app.database.connection import get_company_isolated_sql_database, DatabaseConnection
from app.core.llm import get_llm
from app.agents.specialized_agent_base import SpecializedAgentBase

# Set up logging
logger = logging.getLogger(__name__)


class FinanceAgent(SpecializedAgentBase):
    """Finance agent for handling financial and sales data queries and analysis using database access."""
    
    def __init__(self):
        """Initialize Finance agent with relevant tables."""
        # Check which tables actually exist in the database
        existing_tables = self._get_existing_tables()
        logger.info(f"Found {len(existing_tables)} existing tables in database")
        
        # Define potential tables relevant to Finance and Sales
        potential_finance_tables = [
            # Finance-related tables
            "bills",
            "bill",
            "bill_payments",
            "bill_products",
            "budget",
            "budgets",
            "chart_of_accounts",
            "chart_of_account_types",
            "credit_notes",
            "debit_notes",
            "expenses",
            "expense_types",
            "financial_years",
            "journal_entries",
            "journal_items",
            "payments",
            "payment_methods",
            "revenues",
            "transactions",
            "taxes"
        ]
        
        # Sales-related tables
        potential_sales_tables = [
            "customers",
            "deals",
            "invoices", 
            "invoice_products",
            "invoice_payments", 
            "leads",
            "pipelines",
            "stages",
            "sources",
            "proposals", 
            "product_services",
            "commissions",
            "quotations",
            "orders",
            "coupons",
            "plans",
            "pos",
            "pos_products",
            "contracts",
            "contract_types",
            "estimations"
        ]
        
        # Filter to only include tables that actually exist in the database
        finance_tables = [table for table in potential_finance_tables if table in existing_tables]
        sales_tables = [table for table in potential_sales_tables if table in existing_tables]
        
        # Combined list of tables
        all_tables = finance_tables + sales_tables
        
        # Log the tables being used
        logger.info(f"Finance agent using these tables that exist in database: {', '.join(all_tables)}")
        
        # Initialize base class with Finance-specific settings
        super().__init__(
            agent_type="finance",
            relevant_tables=all_tables,
            fallback_to_sql=True
        )
    
    def _get_existing_tables(self) -> List[str]:
        """Get a list of tables that actually exist in the database."""
        try:
            engine = DatabaseConnection.create_engine()
            with engine.connect() as connection:
                # Get all tables in the database
                result = connection.execute(text("SHOW TABLES"))
                existing_tables = [row[0] for row in result]
                return existing_tables
        except Exception as e:
            logger.error(f"Error checking existing tables: {str(e)}")
            return []
    
    def _generate_system_prompt(self, company_id: int, isolation_instructions: str) -> str:
        """Generate Finance-specific system prompt.
        
        Args:
            company_id: Company ID
            isolation_instructions: Isolation instructions based on table columns
            
        Returns:
            System prompt
        """
        return f"""You are a friendly, helpful Finance & Sales Assistant working with a company's database.
You help users query data about finances, sales, revenue, expenses, budgets, etc. in natural, conversational language.

DATA ISOLATION CRITICAL RULE:
ALWAYS include WHERE created_by = {company_id} in ALL your queries for ALL tables involved.
This is required for security, no exceptions.

{isolation_instructions}

FINANCE DATABASE KNOWLEDGE:
- expenses: Contains expense records with fields like date, amount, category, description, etc.
- revenues: Contains revenue records with fields like date, amount, source, description, etc.
- budget: Contains budget records with fields like period, category, amount, etc.
- chart_of_accounts: Contains account classifications with fields like name, code, type, etc.
- chart_of_account_types: Contains account types with fields like name, etc.
- bill: Contains vendor bills with fields like vendor_id, date, status, etc.
- payments: Contains payment records with fields like date, amount, method, etc.
- journal_entries: Contains accounting entries with fields like date, reference, description, etc.
- journal_items: Contains journal entry line items with fields like journal_id, account_id, debit, credit, etc.

SALES DATABASE KNOWLEDGE:
- customers: Contains customer information with fields like name, email, phone, company, etc.
- deals: Sales opportunities with fields like name, customer_id, amount, status, etc.
- invoices: Customer invoices with fields like customer_id, amount, date, status, etc.
- invoice_products: Line items on invoices with fields like invoice_id, product_id, quantity, price, etc.
- invoice_payments: Payments against invoices with fields like invoice_id, amount, date, etc.
- leads: Prospective customers with fields like name, email, phone, source, status, etc.
- product_services: Products and services with fields like name, price, description, etc.

JOIN RELATIONSHIPS (FINANCE):
- bill.vendor_id connects to venders.id
- bill_products.bill_id connects to bill.id
- bill_products.product_id connects to product_services.id
- journal_items.journal_id connects to journal_entries.id
- journal_items.account_id connects to chart_of_accounts.id
- chart_of_accounts.type connects to chart_of_account_types.id

JOIN RELATIONSHIPS (SALES):
- deals.customer_id connects to customers.id
- deals.stage_id connects to stages.id
- stages.pipeline_id connects to pipelines.id
- leads.stage_id connects to lead_stages.id
- invoices.customer_id connects to customers.id
- invoice_products.invoice_id connects to invoices.id
- invoice_products.product_id connects to product_services.id
- invoice_payments.invoice_id connects to invoices.id

ID RESOLUTION - ALWAYS FOLLOW THIS RULE:
When your query returns IDs (like customer_id, vendor_id, product_id, etc.), ALWAYS perform additional queries to resolve these IDs into human-readable names.
Example:
1. If your first query returns a customer_id = 5, run a second query: "SELECT name FROM customers WHERE id = 5 AND created_by = {company_id}"
2. Then replace "customer_id: 5" with "Customer: [actual customer name]" in your response
3. Do this for ALL foreign key IDs in your results to make the information user-friendly
4. For status codes, convert them to descriptive text (e.g., "Status: 1" becomes "Status: Active" if you know the mapping)

FINANCE AND SALES METRICS KNOWLEDGE:
- Revenue Growth: Percentage change in revenue over time periods
- Profit Margin: Net profit divided by revenue * 100
- Cash Flow: Net cash movement over a time period
- Operating Expenses: Total expenses related to core business operations
- Budget Variance: Difference between budgeted and actual amounts
- Total Revenue: Sum of invoice amounts
- Average Deal Size: Average value of closed deals
- Conversion Rate: Percentage of leads converted to customers
- Win Rate: Percentage of deals closed won vs. total closed deals
- Revenue by Product: Revenue breakdown by product/service

IMPORTANT GUIDELINES:
1. NEVER make up information. Only return data that actually exists in the database.
2. If you don't find relevant data, clearly say so rather than making up a response.
3. Present your answers in a clear, concise way for business users.
4. NEVER reveal SQL queries to the end user - only show the information they asked for.
5. Format your responses in a readable way, with proper capitalization and punctuation.
6. When appropriate, format financial data in tables for clarity.
7. Always include currency symbols ($ by default) for monetary values.
8. For financial reports or trends, consider if a visualization might help the user understand the data better.
9. ALWAYS RESOLVE IDs to actual names when presenting data. For example, show customer names instead of customer_ids.

USE DATABASE TOOLS ALWAYS:
- When asked about finances, sales, customers, invoices, etc., ALWAYS search the database for real data.
- Never fabricate information or say "I would need to search the database" - actually search it.
- Use SQL queries to get actual data for every information request.
- If data doesn't exist, clearly state that no matching records were found.

Today's date is {self._get_current_date()}
"""

    def _clean_technical_references(self, text: str) -> str:
        """Clean technical references from response text."""
        # List of phrases that indicate technical language we want to remove
        technical_phrases = [
            "I would need to query",
            "After querying",
            "I can query",
            "I need to access",
            "in the database",
            "from the database",
            "database query",
            "SQL query",
            "to calculate",
            "I would need to check",
            "I would have to look",
            "I'll check",
            "I need to search",
            "will require accessing",
            "I'll need to query",
            "I would need to access",
            "I will need to retrieve",
            "revenues table",
            "expenses table",
            "transactions table",
            "bills table",
            "invoices table",
            "chart_of_accounts table"
        ]
        
        # Remove sentences containing these phrases
        for phrase in technical_phrases:
            # Find all sentences containing the phrase - using case-insensitive matching
            pattern = r'(?i)[^.!?]*' + re.escape(phrase) + r'[^.!?]*[.!?]'
            text = re.sub(pattern, '', text)
        
        # Clean up double spaces and extra periods
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\.+', '.', text)
        text = re.sub(r'\s+\.', '.', text)
        text = re.sub(r'\.\s+\.', '.', text)
        
        # If the text starts with conjunctions like "Based on" or "According to", remove them
        text = re.sub(r'(?i)^(Based on|According to|From|As per).*?,\s*', '', text)
        
        # If after all this cleaning we've lost content, generate a generic response
        if len(text.strip()) < 20:
            text = "I don't have enough information to provide specific financial data at this time."
        
        return text.strip() 