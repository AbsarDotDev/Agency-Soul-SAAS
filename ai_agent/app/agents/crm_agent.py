from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
import logging
import uuid
import json
import re
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


class CRMAgent(SpecializedAgentBase):
    """CRM agent for handling customer relationship management queries using database access."""
    
    def __init__(self):
        """Initialize CRM agent with relevant tables."""
        # Check which tables actually exist in the database
        existing_tables = self._get_existing_tables()
        
        # Define tables relevant to CRM
        crm_tables = [
            "customers",
            "clients",
            "leads",
            "lead_calls",
            "lead_discussions", 
            "lead_emails",
            "lead_files",
            "lead_activity_logs",
            "user_leads",
            "deals",
            "deal_calls",
            "deal_discussions",
            "deal_emails",
            "deal_files",
            "deal_tasks",
            "client_deals",
            "user_deals",
            "lead_stages",
            "pipelines",
            "stages",
            "supports",
            "bugs",
            "contracts",
            "contract_types",
            "proposals",
            "sources",
            "labels"
        ]
        
        # Filter to tables that actually exist in the database
        crm_tables = [table for table in crm_tables if table in existing_tables]
        
        # Initialize base class with CRM-specific settings
        super().__init__(
            agent_type="crm",
            relevant_tables=crm_tables,
            fallback_to_sql=True
        )
    
    def _get_existing_tables(self) -> List[str]:
        """Get list of tables that actually exist in the database."""
        engine = DatabaseConnection.create_engine()
        with engine.connect() as conn:
            try:
                result = conn.execute(text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = DATABASE()
                    AND table_type = 'BASE TABLE'
                """))
                return [row[0] for row in result]
            except Exception as e:
                logger.error(f"Error getting tables: {str(e)}")
                return []
    
    def _generate_system_prompt(self, company_id: int, isolation_instructions: str) -> str:
        """Generate CRM-specific system prompt.
        
        Args:
            company_id: Company ID
            isolation_instructions: Isolation instructions based on table columns
            
        Returns:
            System prompt
        """
        return f"""You are a friendly, helpful CRM Assistant working with a company's database.
You help users query data about customers, leads, deals, etc. in natural, conversational language.

DATA ISOLATION CRITICAL RULE:
ALWAYS include WHERE created_by = {company_id} in ALL your queries for ALL tables involved.
This is required for security, no exceptions.

{isolation_instructions}

CRM DATABASE KNOWLEDGE:
- customers: Contains customer information with fields like name, email, phone, company, etc.
- clients: Often used interchangeably with customers
- leads: Prospective customers with fields like name, email, phone, source, status, etc.
- lead_calls: Call records related to leads with fields like lead_id, subject, call_type, duration, etc.
- lead_discussions: Discussion comments on leads with fields like lead_id, comment, created_by, etc.
- lead_emails: Email correspondence related to leads with fields like lead_id, to, subject, description, etc.
- lead_files: Files attached to leads with fields like lead_id, file_name, file_path, etc.
- lead_activity_logs: Activity logs for leads with fields like lead_id, user_id, log_type, remark, etc.
- user_leads: Junction table connecting users to leads
- deals: Sales opportunities with fields like name, phone, price, pipeline_id, stage_id, status, etc.
- deal_calls: Call records related to deals with fields like deal_id, subject, call_type, duration, etc.
- deal_discussions: Discussion comments on deals with fields like deal_id, comment, created_by, etc.
- deal_emails: Email correspondence related to deals with fields like deal_id, to, subject, description, etc.
- deal_files: Files attached to deals with fields like deal_id, file_name, file_path, etc.
- deal_tasks: Tasks related to deals with fields like deal_id, name, date, time, priority, status, etc.
- client_deals: Junction table connecting clients to deals
- user_deals: Junction table connecting users to deals
- lead_stages: Stages in the lead nurturing process with fields like name, order, etc.
- pipelines: Sales pipelines with fields like name, etc.
- stages: Stages in the sales process with fields like name, pipeline_id, order, etc.
- supports/bugs: Support tickets with fields like customer_id, subject, status, priority, etc.
- contracts: Customer contracts with fields like customer_id, type_id, value, start_date, end_date, etc.
- proposals: Sales proposals with fields like customer_id, amount, status, etc.
- invoices: Customer invoices with fields like customer_id, amount, status, etc.

JOIN RELATIONSHIPS:
- deals.customer_id connects to customers.id
- deals.stage_id connects to stages.id
- deal_calls.deal_id connects to deals.id
- deal_discussions.deal_id connects to deals.id
- deal_emails.deal_id connects to deals.id
- deal_files.deal_id connects to deals.id
- deal_tasks.deal_id connects to deals.id
- client_deals connects clients.id to deals.id
- user_deals connects users.id to deals.id
- lead_calls.lead_id connects to leads.id
- lead_discussions.lead_id connects to leads.id
- lead_emails.lead_id connects to leads.id
- lead_files.lead_id connects to leads.id
- lead_activity_logs.lead_id connects to leads.id
- user_leads connects users.id to leads.id
- stages.pipeline_id connects to pipelines.id
- leads.stage_id connects to lead_stages.id
- supports.customer_id connects to customers.id
- proposals.customer_id connects to customers.id
- contracts.customer_id connects to customers.id
- contracts.type_id connects to contract_types.id
- invoices.customer_id connects to customers.id

CRM METRICS KNOWLEDGE:
- Lead Conversion Rate: Percentage of leads converted to customers
- Deal Close Rate: Percentage of deals won
- Average Deal Size: Average amount of closed deals
- Customer Lifetime Value: Average total revenue from a customer
- Customer Acquisition Cost: Average cost to acquire a new customer
- Sales Cycle Length: Average time from lead to closed deal
- Customer Retention Rate: Percentage of customers retained year over year

IMPORTANT GUIDELINES:
1. NEVER make up information. Only return data that actually exists in the database.
2. If you don't find relevant data, clearly say so rather than making up a response.
3. Present your answers in a clear, concise way for business users.
4. NEVER reveal SQL queries to the end user - only show the information they asked for.
5. Format your responses in a readable way, with proper capitalization and punctuation.
6. When appropriate, format customer data in tables for clarity.
7. For questions about lead counts, sales funnel, deal stages, or customer metrics,
   consider if a visualization might help the user understand the data better.
8. ALWAYS RESOLVE IDs to actual names when presenting data. For example, show customer names instead of customer_ids.

USE DATABASE TOOLS ALWAYS:
- When asked about customers, leads, deals, or any CRM data, ALWAYS search the database for real data.
- Never fabricate information or say "I would need to search the database" - actually search it.
- Use SQL queries to get actual data for every information request.
- If data doesn't exist, clearly state that no matching records were found.

Today's date is {self._get_current_date()}
"""
    
    async def perform_action(
        self, 
        action: str, 
        parameters: Dict[str, Any], 
        company_id: int, 
        user_id: str,
        session: Optional[Session] = None
    ) -> ActionResult:
        """Perform CRM-related action.
        
        Args:
            action: Action to perform
            parameters: Action parameters
            company_id: Company ID
            user_id: User ID
            session: Optional database session
            
        Returns:
            Action result
        """
        # Only supporting read-only actions for now
        return ActionResult(
            success=False,
            message="CRM action support is coming soon. Currently, only read-only operations are supported."
        )
    
    def _clean_technical_references(self, text: str) -> str:
        """Clean technical references from text.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        # Remove SQL query sections
        text = re.sub(r"```sql\s*.*?\s*```", "", text, flags=re.DOTALL)
        
        # Remove other code blocks
        text = re.sub(r"```\s*.*?\s*```", "", text, flags=re.DOTALL)
        
        # Remove inline SQL
        text = re.sub(r"`SELECT.*?;`", "", text, flags=re.DOTALL)
        
        # Remove technical phrases
        technical_phrases = [
            "query the database",
            "execute a query",
            "run a query",
            "SQL query",
            "database query",
            "query results",
            "database results",
            "database shows",
            "according to the database",
            "from the database",
            "database records",
            "queried the"
        ]
        
        for phrase in technical_phrases:
            text = text.replace(phrase, "")
        
        # Consolidate multiple newlines and spaces
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"\s{2,}", " ", text)
        
        return text.strip() 