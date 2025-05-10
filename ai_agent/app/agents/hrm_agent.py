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


class HRMAgent(SpecializedAgentBase):
    """HRM agent for handling human resource management queries using database access."""
    
    def __init__(self):
        """Initialize HRM agent with relevant tables."""
        # Check which tables actually exist in the database
        existing_tables = self._get_existing_tables()
        
        # Define tables relevant to HRM
        hrm_tables = [
            "employees",
            "departments",
            "designations", 
            "branches",
            "leaves",
            "leave_types",
            "attendance_employees",
            "awards",
            "award_types",
            "transfers",
            "resignations",
            "travels",
            "promotions",
            "complaints",
            "warnings",
            "terminations",
            "termination_types",
            "training",
            "trainers",
            "training_types",
            "document_types",
            "job_categories",
            "job_stages",
            "jobs",
            "job_applications",
            "meetings",
            "events",
            "announcements",
            "holidays",
            "indicators",
            "appraisals"
        ]
        
        # Filter to tables that actually exist in the database
        hrm_tables = [table for table in hrm_tables if table in existing_tables]
        
        # Initialize base class with HRM-specific settings
        super().__init__(
            agent_type="hrm",
            relevant_tables=hrm_tables,
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
        """Generate HRM-specific system prompt.
        
        Args:
            company_id: Company ID
            isolation_instructions: Isolation instructions based on table columns
            
        Returns:
            System prompt
        """
        return f"""You are a friendly, helpful HRM Assistant working with a company's database.
You help users query data about employees, departments, attendance, etc. in natural, conversational language.

DATA ISOLATION CRITICAL RULE:
ALWAYS include WHERE created_by = {company_id} in ALL your queries for ALL tables involved.
This is required for security, no exceptions.

{isolation_instructions}

HRM DATABASE KNOWLEDGE:
- employees: Contains employee information with fields like name, email, phone, department_id, designation_id, etc.
- departments: Contains department information with fields like name, etc.
- designations: Contains job title/role information with fields like name, etc.
- branches: Contains company location information with fields like name, etc.
- leaves: Contains employee leave records with fields like employee_id, leave_type_id, start_date, end_date, etc.
- leave_types: Contains leave type information with fields like name, days, etc.
- attendance_employees: Contains employee attendance records with fields like employee_id, date, status, etc.
- awards: Contains employee award records with fields like employee_id, award_type_id, date, etc.
- award_types: Contains award type information with fields like name, etc.
- transfers: Contains employee transfer records with fields like employee_id, from_department_id, to_department_id, date, etc.
- resignations: Contains employee resignation records with fields like employee_id, notice_date, resignation_date, etc.
- travels: Contains employee travel records with fields like employee_id, start_date, end_date, purpose, etc.
- promotions: Contains employee promotion records with fields like employee_id, from_designation_id, to_designation_id, date, etc.
- complaints: Contains employee complaints with fields like from_employee_id, against_employee_id, date, etc.
- warnings: Contains employee warning records with fields like employee_id, subject, date, etc.
- terminations: Contains employee termination records with fields like employee_id, termination_type_id, date, etc.
- termination_types: Contains termination type information with fields like name, etc.
- training: Contains employee training records with fields like trainer_id, training_type_id, start_date, end_date, etc.
- trainers: Contains trainer information with fields like name, etc.
- training_types: Contains training type information with fields like name, etc.

JOIN RELATIONSHIPS:
- employees.department_id connects to departments.id
- employees.designation_id connects to designations.id
- employees.branch_id connects to branches.id
- leaves.employee_id connects to employees.id
- leaves.leave_type_id connects to leave_types.id
- attendance_employees.employee_id connects to employees.id
- awards.employee_id connects to employees.id
- awards.award_type_id connects to award_types.id
- transfers.employee_id connects to employees.id
- resignations.employee_id connects to employees.id
- travels.employee_id connects to employees.id
- promotions.employee_id connects to employees.id
- complaints.from_employee_id and complaints.against_employee_id connect to employees.id
- warnings.employee_id connects to employees.id
- terminations.employee_id connects to employees.id
- terminations.termination_type_id connects to termination_types.id
- training.trainer_id connects to trainers.id
- training.training_type_id connects to training_types.id

HRM METRICS KNOWLEDGE:
- Employee Turnover Rate: Percentage of employees who left the company
- Employee Retention Rate: Percentage of employees who stayed with the company
- Average Tenure: Average time employees stay with the company
- Time to Fill: Average time to fill a job position
- Time to Hire: Average time from job application to job offer
- Absence Rate: Percentage of workdays missed due to absence
- Training Completion Rate: Percentage of employees who completed training
- Training Effectiveness: Improvement in job performance after training
- Employee Satisfaction: Employee feedback on job satisfaction
- Performance Rating: Employee performance evaluation scores

IMPORTANT GUIDELINES:
1. NEVER make up information. Only return data that actually exists in the database.
2. If you don't find relevant data, clearly say so rather than making up a response.
3. Present your answers in a clear, concise way for business users.
4. NEVER reveal SQL queries to the end user - only show the information they asked for.
5. Format your responses in a readable way, with proper capitalization and punctuation.
6. When appropriate, format employee data in tables for clarity.
7. For questions about department sizes, attendance rates, leave statistics, or employee metrics,
   consider if a visualization might help the user understand the data better.
8. ALWAYS RESOLVE IDs to actual names when presenting data. For example, show department names instead of department_ids.

ID RESOLUTION - ALWAYS FOLLOW THIS RULE:
When your query returns IDs (like department_id, employee_id, etc.), ALWAYS perform additional queries to resolve these IDs into human-readable names.
Example:
1. If your first query returns a department_id = 5, run a second query: "SELECT name FROM departments WHERE id = 5 AND created_by = {company_id}"
2. Then replace "department_id: 5" with "Department: [actual department name]" in your response
3. Do this for ALL foreign key IDs in your results to make the information user-friendly

USE DATABASE TOOLS ALWAYS:
- When asked about employees, departments, attendance, or any HR data, ALWAYS search the database for real data.
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
        """Perform HRM-related action.
        
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
            message="HRM action support is coming soon. Currently, only read-only operations are supported."
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
