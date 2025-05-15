from typing import Dict, Any, List, Optional
import logging
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.agents.specialized_agent_base import SpecializedAgentBase
from app.agents.base_agent import AgentResponse, VisualizationResult, ActionResult
from app.database.connection import DatabaseConnection

# Set up logging
logger = logging.getLogger(__name__)

class ProjectManagementAgent(SpecializedAgentBase):
    """Project management agent for handling project-related queries using database access."""
    
    def __init__(self):
        """Initialize Project Management agent with relevant tables."""
        # Check which tables actually exist in the database
        existing_tables = self._get_existing_tables()
        logger.info(f"Found {len(existing_tables)} existing tables in database")
        
        # Define potential tables relevant to Project Management
        potential_project_tables = [
            "projects", # For project information
            "project_users", # For project team members
            "project_tasks", # For tasks in projects 
            "project_expenses", # For project expenses
            "tasks", # For general tasks
            "task_stages", # For task workflow stages
            "task_checklists", # For task checklists
            "task_comments", # For task comments
            "task_files", # For task files
            "time_trackers", # For time tracking on projects/tasks
            "milestones", # For project milestones
            "project_reports", # For project reports
            "projectstages", # For project workflow stages
            "project_invoices", # For project invoices
            "project_email_templates", # For project email templates
            "invoices", # For general invoices that might relate to projects
            "users", # For user information related to projects
            "labels", # For project/task labeling
            "notes", # For project notes
            "comments", # For comments on tasks
            "checklists", # For task checklists
            "checklist_items", # For individual checklist items
            "estimates", # For project estimates
            "bugs", # For bug tracking
            "bug_comments", # For comments on bugs
            "bug_files", # For bug attachments
            "bug_statuses" # For bug status tracking
        ]
        
        # Filter to only include tables that actually exist in the database
        project_tables = [table for table in potential_project_tables if table in existing_tables]
        
        # Log the tables being used
        logger.info(f"Project Management agent using these tables that exist in database: {', '.join(project_tables)}")
        
        # Initialize base class with Project Management-specific settings
        super().__init__(
            agent_type="project",
            relevant_tables=project_tables,
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
        """Generate Project Management-specific system prompt.
        
        Args:
            company_id: Company ID
            isolation_instructions: Isolation instructions based on table columns
            
        Returns:
            System prompt
        """
        return f"""You are a friendly, helpful Project Management Assistant working with a company's database.
You help users query data about projects, tasks, team members, etc. in natural, conversational language.

DATA ISOLATION CRITICAL RULE:
ALWAYS include WHERE created_by = {company_id} in ALL your queries for ALL tables involved.
This is required for security, no exceptions.

{isolation_instructions}

PROJECT MANAGEMENT DATABASE KNOWLEDGE:
- projects: Contains project information with fields like name, description, start_date, end_date, status, etc.
- project_users: Maps users to projects with fields like project_id, user_id, role, etc.
- project_tasks: Tasks within projects with fields like name, description, due_date, status, assignee_id, etc.
- tasks: General tasks with fields like name, description, due_date, status, assignee_id, etc.
- task_stages: Workflow stages for tasks with fields like name, order, etc.
- time_trackers: Time tracking entries with fields like project_id, task_id, user_id, start_time, end_time, etc.
- milestones: Project milestones with fields like project_id, name, due_date, status, etc.
- project_expenses: Expenses related to projects with fields like project_id, amount, date, category, etc.

JOIN RELATIONSHIPS:
- projects.id connects to project_users.project_id
- projects.id connects to project_tasks.project_id
- projects.id connects to milestones.project_id
- projects.id connects to project_expenses.project_id
- project_users.user_id connects to users.id
- project_tasks.task_id connects to tasks.id
- tasks.stage_id connects to task_stages.id
- tasks.assignee_id connects to users.id

PROJECT METRICS KNOWLEDGE:
- Project Progress: Percentage of completed tasks in a project
- On-Time Completion Rate: Percentage of tasks completed before or on deadline
- Budget Variance: Planned budget vs. actual expenses
- Resource Utilization: Tracked time compared to allocated time
- Milestone Completion Rate: Percentage of milestones reached on schedule

IMPORTANT GUIDELINES:
1. NEVER make up information. Only return data that actually exists in the database.
2. If you don't find relevant data, clearly say so rather than making up a response.
3. Present your answers in a clear, concise way for business users.
4. NEVER reveal SQL queries to the end user - only show the information they asked for.
5. Format your responses in a readable way, with proper capitalization and punctuation.
6. When appropriate, format project data in tables for clarity.
7. For questions about project progress, task distribution, or resource allocation,
   consider if a visualization might help the user understand the data better.

USE DATABASE TOOLS ALWAYS:
- When asked about projects, tasks, milestones, etc., ALWAYS search the database for real data.
- Never fabricate information or say "I would need to search the database" - actually search it.
- Use SQL queries to get actual data for every information request.
- If data doesn't exist, clearly state that no matching records were found.

Today's date is {self._get_current_date()}
""" 