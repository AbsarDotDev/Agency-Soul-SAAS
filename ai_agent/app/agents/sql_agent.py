from typing import Dict, Any, List, Optional, Union
import logging
import uuid
import json
from datetime import datetime
import re

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_community.utilities import SQLDatabase
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import text
from langchain.tools import Tool

from app.agents.base_agent import BaseAgent, AgentResponse, VisualizationResult, ActionResult
from app.database.connection import DatabaseConnection, get_company_isolated_sql_database, get_company_isolation_column
from app.core.llm import get_llm, get_embedding_model
from app.visualizations.generator import generate_visualization_from_data
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.memory import ConversationBufferMemory
from app.core.token_manager import TokenManager

# Set up logging
logger = logging.getLogger(__name__)

class SQLQueryResult(BaseModel):
    """SQL query result."""
    query: str = Field(description="The SQL query that was executed")
    result: List[Dict[str, Any]] = Field(description="The result of the SQL query")
    explanation: str = Field(description="A natural language explanation of the result")

class MCPMySQLTool:
    """Tool for interacting with the MCP MySQL server."""
    
    @staticmethod
    async def list_tables(company_id: int) -> List[str]:
        """List all tables in the database.
        
        Args:
            company_id: Company ID
            
        Returns:
            List of table names
        """
        engine = DatabaseConnection.create_engine()
        session = Session(engine)
        try:
            result = session.execute(text("""
                SELECT TABLE_NAME 
                FROM information_schema.tables 
                WHERE table_schema = DATABASE()
                AND table_type = 'BASE TABLE'
                ORDER BY TABLE_NAME
            """)).fetchall()
            
            return [row[0] for row in result]
        except Exception as e:
            logger.error(f"Error listing tables: {str(e)}")
            return []
        finally:
            session.close()
    
    @staticmethod
    async def get_table_schema(table_name: str, company_id: int) -> List[Dict[str, Any]]:
        """Get schema information for a table.
        
        Args:
            table_name: Table name
            company_id: Company ID
            
        Returns:
            List of column details
        """
        engine = DatabaseConnection.create_engine()
        session = Session(engine)
        try:
            result = session.execute(text("""
                SELECT COLUMN_NAME, DATA_TYPE, COLUMN_TYPE, IS_NULLABLE, COLUMN_KEY, COLUMN_DEFAULT, EXTRA
                FROM information_schema.columns
                WHERE table_schema = DATABASE()
                AND table_name = :table_name
                ORDER BY ORDINAL_POSITION
            """), {"table_name": table_name}).fetchall()
            
            columns = []
            for row in result:
                columns.append({
                    "name": row[0],
                    "data_type": row[1],
                    "column_type": row[2],
                    "is_nullable": row[3],
                    "column_key": row[4],
                    "default": row[5],
                    "extra": row[6]
                })
            
            # Check if this table has a company isolation column
            isolation_column = get_company_isolation_column(table_name)
            
            return {
                "columns": columns,
                "isolation_column": isolation_column
            }
        except Exception as e:
            logger.error(f"Error getting schema for table {table_name}: {str(e)}")
            return {"columns": [], "isolation_column": None}
        finally:
            session.close()
    
    @staticmethod
    async def execute_query(query: str, company_id: int) -> Dict[str, Any]:
        """Execute a SQL query with company isolation.
        
        Args:
            query: SQL query
            company_id: Company ID
            
        Returns:
            Query results
        """
        # Use existing company isolated database for query execution
        sql_database = get_company_isolated_sql_database(
            company_id=company_id,
            sample_rows_in_table_info=3
        )
        
        try:
            # Execute the query
            results = sql_database.run(query)
            return {"status": "success", "results": results}
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}\nQuery: {query}")
            return {"status": "error", "error": str(e)}

class SQLAgent(BaseAgent):
    """SQL agent for database queries and data visualization."""
    
    def __init__(self):
        """Initialize SQL agent."""
        super().__init__()
        self.type = "sql"
        self.mcp_mysql_tool = MCPMySQLTool()
    
    async def process_message(
        self, 
        message: str, 
        company_id: int, 
        user_id: str,
        conversation_id: Optional[str] = None,
        session: Optional[Session] = None
    ) -> AgentResponse:
        """Process message related to database queries.
        
        Args:
            message: User message
            company_id: Company ID
            user_id: User ID
            conversation_id: Optional conversation ID
            session: Optional database session
            
        Returns:
            Agent response
        """
        try:
            # Generate new conversation ID if not provided
            if not conversation_id:
                conversation_id = str(uuid.uuid4())
            
            # Initialize LLM
            llm = get_llm()
            
            # Create SQL database with company isolation
            sql_database = get_company_isolated_sql_database(
                company_id=company_id,
                sample_rows_in_table_info=3
            )
            
            # Get isolation columns information for key tables
            isolation_info = await self._get_isolation_columns_info(company_id)
            
            # Create SQL toolkit
            toolkit = SQLDatabaseToolkit(
                db=sql_database,
                llm=llm
            )
            
            # Create memory for conversation history
            memory = ConversationBufferMemory(
                memory_key="chat_history", 
                return_messages=True
            )
            
            # Retrieve and load conversation history
            if session:
                conversation_history = await self._get_conversation_history(
                    conversation_id=conversation_id,
                    company_id=company_id,
                    session=session
                )
                for exchange in conversation_history:
                    memory.chat_memory.add_user_message(exchange['message'])
                    memory.chat_memory.add_ai_message(exchange['response'])
            
            # Create additional schema info tools
            list_tables_tool = Tool(
                name="List_Database_Tables",
                description="Lists all tables in the database to understand what data is available",
                func=lambda _: self.mcp_mysql_tool.list_tables(company_id)
            )
            
            get_table_schema_tool = Tool(
                name="Get_Table_Schema",
                description="Gets detailed schema information for a specific table including column names, types, and isolation information",
                func=lambda table_name: self.mcp_mysql_tool.get_table_schema(table_name, company_id)
            )
            
            # Add custom tools to the toolkit
            toolkit.get_tools().extend([list_tables_tool, get_table_schema_tool])
            
            # Enhance system message with company-specific context and SQL safety
            isolation_instructions = self._format_isolation_instructions(isolation_info, company_id)
            
            system_message_content = f"""You are a friendly, helpful AI assistant working with a company's database. 
You help users query data and analyze their business information in natural, conversational language.
            
DATA ISOLATION CRITICAL RULE:
ALWAYS include WHERE created_by = {company_id} in ALL your queries for ALL tables involved.
This is required for security, no exceptions.

QUERY EXAMPLE:
SELECT * FROM projects WHERE created_by = {company_id};
SELECT * FROM employees WHERE created_by = {company_id};
SELECT p.name, u.name FROM projects p JOIN project_users pu ON p.id = pu.project_id JOIN users u ON pu.user_id = u.id WHERE p.created_by = {company_id} AND u.created_by = {company_id};

IMPORTANT TABLE USAGE RULES:
1. For user information like names and emails, ALWAYS prefer joining with the `users` table using `users.id`.
2. Only use the `employees` table if the query specifically asks for employee-specific details (like salary, department, designation, hire date, etc.).
3. Common join for project members: `projects` -> `project_users` -> `users`.

RESPONSE GUIDELINES:
1. Always use natural, conversational language
2. Explain insights in business terms, not database terms
3. Present numerical results in a readable format
4. Keep responses friendly and helpful

TECHNICAL RULES:
1. Write SQL compatible with MySQL 5.7
2. Only use SELECT statements (no INSERT, UPDATE, DELETE, etc.)
3. When joining tables, ensure proper join conditions AND company isolation for ALL tables.
"""

            # Create agent executor with enhanced prompt
            agent_executor = create_sql_agent(
                llm=llm,
                toolkit=toolkit,
                verbose=True,
                agent_type="openai-tools",
                handle_parsing_errors=True,
                memory=memory,
                max_iterations=5,
                prefix=system_message_content  # Use our enhanced prompt
            )
            
            tokens_used_in_request = 0
            tokens_remaining_after = None

            try:
                # Execute agent and capture full response including potential token usage
                # We use ainvoke which returns a richer dictionary than arun
                agent_result = await agent_executor.ainvoke({"input": message})
                
                # Extract the final answer and potential token usage
                agent_final_answer = agent_result.get("output", "Sorry, I could not process that.")
                
                # --- Attempt to get token usage (adapt based on actual LLM/agent output structure) ---
                # Example: Check common keys where token info might be stored
                # You might need to inspect the actual 'agent_result' dict structure
                # from your specific LLM provider (Gemini, OpenAI) to find the exact keys.
                llm_output = agent_result.get("llm_output", {}) # Or similar key if agent provides it
                if llm_output and isinstance(llm_output, dict):
                     # Example for OpenAI structure (adapt for Gemini if different)
                     usage_metadata = llm_output.get("token_usage", {}) 
                     if isinstance(usage_metadata, dict):
                          tokens_used_in_request = usage_metadata.get("total_tokens", 1) # Default to 1 if not found
                
                if tokens_used_in_request == 0:
                     # Fallback if LLM didn't provide usage, estimate based on message length?
                     # Or set a default minimum cost
                     tokens_used_in_request = max(1, len(message.split()) // 10) # Simple estimate
                     logger.warning(f"Could not determine exact token usage from LLM for company {company_id}. Estimated: {tokens_used_in_request}")
                else:
                     logger.info(f"Tokens used by LLM for company {company_id}: {tokens_used_in_request}")

                cleaned_response = self._clean_response_for_end_user(agent_final_answer)

                # --- Update token count in DB ---
                if session:
                    token_update_success = await self._update_token_usage(session, company_id, tokens_used_in_request)
                    if token_update_success:
                         tokens_remaining_after = await TokenManager.get_token_count(company_id, session)
                    else:
                         logger.error(f"Failed to update token usage for company {company_id} in database.")
                         # Handle appropriately - maybe return error or proceed with null remaining tokens?
                         # For now, we proceed but tokens_remaining might be inaccurate.

                # --- Save Conversation ---
                conversation_title = await self._generate_conversation_title(
                     conversation_id, company_id, message, cleaned_response
                )
                if session:
                    await self._save_conversation(
                        session=session,
                        conversation_id=conversation_id,
                        company_id=company_id,
                        user_id=user_id,
                        message=message,
                        response=cleaned_response,
                        agent_type=self.type,
                        tokens_used=tokens_used_in_request, # Save actual/estimated usage
                        title=conversation_title # Pass the generated title
                    )

                # --- Return Final Response ---
                return AgentResponse(
                    conversation_id=conversation_id,
                    response=cleaned_response,
                    conversation_title=conversation_title,
                    tokens_used=tokens_used_in_request,
                    tokens_remaining=tokens_remaining_after 
                )

            except Exception as e:
                logger.error(f"Agent execution or processing error: {str(e)}", exc_info=True)
                # Ensure error responses also conform to the model structure
                return AgentResponse(
                    conversation_id=conversation_id or str(uuid.uuid4()), # Generate if needed
                    response=f"I encountered an error processing your request. Details: {str(e)}",
                    conversation_title=None,
                    tokens_used=0, # No tokens used if error occurred before/during execution
                    tokens_remaining=await TokenManager.get_token_count(company_id, session) if session else None # Show current count if possible
                )
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}", exc_info=True)
            return AgentResponse(
                conversation_id=str(uuid.uuid4()) if not conversation_id else conversation_id,
                response="I apologize, but I encountered an internal error processing your request.",
                conversation_title=None,
                tokens_used=0,
                tokens_remaining=await TokenManager.get_token_count(company_id, session) if session else None
            )
    
    async def generate_visualization(
        self, 
        query: str, 
        company_id: int, 
        user_id: str,
        visualization_type: Optional[str] = None,
        session: Optional[Session] = None
    ) -> VisualizationResult:
        """Generate visualization from data.
        
        Args:
            query: Visualization query
            company_id: Company ID
            user_id: User ID
            visualization_type: Optional visualization type
            session: Optional database session
            
        Returns:
            Visualization result
        """
        # Initialize LLM
        llm = get_llm()
        
        # Create SQL database with company isolation
        sql_database = get_company_isolated_sql_database(
            company_id=company_id,
            sample_rows_in_table_info=3
        )
        
        # Create SQL toolkit
        toolkit = SQLDatabaseToolkit(
            db=sql_database,
            llm=llm
        )
        
        # Add custom tools for schema exploration
        list_tables_tool = Tool(
            name="List_Database_Tables",
            description="Lists all tables in the database to understand what data is available",
            func=lambda _: self.mcp_mysql_tool.list_tables(company_id)
        )
        
        get_table_schema_tool = Tool(
            name="Get_Table_Schema",
            description="Gets detailed schema information for a specific table including column names, types, and isolation information",
            func=lambda table_name: self.mcp_mysql_tool.get_table_schema(table_name, company_id)
        )
        
        # Add custom tools to the toolkit
        toolkit.get_tools().extend([list_tables_tool, get_table_schema_tool])
        
        # Create agent executor
        agent_executor = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=True,
            agent_type="openai-tools",
            handle_parsing_errors=True
        )
        
        # Modify the query to explicitly ask for SQL to generate visualization
        sql_query_prompt = f"Write a SQL query to get data for: {query}. Only return the SQL query, no explanations."
        
        try:
            # First get SQL query
            response = await agent_executor.ainvoke({"input": sql_query_prompt})
            sql_query = self._extract_sql_query(response.get("output", ""))
            
            if not sql_query:
                return VisualizationResult(
                    data=None,
                    explanation="Could not generate a SQL query for this visualization request."
                )
            
            # Execute the SQL query to get data
            data = sql_database.run(sql_query)
            
            # Generate visualization from data
            return await generate_visualization_from_data(
                data=data,
                query=query,
                visualization_type=visualization_type,
                llm=llm
            )
            
        except Exception as e:
            logger.error(f"Error generating visualization: {str(e)}")
            return VisualizationResult(
                data=None,
                explanation=f"Error generating visualization: {str(e)}"
            )
    
    async def perform_action(
        self, 
        action: str, 
        parameters: Dict[str, Any], 
        company_id: int, 
        user_id: str,
        session: Optional[Session] = None
    ) -> ActionResult:
        """SQL agent doesn't support actions.
        
        Args:
            action: Action to perform
            parameters: Action parameters
            company_id: Company ID
            user_id: User ID
            session: Optional database session
            
        Returns:
            Action result
        """
        return ActionResult(
            success=False,
            message="SQL agent doesn't support direct actions. Please use a specific query instead."
        )
    
    def _extract_sql_query(self, text: str) -> Optional[str]:
        """Extract SQL query from text.
        
        Args:
            text: Text to extract SQL query from
            
        Returns:
            SQL query if found, None otherwise
        """
        # Try to extract from code blocks
        code_block_pattern = r"```sql(.*?)```"
        matches = re.findall(code_block_pattern, text, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # Try alternative code block format
        alt_code_block_pattern = r"```(.*?)```"
        matches = re.findall(alt_code_block_pattern, text, re.DOTALL)
        
        if matches:
            for match in matches:
                if match.strip().upper().startswith("SELECT"):
                    return match.strip()
        
        # Look for SELECT statements
        select_pattern = r"(SELECT\s+.+?FROM\s+.+?)(;|$)"
        matches = re.findall(select_pattern, text, re.DOTALL | re.IGNORECASE)
        
        if matches:
            return matches[0][0].strip()
        
        return None
    
    def _check_visualization_intent(self, message: str) -> tuple[bool, Optional[str]]:
        """Check if the message has visualization intent.
        
        Args:
            message: User message
            
        Returns:
            Tuple of (has_visualization_intent, visualization_type)
        """
        message_lower = message.lower()
        
        # Keywords that suggest visualization intent
        viz_keywords = [
            "graph", "chart", "plot", "visualize", "visualization", "display",
            "show me", "diagram", "histogram", "bar chart", "line graph",
            "pie chart", "scatter plot"
        ]
        
        # Chart type mapping
        chart_types = {
            "bar": ["bar chart", "bar graph", "column chart"],
            "line": ["line chart", "line graph", "trend", "time series"],
            "pie": ["pie chart", "pie graph", "donut", "distribution"],
            "scatter": ["scatter plot", "scatter chart", "scatter graph"],
            "radar": ["radar chart", "radar graph", "spider chart"],
            "bubble": ["bubble chart", "bubble graph"]
        }
        
        # Check for visualization intent
        has_viz_intent = any(keyword in message_lower for keyword in viz_keywords)
        
        # Determine chart type
        viz_type = None
        if has_viz_intent:
            for chart_type, keywords in chart_types.items():
                if any(keyword in message_lower for keyword in keywords):
                    viz_type = chart_type
                    break
        
        return has_viz_intent, viz_type
    
    async def _get_conversation_history(
        self,
        conversation_id: str,
        company_id: int,
        session: Session
    ) -> List[Dict[str, Any]]:
        """Get conversation history for the given conversation ID and company ID.
        
        Args:
            conversation_id: Conversation ID
            company_id: Company ID
            session: Database session
            
        Returns:
            List of conversation messages
        """
        # Use BaseAgent's implementation
        return await super()._get_conversation_history(conversation_id, company_id, session)

    async def _get_remaining_tokens(self, session: Session, company_id: int) -> Optional[int]:
        """Helper to get the current remaining tokens."""
        # This method seems unused and token management is handled elsewhere.
        # Consider removing or implementing if needed.
        pass

    async def _get_isolation_columns_info(self, company_id: int) -> Dict[str, str]:
        """Get information about isolation columns for commonly used tables.
        
        Args:
            company_id: The company ID
            
        Returns:
            Dictionary mapping table names to their isolation columns
        """
        # List of commonly used tables to get isolation info for
        common_tables = [
            "employees", "users", "departments", "projects", "tasks", 
            "customers", "invoices", "products", "leads", "deals",
            "expenses", "payments", "orders", "leaves", "attendance_employees"
        ]
        
        isolation_info = {}
        engine = DatabaseConnection.create_engine()
        
        for table in common_tables:
            # Use the existing function to get the isolation column
            isolation_column = get_company_isolation_column(table, engine)
            if isolation_column:
                isolation_info[table] = isolation_column
        
        return isolation_info
    
    def _format_isolation_instructions(self, isolation_info: Dict[str, str], company_id: int) -> str:
        """Format isolation information into clear instructions for the LLM.
        
        Args:
            isolation_info: Dictionary mapping table names to isolation columns
            company_id: The company ID to use in filters
            
        Returns:
            Formatted instruction string
        """
        instructions = []
        
        for table, column in isolation_info.items():
            instructions.append(f"- Table '{table}': MUST include WHERE {table}.{column} = {company_id}")
        
        # Add generic instruction for tables not specifically listed
        instructions.append(f"- For ANY table not listed above: Check if it has columns like 'created_by', 'company_id', or 'user_id' and include the appropriate filter")
        
        # Join with newlines for readability in the prompt
        return "\n".join(instructions)
    
    def _clean_response_for_end_user(self, response: str) -> str:
        """Clean the response by removing SQL code blocks and technical details.
        
        Args:
            response: The raw response from the agent
            
        Returns:
            Cleaned response suitable for end users
        """
        # Remove SQL code blocks (```sql...```)
        response = re.sub(r'```sql.*?```', '', response, flags=re.DOTALL)
        
        # Remove any remaining code blocks
        response = re.sub(r'```.*?```', '', response, flags=re.DOTALL)
        
        # Remove SQL-like statements
        response = re.sub(r'SELECT.*?FROM.*?;', '', response, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove references to SQL or query execution
        response = re.sub(r'(executing|running|using|writing)(\s+the)?\s+query', 'checking', response, flags=re.IGNORECASE)
        response = re.sub(r'(the\s+)?(sql|database)\s+(query|results?|tables?|columns?)', 'the data', response, flags=re.IGNORECASE)
        
        # Replace multiple newlines with double newlines for readability
        response = re.sub(r'\n{3,}', '\n\n', response)
        
        # Trim extra whitespace
        response = response.strip()
        
        return response

    async def _generate_conversation_title(
        self,
        conversation_id: str,
        company_id: int,
        user_message: str,
        agent_response: str
    ) -> str:
        """Generate a title for the conversation based on the user's message and the agent's response.
        
        Args:
            conversation_id: Conversation ID
            company_id: Company ID
            user_message: User's original message
            agent_response: Agent's response
            
        Returns:
            Generated conversation title
        """
        # Extract keywords from the user's message and the agent's response
        user_keywords = re.findall(r'\b\w+\b', user_message.lower())
        agent_keywords = re.findall(r'\b\w+\b', agent_response.lower())
        
        # Combine keywords into a single string
        all_keywords = user_keywords + agent_keywords
        
        # Generate a title based on the keywords
        title = " ".join(all_keywords[:5])  # Take the first 5 words
        
        # Add company ID to the title
        title += f" - {company_id}"
        
        return title 