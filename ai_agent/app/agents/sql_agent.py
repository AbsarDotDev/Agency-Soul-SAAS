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
        is_new_conversation = False # Track if it's a new conversation
        try:
            # Generate new conversation ID if not provided
            if not conversation_id:
                is_new_conversation = True
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
2. Only use the `employees` table if the query *specifically and explicitly* asks for employee-only details (like salary, department, designation, hire date, etc.) OR if the context of the conversation (e.g., a previous question about employees) clearly indicates that a pronoun like "their" or "them" refers to employees. In such contextual cases, prioritize using the `employees` table for names if appropriate.
3. Common join for project members: `projects` -> `project_users` -> `users`.

CONTEXTUAL UNDERSTANDING:
- Pay close attention to pronouns (e.g., "their", "them", "those"). If the user just asked about a specific group (e.g., "employees", "projects named X"), and then asks a follow-up question using a pronoun, assume the pronoun refers to that previously mentioned group.
- Example: If user asks "How many employees are there?" and then "Tell me their names", "their" refers to employees. Query the `employees` table for names in this case.

VISUALIZATION RULES:
1. If the user asks to "visualize", "chart", "plot", or "graph" data, first generate the SQL query and get the results.
2. Formulate your natural language response explaining the findings.
3. **CRITICAL:** After your text response, if visualization was requested, add a separate JSON block containing the raw data suitable for charting. Format it like this:
```json
{{{{  # Escaped double curly braces
  "visualization_data": [
    {{{{ "label": "Category1", "value": 10 }}}}, # Escaped double curly braces
    {{{{ "label": "Category2", "value": 25 }}}}  # Escaped double curly braces
    // ... more data points ...
  ]
}}}} # Escaped double curly braces
```
Replace "label" and "value" with appropriate keys based on the query results (e.g., "department", "count"; "month", "sales"; "status", "project_count"). Include ALL relevant data points.

RESPONSE GUIDELINES:
1. Always use natural, conversational language for the main text response.
2. Explain insights in business terms, not database terms.
3. Present numerical results readably in the text.
4. Keep responses friendly and helpful.
5. If the user asks for "names" along with other attributes (e.g., "names and designations", "names and salaries"), ensure your textual response includes a list of the individual names. Do not just summarize the other attributes without listing the names if they were requested. For example, for "Tell their names and designation", an acceptable response would list each employee and their designation.

TECHNICAL RULES:
1. Write SQL compatible with MySQL 5.7
2. Only use SELECT statements.
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
                # Execute agent and capture full respsonse including potential token usage
                # We use ainvoke which returns a richer dictionary than arun
                agent_result = await agent_executor.ainvoke({"input": message})
                
                # Extract the final answer
                agent_final_answer = agent_result.get("output", "Sorry, I could not process that.")
                
                # Estimate tokens used by the main agent chain (Langchain SQL agent doesn't easily exposes this)
                # We can estimate based on input/output or integrate a callback handler later if precise count is needed
                # For now, let's use a basic estimate or assume 0 until a better method is found
                main_agent_tokens_used = 0 # Placeholder - needs improvement
                logger.warning("Token usage for main SQL agent chain is currently estimated as 0.")

                # --- Extract Visualization Data and Clean Text Response ---
                visualization_chart_data = None
                visualization_tokens_used = 0 # Initialize viz tokens
                text_response_part = agent_final_answer # Default to full answer
                
                # Regex to find the specific JSON block for visualization - more robust
                # Handles variations in whitespace around ```json and the { character, and case-insensitivity for 'json'
                viz_json_match = re.search(
                    r"```(?:json)?\s*(\{\s*\"visualization_data\":.*?\})\s*```", 
                    agent_final_answer, 
                    re.DOTALL | re.IGNORECASE
                )
                
                # First try to extract structured visualization data
                if viz_json_match:
                    # Extract group 1 which contains the JSON object itself
                    json_string = viz_json_match.group(1).strip() 
                    try:
                        # Parse the extracted JSON string
                        viz_payload = json.loads(json_string)
                        raw_viz_data = viz_payload.get("visualization_data")
                        
                        if raw_viz_data and isinstance(raw_viz_data, list):
                            logger.info(f"Extracted raw visualization data: {raw_viz_data}")
                            # Determine visualization type
                            viz_keywords, requested_type = self._check_visualization_intent(message)
                            
                            # Generate Chart.js compatible data
                            viz_result: VisualizationResult = await generate_visualization_from_data(
                                data=raw_viz_data, 
                                query=message, 
                                visualization_type=requested_type,
                                llm=llm 
                            )
                            visualization_chart_data = viz_result.data
                            # Fixed token usage for visualization as requested - always use 2 tokens for visualizations
                            visualization_tokens_used = 2
                            logger.info(f"Visualization step used {visualization_tokens_used} tokens.")
                            
                            # Remove the JSON block from the text response shown to the user
                            text_response_part = agent_final_answer[:viz_json_match.start()].strip()
                            logger.info(f"Cleaned text response part: {text_response_part}")
                        else:
                            logger.warning("Found viz block, but 'visualization_data' key missing or not a list.")
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse visualization JSON block: {e}. JSON string: {json_string}")
                    except Exception as viz_e:
                         logger.error(f"Error processing visualization data: {viz_e}", exc_info=True)
                else:
                    # Try to detect visualization intent directly from the message
                    has_viz_intent, viz_type = self._check_visualization_intent(message)
                    
                    if has_viz_intent:
                        logger.info(f"No visualization JSON block found, but visualization intent detected. Trying direct visualization.")
                        try:
                            # Try to extract table data from the LLM response
                            # This looks for patterns like "Department: 5 employees" or "IT: 10, HR: 5"
                            data_points = []
                            
                            # Pattern 1: Look for "Category: Number" patterns
                            pattern1 = r'(\w+):\s*(\d+)'
                            matches = re.findall(pattern1, agent_final_answer)
                            if matches:
                                for category, value in matches:
                                    data_points.append({"label": category, "value": int(value)})
                            
                            # Pattern 2: Look for bullet points with numbers
                            pattern2 = r'[•\*-]\s*\*\*([^:]+)\*\*:\s*(\d+)'
                            matches = re.findall(pattern2, agent_final_answer)
                            if matches:
                                for category, value in matches:
                                    data_points.append({"label": category.strip(), "value": int(value)})
                            
                            # If data points were found, generate visualization
                            if data_points:
                                logger.info(f"Extracted {len(data_points)} data points from text response for visualization: {data_points}")
                                viz_result: VisualizationResult = await generate_visualization_from_data(
                                    data=data_points,
                                    query=message,
                                    visualization_type=viz_type,
                                    llm=llm
                                )
                                visualization_chart_data = viz_result.data
                                visualization_tokens_used = 2
                                logger.info(f"Created visualization from extracted data points.")
                            else:
                                logger.info("No visualization JSON block found and couldn't extract data points from text.")
                        except Exception as e:
                            logger.error(f"Error attempting fallback visualization generation: {e}", exc_info=True)
                    else:
                         logger.info("No visualization JSON block found in agent output and no visualization intent detected.")
                
                # Ensure options is always an object, never an array
                if visualization_chart_data and 'options' in visualization_chart_data:
                    if visualization_chart_data['options'] is None or (
                        isinstance(visualization_chart_data['options'], list) and 
                        len(visualization_chart_data['options']) == 0
                    ):
                        visualization_chart_data['options'] = {}

                # --- Calculate Total Token Usage ---
                # Always use 1 token for main agent plus visualization tokens
                main_agent_tokens_used = 1
                total_tokens_used = main_agent_tokens_used + visualization_tokens_used
                logger.info(f"Total tokens used for request: {total_tokens_used}")

                # Clean the final text response part
                cleaned_response = self._clean_response_for_end_user(text_response_part)

                # --- Update token count in DB ---
                tokens_remaining_after = None # Initialize
                if session:
                    # Use the *total* tokens used
                    token_update_success = await self._update_token_usage(session, company_id, total_tokens_used)
                    if token_update_success:
                         tokens_remaining_after = await TokenManager.get_token_count(company_id, session)
                         logger.info(f"Tokens updated. Remaining for company {company_id}: {tokens_remaining_after}")
                    else:
                         logger.error(f"Failed to update token usage for company {company_id} in database.")
                         # Fetch current count even if update failed, for informational purposes
                         tokens_remaining_after = await TokenManager.get_token_count(company_id, session)

                # --- Save Conversation ---
                actual_title_saved = await self._save_conversation(
                    session=session,
                    conversation_id=conversation_id,
                    company_id=company_id,
                    user_id=user_id,
                    message=message, 
                    response=cleaned_response, 
                    agent_type=self.type,
                    tokens_used=total_tokens_used,
                    visualization=visualization_chart_data,
                    title=None # Let _save_conversation logic decide based on whether it's new
                )
                
                title_for_response = actual_title_saved

                if not is_new_conversation and actual_title_saved is None:
                    # For an existing conversation, _save_conversation might return None if it didn't generate a new title for this specific turn.
                    # We need to fetch the original, established title for the AgentResponse.
                    try:
                        stmt = text("""SELECT title FROM agent_conversations 
                                     WHERE conversation_id = :conv_id AND title IS NOT NULL 
                                     ORDER BY created_at ASC LIMIT 1""")
                        result = session.execute(stmt, {"conv_id": conversation_id}).scalar_one_or_none()
                        if result:
                            title_for_response = result
                            logger.info(f"Fetched original title '{title_for_response}' for existing conversation {conversation_id}")
                        else:
                            # Should not happen if conversation exists and had a title, but as a fallback:
                            logger.warning(f"Could not fetch original title for existing conversation {conversation_id}. Using current message as fallback.")
                            title_for_response = message[:75] + "..." # Fallback, less ideal
                    except Exception as e:
                        logger.error(f"Error fetching original title for conversation {conversation_id}: {e}. Using current message as fallback.")
                        title_for_response = message[:75] + "..." # Fallback, less ideal
                elif is_new_conversation and actual_title_saved is None:
                    # If it was a new conversation but _save_conversation somehow failed to return a title (e.g., error during title gen)
                    logger.warning(f"New conversation {conversation_id} saved, but no title returned from _save_conversation. Using current message as fallback.")
                    title_for_response = message[:75] + "..."

                # --- Return Final Response ---
                return AgentResponse(
                    conversation_id=conversation_id,
                    response=cleaned_response,
                    conversation_title=title_for_response, # Use the determined title
                    tokens_used=total_tokens_used, # Return the total
                    tokens_remaining=tokens_remaining_after,
                    visualization=visualization_chart_data
                )

            except Exception as e:
                logger.error(f"Agent execution or processing error: {str(e)}", exc_info=True)
                # Ensure total_tokens_used is defined even in error paths for the return model
                total_tokens_used = 0 
                tokens_remaining_after = await TokenManager.get_token_count(company_id, session) if session else None
                return AgentResponse(
                    conversation_id=conversation_id or str(uuid.uuid4()),
                    response=f"I encountered an error processing your request. Details: {str(e)}",
                    conversation_title=None,
                    tokens_used=total_tokens_used, # Use 0 on error
                    tokens_remaining=tokens_remaining_after,
                    visualization=None 
                )
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}", exc_info=True)
            total_tokens_used = 0
            tokens_remaining_after = await TokenManager.get_token_count(company_id, session) if session else None
            return AgentResponse(
                conversation_id=str(uuid.uuid4()) if not conversation_id else conversation_id,
                response="I apologize, but I encountered an internal error processing your request.",
                conversation_title=None,
                tokens_used=total_tokens_used,
                tokens_remaining=tokens_remaining_after,
                visualization=None
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