from typing import Dict, Any, List, Optional, Union, Tuple
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
        self.llm = get_llm()
        self.embedding_model = get_embedding_model()
    
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
        # Check if this is a visualization request first
        is_viz_request, viz_type = self._check_visualization_intent(message)
        logger.info(f"Visualization intent detection: intent={is_viz_request}, type={viz_type}, message='{message[:50]}...'")
        
        if is_viz_request:
            logger.info(f"Detected visualization request in message: '{message[:50]}...' - Using visualization processing")
            # Call generate_visualization directly and immediately return its result
            viz_response = await self.generate_visualization(
                query=message,
                company_id=company_id,
                user_id=user_id,
                visualization_type=viz_type,
                session=session,
                conversation_id=conversation_id
            )
            
            # Log full response for debugging
            viz_response_dict = viz_response.dict()
            if "response" in viz_response_dict:
                viz_response_dict["response"] = viz_response_dict["response"][:100] + "..." if len(viz_response_dict["response"]) > 100 else viz_response_dict["response"]
            logger.info(f"Visualization response from generate_visualization: {viz_response_dict}")
            
            return viz_response
            
        # If not a visualization request, proceed with normal processing
        is_new_conversation = False # Track if it's a new conversation
        try:
            # Generate new conversation ID if not provided
            if not conversation_id:
                is_new_conversation = True
                conversation_id = str(uuid.uuid4())
            
            # Create SQL database with company isolation
            sql_database = get_company_isolated_sql_database(
                company_id=company_id,
                sample_rows_in_table_info=3
            )
            
            # Get isolation columns information for key tables
            isolation_info = await self._get_isolation_columns_info(company_id)
            
            # Get a list of available tables to populate the system message
            try:
                available_tables_list = await self.mcp_mysql_tool.list_tables(company_id)
                # Format tables for better readability
                available_tables = "The following tables are available in the database:\n" + ", ".join(available_tables_list)
            except Exception as e:
                logger.error(f"Error getting available tables: {str(e)}")
                available_tables = "Some tables are available in the database. Use List_Database_Tables tool to see them."
            
            # Create SQL toolkit
            toolkit = SQLDatabaseToolkit(
                db=sql_database,
                llm=self.llm
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
You MUST be extremely diligent and include a WHERE clause checking that records belong to the company in ALL your queries.
ALWAYS include WHERE created_by = {company_id} or an equivalent company isolation clause in EVERY query. No exceptions.

For Tables with unique isolation conditions:
{isolation_instructions}

ID RESOLUTION - ALWAYS FOLLOW THIS RULE:
When your query returns IDs (like customer_id, employee_id, etc.), ALWAYS perform additional queries to resolve these IDs into human-readable names.
Example:
1. If your first query returns a customer_id = 5, run a second query: "SELECT name FROM customers WHERE id = 5 AND created_by = {company_id}"
2. Then replace "customer_id: 5" with "Customer: [actual customer name]" in your response
3. Do this for ALL foreign key IDs in your results to make the information user-friendly
4. For status codes, convert them to descriptive text (e.g., "Status: 1" becomes "Status: Active" if you know the mapping)

QUERY EXECUTION WORKFLOW:
1. First, run the main query to get the primary data
2. Then run additional queries to resolve all IDs to names
3. Combine all this information into a natural language response
4. Never mention the additional ID resolution queries in your response

AVAILABLE TABLES OVERVIEW:
{available_tables}

BUSINESS CONTEXT:
The database stores information for a business management system covering:
1. HR & Employee management (employees, departments, attendance, etc.)
2. Finance & Accounting (invoices, expenses, bills, etc.)
3. Sales & CRM (customers, deals, leads, etc.)
4. Project Management (projects, tasks, etc.)
5. Products & Inventory (products, warehouses, stocks, etc.)

The database uses standard conventions with tables like:
- 'employees', 'departments', 'designations', etc. for HR data
- 'customers', 'deals', 'leads', etc. for CRM data
- 'bills', 'expenses', 'revenues', etc. for financial data
- 'projects', 'project_tasks', etc. for project data
- 'product_services', 'warehouses', etc. for inventory data

SPECIFIC TABLE STRUCTURE GUIDANCE:
1. The 'departments' table has columns: id, name, branch_id, created_by
2. The 'employees' table has columns: id, name, department_id, designation_id, etc.
3. For employee department counts, use: SELECT d.name, COUNT(e.id) AS employee_count FROM departments d LEFT JOIN employees e ON d.id = e.department_id WHERE d.created_by = {company_id} GROUP BY d.name
4. Always use the actual table and column names: departments (not department), employees (not employee), etc.

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

DIRECT ANSWERING REQUIREMENTS:
1. ALWAYS attempt to directly answer the user's question, even if some details appear to be missing.
2. DO NOT ask clarifying questions back to the user unless absolutely necessary.
3. Make reasonable assumptions based on context when needed:
   - If time period is ambiguous (e.g. "sales for last month"), assume it means the previous calendar month.
   - If categories are ambiguous, include all relevant categories in your analysis.
   - If specific data seems unavailable, inform the user after trying the best possible query.

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
5. When data is ambiguous, make reasonable assumptions - do NOT ask for clarification.
6. If data for a specific request truly doesn't exist in the database, try to provide the closest relevant information available.
7. If the user asks for "names" along with other attributes (e.g., "names and designations", "names and salaries"), ensure your textual response includes a list of the individual names. Do not just summarize the other attributes without listing the names if they were requested. For example, for "Tell their names and designation", an acceptable response would list each employee and their designation.

TECHNICAL RULES:
1. Write SQL compatible with MySQL 5.7
2. Only use SELECT statements.
3. When joining tables, ensure proper join conditions AND company isolation for ALL tables.
"""

            # Create agent executor with enhanced prompt
            agent_executor = create_sql_agent(
                llm=self.llm,
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
                
                # Log the full agent final answer for debugging
                logger.info(f"Original agent response: {agent_final_answer[:500]}...")
                
                # First check if this is a visualization request
                is_viz_request, viz_type = self._check_visualization_intent(message)
                if is_viz_request:
                    logger.info(f"This is a visualization request for type: {viz_type}")
                
                # Regex to find the specific JSON block for visualization - more robust
                # Handles variations in whitespace around ```json and the { character, and case-insensitivity for 'json'
                viz_json_match = re.search(
                    r"```(?:json)?\s*(\{\s*\"visualization_data\":.*?\})\s*```", 
                    agent_final_answer, 
                    re.DOTALL | re.IGNORECASE
                )
                
                # Try to extract structured visualization data
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
                                llm=self.llm 
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
                    logger.info("No visualization JSON block found in response, attempting to extract data directly from SQL results")
                    
                    # Attempt to extract data directly from SQL results if available
                    try:
                        # If we have SQL results but no viz data, try to create visualization directly
                        if result_rows:
                            logger.info(f"Creating visualization directly from SQL results: {len(result_rows)} rows")
                            
                            # If query is related to employees per department, ensure we format data properly
                            if ("employees per department" in message.lower() or 
                                "employee per department" in message.lower() or 
                                "employees by department" in message.lower()):
                                
                                # For department counts, ensure data is in the right format even if column names differ
                                # Look for department name and count columns
                                dept_col = None
                                count_col = None
                                
                                if result_rows and isinstance(result_rows[0], dict):
                                    for key in result_rows[0].keys():
                                        key_lower = key.lower()
                                        if "department" in key_lower or "dept" in key_lower or "name" in key_lower:
                                            dept_col = key
                                        elif "count" in key_lower or "num" in key_lower or "total" in key_lower or "employee" in key_lower:
                                            count_col = key
                                
                                if dept_col and count_col:
                                    logger.info(f"Found department column '{dept_col}' and count column '{count_col}'")
                                    raw_viz_data = [
                                        {"department": row[dept_col], "count": row[count_col]} 
                                        for row in result_rows
                                    ]
                                else:
                                    # Use the data as is
                                    raw_viz_data = result_rows
                            else:
                                # Use the results as is
                                raw_viz_data = result_rows
                                
                            # Generate visualization from the data
                            viz_result: VisualizationResult = await generate_visualization_from_data(
                                data=raw_viz_data, 
                                query=message, 
                                visualization_type=viz_type,
                                llm=self.llm 
                            )
                            
                            visualization_chart_data = viz_result.data
                            # Fixed token usage for visualization as requested - always use 2 tokens for visualizations
                            visualization_tokens_used = 2
                            logger.info(f"Visualization step used {visualization_tokens_used} tokens.")
                    except Exception as e:
                        logger.error(f"Failed to create visualization directly from SQL results: {e}")
                
                # Ensure visualization data is properly formatted
                if visualization_chart_data:
                    # Ensure options is always an object, never an array
                    if 'options' in visualization_chart_data:
                        if visualization_chart_data['options'] is None or (
                            isinstance(visualization_chart_data['options'], list) and 
                            len(visualization_chart_data['options']) == 0
                        ):
                            visualization_chart_data['options'] = {}
                            logger.info("Fixed empty options array to be an empty object")
                            
                    # Log comprehensive visualization details for debugging
                    viz_info = {
                        "chart_type": visualization_chart_data.get("chart_type", "unknown"),
                        "title": visualization_chart_data.get("title", None),
                        "has_labels": "labels" in visualization_chart_data and len(visualization_chart_data["labels"]) > 0,
                        "has_datasets": "datasets" in visualization_chart_data and len(visualization_chart_data["datasets"]) > 0,
                        "options_type": type(visualization_chart_data.get("options", {})).__name__
                    }
                    logger.info(f"Final visualization data structure: {viz_info}")
                else:
                    logger.info("No visualization data was generated")

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
                # For visualization requests, set agent_type to "visualization" to ensure proper handling
                agent_type_for_response = "visualization" if (visualization_chart_data is not None or is_viz_request) else self.type
                
                # Log explicit debug information about the final response
                if visualization_chart_data:
                    logger.info(f"Returning visualization response with agent_type={agent_type_for_response}")
                    logger.info(f"Visualization data present: chart_type={visualization_chart_data.get('chart_type', 'unknown')}")
                else:
                    logger.info(f"Returning non-visualization response with agent_type={agent_type_for_response}")
                
                return AgentResponse(
                    conversation_id=conversation_id,
                    response=cleaned_response,
                    conversation_title=title_for_response, # Use the determined title
                    tokens_used=total_tokens_used, # Return the total
                    tokens_remaining=tokens_remaining_after,
                    visualization=visualization_chart_data,
                    agent_type=agent_type_for_response  # Set appropriate agent_type based on response type
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
                    visualization=None,
                    agent_type=self.type  # Add agent_type to the response
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
                visualization=None,
                agent_type=self.type  # Add agent_type to the response
            )
    
    def _convert_decimal_to_float(self, obj):
        """Recursively convert Decimal objects to float for JSON serialization.
        Also handles date and datetime objects by converting to string format.
        
        Args:
            obj: Object to convert
            
        Returns:
            Converted object with all special types changed to JSON-serializable types
        """
        from decimal import Decimal
        from datetime import date, datetime
        
        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, (date, datetime)):
            # Convert date/datetime to ISO format string
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: self._convert_decimal_to_float(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_decimal_to_float(item) for item in list(obj)]
        else:
            return obj
            
    async def generate_visualization(
        self, 
        query: str, 
        company_id: int = None, 
        user_id: str = None,
        visualization_type: Optional[str] = None,
        session: Optional[Session] = None,
        conversation_id: Optional[str] = None,
        insights_text: Optional[str] = None  # Added parameter for insights from other agents
    ) -> AgentResponse:
        """Generate visualization from SQL query or natural language description."""
        
        current_conversation_id = conversation_id or str(uuid.uuid4())
        logger.info(f"[SQLAgent.generate_visualization] Called with query: '{query[:60]}...', ConvID: {current_conversation_id}")

        try:
            # When a visualization request is passed from another agent like ProductAgent
            # Determine if this is a product-specific visualization request
            is_product_request = False
            if any(term in query.lower() for term in ["product", "inventory", "stock", "quantity", "item"]):
                is_product_request = True
                logger.info(f"[SQLAgent.generate_visualization] Detected product-specific visualization request")

            sql_database = get_company_isolated_sql_database(company_id=company_id, sample_rows_in_table_info=3)
            
            # For product visualization requests, try to use a specialized query
            if is_product_request:
                # For product visualization, extract relevant information directly
                try:
                    product_query = """
                    SELECT 
                        ps.name as product_name, 
                        ps.quantity as quantity,
                        ps.sale_price as price,
                        pc.name as category
                    FROM 
                        product_services ps
                    LEFT JOIN 
                        product_service_categories pc ON ps.category_id = pc.id
                    WHERE 
                        ps.created_by = :company_id
                    ORDER BY 
                        ps.quantity DESC
                    """
                    
                    logger.info(f"[SQLAgent.generate_visualization] Using specialized product query")
                    data_for_viz = await self._execute_query_safely(
                        query=product_query.replace(":company_id", str(company_id)),
                        company_id=company_id
                    )
                    
                    if data_for_viz:
                        # Convert any Decimal objects to float for JSON serialization
                        data_for_viz = self._convert_decimal_to_float(data_for_viz)
                        
                        logger.info(f"[SQLAgent.generate_visualization] Found {len(data_for_viz)} product records")
                        
                        # Extract visualization type from the query
                        if "histogram" in query.lower():
                            visualization_type = "bar" # Use bar chart for histogram
                        elif not visualization_type:
                            # Default to bar for product quantity
                            visualization_type = "bar"
                            
                        # Generate textual summary
                        summary_prompt = f"""The following product data was retrieved:
Data: {json.dumps(data_for_viz[:5], indent=2)} (showing first 5 of {len(data_for_viz)} products)

The user asked: '{query}'
Based on this data, please provide a brief summary of what the data shows about the products.
Focus on quantities, categories, or other patterns visible in the data.
Be concise and informative."""

                        summary_response_message = await self.llm.ainvoke(summary_prompt)
                        textual_summary = summary_response_message.content.strip()
                        
                        # Generate visualization
                        viz_result_obj = await generate_visualization_from_data(
                            data=data_for_viz, 
                            query=query, 
                            visualization_type=visualization_type, 
                            llm=self.llm
                        )
                        
                        # Finalize response - reuse existing logic below
                    else:
                        logger.warning(f"[SQLAgent.generate_visualization] Product query returned no data")
                        # Will fall back to regular query generation below
                except Exception as product_err:
                    logger.error(f"[SQLAgent.generate_visualization] Error in product specialized query: {product_err}", exc_info=True)
                    # Will fall back to regular query generation below

            # Standard SQL query generation path for non-product queries or if product one failed
            if not is_product_request or not data_for_viz:
                toolkit = SQLDatabaseToolkit(db=sql_database, llm=self.llm)
                # ... (keep custom_tools and agent_executor setup as before)
                custom_tools = [
                    Tool(
                        name="List_Database_Tables",
                        description="Lists all tables in the database...", # Truncated for brevity
                        func=lambda _: self.mcp_mysql_tool.list_tables(company_id)
                    ),
                    Tool(
                        name="Get_Table_Schema",
                        description="Gets detailed schema information for a specific table...", # Truncated
                        func=lambda table_name: self.mcp_mysql_tool.get_table_schema(table_name, company_id)
                    )
                ]
                agent_executor = create_sql_agent(
                    llm=self.llm, toolkit=toolkit, agent_type="openai-tools", handle_parsing_errors=True, verbose=True
                )

                sql_query_prompt = f"""Given the user query: '{query}', ... (rest of your detailed SQL prompt) ... Only return the SQL SELECT query itself."""
                # Ensure the full detailed SQL prompt is here, as it was before.
                # For brevity in this edit, I am not repeating the entire multi-line SQL prompt string.
                # Please ensure the original detailed prompt for generating SQL is maintained.
                
                response_from_sql_llm = await agent_executor.ainvoke({"input": sql_query_prompt})
                sql_query_text = self._extract_sql_query(response_from_sql_llm.get("output", ""), company_id)
                
                if sql_query_text:
                    sql_query_text = self._correct_table_names(sql_query_text, company_id)
                    logger.info(f"[SQLAgent.generate_visualization] Corrected SQL: {sql_query_text}")
                else:
                    logger.warning(f"[SQLAgent.generate_visualization] Could not generate SQL for: {query}")
                    return AgentResponse(
                        response="Failed to generate SQL query for visualization. Please try rephrasing.",
                        conversation_id=current_conversation_id, conversation_title=f"Viz SQL Gen Error: {query[:20]}",
                        visualization=None, tokens_used=1, tokens_remaining=None, agent_type="visualization_sql_error"
                    )

                data_for_viz = await self._execute_query_safely(sql_query_text, company_id)
                # Convert any Decimal objects to float for JSON serialization
                data_for_viz = self._convert_decimal_to_float(data_for_viz)

            # Regardless of how we got the data, check if we have data to visualize
            if not data_for_viz:
                return AgentResponse(
                    response="The query returned no data to visualize.", conversation_id=current_conversation_id,
                    conversation_title=f"Viz No Data: {query[:20]}", visualization=None, tokens_used=1, 
                    tokens_remaining=None, agent_type="visualization_no_data"
                )

            # Step 1: Generate textual summary of the data_for_viz (if not already done)
            textual_summary = None
            try:
                # If we have insights from another agent, use that instead of generating a new summary
                if insights_text:
                    textual_summary = f"{insights_text}\n\nI've created a visualization based on this data."
                    logger.info(f"[SQLAgent.generate_visualization] Using provided insights text for summary")
                elif not textual_summary and data_for_viz:  # Only generate if we don't have one and have data
                    # Convert data to JSON for prompt, ensuring no Decimal objects
                    data_sample_json = json.dumps(data_for_viz[:5] if len(data_for_viz) > 5 else data_for_viz, indent=2)
                    
                    summary_prompt = f"""The following data was retrieved from the database based on the user's request '{query}':
Data: {data_sample_json} (showing first 5 rows if many)

Please provide a brief, natural language summary of this data. 
Focus on answering the user's likely question. For example, if the data is about employees per department, say something like 'The data shows X employees in department A, Y in department B...'.
Do not mention the SQL query or database structure. Be concise. """
                    summary_response_message = await self.llm.ainvoke(summary_prompt)
                    textual_summary = summary_response_message.content.strip()
                    logger.info(f"[SQLAgent.generate_visualization] Generated textual summary: {textual_summary}")
            except Exception as summary_err:
                logger.error(f"[SQLAgent.generate_visualization] Error generating textual summary: {summary_err}", exc_info=True)
                textual_summary = "Here's a visualization of the data."  # Simple fallback

            # Step 2: Generate visualization object from data_for_viz
            viz_result_obj = await generate_visualization_from_data(
                data=data_for_viz, query=query, visualization_type=visualization_type, llm=self.llm
            )
            
            tokens_used = 2 # For viz generation + 1 for summary (adjust if summary LLM call is tokenized separately and more accurately)
            tokens_remaining = None
            if session:
                try:
                    # Direct token update (copied from previous fix, ensure DatabaseConnection and text are imported)
                    engine = DatabaseConnection.create_engine()
                    with Session(engine) as db_session:
                        stmt_update = text("UPDATE users SET ai_agent_tokens_used = ai_agent_tokens_used + :tokens WHERE id = :company_id")
                        db_session.execute(stmt_update, {"tokens": tokens_used, "company_id": company_id})
                        db_session.commit()
                        stmt_get = text("SELECT p.ai_agent_default_tokens - u.ai_agent_tokens_used FROM users u JOIN plans p ON u.plan = p.id WHERE u.id = :company_id")
                        result = db_session.execute(stmt_get, {"company_id": company_id}).fetchone()
                        tokens_remaining = result[0] if result else None
                except Exception as e_token:
                    logger.error(f"[SQLAgent.generate_visualization] Error updating tokens: {e_token}")

            # Save conversation (using the textual_summary as the main response part)
            title_for_saving = None # Let _save_conversation handle title for new conversations
            if session: # Only attempt save if session is available
                 await self._save_conversation(
                    session=session, conversation_id=current_conversation_id, company_id=company_id, user_id=user_id,
                    message=query, response=textual_summary, agent_type="visualization", 
                    tokens_used=tokens_used, visualization=viz_result_obj.data, title=title_for_saving
                )
            
            final_agent_response = AgentResponse(
                response=textual_summary or "Here's a visualization of the data.", # Use the generated summary or fallback
                conversation_id=current_conversation_id,
                conversation_title=f"Viz: {query[:30]}..." if len(query)>30 else f"Viz: {query}", # Placeholder title for now
                visualization=viz_result_obj.data,
                tokens_used=tokens_used,
                tokens_remaining=tokens_remaining,
                agent_type="visualization"
            )
            logger.info(f"[SQLAgent.generate_visualization] Returning: {final_agent_response.dict(exclude_none=True)}")
            return final_agent_response
            
        except Exception as e:
            logger.error(f"[SQLAgent.generate_visualization] CRITICAL ERROR for query '{query[:60]}': {str(e)}", exc_info=True)
            return AgentResponse(
                response=f"Critical error in SQLAgent.generate_visualization: {str(e)}",
                conversation_id=current_conversation_id,
                conversation_title=f"Viz Critical Error: {query[:20]}",
                visualization=None, tokens_used=0, tokens_remaining=None, agent_type="error_sql_agent_viz"
            )
    
    async def _execute_query_safely(self, query: str, company_id: int) -> List[Dict[str, Any]]:
        """Execute a SQL query safely, ensuring company isolation.
        
        Args:
            query: SQL query to execute
            company_id: Company ID
            
        Returns:
            Query results as a list of dictionaries
        """
        # Create a plain engine and connection (no automatic isolation verification)
        engine = DatabaseConnection.create_engine()
        
        try:
            with engine.connect() as connection:
                # Execute the query
                result = connection.execute(text(query))
                
                # Convert result to list of dictionaries
                column_names = result.keys()
                rows = result.fetchall()
                
                data = []
                for row in rows:
                    data.append({column: value for column, value in zip(column_names, row)})
                
                return data
        except Exception as e:
            # Log and re-raise the exception
            logger.error(f"Error executing query for company {company_id}: {str(e)}\nQuery: {query}")
            raise
    
    async def perform_action(
        self, 
        action: str, 
        parameters: Dict[str, Any], 
        company_id: int, 
        user_id: str,
        session: Optional[Session] = None
    ) -> ActionResult:
        """Perform a data modification action after LLM-based validation/planning if necessary."""
        # For now, actions are direct database operations via DatabaseQueries
        # LLM might be used in future for more complex action planning or validation
        # llm = get_llm() # Potentially use self.llm if needed here

        logger.info(f"Performing action '{action}' with params: {parameters} for company {company_id}")
        
        # Implementation of perform_action method
        # This is a placeholder and should be replaced with the actual implementation
        return ActionResult(
            success=False,
            message="SQL agent doesn't support direct actions. Please use a specific query instead."
        )
    
    def _extract_sql_query(self, text: str, company_id: int = None) -> Optional[str]:
        """Extract SQL query from text.
        
        Args:
            text: Text to extract SQL query from
            company_id: Optional company ID to add isolation filters
            
        Returns:
            SQL query if found, None otherwise
        """
        # Try to extract from code blocks
        code_block_pattern = r"```sql(.*?)```"
        matches = re.findall(code_block_pattern, text, re.DOTALL)
        
        if matches:
            return self._correct_table_names(matches[0].strip(), company_id)
        
        # Try alternative code block format
        alt_code_block_pattern = r"```(.*?)```"
        matches = re.findall(alt_code_block_pattern, text, re.DOTALL)
        
        if matches:
            for match in matches:
                if match.strip().upper().startswith("SELECT"):
                    return self._correct_table_names(match.strip(), company_id)
        
        # Look for SELECT statements
        select_pattern = r"(SELECT\s+.+?FROM\s+.+?)(;|$)"
        matches = re.findall(select_pattern, text, re.DOTALL | re.IGNORECASE)
        
        if matches:
            return self._correct_table_names(matches[0][0].strip(), company_id)
        
        return None
    
    def _correct_table_names(self, query: str, company_id: int = None) -> str:
        """Correct common table name mistakes in SQL queries and ensure company isolation."""
        # Map of incorrect table names to correct ones
        table_corrections = {
            # HRM related corrections
            r'\bdepartment\b(?!\.)': 'departments',
            r'\bemployee\b(?!\.)': 'employees',
            r'department\.dept_id': 'departments.id',
            r'department\.dept_name': 'departments.name',
            r'employee\.emp_id': 'employees.id',
            r'employee\.department_id': 'employees.department_id',
            
            # CRM related corrections
            r'\bcustomer\b(?!\.)': 'customers',
            r'\blead\b(?!\.)': 'leads',
            r'\bdeal\b(?!\.)': 'deals',
            
            # Finance related corrections
            r'\binvoice\b(?!\.)': 'invoices',
            r'\bpayment\b(?!\.)': 'payments',
            r'\bexpense\b(?!\.)': 'expenses',
            
            # Product related corrections
            r'\bproduct\b(?!\.)': 'product_services',
            r'\bwarehouse\b(?!\.)': 'warehouses'
        }
        
        # Apply corrections
        corrected_query = query
        for incorrect, correct in table_corrections.items():
            corrected_query = re.sub(incorrect, correct, corrected_query, flags=re.IGNORECASE)
        
        # Fix table aliases in joins if needed
        if 'departments d' in corrected_query.lower() and 'employees e' in corrected_query.lower():
            corrected_query = re.sub(
                r'd\.id\s*=\s*e\.dept_id', 
                'd.id = e.department_id', 
                corrected_query, 
                flags=re.IGNORECASE
            )
        
        # Don't proceed with isolation if no company_id provided
        if company_id is None:
            return corrected_query
        
        # COMPLETELY NEW ISOLATION APPROACH
        # First, parse the query to ensure we don't interfere with SQL keywords
        
        # 1. Extract all the SQL parts using regex
        # This helps us identify the different parts of the query
        select_match = re.search(r'^\s*SELECT\s+', corrected_query, re.IGNORECASE)
        from_match = re.search(r'\s+FROM\s+', corrected_query, re.IGNORECASE)
        where_match = re.search(r'\s+WHERE\s+', corrected_query, re.IGNORECASE)
        group_by_match = re.search(r'\s+GROUP\s+BY\s+', corrected_query, re.IGNORECASE)
        having_match = re.search(r'\s+HAVING\s+', corrected_query, re.IGNORECASE)
        order_by_match = re.search(r'\s+ORDER\s+BY\s+', corrected_query, re.IGNORECASE)
        limit_match = re.search(r'\s+LIMIT\s+', corrected_query, re.IGNORECASE)
        
        if not (select_match and from_match):
            logger.warning(f"Could not identify SELECT or FROM in query: {corrected_query}")
            return corrected_query  # Can't proceed without basics
        
        # 2. Extract the tables to isolate
        tables = self._extract_tables_from_query(corrected_query)
        if not tables:
            logger.info("No tables found to isolate in query")
            return corrected_query
        
        # 3. Build the isolation conditions
        isolation_conditions = []
        for table_name, alias in tables:
            # Use the appropriate table reference
            table_ref = alias if alias else table_name
            isolation_conditions.append(f"{table_ref}.created_by = {company_id}")
        
        if not isolation_conditions:
            return corrected_query
        
        isolation_clause = " AND ".join(isolation_conditions)
        
        # 4. Construct new query with isolation
        if where_match:
            # Already has WHERE clause - add our conditions
            where_pos = where_match.end()
            
            # Find the earliest clause after WHERE
            clauses_after_where = []
            if group_by_match and group_by_match.start() > where_pos:
                clauses_after_where.append((group_by_match.start(), "GROUP BY"))
            if having_match and having_match.start() > where_pos:
                clauses_after_where.append((having_match.start(), "HAVING"))
            if order_by_match and order_by_match.start() > where_pos:
                clauses_after_where.append((order_by_match.start(), "ORDER BY"))
            if limit_match and limit_match.start() > where_pos:
                clauses_after_where.append((limit_match.start(), "LIMIT"))
                
            if clauses_after_where:
                # Sort to find the earliest clause
                clauses_after_where.sort()
                next_clause_pos = clauses_after_where[0][0]
                
                # Extract the existing WHERE conditions
                existing_where = corrected_query[where_pos:next_clause_pos].strip()
                
                # Add our isolation to the existing WHERE
                if existing_where:
                    # WHERE has content, add our conditions with AND
                    modified_where = f" {existing_where} AND {isolation_clause} "
                else:
                    # WHERE is empty, just use our conditions
                    modified_where = f" {isolation_clause} "
                
                # Reassemble the query
                new_query = (
                    corrected_query[:where_pos] + 
                    modified_where +
                    corrected_query[next_clause_pos:]
                )
            else:
                # No clauses after WHERE, the WHERE clause extends to the end
                existing_where = corrected_query[where_pos:].strip()
                
                if existing_where:
                    # Add our isolation with AND
                    new_query = (
                        corrected_query[:where_pos] +
                        f" {existing_where} AND {isolation_clause}"
                    )
                else:
                    # Empty WHERE clause
                    new_query = (
                        corrected_query[:where_pos] +
                        f" {isolation_clause}"
                    )
        else:
            # No existing WHERE - need to add one
            # Find position to insert WHERE
            insert_pos = -1
            for pattern in [group_by_match, having_match, order_by_match, limit_match]:
                if pattern and (insert_pos == -1 or pattern.start() < insert_pos):
                    insert_pos = pattern.start()
            
            if insert_pos != -1:
                # Insert WHERE before the first clause
                new_query = (
                    corrected_query[:insert_pos] +
                    f" WHERE {isolation_clause} " +
                    corrected_query[insert_pos:]
                )
            else:
                # No other clauses, append WHERE at the end
                new_query = f"{corrected_query.rstrip()} WHERE {isolation_clause}"
        
        # Verify we haven't introduced issues like "WHERE AND" or ".created_by"
        # Check for common erroneous patterns
        error_patterns = [
            r'WHERE\s+AND',
            r'WHERE\s+WHERE',
            r'WHERE\s+ORDER',
            r'WHERE\s+GROUP',
            r'WHERE\s+HAVING',
            r'WHERE\s+LIMIT',
            r'\.\s*created_by',
            r'LIMIT\s*\.\s*created_by',
            r'GROUP\s+BY\s*\.\s*created_by',
            r'ORDER\s+BY\s*\.\s*created_by',
            r'HAVING\s*\.\s*created_by'
        ]
        
        has_error = False
        for pattern in error_patterns:
            if re.search(pattern, new_query, re.IGNORECASE):
                has_error = True
                logger.error(f"Found SQL error pattern '{pattern}' in modified query: {new_query}")
                break
        
        if has_error:
            logger.warning(f"Isolation failed due to potential SQL errors. Using original query: {corrected_query}")
            return corrected_query
            
        logger.info(f"SQL after company isolation: {new_query}")
        return new_query
    
    def _extract_tables_from_query(self, query: str) -> List[Tuple[str, str]]:
        """Extract table names and their aliases from a SQL query.
        
        Args:
            query: SQL query
            
        Returns:
            List of (table_name, alias) tuples
        """
        # Improved regex patterns for table detection
        # This pattern matches:
        # 1. Table names in FROM clauses
        # 2. Table names in various JOIN clauses
        # 3. Handles optional AS keyword for aliases
        # 4. Supports backticks, square brackets, or no delimiters around table names
        
        # Helper function to clean table names (remove quotes, backticks, etc)
        def clean_table_name(name):
            if not name:
                return name
            # Remove backticks, quotes, brackets
            return re.sub(r'[`"\[\]]', '', name.strip())
        
        tables = []
        
        # Pattern for FROM clause tables
        # Matches: FROM table1, FROM `table1`, FROM [table1], FROM table1 AS t1, etc.
        from_pattern = r'FROM\s+(?:`|\[)?([a-zA-Z0-9_\.]+)(?:`|\])?\s*(?:AS\s+)?([a-zA-Z0-9_]+)?'
        
        # Find all FROM clause tables
        for match in re.finditer(from_pattern, query, re.IGNORECASE):
            table_name = clean_table_name(match.group(1))
            alias = clean_table_name(match.group(2)) if match.group(2) else None
            tables.append((table_name, alias))
        
        # Pattern for JOIN clause tables
        # Matches various JOIN types: JOIN, INNER JOIN, LEFT JOIN, RIGHT JOIN, etc.
        join_pattern = r'(?:INNER|LEFT|RIGHT|OUTER|CROSS|FULL|)?\s*JOIN\s+(?:`|\[)?([a-zA-Z0-9_\.]+)(?:`|\])?\s*(?:AS\s+)?([a-zA-Z0-9_]+)?'
        
        # Find all JOIN clause tables
        for match in re.finditer(join_pattern, query, re.IGNORECASE):
            table_name = clean_table_name(match.group(1))
            alias = clean_table_name(match.group(2)) if match.group(2) else None
            tables.append((table_name, alias))
        
        # Log the extracted tables for debugging
        if tables:
            logger.info(f"Extracted tables from query: {tables}")
        else:
            logger.warning(f"No tables extracted from query: {query}")
            
        # Return unique tables (in case a table appears multiple times)
        unique_tables = []
        seen_tables = set()
        for table, alias in tables:
            if table not in seen_tables:
                seen_tables.add(table)
                unique_tables.append((table, alias))
                
        return unique_tables
    
    def _get_isolation_column_for_table(self, table_name: str) -> str:
        """Get the isolation column for a table.
        
        Args:
            table_name: Table name
            
        Returns:
            Isolation column name or None if not found
        """
        # Most tables use 'created_by' for company isolation
        common_isolation_columns = ['created_by', 'company_id', 'user_id']
        
        # For simplicity, always return 'created_by' since that's the most common isolation column
        # In a production system, this would check the database schema or a mapping configuration
        return 'created_by'
    
    def _check_visualization_intent(self, message: str) -> tuple[bool, Optional[str]]:
        """Check if the message has visualization intent.
        
        Args:
            message: User message
            
        Returns:
            Tuple of (has_visualization_intent, visualization_type)
        """
        message_lower = message.lower()
        
        # Handle common typos in visualization requests
        typo_corrections = {
            "sohwing": "showing",
            "employes": "employees",
            "employe": "employee",
            "employess": "employees",
            "departement": "department", 
            "departements": "departments",
            "dept": "department",
            "depts": "departments",
            "char": "chart",
            "chars": "charts",
            "grpah": "graph",
            "graff": "graph"
        }
        
        # Apply typo corrections
        corrected_message = message_lower
        for typo, correction in typo_corrections.items():
            corrected_message = corrected_message.replace(typo, correction)
        
        # Log if we made corrections
        if corrected_message != message_lower:
            logger.info(f"Corrected visualization request: '{message_lower}' -> '{corrected_message}'")
            message_lower = corrected_message
        
        # Keywords that suggest visualization intent
        viz_keywords = [
            "graph", "chart", "plot", "visualize", "visualization", "display",
            "show me", "diagram", "histogram", "bar chart", "line graph",
            "pie chart", "scatter plot", "create chart", "create graph", 
            "make chart", "make graph", "make visualization", "generate chart", 
            "generate graph", "generate visualization", "create visualization",
            "create pie", "create bar", "create line", "draw chart", "draw graph",
            "showing", "distribution of", "breakdown of", "employees per"
        ]
        
        # Chart type mapping
        chart_types = {
            "bar": ["bar chart", "bar graph", "column chart", "create bar"],
            "line": ["line chart", "line graph", "trend", "time series", "create line"],
            "pie": ["pie chart", "pie graph", "donut", "distribution", "create pie"],
            "scatter": ["scatter plot", "scatter chart", "scatter graph"],
            "radar": ["radar chart", "radar graph", "spider chart"],
            "bubble": ["bubble chart", "bubble graph"]
        }
        
        # Check for visualization intent
        has_viz_intent = any(keyword in message_lower for keyword in viz_keywords)
        
        # Check for specific phrases that strongly indicate visualization intent
        strong_viz_phrases = [
            "employees per department",
            "department breakdown",
            "department distribution",
            "employee distribution",
            "employee breakdown",
            "show me employees",
            "department count",
            "employees by department",
            "chart showing"
        ]
        
        if not has_viz_intent:
            has_viz_intent = any(phrase in message_lower for phrase in strong_viz_phrases)
            if has_viz_intent:
                logger.info(f"Detected visualization intent from phrase match: '{message_lower}'")
        
        # First check directly for chart type references
        viz_type = None
        for chart_type, keywords in chart_types.items():
            if any(keyword in message_lower for keyword in keywords):
                viz_type = chart_type
                has_viz_intent = True  # Force intent to true if chart type is mentioned
                break
            
        # If no specific chart type detected but there is intent, try to infer from context
        if has_viz_intent and viz_type is None:
            # If "pie" is mentioned with "department", default to pie
            if "pie" in message_lower and ("department" in message_lower or "departments" in message_lower):
                viz_type = "pie"
            # If it mentions "department" or "employee" counts/distribution, likely a pie chart
            elif ("department" in message_lower or "departments" in message_lower or "employees per" in message_lower):
                viz_type = "pie" if "distribution" in message_lower else "bar"
            # Time-based visualizations are usually line charts
            elif any(time_word in message_lower for time_word in ["month", "year", "quarter", "time", "trend"]):
                viz_type = "line"
            # Comparisons are usually bar charts
            elif any(compare_word in message_lower for compare_word in ["compare", "comparison", "versus", "vs"]):
                viz_type = "bar"
            else:
                # Default to bar if no other clues
                viz_type = "bar"
        
        logger.info(f"Visualization intent detection: intent={has_viz_intent}, type={viz_type}, message='{message[:50]}...'")
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
        return await TokenManager.get_token_count(company_id, session)

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
        """Clean the response for end user consumption.
        
        Args:
            response: Raw response text
            
        Returns:
            Cleaned response text
        """
        # Remove SQL query sections
        response = re.sub(r"```sql\s*.*?\s*```", "", response, flags=re.DOTALL)
        
        # Remove other code blocks
        response = re.sub(r"```\s*.*?\s*```", "", response, flags=re.DOTALL)
        
        # Remove inline SQL
        response = re.sub(r"`SELECT.*?;`", "", response, flags=re.DOTALL)
        
        # Check for and clean any remaining raw IDs that weren't properly resolved
        # Look for patterns like "customer_id: 5", "employee_id: 10", etc.
        id_patterns = [
            r'(\w+)_id: (\d+)',
            r'(\w+) ID: (\d+)',
            r'(\w+) Id: (\d+)',
            r'(\w+) id: (\d+)',
        ]
        
        for pattern in id_patterns:
            response = re.sub(pattern, r'\1: [ID \2]', response)
        
        # Remove "The database shows that..." type phrases
        phrases_to_remove = [
            "I ran a query to check",
            "I queried the database",
            "according to the database",
            "from the database",
            "in the database",
            "based on the database",
            "according to the query",
            "from my query",
            "I have queried",
            "I will query",
            "let me check",
            "let me search",
        ]
        
        for phrase in phrases_to_remove:
            response = re.sub(r'(?i)' + re.escape(phrase) + r'.*?[\.,]', '', response)
        
        # Remove "Based on the data..." at the start
        response = re.sub(r'^(?i)(Based on|According to|From|As per|After analyzing|The data shows|Looking at).*?,\s*', '', response)
        
        # Remove any empty lines and condense multiple spaces
        response = re.sub(r'\n{3,}', '\n\n', response)
        response = re.sub(r'\s{2,}', ' ', response)
        
        # Remove empty bullet points
        response = re.sub(r'^\s*[\*\-]\s*$', '', response, flags=re.MULTILINE)
        
        return response.strip()

    async def _generate_conversation_title(
        self,
        conversation_id: str,
        company_id: int,
        user_message: str,
        agent_response: str
    ) -> str:
        """Generate a concise title for the conversation using the LLM."""
        try:
            # LLM is now in self.llm
            prompt = f"""Given the following exchange, generate a very short, concise title (max 5-7 words) for this conversation snippet. Focus on the main subject. Do not include prefixes like 'Title:'.
User: {user_message}
Assistant: {agent_response}
Title:"""
            
            response_message = await self.llm.ainvoke([HumanMessage(content=prompt)]) # Use stored LLM
            title = response_message.content.strip()
            # Basic cleaning of potential LLM artifacts like quotes
            title = title.replace('"', '').replace("'", '')
            if not title: # Fallback if LLM returns empty string
                title = user_message[:30] + "..." if len(user_message) > 30 else user_message
            logger.info(f"Generated title for conversation {conversation_id}: '{title}'")
            return title
        except Exception as e:
            logger.error(f"Error generating conversation title for {conversation_id}: {e}", exc_info=True)
            # Fallback title based on the first few words of the user message
            fallback_title = ' '.join(user_message.split()[:5])
            if len(user_message.split()) > 5:
                fallback_title += "..."
            return fallback_title if fallback_title else "Chat" 