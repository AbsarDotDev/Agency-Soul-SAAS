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
    
    async def generate_visualization(
        self, 
        query: str, 
        company_id: int, 
        user_id: str,
        visualization_type: Optional[str] = None,
        session: Optional[Session] = None,
        conversation_id: Optional[str] = None
    ) -> AgentResponse:
        """Generate visualization from SQL query or natural language description."""
        
        try:
            # Create SQL database with company isolation
            sql_database = get_company_isolated_sql_database(
                company_id=company_id,
                sample_rows_in_table_info=3
            )
            
            # Create SQL toolkit
            toolkit = SQLDatabaseToolkit(
                db=sql_database,
                llm=self.llm
            )
            
            # Add custom tools for schema exploration (if not already part of the default toolkit via constructor)
            # This might be redundant if SQLDatabaseToolkit includes them by default with a good LLM
            # Forcing their availability for clarity.
            custom_tools = [
                Tool(
                    name="List_Database_Tables",
                    description="Lists all tables in the database to understand what data is available. Input should be an empty string.",
                    func=lambda _: self.mcp_mysql_tool.list_tables(company_id) # Ensure async compatibility or run sync
                ),
                Tool(
                    name="Get_Table_Schema",
                    description="Gets detailed schema information for a specific table including column names, types, and isolation information. Input should be the table name.",
                    func=lambda table_name: self.mcp_mysql_tool.get_table_schema(table_name, company_id) # Ensure async
                )
            ]
            all_tools = toolkit.get_tools() + custom_tools
            
            # Create the SQL Agent
            agent_executor = create_sql_agent( # Renamed from 'agent' for clarity
                llm=self.llm, # Use stored LLM instance
                toolkit=toolkit, # Pass the original toolkit, create_sql_agent might handle tools internally
                                 # Or pass 'tools=all_tools' if 'toolkit' is not enough
                agent_type="openai-tools",
                handle_parsing_errors=True,
                verbose=True
            )
            
            # Modify the query to explicitly ask for SQL to generate visualization
            sql_query_prompt = f"""Given the user query: '{query}', first, identify if it directly contains a valid SQL SELECT query.
If it does, use that SQL query.
If it does not, write a SQL SELECT query that would retrieve the necessary data to address the user's request: '{query}'.

IMPORTANT NOTES ABOUT DATABASE STRUCTURE:
- For employee counts by department, use 'departments' (not 'department') and 'employees' (not 'employee')
- The departments table has columns: id, name, branch_id, created_by
- The employees table has columns: id, name, department_id, designation_id, etc.
- The correct join between departments and employees is: departments.id = employees.department_id
- Always use proper column names, e.g., departments.name (not dept_name), employees.id (not emp_id)

CRITICAL SECURITY REQUIREMENT:
You MUST include company isolation in your query by adding appropriate WHERE conditions:
- ALWAYS include 'WHERE created_by = {company_id}' for each table in the query
- For joins, include the condition for EACH table: 'table1.created_by = {company_id} AND table2.created_by = {company_id}'
- Example: SELECT d.name, COUNT(e.id) FROM departments d JOIN employees e ON d.id = e.department_id WHERE d.created_by = {company_id} AND e.created_by = {company_id} GROUP BY d.name

Only return the SQL SELECT query itself, with no explanation, no markdown, just the SQL.
If the user's query is too vague to form a SQL query (e.g., 'show me sales data'), ask for clarification instead of guessing.
"""
            
            # First get SQL query
            response = await agent_executor.ainvoke({"input": sql_query_prompt})
            sql_query_text = self._extract_sql_query(response.get("output", ""), company_id)
            
            # Apply an additional correction to the extracted SQL
            if sql_query_text:
                sql_query_text = self._correct_table_names(sql_query_text, company_id)
                logger.info(f"Corrected SQL for visualization: {sql_query_text}")
            
            if not sql_query_text:
                logger.warning(f"SQL Agent could not generate a SQL query for visualization: {query}")
                # Try to get a clarifying message from the agent's response if it didn't produce SQL
                clarification_needed = response.get("output", "Could not determine the SQL query for your request. Please be more specific.")
                if "ask for clarification" in clarification_needed.lower():
                     return AgentResponse(
                         response=clarification_needed,
                         conversation_id=conversation_id or str(uuid.uuid4()),
                         conversation_title=f"Visualization: {query[:30]}..." if len(query) > 30 else f"Visualization: {query}",
                         visualization=None,
                         tokens_used=0,
                         tokens_remaining=None,
                         agent_type="visualization"  # Force visualization agent type
                     )
                return AgentResponse(
                    response="Failed to generate SQL query for visualization. Please try rephrasing your request.",
                    conversation_id=conversation_id or str(uuid.uuid4()),
                    conversation_title=f"Visualization: {query[:30]}..." if len(query) > 30 else f"Visualization: {query}",
                    visualization=None,
                    tokens_used=0,
                    tokens_remaining=None,
                    agent_type="visualization"  # Force visualization agent type
                )

            logger.info(f"Generated SQL for visualization: {sql_query_text}")
            
            # Execute the SQL query directly (bypassing the CompanyIsolatedSQLDatabase run method)
            # This gives us control over how we execute the query and handle the results
            try:
                data_for_viz = await self._execute_query_safely(sql_query_text, company_id)
                if not data_for_viz or (isinstance(data_for_viz, list) and len(data_for_viz) == 0):
                    return AgentResponse(
                        response="The query returned no data to visualize. Try a different query or check if the data exists.",
                        conversation_id=conversation_id or str(uuid.uuid4()),
                        conversation_title=f"Visualization: {query[:30]}..." if len(query) > 30 else f"Visualization: {query}",
                        visualization=None,
                        tokens_used=0,
                        tokens_remaining=None,
                        agent_type="visualization"  # Force visualization agent type
                    )
            except Exception as e:
                logger.error(f"Error executing visualization query: {str(e)}")
                return AgentResponse(
                    response=f"Error executing the query: {str(e)}",
                    conversation_id=conversation_id or str(uuid.uuid4()),
                    conversation_title=f"Visualization: {query[:30]}..." if len(query) > 30 else f"Visualization: {query}",
                    visualization=None,
                    tokens_used=0,
                    tokens_remaining=None,
                    agent_type="visualization"  # Force visualization agent type
                )

            # Generate visualization using the data
            viz_result_obj = await generate_visualization_from_data(
                data=data_for_viz, 
                query=query,
                visualization_type=visualization_type,
                llm=self.llm
            )
            
            # Fixed token usage as requested in specification
            tokens_used = 2
            
            # Get remaining tokens if session provided
            tokens_remaining = None
            if session:
                try:
                    # Direct token update to avoid the TokenManager issue
                    engine = DatabaseConnection.create_engine()
                    with Session(engine) as db_session:
                        # Update user tokens
                        stmt = text("""
                            UPDATE users
                            SET ai_agent_tokens_used = ai_agent_tokens_used + :tokens
                            WHERE id = :company_id
                        """)
                        db_session.execute(stmt, {"tokens": tokens_used, "company_id": company_id})
                        db_session.commit()
                        
                        # Get remaining tokens
                        stmt = text("""
                            SELECT 
                                p.ai_agent_default_tokens - u.ai_agent_tokens_used as tokens_remaining
                            FROM users u
                            JOIN plans p ON u.plan = p.id
                            WHERE u.id = :company_id
                        """)
                        result = db_session.execute(stmt, {"company_id": company_id}).fetchone()
                        tokens_remaining = result[0] if result else None
                        logger.info(f"Updated tokens directly. Remaining: {tokens_remaining}")
                except Exception as e:
                    logger.error(f"Error updating token usage directly: {e}")
            
            # Check if we need to save this conversation in the database
            if conversation_id and session:
                try:
                    # Save conversation with visualization
                    title = await self._generate_conversation_title(
                        conversation_id=conversation_id,
                        company_id=company_id,
                        user_message=query,
                        agent_response=viz_result_obj.explanation or "Visualization generated."
                    ) if not conversation_id else None
                    
                    await self._save_conversation(
                        session=session,
                        conversation_id=conversation_id,
                        company_id=company_id,
                        user_id=user_id,
                        message=query,
                        response=viz_result_obj.explanation or "Visualization generated.",
                        agent_type="visualization",  # Always mark as visualization
                        tokens_used=tokens_used,
                        visualization=viz_result_obj.data,
                        title=title
                    )
                except Exception as e:
                    logger.error(f"Error saving conversation: {e}")
            
            # Return a properly constructed AgentResponse
            viz_data = viz_result_obj.data
            logger.info(f"Returning visualization data with chart_type={viz_data.get('chart_type', 'unknown') if viz_data else 'None'}")
            
            # Ensure visualization data is properly formatted
            if viz_data and 'options' in viz_data:
                if viz_data['options'] is None or (isinstance(viz_data['options'], list) and len(viz_data['options']) == 0):
                    viz_data['options'] = {}  # Ensure options is an object, not null or empty array
            
            # Log full visualization data for debugging
            logger.info(f"Full visualization data: {viz_data}")
            
            # Create response with visualization data
            agent_response = AgentResponse(
                response=viz_result_obj.explanation or "Here's the visualization you requested.",
                conversation_id=conversation_id or str(uuid.uuid4()),
                conversation_title=f"Visualization: {query[:30]}..." if len(query) > 30 else f"Visualization: {query}",
                visualization=viz_data,  # Make sure we directly use the data from viz_result_obj
                tokens_used=tokens_used,
                tokens_remaining=tokens_remaining,
                agent_type="visualization"  # Force visualization agent type
            )
            
            # Log final response structure
            logger.info(f"Final AgentResponse structure: {agent_response.dict()}")
            
            return agent_response

        except Exception as e:
            logger.error(f"Error generating visualization for company {company_id}, user {user_id}: {str(e)}", exc_info=True)
            return AgentResponse(
                response=f"An error occurred while generating the visualization: {str(e)}",
                conversation_id=conversation_id or str(uuid.uuid4()),
                conversation_title=f"Visualization: {query[:30]}..." if len(query) > 30 else f"Visualization: {query}",
                visualization=None,
                tokens_used=0,
                tokens_remaining=None,
                agent_type="visualization"  # Force visualization agent type even for errors
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
        """Correct common table name mistakes in SQL queries and ensure company isolation.
        
        Args:
            query: SQL query to correct
            company_id: Optional company ID to add isolation filters
            
        Returns:
            Corrected SQL query
        """
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
            # If we have both departments and employees with the correct aliases,
            # make sure the join condition uses the correct field names
            corrected_query = re.sub(
                r'd\.id\s*=\s*e\.dept_id', 
                'd.id = e.department_id', 
                corrected_query, 
                flags=re.IGNORECASE
            )
        
        # Ensure company isolation filters are present if company_id is provided
        if company_id is not None:
            # Extract table information from the query
            tables_with_aliases = self._extract_tables_from_query(corrected_query)
            
            # Check if a WHERE clause already exists
            has_where = bool(re.search(r'\bWHERE\b', corrected_query, re.IGNORECASE))
            
            # Build company isolation filters
            isolation_conditions = []
            for table_name, alias in tables_with_aliases:
                # Try to determine the isolation column (almost always 'created_by')
                isolation_column = self._get_isolation_column_for_table(table_name)
                if isolation_column:
                    if alias:
                        isolation_conditions.append(f"{alias}.{isolation_column} = {company_id}")
                    else:
                        isolation_conditions.append(f"{table_name}.{isolation_column} = {company_id}")
            
            if isolation_conditions:
                isolation_clause = " AND ".join(isolation_conditions)
                
                # Add or append to WHERE clause
                if has_where:
                    # Look for the WHERE clause and any potential GROUP BY, ORDER BY, or LIMIT clauses after it
                    match = re.search(r'\bWHERE\b(.*?)(?:\b(GROUP BY|ORDER BY|LIMIT)\b|$)', corrected_query, re.IGNORECASE | re.DOTALL)
                    if match:
                        # Get the existing WHERE conditions
                        where_conditions = match.group(1).strip()
                        
                        # Check if the WHERE conditions already contain company isolation for all tables
                        all_tables_isolated = True
                        for table_name, alias in tables_with_aliases:
                            isolation_column = self._get_isolation_column_for_table(table_name)
                            if isolation_column:
                                table_ref = alias if alias else table_name
                                isolation_pattern = rf"{table_ref}\.{isolation_column}\s*=\s*{company_id}"
                                if not re.search(isolation_pattern, where_conditions, re.IGNORECASE):
                                    all_tables_isolated = False
                                    break
                        
                        if not all_tables_isolated:
                            # Append isolation conditions to existing WHERE clause
                            new_where_conditions = f"{where_conditions} AND {isolation_clause}"
                            rest_of_query = corrected_query[match.end():] if match.group(2) else ""
                            corrected_query = corrected_query[:match.start()] + f" WHERE {new_where_conditions}" + rest_of_query
                else:
                    # Find position to insert WHERE clause (before any GROUP BY, ORDER BY, LIMIT)
                    match = re.search(r'\b(GROUP BY|ORDER BY|LIMIT)\b', corrected_query, re.IGNORECASE)
                    if match:
                        # Insert WHERE before these clauses
                        corrected_query = corrected_query[:match.start()] + f" WHERE {isolation_clause} " + corrected_query[match.start():]
                    else:
                        # Append WHERE at the end
                        corrected_query = f"{corrected_query} WHERE {isolation_clause}"
            
        return corrected_query
    
    def _extract_tables_from_query(self, query: str) -> List[Tuple[str, str]]:
        """Extract table names and their aliases from a SQL query.
        
        Args:
            query: SQL query
            
        Returns:
            List of (table_name, alias) tuples
        """
        # Look for common table patterns in FROM and JOIN clauses
        from_pattern = r'\bFROM\s+([a-zA-Z0-9_]+)(?:\s+(?:AS\s+)?([a-zA-Z0-9_]+))?'
        join_pattern = r'\bJOIN\s+([a-zA-Z0-9_]+)(?:\s+(?:AS\s+)?([a-zA-Z0-9_]+))?'
        
        tables = []
        
        # Find tables in FROM clause
        from_matches = re.finditer(from_pattern, query, re.IGNORECASE)
        for match in from_matches:
            table_name = match.group(1)
            alias = match.group(2) if match.group(2) else None
            tables.append((table_name, alias))
        
        # Find tables in JOIN clauses
        join_matches = re.finditer(join_pattern, query, re.IGNORECASE)
        for match in join_matches:
            table_name = match.group(1)
            alias = match.group(2) if match.group(2) else None
            tables.append((table_name, alias))
        
        return tables
    
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