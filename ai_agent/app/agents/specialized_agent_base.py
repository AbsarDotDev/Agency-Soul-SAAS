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

class SpecializedAgentBase(BaseAgent):
    """Base class for all specialized domain agents using SQL Database querying."""
    
    def __init__(self, agent_type: str, relevant_tables: List[str], fallback_to_sql: bool = True):
        """Initialize specialized agent base.
        
        Args:
            agent_type: Type of agent (e.g., 'hrm', 'finance', etc.)
            relevant_tables: List of database tables relevant to this agent
            fallback_to_sql: Whether to fall back to SQL agent if needed
        """
        super().__init__()
        self.type = agent_type
        self.relevant_tables = relevant_tables
        self.fallback_to_sql = fallback_to_sql
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
        """Process domain-specific message through database queries.
        
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
        tokens_used = 0
        
        try:
            # Generate new conversation ID if not provided
            if not conversation_id:
                is_new_conversation = True
                conversation_id = str(uuid.uuid4())
            
            # Create SQL database with company isolation
            sql_database = get_company_isolated_sql_database(
                company_id=company_id,
                sample_rows_in_table_info=3,
                include_tables=self.relevant_tables
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
            
            # Generate domain-specific system message with company context and SQL safety
            isolation_instructions = self._format_isolation_instructions(isolation_info, company_id)
            system_message = self._generate_system_prompt(company_id, isolation_instructions)
            
            # Create custom tools for this specialized agent
            list_tables_tool = Tool(
                name="List_Database_Tables",
                description=f"Lists all tables in the database relevant to {self.type} to understand what data is available",
                func=lambda _: self._list_tables(company_id)
            )
            
            get_table_schema_tool = Tool(
                name="Get_Table_Schema",
                description="Gets detailed schema information for a specific table including column names, types, and isolation information",
                func=lambda table_name: self._get_table_schema(table_name, company_id)
            )
            
            # Add custom tools to the toolkit
            all_tools = toolkit.get_tools()
            all_tools.extend([list_tables_tool, get_table_schema_tool])
            
            # Check for remaining tokens
            remaining_tokens = None
            if session:
                remaining_tokens = await self._get_remaining_tokens(session, company_id)
                if remaining_tokens is not None and remaining_tokens <= 0:
                    return AgentResponse(
                        response="Sorry, you have used all your available tokens for AI interactions. Please contact your administrator to get more tokens assigned.",
                        conversation_id=conversation_id,
                        agent_type=self.type
                    )
            
            # Set up the agent with higher verbosity and improved settings
            agent_executor = create_sql_agent(
                llm=self.llm,
                toolkit=toolkit,
                verbose=True,
                agent_type="openai-tools",
                top_k=20,  # Increased for better context
                prefix=system_message,
                handle_parsing_errors=True  # Better error handling
            )
            
            # Add reminder to agent's execution wrapper about response formatting
            response_format_reminder = """
After you have the query results, please format your answer in a natural, conversational way.
Do not include SQL syntax or technical database terminology in your response.
Focus on presenting the information clearly and in a business-friendly manner.
Never say you are running a query or searching the database - just provide the information.

VERY IMPORTANT ID RESOLUTION REMINDER:
For ANY and ALL IDs returned in query results (e.g., customer_id, employee_id, product_id, etc.):
1. You MUST run follow-up queries to get the corresponding names
2. Example: If you see "customer_id: 5", run: SELECT name FROM customers WHERE id = 5 AND created_by = [company_id]
3. Then replace all IDs with the actual names in your final response
4. DO NOT show raw IDs to the user - always translate them to names
5. For status values (0, 1, etc.), convert them to descriptive text (e.g., "Inactive", "Active")
"""
            
            # Execute the agent to process the message with the chat history
            chat_history = memory.chat_memory.messages if memory else []
            result = await agent_executor.ainvoke({
                "input": message + "\n\n" + response_format_reminder,
                "chat_history": chat_history
            })
            
            # Log the raw response
            logger.debug(f"Raw agent response: {result.get('output', '')[:200]}...")
            
            # Clean up the response for end-user consumption
            response_text = result.get("output", "")
            response_text = self._clean_response_for_end_user(response_text)
            
            # Apply a secondary response clean pass using LLM if needed
            if "query" in response_text.lower() or "database" in response_text.lower() or response_text.lower().startswith("select"):
                clean_response = await self._get_clean_response_from_llm(response_text, message)
                if clean_response and len(clean_response) > 20:
                    response_text = clean_response
            
            # Save conversation
            if session:
                try:
                    title = await self._save_conversation(
                        session=session,
                        conversation_id=conversation_id,
                        company_id=company_id,
                        user_id=user_id,
                        message=message,
                        response=response_text,
                        agent_type=self.type
                    )
                    
                    # Decrement token usage (simplified token counting for now)
                    if remaining_tokens is not None:
                        token_manager = TokenManager()
                        tokens_used = 1  # Default simplified token count
                        await token_manager.decrement_tokens(session, company_id, tokens_used)
                        logger.info(f"Tokens updated. Remaining for company {company_id}: {remaining_tokens - tokens_used}")
                except Exception as e:
                    logger.error(f"Error saving conversation: {str(e)}")
            
            return AgentResponse(
                response=response_text,
                conversation_id=conversation_id,
                agent_type=self.type,
                tokens_used=tokens_used
            )
        except Exception as e:
            logger.error(f"Error in {self.type} agent: {str(e)}", exc_info=True)
            
            # Try to fall back to SQL agent if enabled
            if self.fallback_to_sql:
                try:
                    from app.agents.sql_agent import SQLAgent
                    logger.info(f"Falling back to SQL agent for query: {message}")
                    sql_agent = SQLAgent()
                    response = await sql_agent.process_message(
                        message=message,
                        company_id=company_id,
                        user_id=user_id,
                        conversation_id=conversation_id,
                        session=session
                    )
                    # Add a note that this was handled by the SQL agent
                    response.response = f"{response.response}\n\n(Answered by SQL assistant due to specialized agent error)"
                    return response
                except Exception as fallback_error:
                    logger.error(f"Error in SQL agent fallback: {str(fallback_error)}", exc_info=True)
            
            # Return error message if all else fails
            return AgentResponse(
                response=f"I apologize, but I encountered an error processing your request related to {self.type}. Please try again with a different question.",
                conversation_id=conversation_id,
                agent_type=self.type
            )
    
    def _generate_system_prompt(self, company_id: int, isolation_instructions: str) -> str:
        """Generate domain-specific system prompt.
        
        Args:
            company_id: Company ID
            isolation_instructions: Isolation instructions based on table columns
            
        Returns:
            System prompt
        """
        # This is meant to be overridden by each specialized agent
        return f"""You are a friendly, helpful AI assistant working with a company's database.
You help users query data and analyze their business information in natural, conversational language.

DATA ISOLATION CRITICAL RULE:
ALWAYS include WHERE created_by = {company_id} or equivalent company isolation conditions in ALL your queries for ALL tables involved.
This is required for security, no exceptions.

{isolation_instructions}

ID RESOLUTION - ALWAYS FOLLOW THIS RULE:
When your query returns IDs (like customer_id, employee_id, warehouse_id, etc.), ALWAYS perform additional queries to resolve these IDs into human-readable names.
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

IMPORTANT GUIDELINES:
1. NEVER make up information. Only return data that actually exists in the database.
2. If you don't find relevant data, clearly say so rather than making up a response.
3. Present your answers in a clear, concise way for business users.
4. NEVER reveal SQL queries to the end user - only show the information they asked for.
5. Format your responses in a readable way, with proper capitalization and punctuation.
6. For numeric data, use appropriate formatting (currency, percentages, etc.)
7. For dates, use a readable format (e.g., "January 15, 2025" instead of "2025-01-15")

USE DATABASE TOOLS ALWAYS:
- When asked for information, ALWAYS search the database for real data.
- Never fabricate information or say "I would need to search the database" - actually search it.
- Use SQL queries to get actual data for every information request.
- If data doesn't exist, clearly state that no matching records were found.

Today's date is {self._get_current_date()}
"""

    async def generate_visualization(
        self, 
        query: str, 
        company_id: int, 
        user_id: str,
        visualization_type: Optional[str] = None,
        session: Optional[Session] = None
    ) -> VisualizationResult:
        """Generate visualization from domain-specific data.
        
        Args:
            query: Visualization query
            company_id: Company ID
            user_id: User ID
            visualization_type: Optional visualization type
            session: Optional database session
            
        Returns:
            Visualization result
        """
        # Check for visualization intent
        has_viz_intent, detected_viz_type = self._check_visualization_intent(query)
        if not has_viz_intent:
            return VisualizationResult(
                data={
                    "error": "No visualization intent detected"
                },
                explanation="Your query doesn't seem to be asking for a visualization. Please try again with a query that specifically asks for a chart, graph, or visualization."
            )
        
        # Use detected visualization type if not specified
        if not visualization_type and detected_viz_type:
            visualization_type = detected_viz_type
        
        try:
            # Get SQL database with proper isolation
            sql_database = get_company_isolated_sql_database(
                company_id=company_id,
                include_tables=self.relevant_tables
            )
            
            # Generate SQL query from natural language
            system_context = f"""
You are a data analyst working with SQL to extract data for visualizations.
You need to convert natural language queries into SQL. The query should be focused on {self.type} data.

IMPORTANT DATABASE CONSTRAINTS:
1. Always include WHERE created_by = {company_id} in ALL your SQL queries
2. Only query tables relevant to {self.type}: {', '.join(self.relevant_tables)}
3. Make sure your query returns data in a format suitable for visualization
4. Include appropriate GROUP BY clauses for aggregate data
5. Include ORDER BY for time series or ranked data
6. Limit the result set to a reasonable size (e.g., top 10) if appropriate

Just output the SQL query, nothing else. Do not include any explanations.
"""
            sql_query_response = await self.llm.ainvoke([
                SystemMessage(content=system_context),
                HumanMessage(content=f"Create a SQL query to visualize: {query}")
            ])
            sql_query = sql_query_response.content.strip()
            
            # Extract just the SQL if the response has additional text
            sql_query = self._extract_sql_query(sql_query)
            if not sql_query:
                return VisualizationResult(
                    data={"error": "Could not generate a valid SQL query for visualization"},
                    explanation="I couldn't create a visualization because I couldn't formulate a proper database query for your request."
                )
            
            try:
                # Execute the query
                raw_data = sql_database.run(sql_query)
                
                if not raw_data:
                    return VisualizationResult(
                        data={"error": "No data found", "chart_type": visualization_type or "bar"},
                        explanation="I couldn't create the visualization because no data was found that matches your request."
                    )
                
                # Generate visualization from the data
                return await generate_visualization_from_data(
                    data=raw_data,
                    query=query,
                    visualization_type=visualization_type,
                    llm=self.llm
                )
                
            except Exception as query_error:
                logger.error(f"Error executing visualization query: {str(query_error)}\nQuery: {sql_query}")
                return VisualizationResult(
                    data={"error": "Database query error", "chart_type": visualization_type or "bar"},
                    explanation=f"I encountered an error while trying to retrieve data for your visualization: {str(query_error)}"
                )
                
        except Exception as e:
            logger.error(f"Error generating visualization: {str(e)}", exc_info=True)
            return VisualizationResult(
                data={"error": "Visualization generation error", "chart_type": visualization_type or "bar"},
                explanation=f"I encountered an error while trying to create your visualization: {str(e)}"
            )
    
    async def perform_action(
        self, 
        action: str, 
        parameters: Dict[str, Any], 
        company_id: int, 
        user_id: str,
        session: Optional[Session] = None
    ) -> ActionResult:
        """Perform domain-specific action.
        
        Args:
            action: Action to perform
            parameters: Action parameters
            company_id: Company ID
            user_id: User ID
            session: Optional database session
            
        Returns:
            Action result
        """
        # No actions implemented by default - read only access!
        return ActionResult(
            success=False,
            message=f"Data modification actions are not supported in {self.type}. The agent can only perform read-only operations.",
            data={"supported_actions": ["read_only_queries", "generate_visualization"]}
        )
    
    def _extract_sql_query(self, text: str) -> Optional[str]:
        """Extract SQL query from text that may contain other content.
        
        Args:
            text: Text that may contain SQL
            
        Returns:
            Extracted SQL query or None if not found
        """
        # Check for SQL within triple backticks
        sql_pattern = r"```sql\s*(.*?)\s*```"
        matches = re.findall(sql_pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
        
        # Check for SQL within single backticks
        sql_pattern = r"`(.*?)`"
        matches = re.findall(sql_pattern, text, re.DOTALL)
        if matches:
            for match in matches:
                if match.strip().upper().startswith("SELECT") or match.strip().upper().startswith("WITH"):
                    return match.strip()
        
        # If the whole text looks like SQL, return it
        if text.strip().upper().startswith("SELECT") or text.strip().upper().startswith("WITH"):
            return text.strip()
        
        return None

    def _check_visualization_intent(self, message: str) -> tuple[bool, Optional[str]]:
        """Check if a message has visualization intent and detect desired type.
        
        Args:
            message: User message
            
        Returns:
            Tuple of (has_intent, visualization_type)
        """
        message = message.lower()
        
        # Check for visualization keywords
        viz_keywords = ["chart", "graph", "plot", "visualization", "visualize", "visualise", "dashboard", "diagram"]
        has_viz_intent = any(keyword in message for keyword in viz_keywords)
        
        # Detect visualization type
        viz_type = None
        if "bar" in message or "column" in message:
            viz_type = "bar"
        elif "line" in message:
            viz_type = "line"
        elif "pie" in message or "distribution" in message:
            viz_type = "pie"
        elif "scatter" in message:
            viz_type = "scatter"
        elif "area" in message:
            viz_type = "area"
        elif "radar" in message:
            viz_type = "radar"
        elif "bubble" in message:
            viz_type = "bubble"
        elif "heatmap" in message:
            viz_type = "heatmap"
        
        return has_viz_intent, viz_type
    
    async def _get_conversation_history(
        self,
        conversation_id: str,
        company_id: int,
        session: Session
    ) -> List[Dict[str, Any]]:
        """Get conversation history for agent.
        
        Args:
            conversation_id: Conversation ID
            company_id: Company ID
            session: Database session
            
        Returns:
            List of conversation messages
        """
        return await super()._get_conversation_history(
            conversation_id=conversation_id,
            company_id=company_id,
            session=session
        )
    
    async def _get_remaining_tokens(self, session: Session, company_id: int) -> Optional[int]:
        """Get remaining tokens for company.
        
        Args:
            session: Database session
            company_id: Company ID
            
        Returns:
            Remaining tokens or None if not applicable
        """
        return await TokenManager.get_token_count(company_id, session)
    
    async def _get_isolation_columns_info(self, company_id: int) -> Dict[str, str]:
        """Get information about isolation columns for tables.
        
        Args:
            company_id: Company ID
            
        Returns:
            Dictionary of table names to isolation column names
        """
        isolation_info = {}
        engine = DatabaseConnection.create_engine()
        for table_name in self.relevant_tables:
            isolation_column = get_company_isolation_column(table_name, engine)
            if isolation_column:
                isolation_info[table_name] = isolation_column
        return isolation_info
    
    def _format_isolation_instructions(self, isolation_info: Dict[str, str], company_id: int) -> str:
        """Format isolation instructions for the agent.
        
        Args:
            isolation_info: Dictionary of table names to isolation column names
            company_id: Company ID
            
        Returns:
            Formatted isolation instructions
        """
        if not isolation_info:
            return ""
        
        instructions = "TABLE ISOLATION REQUIREMENTS:\n"
        for table, column in isolation_info.items():
            instructions += f"- For table `{table}`, always include: WHERE `{column}` = {company_id}\n"
        
        instructions += "\nEXAMPLE QUERIES:\n"
        example_tables = list(isolation_info.keys())[:3]  # Take up to 3 tables for examples
        for table in example_tables:
            column = isolation_info[table]
            instructions += f"SELECT * FROM `{table}` WHERE `{column}` = {company_id};\n"
        
        return instructions
    
    def _clean_response_for_end_user(self, response: str) -> str:
        """Clean response text for end user display.
        
        Args:
            response: Raw response text
            
        Returns:
            Cleaned response text
        """
        # Remove SQL query sections
        sql_pattern = r"```sql\s*.*?\s*```"
        response = re.sub(sql_pattern, "", response, flags=re.DOTALL)
        
        # Remove other code blocks
        code_pattern = r"```\s*.*?\s*```"
        response = re.sub(code_pattern, "", response, flags=re.DOTALL)
        
        # Remove inline SQL
        inline_sql_pattern = r"`SELECT.*?;`"
        response = re.sub(inline_sql_pattern, "", response, flags=re.DOTALL)
        
        # Remove "This is the SQL query I'll use" type phrases
        query_phrases = [
            r"(?i)Here'?s the SQL query I'?ll use:.*?\n",
            r"(?i)I'?ll use the following SQL query:.*?\n",
            r"(?i)I need to run a SQL query .*?\n",
            r"(?i)Let me query the database .*?\n",
            r"(?i)I'll execute this query:.*?\n",
            r"(?i)To answer this question, I'll need to query the .*?\n",
            r"(?i)I'll need to check the database.*?\n",
            r"(?i)I can check this by querying.*?\n",
            r"(?i)Let me search for.*? in the database.*?\n",
            r"(?i)I'll look up.*? in the database.*?\n",
            r"(?i)First, I need to.*?\n",
            r"(?i)Let me execute a query to.*?\n",
            r"(?i)I'll retrieve the information from.*?\n",
            r"(?i)Looking at the database.*?\n",
        ]
        for pattern in query_phrases:
            response = re.sub(pattern, "", response)
        
        # Remove entire sentences that contain technical references
        technical_phrases = [
            "query the database",
            "SQL query",
            "database query",
            "execute a query",
            "run a query",
            "queried the",
            "database shows",
            "according to the database",
            "from the database",
            "database records",
            "database results",
            "results from the query",
            "query results",
            "query returned",
            "tables in the database",
            "found in the database"
        ]
        
        # Remove sentences containing these phrases
        for phrase in technical_phrases:
            pattern = r'(?i)[^.!?]*' + re.escape(phrase) + r'[^.!?]*[.!?]'
            response = re.sub(pattern, '', response)
        
        # Remove phrases like "Based on the data" at the start of the response
        response = re.sub(r"(?i)^(Based on|According to|From|As per|Looking at|After analyzing|From the|The data shows).*?, ?", "", response)
        
        # Consolidate multiple newlines and spaces
        response = re.sub(r"\n{3,}", "\n\n", response)
        response = re.sub(r"\s{2,}", " ", response)
        
        # Remove any empty bullets or list markers
        response = re.sub(r"^[\*\-\+]\s*$", "", response, flags=re.MULTILINE)
        
        # If response is empty after cleaning, return a generic response
        if not response.strip():
            return f"I couldn't find any {self.type} data matching your query."
            
        return response.strip()
    
    async def _get_clean_response_from_llm(self, response: str, original_query: str) -> str:
        """Use LLM to clean the response and remove any remaining technical language.
        
        Args:
            response: The response to clean
            original_query: The original user query
            
        Returns:
            Cleaned response
        """
        try:
            context = f"""
Original user question: {original_query}

Raw response with technical language: {response}

Your task is to rewrite the raw response above to:
1. Remove all references to database operations, SQL, queries, etc.
2. Reformat into clear, direct natural language with a conversational tone
3. Keep all the actual information/data from the response
4. Maintain the same meaning and content
5. Preserve any numbers, statistics, or dates
6. Start with a direct answer, not "Based on the data..." or similar phrases

VERY IMPORTANT: If you see any raw IDs in the response (like customer_id: 5, employee_id: 10, etc.), 
replace them with more human-readable placeholders (like "Customer: [ID 5]", "Employee: [ID 10]"). 
This indicates that these IDs should have been resolved to actual names.

For status codes (like status: 0, status: 1), convert them to probable meanings (like "Status: Inactive", "Status: Active").

DO NOT add any new information not in the original response.
Just reformulate the EXACT SAME information in user-friendly language.
"""
            clean_messages = [
                SystemMessage(content="You are a helpful assistant that reformats technical responses into natural language."),
                HumanMessage(content=context)
            ]
            clean_response = await self.llm.ainvoke(clean_messages)
            return clean_response.content.strip()
        except Exception as e:
            logger.error(f"Error cleaning response with LLM: {str(e)}")
            # If there's an error, return the original response
            return response
    
    def _list_tables(self, company_id: int) -> List[str]:
        """List tables relevant to this agent.
        
        Args:
            company_id: Company ID
            
        Returns:
            List of table names
        """
        return self.relevant_tables
    
    def _get_table_schema(self, table_name: str, company_id: int) -> Dict[str, Any]:
        """Get schema information for a table.
        
        Args:
            table_name: Table name
            company_id: Company ID
            
        Returns:
            Table schema information
        """
        if table_name not in self.relevant_tables:
            return {"error": f"Table {table_name} is not relevant to {self.type} agent"}
        
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