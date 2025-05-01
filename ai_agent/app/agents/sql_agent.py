from typing import Dict, Any, List, Optional, Union
import logging
import uuid
import json
from datetime import datetime

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.chains import create_sql_query_chain
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.agents.base_agent import BaseAgent, AgentResponse, VisualizationResult
from app.database.connection import get_company_isolated_sql_database, COMPANY_ISOLATION_MAPPING
from app.core.llm import get_llm, get_embedding_model
from app.visualizations.generator import generate_visualization_from_data

# Set up logging
logger = logging.getLogger(__name__)

class SQLQueryResult(BaseModel):
    """SQL query result."""
    query: str = Field(description="The SQL query that was executed")
    result: List[Dict[str, Any]] = Field(description="The result of the SQL query")
    explanation: str = Field(description="A natural language explanation of the result")

class SQLAgent(BaseAgent):
    """SQL agent for dynamic database queries."""
    
    async def process_message(
        self, 
        message: str, 
        company_id: int, 
        user_id: str,
        conversation_id: Optional[str] = None,
        session: Optional[Session] = None
    ) -> AgentResponse:
        """Process a message using dynamic SQL query generation.
        
        Args:
            message: User message
            company_id: Company ID for data isolation
            user_id: User ID
            conversation_id: Optional conversation ID
            session: Optional database session
            
        Returns:
            Agent response
        """
        # Generate new conversation ID if not provided
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        
        try:
            # Verify token usage
            if session:
                # Fetch user's token usage and plan's token allocation
                result = session.execute(text("""
                    SELECT u.ai_agent_tokens_used, p.ai_agent_default_tokens
                    FROM users u
                    JOIN plans p ON u.plan = p.id
                    WHERE u.id = :company_id AND p.ai_agent_enabled = 1
                """), {
                    "company_id": company_id
                }).fetchone()
                
                if not result:
                     # This could mean user not found, plan not found, or plan does not have AI enabled
                    logger.warning(f"Could not retrieve token/plan info for company_id: {company_id}. User or valid AI plan might not exist.")
                    return AgentResponse(
                        message="Could not verify AI Agent access or plan status. Please contact support.",
                        conversation_id=conversation_id
                    )
                
                tokens_used, tokens_allocated = result
                tokens_remaining = tokens_allocated - tokens_used
                
                if tokens_remaining <= 0:
                    logger.info(f"Company {company_id} has no remaining AI tokens.")
                    return AgentResponse(
                        message="You have used all your AI Agent tokens. Please contact your administrator.",
                        conversation_id=conversation_id
                    )
            
            # Get conversation history if conversation ID is provided
            conversation_history = []
            if conversation_id and session:
                conversation_history = await self._get_conversation_history(
                    conversation_id=conversation_id,
                    company_id=company_id,
                    session=session
                )
            
            # Create company-isolated SQLDatabase instance
            db = get_company_isolated_sql_database(company_id)
            
            # Get database schema details for the system prompt
            tables_info = self._get_filtered_table_info(db) 
            
            # Create a SQLDatabaseToolkit instance
            llm = get_llm()
            toolkit = SQLDatabaseToolkit(db=db, llm=llm)
            
            # Build system prompt with detailed database context
            system_prompt = f"""
You are an SQL expert and data analyst helping a company (ID: {company_id}) analyze their data. 
You have access to their database schema and can run SQL queries to answer their questions.

# CRITICAL DATA PRIVACY REQUIREMENTS:
EXTREMELY IMPORTANT: You must ONLY access data that belongs to the company with ID {company_id}.
The database contains data from multiple companies, but you should NEVER query data from other companies.

# DATABASE STRUCTURE:
The database is structured with company isolation. Each relevant table has a "created_by" column or a similar company identifier.
Here are the key tables you have access to:

{tables_info}

# DATA ISOLATION ENFORCEMENT:
All your queries will be automatically modified to include company filters (e.g., "WHERE created_by = {company_id}"), 
but you should still be aware of this restriction when writing your queries.

# HOW TO APPROACH QUESTIONS:
1. Think about what tables contain the relevant data based on the schema provided
2. Focus on writing SQL queries that involve only the necessary tables
3. Don't worry about adding company_id or created_by filters - these will be added automatically
4. If appropriate, suggest visualizations based on the data
5. Always explain results in a way that's easy for the user to understand
            """
            
            # Include conversation history in prompt if available
            user_prompt = message
            if conversation_history:
                context = "Previous conversation:\n"
                for msg in conversation_history[-3:]:  # Include the 3 most recent exchanges
                    context += f"User: {msg['message']}\nAssistant: {msg['response']}\n\n"
                user_prompt = f"{context}\nNew question: {message}"
            
            # Create a SQL agent
            agent_executor = create_sql_agent(
                llm=llm,
                toolkit=toolkit,
                verbose=True,
                agent_executor_kwargs={"handle_parsing_errors": True},
                prefix=system_prompt
            )
            
            # Execute the agent
            result = await agent_executor.ainvoke({"input": user_prompt})
            response_text = result["output"]
            
            # Update token usage (placeholder for actual token counting)
            tokens_to_consume = 1  # This would be calculated based on actual token usage
            if session:
                update_success = await self._update_token_usage(
                    session=session,
                    company_id=company_id,
                    tokens_used=tokens_to_consume # Use a different name
                )
                if not update_success:
                     logger.error(f"Failed to update token usage for company {company_id}")
                     # Decide how to handle this - maybe still save convo but log error?
                
                # Save conversation
                await self._save_conversation(
                    session=session,
                    conversation_id=conversation_id,
                    company_id=company_id,
                    user_id=user_id,
                    message=message,
                    response=response_text,
                    agent_type="SQL",
                    tokens_used=tokens_to_consume
                )
            
            # Return response including remaining tokens
            # We fetch the latest count AFTER decrementing
            final_tokens_remaining = await self._get_remaining_tokens(session, company_id)
            
            response_payload = AgentResponse(
                message=response_text,
                conversation_id=conversation_id,
                tokens_remaining=final_tokens_remaining,
                tokens_used=tokens_to_consume
            )
            
            # Log the payload just before returning
            logger.info(f"Returning payload to API: {response_payload.model_dump()}")
            
            return response_payload
            
        except Exception as e:
            logger.error(f"Error in SQL agent: {str(e)}")
            return AgentResponse(
                message=f"I encountered an error while trying to answer your question: {str(e)}. Please try again or rephrase your question.",
                conversation_id=conversation_id
            )
    
    def _get_filtered_table_info(self, db: Any) -> str:
        """Get filtered table information from the database.
        
        Args:
            db: Database instance
            
        Returns:
            Formatted string with table information
        """
        try:
            # Get table information from the database
            tables_with_isolation = []
            
            # For each table in the database, check if it has company isolation
            for table_name in db.get_usable_table_names():
                if table_name.lower() in COMPANY_ISOLATION_MAPPING:
                    tables_with_isolation.append(table_name)
            
            # Get table schema information
            table_info = []
            for table_name in tables_with_isolation:
                columns = db.get_table_info(table_names=[table_name])
                # Format the table information with indentation for better readability
                formatted_columns = "  - " + "\n  - ".join(
                    [col.strip() for col in columns.split(",")]
                )
                table_info.append(f"Table: {table_name}\nColumns:\n{formatted_columns}\n")
            
            return "\n".join(table_info)
        except Exception as e:
            logger.error(f"Error getting filtered table info: {str(e)}")
            return "Table information not available"
    
    async def generate_visualization(
        self, 
        query: str, 
        company_id: int, 
        user_id: str,
        visualization_type: Optional[str] = None,
        session: Optional[Session] = None
    ) -> VisualizationResult:
        """Generate visualization based on SQL query results.
        
        Args:
            query: Visualization query in natural language
            company_id: Company ID for data isolation
            user_id: User ID
            visualization_type: Optional visualization type (bar, line, pie, etc.)
            session: Optional database session
            
        Returns:
            Visualization result
        """
        try:
            # Verify token usage
            if session:
                result = session.execute(text("""
                    SELECT u.ai_agent_tokens_used, p.ai_agent_default_tokens
                    FROM users u
                    JOIN plans p ON u.plan = p.id
                    WHERE u.id = :company_id AND p.ai_agent_enabled = 1
                """), {
                    "company_id": company_id
                }).fetchone()
                
                if not result:
                    logger.warning(f"Could not retrieve token/plan info for company_id: {company_id} for visualization.")
                    return VisualizationResult(
                        data=None,
                        explanation="Could not verify AI Agent access or plan status."
                    )
                
                tokens_used, tokens_allocated = result
                tokens_remaining = tokens_allocated - tokens_used

                # Check specifically for visualization token cost
                visualization_token_cost = 2 # Assume higher cost
                if tokens_remaining < visualization_token_cost:
                    logger.info(f"Company {company_id} has insufficient tokens ({tokens_remaining}) for visualization (cost: {visualization_token_cost}).")
                    return VisualizationResult(
                        data=None,
                        explanation=f"You need at least {visualization_token_cost} tokens for visualization, but only have {tokens_remaining} left."
                    )
            
            # Create company-isolated SQLDatabase instance
            db = get_company_isolated_sql_database(company_id)
            
            # Create LLM
            llm = get_llm()
            
            # Get database schema details for the system prompt
            tables_info = self._get_filtered_table_info(db)
            
            # Build a prompt for SQL query generation specifically for visualization
            system_prompt = f"""
You are an SQL expert helping generate visualizations from database data.
You need to write a SQL query that will provide the necessary data for creating a visualization.

# CRITICAL DATA PRIVACY REQUIREMENTS:
EXTREMELY IMPORTANT: You must ONLY access data that belongs to the company with ID {company_id}.

# DATABASE STRUCTURE:
Here are the key tables you have access to:

{tables_info}

# DATA ISOLATION ENFORCEMENT:
All your queries will be automatically modified to include company filters (e.g., "WHERE created_by = {company_id}"), 
but you should still be aware of this restriction when writing your queries.

# VISUALIZATION GUIDANCE:
For a {visualization_type or 'suitable'} visualization, focus on:
1. Selecting the right columns that can be visualized effectively
2. Using aggregation functions (COUNT, SUM, AVG) when appropriate
3. Including categories/dimensions for grouping
4. Limiting the result to a reasonable number of data points (10-15 max)

Write ONLY the SQL query, nothing else.
"""
            
            # Create SQL query generation chain with the system prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}")
            ])
            
            # Create SQL query generation chain
            sql_chain = create_sql_query_chain(llm, db, prompt=prompt)
            
            # Generate SQL query
            sql_query = await sql_chain.ainvoke({"input": query})
            
            logger.info(f"Generated SQL query for visualization: {sql_query}")
            
            # Execute the SQL query
            result = db.run(sql_query)
            
            # Parse results into a format suitable for visualization
            if isinstance(result, str):
                try:
                    # Try to parse if returned as string
                    data = json.loads(result)
                except:
                    # If parsing fails, assume it's a formatted string representation
                    logger.warning(f"Could not parse SQL result as JSON: {result}")
                    return VisualizationResult(
                        data=None,
                        explanation=f"Failed to process data for visualization. Result: {result}"
                    )
            else:
                # Convert result to dict/list structure for visualization
                try:
                    if hasattr(result, '_mapping'):  # SQLAlchemy row
                        data = [dict(row._mapping) for row in result]
                    else:
                        data = result  # Assume it's already in a suitable format
                except Exception as e:
                    logger.error(f"Error converting SQL result to visualization data: {str(e)}")
                    return VisualizationResult(
                        data=None,
                        explanation=f"Failed to convert data for visualization: {str(e)}"
                    )
            
            # Generate visualization
            visualization = await generate_visualization_from_data(
                data=data,
                query=query,
                visualization_type=visualization_type,
                llm=llm
            )
            
            # Update token usage
            tokens_to_consume = visualization_token_cost 
            if session:
                 update_success = await self._update_token_usage(
                    session=session,
                    company_id=company_id,
                    tokens_used=tokens_to_consume
                 )
                 if not update_success:
                     logger.error(f"Failed to update token usage for visualization for company {company_id}")

            # Return visualization result including remaining tokens
            final_tokens_remaining = await self._get_remaining_tokens(session, company_id)
            visualization.tokens_remaining = final_tokens_remaining # Assuming VisualizationResult has this field

            return visualization
            
        except Exception as e:
            logger.error(f"Error generating visualization: {str(e)}")
            return VisualizationResult(
                data=None,
                explanation=f"I encountered an error while generating the visualization: {str(e)}. Please try again or rephrase your request."
            )
    
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
        try:
            # First check if the table exists
            table_exists = session.execute(text("""
                SELECT COUNT(*) 
                FROM information_schema.tables 
                WHERE table_name = 'agent_conversations'
            """)).scalar() > 0
            
            if not table_exists:
                return []
            
            # Retrieve conversation history
            result = session.execute(text("""
                SELECT message, response, created_at
                FROM agent_conversations
                WHERE conversation_id = :conversation_id
                AND company_user_id = :company_id
                ORDER BY created_at ASC
            """), {
                "conversation_id": conversation_id,
                "company_id": company_id
            }).fetchall()
            
            # Convert to list of dictionaries
            conversation_history = []
            for row in result:
                conversation_history.append({
                    "message": row[0],
                    "response": row[1],
                    "created_at": row[2]
                })
            
            return conversation_history
            
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {str(e)}")
            return [] 

    async def _get_remaining_tokens(self, session: Session, company_id: int) -> Optional[int]:
        """Helper to get the current remaining tokens."""
        if not session:
            return None
        try:
            result = session.execute(text("""
                SELECT u.ai_agent_tokens_used, p.ai_agent_default_tokens
                FROM users u
                JOIN plans p ON u.plan = p.id
                WHERE u.id = :company_id
            """), {"company_id": company_id}).fetchone()
            if result:
                tokens_used, tokens_allocated = result
                return max(0, tokens_allocated - tokens_used)
            return None # User or plan not found
        except Exception as e:
            logger.error(f"Error fetching remaining tokens for company {company_id}: {e}")
            return None 