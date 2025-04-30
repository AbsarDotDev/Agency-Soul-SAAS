from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
import logging
import uuid
import json

from app.agents.base_agent import BaseAgent, AgentResponse, VisualizationResult, ActionResult
from app.database.queries import DatabaseQueries

# Set up logging
logger = logging.getLogger(__name__)


class SalesAgent(BaseAgent):
    """Sales agent for handling sales and revenue related queries."""
    
    async def process_message(
        self, 
        message: str, 
        company_id: int, 
        user_id: str,
        conversation_id: Optional[str] = None,
        session: Optional[Session] = None
    ) -> AgentResponse:
        """Process sales related message.
        
        Args:
            message: User message
            company_id: Company ID
            user_id: User ID
            conversation_id: Optional conversation ID
            session: Optional database session
            
        Returns:
            Agent response
        """
        # Generate new conversation ID if not provided
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        
        # Get conversation history if conversation ID is provided
        conversation_history = []
        if conversation_id and session:
            conversation_history = await self._get_conversation_history(
                conversation_id=conversation_id,
                company_id=company_id,
                session=session
            )
        
        # Build chat context
        context = {
            "agent_type": "Sales",
            "user_id": user_id,
        }
        
        # Get company information
        company_info = None
        if session:
            try:
                company_info = await DatabaseQueries.get_company_info(
                    company_id=company_id,
                    session=session
                )
                context["company_name"] = company_info.get("company_name", "")
            except Exception as e:
                logger.error(f"Error getting company info: {str(e)}")
        
        # Get sales data for context
        try:
            if session:
                sales_data = await DatabaseQueries.get_sales_data(
                    company_id=company_id,
                    session=session
                )
                
                if sales_data:
                    # Calculate total sales
                    total_sales = sum(item.get("total_amount", 0) for item in sales_data)
                    context["total_sales"] = total_sales
                    context["sales_periods"] = len(sales_data)
        except Exception as e:
            logger.error(f"Error getting sales data: {str(e)}")
        
        # Build system prompt
        system_prompt = self._generate_system_prompt(company_id, context)
        
        # Build user prompt with conversation history
        user_prompt = message
        if conversation_history:
            user_prompt = f"Conversation history:\n"
            for msg in conversation_history:
                user_prompt += f"User: {msg['message']}\nAssistant: {msg['response']}\n\n"
            user_prompt += f"New message: {message}"
        
        # Get response from LLM
        response_text = await self._get_llm_response(user_prompt, system_prompt)
        
        # Save conversation
        if session:
            try:
                # Check if table exists, create if not
                session.execute("""
                    CREATE TABLE IF NOT EXISTS agent_conversations (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        conversation_id VARCHAR(36) NOT NULL,
                        company_id INT NOT NULL,
                        user_id VARCHAR(255) NOT NULL,
                        message TEXT NOT NULL,
                        response TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        INDEX (conversation_id),
                        INDEX (company_id),
                        INDEX (user_id)
                    )
                """)
                
                # Insert conversation
                session.execute("""
                    INSERT INTO agent_conversations 
                    (conversation_id, company_id, user_id, message, response)
                    VALUES (:conversation_id, :company_id, :user_id, :message, :response)
                """, {
                    "conversation_id": conversation_id,
                    "company_id": company_id,
                    "user_id": user_id,
                    "message": message,
                    "response": response_text
                })
                
                session.commit()
            except Exception as e:
                logger.error(f"Error saving conversation: {str(e)}")
                session.rollback()
        
        return AgentResponse(
            message=response_text,
            conversation_id=conversation_id
        )
    
    async def generate_visualization(
        self, 
        query: str, 
        company_id: int, 
        user_id: str,
        visualization_type: Optional[str] = None,
        session: Optional[Session] = None
    ) -> VisualizationResult:
        """Generate sales visualization.
        
        Args:
            query: Visualization query
            company_id: Company ID
            user_id: User ID
            visualization_type: Optional visualization type
            session: Optional database session
            
        Returns:
            Visualization result
        """
        # Get sales data
        sales_data = []
        if session:
            try:
                sales_data = await DatabaseQueries.get_sales_data(
                    company_id=company_id,
                    session=session
                )
            except Exception as e:
                logger.error(f"Error getting sales data: {str(e)}")
        
        # Build context
        context = {
            "agent_type": "Sales",
            "user_id": user_id,
            "visualization_query": query,
            "visualization_type": visualization_type or "auto",
            "sales_periods": len(sales_data)
        }
        
        # Calculate total sales
        if sales_data:
            total_sales = sum(item.get("total_amount", 0) for item in sales_data)
            context["total_sales"] = total_sales
        
        # Build system prompt with visualization instructions
        system_prompt = self._generate_system_prompt(company_id, context)
        system_prompt += """
You are generating a visualization based on sales data.
Your response must be valid JSON with the following structure:
{
  "chart_type": "bar|line|pie|scatter",
  "labels": ["Label1", "Label2", ...],
  "datasets": [
    {
      "label": "Dataset Label",
      "data": [value1, value2, ...],
      "backgroundColor": "color code(s)"
    }
  ],
  "title": "Chart Title",
  "description": "Chart Description",
  "explanation": "Text explanation of the insights from this visualization"
}
Respond ONLY with valid JSON. Do not include any other text.
"""
        
        # Build user prompt
        user_prompt = f"Generate a visualization for: {query}\n"
        if visualization_type:
            user_prompt += f"Visualization type: {visualization_type}\n"
        
        user_prompt += f"Sales data summary: {len(sales_data)} periods"
        
        if sales_data:
            # Format some sample data
            sample_data = sales_data[:5] if len(sales_data) > 5 else sales_data
            user_prompt += f"\nSample sales data: {json.dumps(sample_data)}"
        
        # Get response from LLM
        response_text = await self._get_llm_response(user_prompt, system_prompt)
        
        # Parse JSON response
        try:
            response_json = json.loads(response_text)
            
            # Extract explanation from the JSON
            explanation = response_json.pop("explanation", "No explanation provided.")
            
            # Return visualization data
            return VisualizationResult(
                data=response_json,
                explanation=explanation
            )
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing visualization JSON: {str(e)}")
            
            # Fallback to simple visualization if JSON parsing fails
            # Extract periods and amounts from sales data
            periods = [item.get("period", f"Period {i+1}") for i, item in enumerate(sales_data)]
            amounts = [item.get("total_amount", 0) for item in sales_data]
            
            fallback_data = {
                "chart_type": visualization_type or "line",
                "labels": periods,
                "datasets": [{
                    "label": "Sales Amount",
                    "data": amounts,
                    "borderColor": "rgba(75, 192, 192, 1)",
                    "backgroundColor": "rgba(75, 192, 192, 0.2)"
                }],
                "title": "Sales Trend",
                "description": "Sales trend over time"
            }
            
            return VisualizationResult(
                data=fallback_data,
                explanation="This chart shows the sales trend over time."
            )
    
    async def perform_action(
        self, 
        action: str, 
        parameters: Dict[str, Any], 
        company_id: int, 
        user_id: str,
        session: Optional[Session] = None
    ) -> ActionResult:
        """Perform sales action.
        
        Args:
            action: Action to perform
            parameters: Action parameters
            company_id: Company ID
            user_id: User ID
            session: Optional database session
            
        Returns:
            Action result
        """
        # Placeholder for sales actions
        return ActionResult(
            success=False,
            message="Sales actions not yet implemented",
            data={"supported_actions": ["add_customer", "update_customer", "delete_customer"]}
        )
