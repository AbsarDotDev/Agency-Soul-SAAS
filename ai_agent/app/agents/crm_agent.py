from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
import logging
import uuid
import json

from app.agents.base_agent import BaseAgent, AgentResponse, VisualizationResult, ActionResult
from app.database.queries import DatabaseQueries

# Set up logging
logger = logging.getLogger(__name__)


class CRMAgent(BaseAgent):
    """CRM agent for handling customer relationship management related queries."""
    
    async def process_message(
        self, 
        message: str, 
        company_id: int, 
        user_id: str,
        conversation_id: Optional[str] = None,
        session: Optional[Session] = None
    ) -> AgentResponse:
        """Process CRM related message.
        
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
            "agent_type": "CRM",
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
        
        # Get customer data for context
        try:
            if session:
                customers = await DatabaseQueries.get_customer_data(
                    company_id=company_id,
                    session=session
                )
                context["customer_count"] = len(customers)
                
                # Add geographic distribution stats if available
                regions = {}
                for customer in customers:
                    region = customer.get("billing_state", "Unknown")
                    if region in regions:
                        regions[region] += 1
                    else:
                        regions[region] = 1
                
                context["customer_regions"] = regions
        except Exception as e:
            logger.error(f"Error getting customer data: {str(e)}")
        
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
        """Generate CRM visualization.
        
        Args:
            query: Visualization query
            company_id: Company ID
            user_id: User ID
            visualization_type: Optional visualization type
            session: Optional database session
            
        Returns:
            Visualization result
        """
        # Get customer data
        customers = []
        if session:
            try:
                customers = await DatabaseQueries.get_customer_data(
                    company_id=company_id,
                    session=session
                )
            except Exception as e:
                logger.error(f"Error getting customer data: {str(e)}")
        
        # Build context
        context = {
            "agent_type": "CRM",
            "user_id": user_id,
            "visualization_query": query,
            "visualization_type": visualization_type or "auto",
            "customer_count": len(customers)
        }
        
        # Add geographic distribution stats if available
        regions = {}
        for customer in customers:
            region = customer.get("billing_state", "Unknown")
            if region in regions:
                regions[region] += 1
            else:
                regions[region] = 1
        
        context["customer_regions"] = regions
        
        # Build system prompt with visualization instructions
        system_prompt = self._generate_system_prompt(company_id, context)
        system_prompt += """
You are generating a visualization based on customer data.
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
        
        user_prompt += f"Customer data summary: {len(customers)} customers, {len(regions)} regions\n"
        user_prompt += f"Region breakdown: {json.dumps(regions)}"
        
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
            fallback_data = {
                "chart_type": visualization_type or "pie",
                "labels": list(regions.keys()),
                "datasets": [{
                    "label": "Customer Count",
                    "data": list(regions.values()),
                    "backgroundColor": ["rgba(255, 99, 132, 0.5)", "rgba(54, 162, 235, 0.5)", 
                                       "rgba(255, 206, 86, 0.5)", "rgba(75, 192, 192, 0.5)", 
                                       "rgba(153, 102, 255, 0.5)"]
                }],
                "title": "Customer Distribution by Region",
                "description": "Number of customers in each region"
            }
            
            return VisualizationResult(
                data=fallback_data,
                explanation="This chart shows the distribution of customers across different regions."
            )
    
    async def perform_action(
        self, 
        action: str, 
        parameters: Dict[str, Any], 
        company_id: int, 
        user_id: str,
        session: Optional[Session] = None
    ) -> ActionResult:
        """Perform CRM action.
        
        Args:
            action: Action to perform
            parameters: Action parameters
            company_id: Company ID
            user_id: User ID
            session: Optional database session
            
        Returns:
            Action result
        """
        # Map of supported actions
        supported_actions = {
            "add_customer": self._add_customer,
            "update_customer": self._update_customer,
            "delete_customer": self._delete_customer,
        }
        
        # Check if action is supported
        if action not in supported_actions:
            return ActionResult(
                success=False,
                message=f"Unsupported action: {action}",
                data={"supported_actions": list(supported_actions.keys())}
            )
        
        # Perform action
        try:
            if not session:
                return ActionResult(
                    success=False,
                    message="Database session is required for actions"
                )
            
            # Call the appropriate action method
            result = await supported_actions[action](
                parameters=parameters,
                company_id=company_id,
                user_id=user_id,
                session=session
            )
            
            return result
        except Exception as e:
            logger.error(f"Error performing action {action}: {str(e)}")
            return ActionResult(
                success=False,
                message=f"Error performing action: {str(e)}"
            )
    
    async def _add_customer(
        self, 
        parameters: Dict[str, Any], 
        company_id: int, 
        user_id: str,
        session: Session
    ) -> ActionResult:
        """Add customer action.
        
        Args:
            parameters: Action parameters
            company_id: Company ID
            user_id: User ID
            session: Database session
            
        Returns:
            Action result
        """
        # Required parameters
        required_params = ["name", "email", "contact"]
        
        # Check required parameters
        missing_params = [param for param in required_params if param not in parameters]
        if missing_params:
            return ActionResult(
                success=False,
                message=f"Missing required parameters: {', '.join(missing_params)}",
                data={"required_parameters": required_params}
            )
        
        try:
            # Execute SQL to insert customer
            result = session.execute("""
                INSERT INTO customers (
                    name, email, contact, company_id, created_by
                ) VALUES (
                    :name, :email, :contact, :company_id, :created_by
                )
            """, {
                "name": parameters["name"],
                "email": parameters["email"],
                "contact": parameters["contact"],
                "company_id": company_id,
                "created_by": user_id
            })
            
            session.commit()
            
            # Get the inserted customer ID
            customer_id = result.lastrowid
            
            return ActionResult(
                success=True,
                message=f"Customer {parameters['name']} added successfully",
                data={"customer_id": customer_id, "name": parameters["name"], "email": parameters["email"]}
            )
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding customer: {str(e)}")
            return ActionResult(
                success=False,
                message=f"Error adding customer: {str(e)}"
            )
    
    async def _update_customer(
        self, 
        parameters: Dict[str, Any], 
        company_id: int, 
        user_id: str,
        session: Session
    ) -> ActionResult:
        """Update customer action.
        
        Args:
            parameters: Action parameters
            company_id: Company ID
            user_id: User ID
            session: Database session
            
        Returns:
            Action result
        """
        # Required parameters
        required_params = ["customer_id"]
        
        # Check required parameters
        missing_params = [param for param in required_params if param not in parameters]
        if missing_params:
            return ActionResult(
                success=False,
                message=f"Missing required parameters: {', '.join(missing_params)}",
                data={"required_parameters": required_params}
            )
        
        # Get updatable fields
        updatable_fields = ["name", "email", "contact", "billing_name", "billing_address", 
                           "billing_city", "billing_state", "billing_country", "billing_phone"]
        
        # Build update SET clause
        update_fields = [f"{field} = :{field}" for field in updatable_fields if field in parameters]
        
        if not update_fields:
            return ActionResult(
                success=False,
                message="No fields to update provided",
                data={"updatable_fields": updatable_fields}
            )
        
        try:
            # Verify customer belongs to company
            customer = await DatabaseQueries.get_customer_data(
                company_id=company_id,
                customer_id=parameters["customer_id"],
                session=session
            )
            
            if not customer:
                return ActionResult(
                    success=False,
                    message=f"Customer with ID {parameters['customer_id']} not found or does not belong to this company"
                )
            
            # Build SQL parameters
            sql_params = {field: parameters[field] for field in updatable_fields if field in parameters}
            sql_params["customer_id"] = parameters["customer_id"]
            sql_params["company_id"] = company_id
            
            # Execute SQL to update customer
            session.execute(f"""
                UPDATE customers
                SET {', '.join(update_fields)}
                WHERE id = :customer_id AND company_id = :company_id
            """, sql_params)
            
            session.commit()
            
            return ActionResult(
                success=True,
                message=f"Customer with ID {parameters['customer_id']} updated successfully",
                data={"customer_id": parameters["customer_id"], "updated_fields": list(set(sql_params.keys()) - {"customer_id", "company_id"})}
            )
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating customer: {str(e)}")
            return ActionResult(
                success=False,
                message=f"Error updating customer: {str(e)}"
            )
    
    async def _delete_customer(
        self, 
        parameters: Dict[str, Any], 
        company_id: int, 
        user_id: str,
        session: Session
    ) -> ActionResult:
        """Delete customer action.
        
        Args:
            parameters: Action parameters
            company_id: Company ID
            user_id: User ID
            session: Database session
            
        Returns:
            Action result
        """
        # Required parameters
        required_params = ["customer_id"]
        
        # Check required parameters
        missing_params = [param for param in required_params if param not in parameters]
        if missing_params:
            return ActionResult(
                success=False,
                message=f"Missing required parameters: {', '.join(missing_params)}",
                data={"required_parameters": required_params}
            )
        
        try:
            # Verify customer belongs to company
            customer = await DatabaseQueries.get_customer_data(
                company_id=company_id,
                customer_id=parameters["customer_id"],
                session=session
            )
            
            if not customer:
                return ActionResult(
                    success=False,
                    message=f"Customer with ID {parameters['customer_id']} not found or does not belong to this company"
                )
            
            # Execute SQL to delete customer
            session.execute("""
                DELETE FROM customers
                WHERE id = :customer_id AND company_id = :company_id
            """, {
                "customer_id": parameters["customer_id"],
                "company_id": company_id
            })
            
            session.commit()
            
            return ActionResult(
                success=True,
                message=f"Customer with ID {parameters['customer_id']} deleted successfully",
                data={"customer_id": parameters["customer_id"]}
            )
        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting customer: {str(e)}")
            return ActionResult(
                success=False,
                message=f"Error deleting customer: {str(e)}"
            ) 