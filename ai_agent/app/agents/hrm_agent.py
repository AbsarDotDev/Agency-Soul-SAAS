from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
import logging
import uuid
import json

from app.agents.base_agent import BaseAgent, AgentResponse, VisualizationResult, ActionResult
from app.database.queries import DatabaseQueries

# Set up logging
logger = logging.getLogger(__name__)


class HRMAgent(BaseAgent):
    """HRM agent for handling human resource management related queries."""
    
    async def process_message(
        self, 
        message: str, 
        company_id: int, 
        user_id: str,
        conversation_id: Optional[str] = None,
        session: Optional[Session] = None
    ) -> AgentResponse:
        """Process HRM related message.
        
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
            "agent_type": "HRM",
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
        
        # Get employee data for context
        try:
            if session:
                employees = await DatabaseQueries.get_employee_data(
                    company_id=company_id,
                    session=session
                )
                context["employee_count"] = len(employees)
                
                # Add department stats
                departments = {}
                for emp in employees:
                    dept_name = emp.get("department_name", "Unknown")
                    if dept_name in departments:
                        departments[dept_name] += 1
                    else:
                        departments[dept_name] = 1
                
                context["departments"] = departments
        except Exception as e:
            logger.error(f"Error getting employee data: {str(e)}")
        
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
        """Generate HRM visualization.
        
        Args:
            query: Visualization query
            company_id: Company ID
            user_id: User ID
            visualization_type: Optional visualization type
            session: Optional database session
            
        Returns:
            Visualization result
        """
        # Get employee data
        employees = []
        if session:
            try:
                employees = await DatabaseQueries.get_employee_data(
                    company_id=company_id,
                    session=session
                )
            except Exception as e:
                logger.error(f"Error getting employee data: {str(e)}")
        
        # Build context
        context = {
            "agent_type": "HRM",
            "user_id": user_id,
            "visualization_query": query,
            "visualization_type": visualization_type or "auto",
            "employee_count": len(employees)
        }
        
        # Add department stats
        departments = {}
        for emp in employees:
            dept_name = emp.get("department_name", "Unknown")
            if dept_name in departments:
                departments[dept_name] += 1
            else:
                departments[dept_name] = 1
        
        context["departments"] = departments
        
        # Build system prompt with visualization instructions
        system_prompt = self._generate_system_prompt(company_id, context)
        system_prompt += """
You are generating a visualization based on HR data.
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
        
        user_prompt += f"Employee data summary: {len(employees)} employees, {len(departments)} departments\n"
        user_prompt += f"Department breakdown: {json.dumps(departments)}"
        
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
                "chart_type": visualization_type or "bar",
                "labels": list(departments.keys()),
                "datasets": [{
                    "label": "Employee Count",
                    "data": list(departments.values()),
                    "backgroundColor": "rgba(54, 162, 235, 0.5)"
                }],
                "title": "Employee Distribution by Department",
                "description": "Number of employees in each department"
            }
            
            return VisualizationResult(
                data=fallback_data,
                explanation="This chart shows the distribution of employees across departments."
            )
    
    async def perform_action(
        self, 
        action: str, 
        parameters: Dict[str, Any], 
        company_id: int, 
        user_id: str,
        session: Optional[Session] = None
    ) -> ActionResult:
        """Perform HRM action.
        
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
            "add_employee": self._add_employee,
            "update_employee": self._update_employee,
            "delete_employee": self._delete_employee,
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
    
    async def _add_employee(
        self, 
        parameters: Dict[str, Any], 
        company_id: int, 
        user_id: str,
        session: Session
    ) -> ActionResult:
        """Add employee action.
        
        Args:
            parameters: Action parameters
            company_id: Company ID
            user_id: User ID
            session: Database session
            
        Returns:
            Action result
        """
        # Required parameters
        required_params = ["name", "email"]
        
        # Check required parameters
        missing_params = [param for param in required_params if param not in parameters]
        if missing_params:
            return ActionResult(
                success=False,
                message=f"Missing required parameters: {', '.join(missing_params)}",
                data={"required_parameters": required_params}
            )
        
        try:
            # Execute SQL to insert employee
            result = session.execute("""
                INSERT INTO employees (
                    name, email, company_id, created_by
                ) VALUES (
                    :name, :email, :company_id, :created_by
                )
            """, {
                "name": parameters["name"],
                "email": parameters["email"],
                "company_id": company_id,
                "created_by": user_id
            })
            
            session.commit()
            
            # Get the inserted employee ID
            employee_id = result.lastrowid
            
            return ActionResult(
                success=True,
                message=f"Employee {parameters['name']} added successfully",
                data={"employee_id": employee_id, "name": parameters["name"], "email": parameters["email"]}
            )
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding employee: {str(e)}")
            return ActionResult(
                success=False,
                message=f"Error adding employee: {str(e)}"
            )
    
    async def _update_employee(
        self, 
        parameters: Dict[str, Any], 
        company_id: int, 
        user_id: str,
        session: Session
    ) -> ActionResult:
        """Update employee action.
        
        Args:
            parameters: Action parameters
            company_id: Company ID
            user_id: User ID
            session: Database session
            
        Returns:
            Action result
        """
        # Required parameters
        required_params = ["employee_id"]
        
        # Check required parameters
        missing_params = [param for param in required_params if param not in parameters]
        if missing_params:
            return ActionResult(
                success=False,
                message=f"Missing required parameters: {', '.join(missing_params)}",
                data={"required_parameters": required_params}
            )
        
        # Get updatable fields
        updatable_fields = ["name", "email", "phone", "department_id", "designation_id", "salary"]
        
        # Build update SET clause
        update_fields = [f"{field} = :{field}" for field in updatable_fields if field in parameters]
        
        if not update_fields:
            return ActionResult(
                success=False,
                message="No fields to update provided",
                data={"updatable_fields": updatable_fields}
            )
        
        try:
            # Verify employee belongs to company
            employee = await DatabaseQueries.get_employee_data(
                company_id=company_id,
                employee_id=parameters["employee_id"],
                session=session
            )
            
            if not employee:
                return ActionResult(
                    success=False,
                    message=f"Employee with ID {parameters['employee_id']} not found or does not belong to this company"
                )
            
            # Build SQL parameters
            sql_params = {field: parameters[field] for field in updatable_fields if field in parameters}
            sql_params["employee_id"] = parameters["employee_id"]
            sql_params["company_id"] = company_id
            
            # Execute SQL to update employee
            session.execute(f"""
                UPDATE employees
                SET {', '.join(update_fields)}
                WHERE id = :employee_id AND company_id = :company_id
            """, sql_params)
            
            session.commit()
            
            return ActionResult(
                success=True,
                message=f"Employee with ID {parameters['employee_id']} updated successfully",
                data={"employee_id": parameters["employee_id"], "updated_fields": list(sql_params.keys())}
            )
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating employee: {str(e)}")
            return ActionResult(
                success=False,
                message=f"Error updating employee: {str(e)}"
            )
    
    async def _delete_employee(
        self, 
        parameters: Dict[str, Any], 
        company_id: int, 
        user_id: str,
        session: Session
    ) -> ActionResult:
        """Delete employee action.
        
        Args:
            parameters: Action parameters
            company_id: Company ID
            user_id: User ID
            session: Database session
            
        Returns:
            Action result
        """
        # Required parameters
        required_params = ["employee_id"]
        
        # Check required parameters
        missing_params = [param for param in required_params if param not in parameters]
        if missing_params:
            return ActionResult(
                success=False,
                message=f"Missing required parameters: {', '.join(missing_params)}",
                data={"required_parameters": required_params}
            )
        
        try:
            # Verify employee belongs to company
            employee = await DatabaseQueries.get_employee_data(
                company_id=company_id,
                employee_id=parameters["employee_id"],
                session=session
            )
            
            if not employee:
                return ActionResult(
                    success=False,
                    message=f"Employee with ID {parameters['employee_id']} not found or does not belong to this company"
                )
            
            # Execute SQL to delete employee
            session.execute("""
                DELETE FROM employees
                WHERE id = :employee_id AND company_id = :company_id
            """, {
                "employee_id": parameters["employee_id"],
                "company_id": company_id
            })
            
            session.commit()
            
            return ActionResult(
                success=True,
                message=f"Employee with ID {parameters['employee_id']} deleted successfully",
                data={"employee_id": parameters["employee_id"]}
            )
        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting employee: {str(e)}")
            return ActionResult(
                success=False,
                message=f"Error deleting employee: {str(e)}"
            )
