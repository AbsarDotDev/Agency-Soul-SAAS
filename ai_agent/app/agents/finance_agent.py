from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
import logging
import uuid
import json
from fastapi import HTTPException

from app.agents.base_agent import BaseAgent, AgentResponse, VisualizationResult, ActionResult
from app.database.queries import DatabaseQueries

# Set up logging
logger = logging.getLogger(__name__)


class FinanceAgent(BaseAgent):
    """Finance agent for handling financial data related queries."""

    async def process_message(
        self,
        message: str,
        company_id: int,
        user_id: str,
        conversation_id: Optional[str] = None,
        session: Optional[Session] = None
    ) -> AgentResponse:
        """Process finance related message."""
        if not conversation_id:
            conversation_id = str(uuid.uuid4())

        conversation_history = []
        if conversation_id and session:
            conversation_history = await self._get_conversation_history(
                conversation_id=conversation_id,
                company_id=company_id,
                session=session
            )

        context = {
            "agent_type": "Finance",
            "user_id": user_id,
        }

        # Fetch relevant financial context using DatabaseQueries
        try:
            if session:
                financial_summary = await DatabaseQueries.get_financial_summary(company_id, session)
                context.update(financial_summary)
        except HTTPException as e:
            # Handle potential DB errors from queries gracefully
             logger.error(f"HTTP error getting financial context: {e.detail}")
             context["error"] = f"Could not retrieve financial summary: {e.detail}"
        except Exception as e:
            logger.error(f"Unexpected error getting financial context: {str(e)}")
            context["error"] = "An unexpected error occurred while fetching financial data."


        system_prompt = self._generate_system_prompt(company_id, context)
        user_prompt = self._build_user_prompt_with_history(message, conversation_history)

        response_text = await self._get_llm_response(user_prompt, system_prompt)

        await self._save_conversation(
            conversation_id=conversation_id,
            company_id=company_id,
            user_id=user_id,
            message=message,
            response=response_text,
            session=session
        )

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
        """Generate finance visualization."""
        context = {
            "agent_type": "Finance",
            "user_id": user_id,
            "visualization_query": query,
            "visualization_type": visualization_type or "auto",
        }

        # Fetch data needed for visualization using DatabaseQueries
        finance_data_for_viz = []
        explanation_fallback = "Could not generate specific visualization."
        try:
            if session:
                # Example: Fetch expense breakdown for a pie chart
                finance_data_for_viz = await DatabaseQueries.get_expense_breakdown(company_id, session)
                context["expense_breakdown"] = finance_data_for_viz
            else:
                explanation_fallback = "Database connection not available for visualization data."

        except HTTPException as e:
            logger.error(f"HTTP error getting finance data for visualization: {e.detail}")
            explanation_fallback = f"Failed to retrieve visualization data: {e.detail}"
        except Exception as e:
            logger.error(f"Unexpected error getting finance data for visualization: {str(e)}")
            explanation_fallback = "An unexpected error occurred fetching visualization data."

        # Prepare data in a format the LLM might understand for JSON generation
        viz_data_summary = {item['category']: item['total_amount'] for item in finance_data_for_viz}

        system_prompt = self._generate_system_prompt(company_id, context)
        system_prompt += """
You are generating a financial visualization based on the provided data summary.
Your response must be valid JSON following the specified chart structure (bar, line, pie, etc.).
Respond ONLY with valid JSON.
Example JSON Structure:
{
  "chart_type": "pie",
  "labels": ["Category A", "Category B"],
  "datasets": [{"label": "Dataset Label", "data": [value1, value2], "backgroundColor": ["color1", "color2"] }],
  "title": "Chart Title",
  "description": "Chart Description",
  "explanation": "Text explanation of the insights."
}
"""

        user_prompt = f"Generate a visualization for: {query}\n"
        if visualization_type:
            user_prompt += f"Preferred type: {visualization_type}\n"
        user_prompt += f"Based on available financial data. Data Summary: {json.dumps(viz_data_summary)}"

        response_text = await self._get_llm_response(user_prompt, system_prompt)

        try:
            response_json = json.loads(response_text)
            explanation = response_json.pop("explanation", "No explanation provided.")
            return VisualizationResult(
                data=response_json,
                explanation=explanation
            )
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing finance visualization JSON: {str(e)}")
            # Fallback visualization using the fetched data if parsing fails
            fallback_data = {
                "chart_type": visualization_type or "pie", # Default to pie if not specified
                "labels": list(viz_data_summary.keys()),
                "datasets": [{
                    "label": "Expenses",
                    "data": list(viz_data_summary.values()),
                    # Generate some default colors if needed
                    "backgroundColor": [f'hsl({(i*60)%360}, 70%, 80%)' for i in range(len(viz_data_summary))] 
                }],
                "title": "Expense Breakdown",
                "description": "Distribution of expenses across categories."
            }
            return VisualizationResult(
                data=fallback_data,
                explanation=f"{explanation_fallback} Showing available expense breakdown."
            )

    async def perform_action(
        self,
        action: str,
        parameters: Dict[str, Any],
        company_id: int,
        user_id: str,
        session: Optional[Session] = None
    ) -> ActionResult:
        """Perform finance action."""
        supported_actions = {
            "add_expense": self._add_expense,
            "add_revenue": self._add_revenue,
            # Add more finance actions here
        }

        if action not in supported_actions:
            return ActionResult(
                success=False,
                message=f"Unsupported finance action: {action}",
                data={"supported_actions": list(supported_actions.keys())}
            )

        if not session:
            return ActionResult(success=False, message="Database session is required for actions")

        try:
            # Convert user_id from string (from token potentially) to int for DB
            user_id_int = int(user_id) 
            result = await supported_actions[action](
                parameters=parameters,
                company_id=company_id,
                user_id=user_id_int, # Pass integer user_id
                session=session
            )
            return result
        except ValueError:
             logger.error(f"Invalid user_id format for action {action}: {user_id}")
             return ActionResult(success=False, message="Invalid user ID format.")
        except HTTPException as he:
            # Handle errors raised from DatabaseQueries (e.g., 403 Forbidden, 500)
            logger.error(f"HTTP error during finance action {action}: {he.detail} (Status: {he.status_code})")
            return ActionResult(success=False, message=f"Action failed: {he.detail}")
        except Exception as e:
            logger.error(f"Unexpected error performing finance action {action}: {str(e)}")
            return ActionResult(success=False, message=f"Error performing action: {str(e)}")

    async def _add_expense(
        self,
        parameters: Dict[str, Any],
        company_id: int,
        user_id: int, # Expecting integer user_id
        session: Session
    ) -> ActionResult:
        """Adds an expense record using DatabaseQueries."""
        # Parameters expected by DatabaseQueries.insert_expense:
        # category (maps to name), date, amount, description?, attachment?, project_id?, task_id?
        required_params = ["category", "amount", "date"] 
        missing_params = [p for p in required_params if p not in parameters or parameters[p] is None]
        if missing_params:
            return ActionResult(
                success=False,
                message=f"Missing required parameters for adding expense: {', '.join(missing_params)}",
                data={"required_parameters": required_params}
            )
        
        try:
            expense_id = await DatabaseQueries.insert_expense(
                company_id=company_id,
                user_id=user_id,
                parameters=parameters,
                session=session
            )
            if expense_id:
                 return ActionResult(
                    success=True,
                    message=f"Expense of {parameters.get('amount')} in category '{parameters.get('category')}' added successfully.",
                    data={"expense_id": expense_id, "details": parameters}
                )
            else:
                 # Should not happen if insert_expense raises on failure, but handle just in case
                 return ActionResult(success=False, message="Failed to add expense record.")
        except HTTPException as he:
            # Re-raise for the main perform_action handler
             raise he
        except Exception as e:
            # Catch unexpected errors from DB layer if not handled as HTTPException
            logger.error(f"Unexpected error in _add_expense calling insert_expense: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Internal error adding expense: {str(e)}")


    async def _add_revenue(
        self,
        parameters: Dict[str, Any],
        company_id: int,
        user_id: int, # Expecting integer user_id
        session: Session
    ) -> ActionResult:
        """Adds a revenue record using DatabaseQueries."""
        # Parameters expected by DatabaseQueries.insert_revenue:
        # date, amount, account_id, customer_id, category_id, payment_method, reference?, description/source?, add_receipt?
        required_params = ["source", "amount", "date", "account_id", "customer_id", "category_id", "payment_method"]
        # Map 'source' to 'description' if 'description' isn't provided
        if 'description' not in parameters and 'source' in parameters:
            parameters['description'] = parameters['source']
            
        missing_params = [p for p in required_params if p not in parameters or parameters[p] is None]
        if missing_params:
            return ActionResult(
                success=False,
                message=f"Missing required parameters for adding revenue: {', '.join(missing_params)}",
                data={"required_parameters": [p for p in required_params if p != 'description']} # Show original requirements
            )
            
        try:
            revenue_id = await DatabaseQueries.insert_revenue(
                 company_id=company_id,
                 user_id=user_id,
                 parameters=parameters,
                 session=session
             )
            if revenue_id:
                 return ActionResult(
                    success=True,
                    message=f"Revenue of {parameters.get('amount')} from source '{parameters.get('source', 'N/A')}' added successfully.",
                    data={"revenue_id": revenue_id, "details": parameters}
                )
            else:
                 # Should not happen if insert_revenue raises on failure
                 return ActionResult(success=False, message="Failed to add revenue record.")
        except HTTPException as he:
             # Re-raise for the main perform_action handler
             raise he
        except Exception as e:
             # Catch unexpected errors from DB layer
             logger.error(f"Unexpected error in _add_revenue calling insert_revenue: {str(e)}")
             raise HTTPException(status_code=500, detail=f"Internal error adding revenue: {str(e)}")


    def _build_user_prompt_with_history(self, message: str, history: List[Dict]) -> str:
        """Builds the user prompt including conversation history."""
        if not history:
            return message

        prompt = "Conversation history:\n"
        for entry in history:
            prompt += f"User: {entry['message']}\nAssistant: {entry['response']}\n\n"
        prompt += f"New message: {message}"
        return prompt 