from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from fastapi import HTTPException, status
import logging
from typing import List, Dict, Any, Optional

from .connection import get_company_scoped_query

# Set up logging
logger = logging.getLogger(__name__)


class DatabaseQueries:
    """Database queries for retrieving company-specific data."""

    @staticmethod
    async def _get_user_company_id(user_id: int, session: Session) -> Optional[int]:
        """Helper method to get the company ID for a given user."""
        try:
            # Assuming 'users' table has 'id' and 'created_by' (company user/admin ID)
            # And 'users' table linked to 'companies' or 'created_by' directly maps to company ID
            # Adjust this query based on the actual schema relating users to companies
            query = """
                SELECT created_by 
                FROM users 
                WHERE id = :user_id 
                LIMIT 1
            """
            # If 'created_by' in users is NOT the company_id, you might need a join:
            # query = """
            #     SELECT u.company_id 
            #     FROM users u 
            #     WHERE u.id = :user_id 
            #     LIMIT 1
            # """
            # Or if the user *is* the company record (e.g., type='company')
            # query = """
            #     SELECT id 
            #     FROM users 
            #     WHERE id = :user_id AND type = 'company'
            #     LIMIT 1
            # """
            
            result = session.execute(query, {"user_id": user_id}).fetchone()
            if result:
                # Assuming the fetched ID *is* the company ID. Adjust if necessary.
                return result[0] 
            else:
                logger.warning(f"Could not find company ID for user_id: {user_id}")
                return None
        except SQLAlchemyError as e:
            logger.error(f"Database error getting company ID for user {user_id}: {str(e)}")
            return None
        except IndexError:
             logger.error(f"Index error resolving company ID for user {user_id}. Query result unexpected.")
             return None

    @staticmethod
    async def get_employee_data(company_id: int, session: Session, employee_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get employee data for a company.
        
        Args:
            company_id: Company ID
            session: Database session
            employee_id: Optional employee ID to filter by
            
        Returns:
            List of employee data dictionaries
            
        Raises:
            HTTPException: If there's a database error
        """
        try:
            query = """
                SELECT e.id, e.name, e.email, e.phone, e.doj, e.branch_id, e.department_id, 
                       e.designation_id, e.salary, e.salary_type, e.created_by,
                       b.name as branch_name, d.name as department_name, ds.name as designation_name
                FROM employees e
                LEFT JOIN branches b ON e.branch_id = b.id
                LEFT JOIN departments d ON e.department_id = d.id
                LEFT JOIN designations ds ON e.designation_id = ds.id
                WHERE e.company_id = :company_id
            """
            
            params = {"company_id": company_id}
            
            if employee_id:
                query += " AND e.id = :employee_id"
                params["employee_id"] = employee_id
                
            result = session.execute(query, params).fetchall()
            
            # Convert to list of dictionaries
            employees = [dict(row._mapping) for row in result]
            
            return employees
            
        except SQLAlchemyError as e:
            logger.error(f"Database error when retrieving employee data: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database error when retrieving employee data"
            )
    
    @staticmethod
    async def get_sales_data(company_id: int, session: Session, period: str = 'month') -> List[Dict[str, Any]]:
        """Get sales data for a company.
        
        Args:
            company_id: Company ID
            session: Database session
            period: Time period for aggregation ('day', 'week', 'month', 'year')
            
        Returns:
            List of sales data dictionaries
            
        Raises:
            HTTPException: If there's a database error
        """
        try:
            # Build the date grouping expression based on the period
            date_group = {
                'day': "DATE_FORMAT(i.issue_date, '%Y-%m-%d')",
                'week': "DATE_FORMAT(i.issue_date, '%Y-%u')",
                'month': "DATE_FORMAT(i.issue_date, '%Y-%m')",
                'year': "DATE_FORMAT(i.issue_date, '%Y')"
            }.get(period, "DATE_FORMAT(i.issue_date, '%Y-%m')")
            
            query = f"""
                SELECT {date_group} as period, 
                       SUM(i.amount) as total_amount,
                       COUNT(i.id) as invoice_count
                FROM invoices i
                WHERE i.company_id = :company_id
                GROUP BY period
                ORDER BY MIN(i.issue_date) DESC
                LIMIT 12
            """
            
            result = session.execute(query, {"company_id": company_id}).fetchall()
            
            # Convert to list of dictionaries
            sales_data = [dict(row._mapping) for row in result]
            
            return sales_data
            
        except SQLAlchemyError as e:
            logger.error(f"Database error when retrieving sales data: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database error when retrieving sales data"
            )
    
    @staticmethod
    async def get_customer_data(company_id: int, session: Session, customer_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get customer data for a company.
        
        Args:
            company_id: Company ID
            session: Database session
            customer_id: Optional customer ID to filter by
            
        Returns:
            List of customer data dictionaries
            
        Raises:
            HTTPException: If there's a database error
        """
        try:
            query = """
                SELECT c.id, c.name, c.email, c.contact, c.billing_name, c.billing_country,
                       c.billing_state, c.billing_city, c.billing_phone, c.billing_address,
                       c.shipping_name, c.shipping_country, c.shipping_state, c.shipping_city,
                       c.shipping_phone, c.shipping_address, c.created_by
                FROM customers c
                WHERE c.company_id = :company_id
            """
            
            params = {"company_id": company_id}
            
            if customer_id:
                query += " AND c.id = :customer_id"
                params["customer_id"] = customer_id
                
            result = session.execute(query, params).fetchall()
            
            # Convert to list of dictionaries
            customers = [dict(row._mapping) for row in result]
            
            return customers
            
        except SQLAlchemyError as e:
            logger.error(f"Database error when retrieving customer data: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database error when retrieving customer data"
            )
    
    @staticmethod
    async def get_company_info(company_id: int, session: Session) -> Dict[str, Any]:
        """Get company information.
        
        Args:
            company_id: Company ID
            session: Database session
            
        Returns:
            Company information dictionary
            
        Raises:
            HTTPException: If there's a database error or company not found
        """
        try:
            query = """
                SELECT c.id, c.company_name, c.company_email, c.company_telephone,
                       c.address, c.city, c.state, c.zipcode, c.country,
                       c.company_start_time, c.company_end_time, c.timezone,
                       c.created_at, c.updated_at
                FROM companies c
                WHERE c.id = :company_id
            """
            
            result = session.execute(query, {"company_id": company_id}).fetchone()
            
            if not result:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Company with ID {company_id} not found"
                )
            
            # Convert to dictionary
            company_info = dict(result._mapping)
            
            return company_info
            
        except SQLAlchemyError as e:
            logger.error(f"Database error when retrieving company info: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database error when retrieving company info"
            )

    # --- New Finance Methods ---

    @staticmethod
    async def get_financial_summary(company_id: int, session: Session) -> Dict[str, Any]:
        """Get financial summary (total revenue, total expense) for a company."""
        summary = {"total_revenue": 0.0, "total_expense": 0.0}
        try:
            # Get total revenue - Filter by users belonging to the company
            revenue_query = """
                SELECT SUM(r.amount) 
                FROM revenues r
                JOIN users u ON r.created_by = u.id
                WHERE u.created_by = :company_id -- Assumes user.created_by is company_id
            """
            # Adjust join/where condition based on actual user-company linkage
            # If users table has company_id: WHERE u.company_id = :company_id
            
            revenue_result = session.execute(revenue_query, {"company_id": company_id}).scalar()
            if revenue_result:
                summary["total_revenue"] = float(revenue_result)

            # Get total expense - Filter by users belonging to the company
            expense_query = """
                SELECT SUM(e.amount) 
                FROM expenses e
                JOIN users u ON e.created_by = u.id
                WHERE u.created_by = :company_id -- Assumes user.created_by is company_id
            """
            # Adjust join/where condition based on actual user-company linkage
            
            expense_result = session.execute(expense_query, {"company_id": company_id}).scalar()
            if expense_result:
                summary["total_expense"] = float(expense_result)
                
            return summary

        except SQLAlchemyError as e:
            logger.error(f"Database error retrieving financial summary for company {company_id}: {str(e)}")
            # Return default summary or raise? Returning default for now.
            return summary
            # raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="DB error getting financial summary")


    @staticmethod
    async def get_expense_breakdown(company_id: int, session: Session, limit: int = 10) -> List[Dict[str, Any]]:
        """Get expense breakdown by name/category for a company."""
        try:
             # Group by 'name' as there's no category_id in expenses table
            query = """
                SELECT e.name as category, SUM(e.amount) as total_amount
                FROM expenses e
                JOIN users u ON e.created_by = u.id
                WHERE u.created_by = :company_id -- Assumes user.created_by is company_id
                GROUP BY e.name
                ORDER BY total_amount DESC
                LIMIT :limit
            """
            # Adjust join/where condition based on actual user-company linkage
            
            result = session.execute(query, {"company_id": company_id, "limit": limit}).fetchall()
            return [dict(row._mapping) for row in result]

        except SQLAlchemyError as e:
            logger.error(f"Database error retrieving expense breakdown for company {company_id}: {str(e)}")
            return []
            # raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="DB error getting expense breakdown")


    @staticmethod
    async def insert_expense(company_id: int, user_id: int, parameters: Dict[str, Any], session: Session) -> Optional[int]:
        """Insert a new expense record after verifying user belongs to the company."""
        try:
            # Verify user belongs to the target company
            user_company_id = await DatabaseQueries._get_user_company_id(user_id, session)
            if user_company_id != company_id:
                 logger.error(f"User {user_id} does not belong to company {company_id}. Cannot insert expense.")
                 raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User does not belong to the specified company.")

            # Prepare data for insertion - ensure all required fields from parameters are present
            insert_data = {
                "name": parameters.get("category"), # Map 'category' from agent params to 'name' field
                "date": parameters.get("date"),
                "amount": parameters.get("amount"),
                "description": parameters.get("description"),
                "attachment": parameters.get("attachment"), # Optional
                "project_id": parameters.get("project_id", 0), # Optional, default 0
                "task_id": parameters.get("task_id", 0), # Optional, default 0
                "created_by": user_id # Set the creator
            }
            
            # Validate essential fields
            if not all([insert_data["name"], insert_data["date"], insert_data["amount"] is not None]):
                 raise ValueError("Missing required fields for expense insertion (name/category, date, amount)")

            # Insert into expenses table
            query = """
                INSERT INTO expenses (name, date, amount, description, attachment, project_id, task_id, created_by)
                VALUES (:name, :date, :amount, :description, :attachment, :project_id, :task_id, :created_by)
            """
            
            result = session.execute(query, insert_data)
            session.commit()
            
            expense_id = result.lastrowid
            logger.info(f"Inserted expense record with ID: {expense_id} by user {user_id} for company {company_id}")
            return expense_id

        except (SQLAlchemyError, ValueError) as e:
            session.rollback()
            logger.error(f"Error inserting expense for company {company_id} by user {user_id}: {str(e)}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error inserting expense: {str(e)}")
        except HTTPException as he:
             # Re-raise forbidden error
             raise he


    @staticmethod
    async def insert_revenue(company_id: int, user_id: int, parameters: Dict[str, Any], session: Session) -> Optional[int]:
        """Insert a new revenue record after verifying user and related entities belong to the company."""
        try:
            # 1. Verify user belongs to the target company
            user_company_id = await DatabaseQueries._get_user_company_id(user_id, session)
            if user_company_id != company_id:
                 logger.error(f"User {user_id} does not belong to company {company_id}. Cannot insert revenue.")
                 raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User does not belong to the specified company.")

            # 2. Prepare data and validate required fields from parameters
            insert_data = {
                "date": parameters.get("date"),
                "amount": parameters.get("amount"),
                "account_id": parameters.get("account_id"),
                "customer_id": parameters.get("customer_id"),
                "category_id": parameters.get("category_id"), # Corresponds to ProductServiceCategory
                "payment_method": parameters.get("payment_method"), # Needs validation? Assumes integer ID
                "reference": parameters.get("reference"),
                "description": parameters.get("description", parameters.get("source")), # Use description or source
                "add_receipt": parameters.get("add_receipt"), # Optional file path/name
                "created_by": user_id
            }

            # Validate essential fields
            required = ["date", "amount", "account_id", "customer_id", "category_id", "payment_method"]
            if not all(insert_data.get(k) is not None for k in required):
                 missing = [k for k in required if insert_data.get(k) is None]
                 raise ValueError(f"Missing required fields for revenue insertion: {', '.join(missing)}")

            # 3. (Crucial) Verify related entities (customer, account, category) belong to the company
            #    This requires additional queries. Adjust based on how these entities link to companies.
            # Example Verification (adjust queries based on actual schema):
            # Customer check:
            customer_check_query = "SELECT id FROM customers WHERE id = :customer_id AND created_by IN (SELECT id FROM users WHERE created_by = :company_id)" 
            # Or if Customer has company_id: "SELECT id FROM customers WHERE id = :customer_id AND company_id = :company_id"
            customer_valid = session.execute(customer_check_query, {"customer_id": insert_data["customer_id"], "company_id": company_id}).fetchone()
            if not customer_valid:
                 raise ValueError(f"Customer ID {insert_data['customer_id']} not found or doesn't belong to company {company_id}")

            # Bank Account check (assuming BankAccount links via created_by user):
            account_check_query = "SELECT id FROM bank_accounts WHERE id = :account_id AND created_by IN (SELECT id FROM users WHERE created_by = :company_id)"
            account_valid = session.execute(account_check_query, {"account_id": insert_data["account_id"], "company_id": company_id}).fetchone()
            if not account_valid:
                raise ValueError(f"Bank Account ID {insert_data['account_id']} not found or doesn't belong to company {company_id}")
                
            # Category check (assuming ProductServiceCategory links via created_by user):
            category_check_query = "SELECT id FROM product_service_categories WHERE id = :category_id AND created_by IN (SELECT id FROM users WHERE created_by = :company_id)"
            category_valid = session.execute(category_check_query, {"category_id": insert_data["category_id"], "company_id": company_id}).fetchone()
            if not category_valid:
                 raise ValueError(f"Category ID {insert_data['category_id']} not found or doesn't belong to company {company_id}")


            # 4. Insert into revenues table
            query = """
                INSERT INTO revenues (date, amount, account_id, customer_id, category_id, payment_method, reference, description, add_receipt, created_by)
                VALUES (:date, :amount, :account_id, :customer_id, :category_id, :payment_method, :reference, :description, :add_receipt, :created_by)
            """
            
            result = session.execute(query, insert_data)
            session.commit()
            
            revenue_id = result.lastrowid
            logger.info(f"Inserted revenue record with ID: {revenue_id} by user {user_id} for company {company_id}")
            return revenue_id

        except (SQLAlchemyError, ValueError) as e:
            session.rollback()
            logger.error(f"Error inserting revenue for company {company_id} by user {user_id}: {str(e)}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error inserting revenue: {str(e)}")
        except HTTPException as he:
             # Re-raise forbidden/validation errors
             raise he
