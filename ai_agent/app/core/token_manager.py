from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from fastapi import HTTPException, status
from typing import Optional
import logging
from datetime import datetime
from sqlalchemy import text

# Set up logging
logger = logging.getLogger(__name__)


class TokenManager:
    """Token manager for tracking and limiting AI agent usage using 'users' and 'plans' tables."""

    @staticmethod
    async def check_token_availability(company_id: int, session: Session) -> bool:
        """Check if company has available tokens.
        
        Args:
            company_id: Company ID
            session: Database session
            
        Returns:
            True if tokens are available, False otherwise
            
        Raises:
            HTTPException: If there's a database error
        """
        try:
            # Query users and plans table
            query = text("""
                SELECT u.ai_agent_tokens_used, p.ai_agent_enabled, p.ai_agent_default_tokens
                FROM users u
                JOIN plans p ON u.plan = p.id
                WHERE u.id = :company_id AND u.type = 'company'
            """)
            result = session.execute(query, {"company_id": company_id}).fetchone()
            
            if not result:
                # Log specific reason
                logger.warning(f"Company user not found or not of type 'company' for ID: {company_id}")
                # Raise 404 to indicate the specific user (company) wasn't found as expected
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Company user with ID {company_id} not found or is not a company account."
                )
            
            tokens_used, ai_agent_enabled, tokens_allocated = result
            
            # Check if AI agent is enabled for the company's plan
            if not ai_agent_enabled:
                # Log specific reason
                logger.info(f"AI Agent access denied for company {company_id}: Feature not enabled in plan.")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="AI Agent is not enabled for your company's subscription plan."
                )
            
            # Check if company has tokens available (allocated > used)
            if tokens_allocated <= tokens_used:
                # Log specific reason
                logger.info(f"AI Agent access denied for company {company_id}: Insufficient tokens (Used: {tokens_used}, Allocated: {tokens_allocated}).")
                # Return False, let the caller handle the 403 if needed
                return False
                
            # If enabled and tokens available > used
            logger.debug(f"Token check passed for company {company_id}. Used: {tokens_used}, Allocated: {tokens_allocated}")
            return True
            
        except SQLAlchemyError as e:
            logger.error(f"Database error checking token availability for company {company_id}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database error checking token availability."
            )
    
    @staticmethod
    async def consume_token(company_id: int, session: Session, amount: int = 1) -> bool:
        """DEPRECATED/INCORRECT - Use BaseAgent._update_token_usage instead.
           This method attempts to update a non-existent table/column.
           Keeping for reference but should not be used.
        """
        logger.warning(f"TokenManager.consume_token called for company {company_id}, but this method is deprecated and likely incorrect. Use BaseAgent._update_token_usage.")
        # Returning False to indicate failure, as this logic is wrong.
        # The actual token update happens in BaseAgent._update_token_usage.
        return False
        # --- Original Incorrect Logic (Kept for reference) ---
        # try:
        #     if not await TokenManager.check_token_availability(company_id, session):
        #         return False
        #     
        #     # Incorrect: Tries to update non-existent 'companies' table
        #     # The correct update happens in BaseAgent._update_token_usage
        #     result = session.execute(text(...))
        #     session.commit()
        #     
        #     if result.rowcount > 0:
        #        # Incorrect: Tries to insert into non-existent 'agent_token_usage'
        #        session.execute(text(...))
        #        session.commit()
        #        return True
        #     
        #     return False
        #     
        # except SQLAlchemyError as e:
        #     session.rollback()
        #     logger.error(f"Database error in deprecated consume_token: {str(e)}")
        #     raise HTTPException(...)

    @staticmethod
    async def get_token_count(company_id: int, session: Session) -> Optional[int]:
        """Get available token count for a company.
        
        Args:
            company_id: Company ID
            session: Database session
            
        Returns:
            Number of available tokens, or None if company not found
            
        Raises:
            HTTPException: If there's a database error
        """
        try:
            # Query users and plans table to calculate remaining tokens
            query = text("""
                SELECT u.ai_agent_tokens_used, p.ai_agent_default_tokens
                FROM users u
                JOIN plans p ON u.plan = p.id
                WHERE u.id = :company_id AND u.type = 'company'
            """)
            result = session.execute(query, {"company_id": company_id}).fetchone()
            
            if not result:
                logger.warning(f"Could not get token count: Company user {company_id} not found or not type 'company'.")
                return None # Return None if company/plan not found
            
            tokens_used, tokens_allocated = result
            remaining = max(0, tokens_allocated - tokens_used)
            logger.debug(f"Calculated remaining tokens for company {company_id}: {remaining} (Allocated: {tokens_allocated}, Used: {tokens_used})")
            return remaining
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting token count for company {company_id}: {str(e)}")
            # Don't raise HTTPException here, allow the caller (agent) to handle it
            # Returning None indicates an issue fetching the count.
            return None
