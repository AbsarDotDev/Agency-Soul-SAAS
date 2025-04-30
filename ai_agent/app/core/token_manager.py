from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from fastapi import HTTPException, status
from typing import Optional
import logging
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)


class TokenManager:
    """Token manager for tracking and limiting AI agent usage."""

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
            # Execute raw SQL to get company's token count from the Laravel database
            # This assumes the Laravel database has a companies table with tokens_available column
            result = session.execute(
                "SELECT tokens_available, ai_agent_enabled FROM companies WHERE id = :company_id",
                {"company_id": company_id}
            ).fetchone()
            
            if not result:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Company with ID {company_id} not found"
                )
            
            tokens_available, ai_agent_enabled = result
            
            # Check if AI agent is enabled for the company
            if not ai_agent_enabled:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="AI Agent is not enabled for this company"
                )
            
            # Check if company has tokens available
            if tokens_available <= 0:
                return False
                
            return True
            
        except SQLAlchemyError as e:
            logger.error(f"Database error when checking token availability: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database error when checking token availability"
            )
    
    @staticmethod
    async def consume_token(company_id: int, session: Session, amount: int = 1) -> bool:
        """Consume tokens for a company.
        
        Args:
            company_id: Company ID
            session: Database session
            amount: Number of tokens to consume
            
        Returns:
            True if tokens were consumed successfully, False otherwise
            
        Raises:
            HTTPException: If there's a database error
        """
        try:
            # Check if company has enough tokens
            if not await TokenManager.check_token_availability(company_id, session):
                return False
            
            # Execute raw SQL to decrement tokens_available
            result = session.execute(
                """
                UPDATE companies 
                SET tokens_available = tokens_available - :amount,
                    tokens_used = tokens_used + :amount,
                    last_agent_use = :timestamp
                WHERE id = :company_id AND tokens_available >= :amount
                """,
                {
                    "company_id": company_id,
                    "amount": amount,
                    "timestamp": datetime.utcnow()
                }
            )
            
            session.commit()
            
            # Check if update was successful (affected rows > 0)
            if result.rowcount > 0:
                # Log token consumption
                session.execute(
                    """
                    INSERT INTO agent_token_usage (company_id, tokens_used, created_at)
                    VALUES (:company_id, :amount, :timestamp)
                    """,
                    {
                        "company_id": company_id,
                        "amount": amount,
                        "timestamp": datetime.utcnow()
                    }
                )
                
                session.commit()
                return True
            
            return False
            
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database error when consuming tokens: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database error when consuming tokens"
            )
    
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
            # Execute raw SQL to get company's token count
            result = session.execute(
                "SELECT tokens_available FROM companies WHERE id = :company_id",
                {"company_id": company_id}
            ).fetchone()
            
            if not result:
                return None
                
            return result[0]
            
        except SQLAlchemyError as e:
            logger.error(f"Database error when getting token count: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database error when getting token count"
            )
