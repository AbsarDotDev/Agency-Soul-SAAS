from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from fastapi import HTTPException, status
from typing import Optional
import logging
from datetime import datetime
from sqlalchemy import text
import tiktoken # Import tiktoken

# Set up logging
logger = logging.getLogger(__name__)

# Default encoding for many models like gpt-4, gpt-3.5-turbo, text-embedding-ada-002
# Use cl100k_base for Gemini models as per general recommendations
DEFAULT_ENCODING = "cl100k_base" 

class TokenManager:
    """Token manager for tracking and limiting AI agent usage using 'users' and 'plans' tables."""
    
    _encoding = None

    @classmethod
    def _get_encoding(cls):
        """Get the tiktoken encoding instance, initializing if needed."""
        if cls._encoding is None:
            try:
                cls._encoding = tiktoken.get_encoding(DEFAULT_ENCODING)
            except Exception as e:
                logger.error(f"Failed to get tiktoken encoding '{DEFAULT_ENCODING}': {e}. Falling back to basic split.")
                # Fallback if encoding fails - provides a very rough estimate
                cls._encoding = 'fallback' 
        return cls._encoding

    @staticmethod
    def count_tokens(text: str) -> int:
        """Count the number of tokens in a given text string.
        
        Args:
            text: The text string to count tokens for.
            
        Returns:
            The estimated number of tokens.
        """
        if not text:
            return 0
        
        # Simplified token counting logic as requested:
        # Check if the text likely contains visualization JSON
        if any(term in text.lower() for term in ["chart", "visualization", "datasets", "labels", "chart_type"]):
            return 2  # Use 2 tokens for visualization-related text
        else:
            return 1  # Use 1 token for simple text responses
        
        # Old logic (commented out)
        """
        encoding = TokenManager._get_encoding()
        
        if encoding == 'fallback':
            # Very basic fallback: count words
            return len(text.split())
        elif encoding:
            try:
                return len(encoding.encode(text))
            except Exception as e:
                logger.error(f"Error encoding text with tiktoken: {e}. Falling back to basic split.")
                return len(text.split())
        else:
            # Should not happen if _get_encoding works, but handle defensively
            return len(text.split())
        """

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
            return False
        except Exception as e:
            logger.error(f"Unexpected error checking token availability for company {company_id}: {e}")
            return False
    
    @staticmethod
    async def consume_token(company_id: int, session: Session, tokens_to_consume: int = 1) -> bool:
        """DEPRECATED/INCORRECT - Use BaseAgent._update_token_usage instead.
           This method attempts to update a non-existent table/column.
           Keeping for reference but should not be used.
        """
        logger.warning(f"TokenManager.consume_token called for company {company_id}, but this method is deprecated and likely incorrect. Use BaseAgent._update_token_usage.")
        if tokens_to_consume <= 0:
             logger.warning(f"Attempted to consume {tokens_to_consume} tokens for company {company_id}. Skipping update.")
             return True # Indicate success as no update needed
        try:
            # Update users table
            update_query = text("""
                UPDATE users
                SET ai_agent_tokens_used = ai_agent_tokens_used + :tokens_to_consume
                WHERE id = :company_id AND type = 'company'
            """)
            result = session.execute(update_query, {"company_id": company_id, "tokens_to_consume": tokens_to_consume})
            session.commit()
            
            if result.rowcount > 0:
                logger.debug(f"Successfully consumed {tokens_to_consume} tokens for company {company_id}")
                return True
            else:
                logger.warning(f"No tokens consumed for company {company_id}: No matching records found.")
                return True # Indicate success as no update needed
            
        except SQLAlchemyError as e:
             logger.error(f"Database error consuming {tokens_to_consume} tokens for company {company_id}: {e}")
             return False
        except Exception as e:
             logger.error(f"Unexpected error consuming tokens for company {company_id}: {e}")
             return False

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
