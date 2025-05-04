import os
import logging
from typing import Optional, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import BaseChatModel
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import GooglePalmEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from app.core.config import settings

# Set up logging
logger = logging.getLogger(__name__)

def get_llm(model_name: Optional[str] = None) -> BaseChatModel:
    """Get the Google Gemini language model instance."""
    
    if not settings.GOOGLE_API_KEY:
        logger.error("GOOGLE_API_KEY is not set in the environment.")
        raise ValueError("GOOGLE_API_KEY must be set to use the Gemini model.")

    # Use the explicitly requested model if provided, otherwise default
    target_model = model_name if model_name and model_name.startswith('gemini') else "gemini-2.0-flash"
    logger.info(f"Attempting to initialize Google Gemini model: {target_model}")

    try:
        # Use ChatGoogleGenerativeAI 
        return ChatGoogleGenerativeAI(
            model=target_model, 
            google_api_key=settings.GOOGLE_API_KEY, 
            temperature=0.1
            # convert_system_message_to_human=True # REMOVED deprecated parameter
        )
    except Exception as e:
        logger.error(f"Failed to initialize Google model {target_model}: {str(e)}", exc_info=True)
        raise ValueError(f"Failed to initialize Google Gemini model {target_model}. Check API key and model name.")

def get_embedding_model(provider: str = "google") -> Any:
    """Get an embedding model instance, prioritizing Google."""
    # Prioritize Google
    if settings.GOOGLE_API_KEY:
         logger.info("Using Google Palm Embeddings")
         return GooglePalmEmbeddings(google_api_key=settings.GOOGLE_API_KEY)
    
    # Fallback removed as per request
    # elif settings.OPENAI_API_KEY:
    #     logger.info("Using OpenAI Embeddings as fallback")
    #     return OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)
    
    raise ValueError("No GOOGLE_API_KEY set for the embedding provider.") 