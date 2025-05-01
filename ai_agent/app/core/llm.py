import os
import logging
from typing import Optional, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import BaseChatModel
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import GooglePalmEmbeddings

from app.core.config import settings

# Set up logging
logger = logging.getLogger(__name__)

def get_llm(model_name: Optional[str] = None) -> BaseChatModel:
    """Get a language model instance.
    
    Args:
        model_name: Optional model name. If not provided, will use default models.
            Can be 'gpt-4-turbo', 'gpt-3.5-turbo', 'gemini-pro', etc.
        
    Returns:
        LLM instance
    """
    # If a specific model is requested, determine the provider
    if model_name:
        if model_name.startswith('gpt'):
            return _get_openai_model(model_name)
        elif model_name.startswith('gemini'):
            return _get_google_model(model_name)
    
    # Try to use OpenAI model if API key is set
    if settings.OPENAI_API_KEY:
        try:
            return _get_openai_model()
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI model: {str(e)}")
    
    # Fall back to Google model
    if settings.GOOGLE_API_KEY:
        try:
            return _get_google_model()
        except Exception as e:
            logger.warning(f"Failed to initialize Google model: {str(e)}")
            raise ValueError("No available LLM could be initialized. Check API keys.")
    
    raise ValueError("No API keys set for any LLM provider")

def _get_openai_model(model_name: str = "gpt-4-turbo") -> ChatOpenAI:
    """Get an OpenAI model instance.
    
    Args:
        model_name: Model name
        
    Returns:
        ChatOpenAI instance
    """
    return ChatOpenAI(
        api_key=settings.OPENAI_API_KEY,
        model=model_name,
        temperature=0.1
    )

def _get_google_model(model_name: str = "gemini-2.0-flash") -> ChatGoogleGenerativeAI:
    """Get a Google Generative AI model instance.
    
    Args:
        model_name: Model name
        
    Returns:
        ChatGoogleGenerativeAI instance
    """
    return ChatGoogleGenerativeAI(
        google_api_key=settings.GOOGLE_API_KEY,
        model=model_name,
        temperature=0.1,
        convert_system_message_to_human=True
    )

def get_embedding_model(provider: str = "openai") -> Any:
    """Get an embedding model instance.
    
    Args:
        provider: Embedding provider ('openai' or 'google')
        
    Returns:
        Embedding model instance
    """
    if provider == "openai" and settings.OPENAI_API_KEY:
        return OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)
    elif provider == "google" and settings.GOOGLE_API_KEY:
        return GooglePalmEmbeddings(google_api_key=settings.GOOGLE_API_KEY)
    
    # Fall back to available provider
    if settings.OPENAI_API_KEY:
        return OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)
    elif settings.GOOGLE_API_KEY:
        return GooglePalmEmbeddings(google_api_key=settings.GOOGLE_API_KEY)
    
    raise ValueError("No API keys set for any embedding provider") 