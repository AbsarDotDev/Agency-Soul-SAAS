import os
import logging
from typing import Optional, Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain_core.language_models import BaseChatModel
from langchain_community.embeddings import GooglePalmEmbeddings
from app.core.config import settings

# Set up logging
logger = logging.getLogger(__name__)

_llm_instance: Optional[BaseChatModel] = None
_embedding_model_instance: Optional[Any] = None

def get_llm(model_name: Optional[str] = None) -> BaseChatModel:
    """Get the Google Gemini language model instance. Initializes only once."""
    
    global _llm_instance
    
    default_model_to_use = "gemini-2.0-flash"

    if _llm_instance is not None and model_name is None: 
        logger.info("Returning cached LLM instance.")
        return _llm_instance

    if not settings.GOOGLE_API_KEY:
        logger.error("GOOGLE_API_KEY is not set in the environment.")
        raise ValueError("GOOGLE_API_KEY must be set to use the Gemini model.")

    target_model = model_name if model_name and model_name.startswith('gemini') else default_model_to_use
    logger.info(f"Attempting to initialize Google Gemini model: {target_model}")

    try:
        current_safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        llm_to_cache = ChatGoogleGenerativeAI(
            model=target_model, 
            google_api_key=settings.GOOGLE_API_KEY, 
            temperature=0.1,
            safety_settings=current_safety_settings
        )
        if model_name is None: 
            _llm_instance = llm_to_cache
        logger.info(f"Successfully initialized Google model: {target_model} with safety settings.")
        return llm_to_cache
    except Exception as e:
        logger.error(f"Failed to initialize Google model {target_model}: {str(e)}", exc_info=True)
        raise ValueError(f"Failed to initialize Google Gemini model {target_model}. Check API key, model name, and safety settings. Error: {str(e)}")

def get_embedding_model(provider: str = "google") -> Any:
    """Get an embedding model instance, prioritizing Google. Initializes only once."""
    global _embedding_model_instance

    if _embedding_model_instance is not None and provider == "google":
        logger.info("Returning cached embedding model instance.")
        return _embedding_model_instance
    
    if settings.GOOGLE_API_KEY:
         logger.info("Using Google Palm Embeddings")
         embedding_model_to_cache = GooglePalmEmbeddings(google_api_key=settings.GOOGLE_API_KEY)
         if provider == "google": 
            _embedding_model_instance = embedding_model_to_cache
         return embedding_model_to_cache
    
    raise ValueError("No GOOGLE_API_KEY set for the embedding provider.") 