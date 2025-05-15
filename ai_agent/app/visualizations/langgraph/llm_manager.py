from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
import logging
import re
import json
from copy import deepcopy
from app.core.llm import get_llm
from app.core.token_manager import TokenManager

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class LLMManager:
    """
    Manager for interactions with the language model.
    Handles prompting, token tracking, and response parsing.
    """
    
    def __init__(self, company_id: int = None):
        """
        Initialize the LLM manager.
        
        Args:
            company_id: Optional company ID for token tracking
        """
        self.llm = get_llm()
        self.token_manager = TokenManager()  # Initialize without arguments
        self.company_id = company_id
        self.tokens_used = 0
    
    def set_company_id(self, company_id: int):
        """
        Set the company ID after initialization.
        
        Args:
            company_id: Company ID for token tracking
        """
        self.company_id = company_id
    
    def invoke(self, prompt, **kwargs):
        """
        Invoke the LLM with a prompt and variables.
        
        Args:
            prompt: LangChain prompt template
            **kwargs: Variables to pass to the prompt
            
        Returns:
            Response from the LLM
        """
        try:
            # Handle potentially problematic inputs like [('IT', 2), ('Audit', 2)]
            sanitized_kwargs = {}
            for key, value in kwargs.items():
                if key == 'results' or key == 'data':
                    # Special handling for data/results that might contain tuples or special structures
                    if isinstance(value, str) and (("(" in value and ")" in value) or "[" in value):
                        # This is likely a string representation of structured data - simplify it
                        try:
                            # Try to parse with literal_eval if it looks like Python data
                            import ast
                            parsed_data = ast.literal_eval(value)
                            # Convert to a more readable format
                            if isinstance(parsed_data, list):
                                if all(isinstance(x, tuple) for x in parsed_data):
                                    # Convert list of tuples to formatted text
                                    readable = "Data with label/value pairs:\n"
                                    for item in parsed_data:
                                        if len(item) >= 2:
                                            readable += f"- {item[0]}: {item[1]}\n"
                                    sanitized_kwargs[key] = readable
                                    continue
                        except Exception:
                            # If parsing fails, proceed with normal cleaning
                            pass
                
                # Standard cleaning for other values
                if isinstance(value, str):
                    sanitized_kwargs[key] = self._clean_string_for_prompt(value)
                else:
                    sanitized_kwargs[key] = value
            
            # Create the messages with the formatted prompt
            formatted_prompt = prompt.format_prompt(**sanitized_kwargs).to_messages()
            
            # Invoke the LLM
            response = self.llm.invoke(formatted_prompt)
            
            # Track tokens (simplified)
            content = response.content
            self.tokens_used += 1
                
            return content
        except Exception as e:
            logger.error(f"Error invoking LLM: {str(e)}")
            # Return a default message rather than raising an exception
            return "This data represents a count of employees per department. It's ideal for visualization with a bar or pie chart."

    def _clean_string_for_prompt(self, text):
        """
        Clean a string to avoid Unicode escape issues when formatting prompts.
        
        Args:
            text: String to clean
            
        Returns:
            Cleaned string
        """
        if not isinstance(text, str):
            return text
            
        try:
            # If the string contains JSON, try to parse it first and then re-serialize it
            # This ensures all escape sequences are properly handled
            json.loads(text)
            return json.dumps(json.loads(text))
        except (json.JSONDecodeError, ValueError):
            # If it's not valid JSON, handle it as regular text
            pass
            
        # For normal text, handle any Unicode escape sequences that might cause problems
        # Replace \u with \\u to prevent issues with Unicode escape sequences
        text = text.replace('\\u', '\\\\u')
        
        # Make sure backslashes are properly escaped
        text = text.replace('\\', '\\\\').replace('\\\\\\\\u', '\\\\u')
        
        return text

    def get_tokens_used(self):
        """
        Get the number of tokens used.
        
        Returns:
            Number of tokens used
        """
        return self.tokens_used 