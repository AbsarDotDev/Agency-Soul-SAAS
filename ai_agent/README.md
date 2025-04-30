# AgenySoul AI Agent

AgenySoul AI Agent is a company-specific chatbot system that provides insights, answers questions, creates visualizations, and can perform actions - all strictly limited to each company's own data.

## Overview

The system follows a multi-agent architecture with specialized agents for different domains:
- HRM Agent: Provides insights on human resource management data
- Sales Agent: Analyzes sales data and provides forecasts
- Finance Agent: Handles financial data analysis
- CRM Agent: Analyzes customer data and provides insights

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables in a `.env` file:
```
# Database configuration
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=ageny_soul

# Google Generative AI
GOOGLE_API_KEY=your_google_api_key

# Token settings
DEFAULT_TOKEN_ALLOCATION=100
```

3. Run the FastAPI server:
```bash
uvicorn app.main:app --reload
```

## Architecture

- **API Layer**: FastAPI-based RESTful API interface
- **Agent Core**: Multi-agent orchestration using LangGraph
- **Data Access Layer**: Secure database integration with company isolation
- **Visualization Module**: Creates chart data for frontend rendering
- **Token Management**: Tracks usage based on company subscription plans

## Security Features

- **Company Data Isolation**: Each company can only access their own data
- **Authentication**: JWT-based authentication for secure API access
- **Token Management**: Usage tracking and limiting based on company subscription plans

## Integration with Laravel

The AI agent microservice integrates with the main Laravel application through a RESTful API, allowing for:
- User authentication
- Conversation history management
- Token usage tracking
- Visualization rendering in the frontend

## Development

This project is built with Python and uses the following technologies:
- LangChain framework for the foundation
- LangGraph for multi-agent orchestration
- Google Gemini 2.5 Pro as the LLM provider
- FastAPI for the API interface
- SQLAlchemy for database interactions
