# AgenySoul AI Agent

This is the FastAPI + Langchain agent for AgenySoul, ready for production deployment on Azure Container Apps.

## Features
- Modular multi-agent architecture (HRM, Sales, CRM, etc.)
- Strict company data isolation
- Token-based usage control
- Health checks and diagnostics
- Production-ready Dockerfile
- Python 3.12 compatibility
- Latest LangChain v0.3 framework

## Environment Files
- `.env.example`: Template for local development
- `.env`: (local development, not committed)
- `.env.prod`: For production (upload to Azure as environment variables)

## Dependency Management
This project uses [Poetry](https://python-poetry.org/) for Python dependency management.

### Install dependencies locally
```bash
poetry install
```

### Run locally
```bash
poetry run uvicorn main:app --reload
```

## Docker Build & Run

### Build the Docker image
```bash
docker build -t ageny-ai-agent:latest .
```

### Run the Docker container
```bash
docker run --env-file .env.prod -p 8000:8000 ageny-ai-agent:latest
```

## Azure Container Apps Deployment
1. Build and push the Docker image to your registry:
   ```bash
   docker build -t yourregistry.azurecr.io/ageny-ai-agent:latest .
   docker push yourregistry.azurecr.io/ageny-ai-agent:latest
   ```

2. Deploy to Azure Container Apps:
   - Create a new Container App in Azure Portal or via CLI
   - Specify your Docker image from the registry
   - Set your environment variables from `.env.prod`
   - Set the port to 8000
   - Configure scaling rules as needed

3. Verify deployment:
   - Check the health endpoint: `https://your-app-url/health`
   - Test the API endpoints with sample queries

## Project Structure
- `main.py`: FastAPI entrypoint
- `app/`: All agent, API, and core logic
- `tests/`: Test suite

## Notes
- All company data is strictly isolated (see `COMPANY_ISOLATION.md`)
- Only `.env.example` and `.env.prod` are tracked in git
- Poetry is used for all dependency management

---

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

## Running with Conda Environment

The application uses LangGraph which has specific dependencies. We've set up a conda environment named `nomessos` to manage these dependencies.

### Using the run script

The easiest way to run the application is to use the provided run script:

```bash
./run_app.sh
```

This script will:
1. Activate the `nomessos` conda environment
2. Start the FastAPI application

### Manual setup

If you prefer to set up things manually:

1. Activate the conda environment:
   ```bash
   conda activate nomessos
   ```

2. Run the application:
   ```bash
   python main.py
   ```

### Testing imports

To verify that your conda environment is correctly set up, you can run:

```bash
python test_imports.py
```

This will test all the necessary imports and confirm that the environment is correctly configured.
