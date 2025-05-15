import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import time

from app.api.router import router
from app.core.llm import get_llm, get_embedding_model
from app.core.langgraph_agents import get_langgraph

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for FastAPI.
    This runs before the application starts and after it stops.
    """
    # Pre-load LLM, embeddings, and LangGraph on startup
    logger.info("Initializing LLM and embedding model...")
    start_time = time.time()
    
    # Initialize LLM
    llm = get_llm()
    logger.info(f"LLM initialized in {time.time() - start_time:.2f} seconds")
    
    # Initialize embedding model
    embedding_model = get_embedding_model()
    logger.info(f"Embedding model initialized in {time.time() - start_time:.2f} seconds")
    
    # Pre-compile and cache LangGraph
    logger.info("Compiling LangGraph...")
    graph_start_time = time.time()
    langgraph = get_langgraph()
    logger.info(f"LangGraph compiled in {time.time() - graph_start_time:.2f} seconds")
    
    # Initialize Visualization LangGraph Agent
    logger.info("Initializing Visualization LangGraph Agent...")
    viz_start_time = time.time()
    from app.core.dependencies import get_visualization_langgraph_agent
    viz_agent = get_visualization_langgraph_agent()
    logger.info(f"Visualization LangGraph agent initialized in {time.time() - viz_start_time:.2f} seconds")
    
    logger.info(f"Total initialization time: {time.time() - start_time:.2f} seconds")
    
    # Application is now ready to receive requests
    logger.info("FastAPI application is starting up!")
    yield
    # Clean up when the application is shutting down
    logger.info("FastAPI application is shutting down")

app = FastAPI(
    title="AgentSoul AI API",
    description="AI agent API for AgentSoul",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
if os.getenv("CORS_ORIGINS"):
    origins = os.getenv("CORS_ORIGINS", "").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    # Allow all origins in development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Add global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An unexpected error occurred. Please try again later."
        }
    )

# Include API router
app.include_router(router, prefix="/api")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    
    # Determine port
    port = int(os.environ.get("PORT", 8080))
    
    # Start server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        workers=1
    )
