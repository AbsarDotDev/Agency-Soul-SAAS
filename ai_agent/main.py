import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.api.routes import router as api_router
from app.core.config import settings
from app.database.connection import DatabaseConnection

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run on startup
    logger.info("Starting up AI Agent service...")
    # Test database connection
    try:
        engine = DatabaseConnection.create_engine()
        with engine.connect() as conn:
            logger.info("Database connection successful.")
    except Exception as e:
        logger.error(f"Failed to connect to database during startup: {str(e)}")
        # Optionally, you might want to raise an exception here to prevent startup
        # raise RuntimeError("Database connection failed on startup")
    yield
    # Code to run on shutdown
    logger.info("Shutting down AI Agent service...")

# Create the FastAPI application with the lifespan manager
app = FastAPI(
    title="AgenySoul AI Agent API",
    description="AI Agent for interacting with AgenySoul company data",
    version="0.1.0",
    lifespan=lifespan
)

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, you should specify domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create database session factory (no need to create the session here)
# Session = DatabaseConnection.create_session_factory()

# No longer needed with lifespan
# @app.on_event("startup")
# async def startup_event():
#     pass
#
# @app.on_event("shutdown")
# async def shutdown_event():
#     pass

# Include API routes
app.include_router(api_router)

# Add a health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint for the AI Agent API."""
    return {"status": "healthy"}

# Add a route for database connection test
@app.get("/test-db", tags=["Health"])
async def test_db():
    """Test database connection."""
    try:
        engine = DatabaseConnection.create_engine()
        with engine.connect() as conn:
            result = conn.execute("SELECT 1")
            return {"status": "Database connection successful"}
    except Exception as e:
        logger.error(f"Failed to connect to database: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database connection failed: {str(e)}"
        )

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
