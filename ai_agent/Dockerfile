FROM python:3.12-slim

WORKDIR /app

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy only the poetry files first for better caching
COPY pyproject.toml poetry.lock* ./

# Install dependencies (no dev dependencies)
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root --only main

# Copy application code
COPY main.py ./
COPY app/ ./app/
# COPY .env ./ # Avoid copying .env with potential secrets
# COPY .env.prod ./ # Avoid copying .env.prod with potential secrets
COPY .env.example ./

# Set environment variable for port
ENV PORT=8000

# Expose port for FastAPI
EXPOSE 8000

# Run the application directly with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"] 