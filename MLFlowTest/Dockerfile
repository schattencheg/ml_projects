FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ ./src/
COPY api/ ./api/
COPY config/ ./config/
COPY .env.example ./.env

# Create necessary directories
RUN mkdir -p models data logs artifacts

# Expose API port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app/src
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the API server
CMD ["python", "api/app.py"]
