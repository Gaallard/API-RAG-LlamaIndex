# Multi-stage build for RAG FastAPI Application
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies (for PDF processing)
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY app/ ./app/
COPY pyproject.toml ./

# Create data directory
RUN mkdir -p /app/data

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DATA_DIR=/app/data

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/')" || exit 1

# Run uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

