# ============================================================
# Stage 1: Builder - Install dependencies
# ============================================================
FROM python:3.9-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ============================================================
# Stage 2: Runner - Production image
# ============================================================
FROM python:3.9-slim

LABEL maintainer="RVC ONNX Exporter"
LABEL description="HTTP API for batch converting RVC .pth model files to ONNX format"

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    mkdir -p /app/data && \
    chown -R appuser:appuser /app

WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser api.py .
COPY --chown=appuser:appuser infer_pack/ ./infer_pack/
COPY --chown=appuser:appuser static/ ./static/

# Ensure data directories exist with proper permissions
RUN mkdir -p /app/data/exported_onnx /app/data/uploaded_pth /app/data/static && \
    chown -R appuser:appuser /app/data

# Switch to non-root user
USER appuser

# Expose port (can be overridden via PORT env var)
EXPOSE 8000

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PORT=8000 \
    HOST=0.0.0.0 \
    DATA_DIR=/app/data \
    CLEANUP_INTERVAL_MINUTES=5 \
    FILE_EXPIRE_MINUTES=30

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run the application
CMD ["python", "api.py"]
