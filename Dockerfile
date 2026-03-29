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

# Install CPU-only PyTorch from official index
# IMPORTANT: Install numpy<2 BEFORE torch to avoid compatibility issues
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "numpy<2.0.0" && \
    pip install --no-cache-dir torch==2.1.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html && \
    pip install --no-cache-dir fastapi>=0.95.0 && \
    pip install --no-cache-dir uvicorn>=0.21.0 && \
    pip install --no-cache-dir python-multipart>=0.0.6 && \
    pip install --no-cache-dir onnx>=1.14.0 && \
    pip install --no-cache-dir onnxconverter-common>=1.14.0


# ============================================================
# Stage 2: Runner - Production image
# ============================================================
FROM python:3.9-slim

LABEL maintainer="RVC ONNX Exporter"
LABEL description="HTTP API for batch converting RVC .pth model files to ONNX format"

# Install runtime dependencies (no build tools needed)
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
