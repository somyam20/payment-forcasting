# # Multi-stage build for Meeting Document Generator
# FROM python:3.11-slim as base

# # Set working directory
# WORKDIR /app

# # Set environment variables
# ENV PYTHONUNBUFFERED=1 \
#     PYTHONDONTWRITEBYTECODE=1 \
#     PIP_NO_CACHE_DIR=1 \
#     PIP_DISABLE_PIP_VERSION_CHECK=1

# # Install system dependencies
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     # Build essentials
#     build-essential \
#     gcc \
#     g++ \
#     # FFmpeg for video/audio processing
#     ffmpeg \
#     # Tesseract OCR for text detection
#     tesseract-ocr \
#     # Graphviz for diagram generation
#     graphviz \
#     # OpenCV dependencies
#     libglib2.0-0 \
#     libsm6 \
#     libxrender1 \
#     libxext6 \
#     # libgl1-mesa-glx \
#     libglib2.0-0 \
#     # Audio processing libraries
#     libsndfile1 \
#     # Node.js and npm for Mermaid CLI
#     curl \
#     ca-certificates \
#     && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
#     && apt-get install -y nodejs \
#     && npm install -g @mermaid-js/mermaid-cli \
#     # Cleanup
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

# # Copy requirements first for better caching
# COPY requirements.txt .

# # Install Python dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy application code
# COPY . .

# # Create necessary directories
# RUN mkdir -p data/outputs local_storage config

# # Expose Streamlit port
# EXPOSE 8000

# CMD ["python","app.py"]


# Multi-stage build for Meeting Document Generator
FROM python:3.12-slim AS builder

# Install build dependencies only in builder stage
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Build Python wheels
COPY requirements-freeze.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements-freeze.txt

# Runtime stage
FROM python:3.12-slim

WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    tesseract-ocr \
    graphviz \
    libsm6 \
    libxrender1 \
    libxext6 \
    libglib2.0-0 \
    libgomp1 \
    libsndfile1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js and Mermaid CLI (optimized)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    npm install -g @mermaid-js/mermaid-cli && \
    npm cache clean --force && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /root/.npm /tmp/*

# Copy wheels from builder and install Python packages
COPY --from=builder /wheels /wheels
COPY requirements-freeze.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements-freeze.txt && \
    rm -rf /wheels

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/outputs local_storage config

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose API port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]