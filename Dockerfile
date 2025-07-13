# Multi-stage Docker build for Informer Option Pricing Model
# Stage 1: Base image with Python and system dependencies
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libhdf5-dev \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Development image
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest>=7.0.0 \
    pytest-cov>=4.0.0 \
    black>=23.0.0 \
    isort>=5.12.0 \
    flake8>=6.0.0 \
    mypy>=1.0.0 \
    pre-commit>=3.0.0 \
    jupyter>=1.0.0 \
    ipykernel>=6.0.0

# Copy source code
COPY . .

# Install the package in development mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p logs checkpoints experiments data

# Set up pre-commit hooks
RUN pre-commit install

# Default command for development
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Stage 3: Production image
FROM base as production

# Copy only necessary files
COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/
COPY pyproject.toml .

# Install the package
RUN pip install .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app

# Create necessary directories with proper permissions
RUN mkdir -p logs checkpoints experiments data && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port for API (if needed)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print('PyTorch version:', torch.__version__)" || exit 1

# Default command for production
CMD ["python", "scripts/train.py"]

# Stage 4: Inference/API image
FROM production as inference

# Install additional dependencies for API
USER root
RUN pip install --no-cache-dir \
    fastapi>=0.104.0 \
    uvicorn>=0.24.0 \
    pydantic>=2.5.0

# Copy API files (if they exist)
COPY api/ api/ 2>/dev/null || true

# Switch back to non-root user
USER appuser

# Expose API port
EXPOSE 8000

# Command for API service
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Stage 5: GPU-enabled image
FROM nvidia/cuda:11.8-devel-ubuntu22.04 as gpu

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_VISIBLE_DEVICES=0

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    curl \
    git \
    libhdf5-dev \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set up Python
RUN ln -s /usr/bin/python3.10 /usr/bin/python
RUN python -m pip install --upgrade pip

# Set working directory
WORKDIR /app

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install the package
RUN pip install .

# Create necessary directories
RUN mkdir -p logs checkpoints experiments data

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app

USER appuser

# Health check for GPU
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('CUDA devices:', torch.cuda.device_count())" || exit 1

# Default command for GPU training
CMD ["python", "scripts/train.py"]