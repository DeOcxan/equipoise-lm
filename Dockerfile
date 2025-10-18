# Energy-Based Transformers Mastery - Production Docker Image
# Ensures reproducibility across different environments
# Supports both CPU and GPU training

# Base image with Python 3.12 and CUDA support
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0" \
    FORCE_CUDA=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3-pip \
    python3.12-venv \
    git \
    wget \
    curl \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Set working directory
WORKDIR /workspace/EBT-Mastery

# Copy requirements first (for layer caching)
COPY requirements.txt .
COPY environment.yml .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Install project in development mode
RUN pip install -e .

# Create directories for data and outputs
RUN mkdir -p /workspace/data /workspace/outputs

# Set permissions
RUN chmod -R 755 /workspace

# Expose port for Jupyter (optional)
EXPOSE 8888

# Default command (can be overridden)
CMD ["/bin/bash"]

