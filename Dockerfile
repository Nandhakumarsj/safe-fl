# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Avoid interactive prompts during apt package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX"

WORKDIR /workspace

# Install system utilities, C++ build toolchain, Python, and Ninja
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    curl \
    ca-certificates \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install PyTorch with CUDA 12.4 support
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python3 -m pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Copy dependency definition and install Python requirements
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt && \
    python3 -m pip install --no-cache-dir ninja python-docx

# Copy project source code
COPY . /workspace

# Build and install the CUDA extension modules ahead of time
RUN python3 setup.py build_ext --inplace

# Default smoke test verification
CMD ["python3", "run_all_smoke.py"]
