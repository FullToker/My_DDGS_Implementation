# Dockerfile for D2GS (Depth-and-Density Guided Gaussian Splatting)
# Uses pip instead of conda

FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Set CUDA architectures for compilation (required when building without GPU)
# Covers: V100(7.0), T4/RTX20xx(7.5), A100(8.0), RTX30xx(8.6), RTX40xx(8.9)
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9+PTX"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch with CUDA 12.1 support
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Install Python dependencies
RUN pip install --no-cache-dir \
    tqdm \
    plyfile \
    matplotlib \
    torchmetrics==1.2.0 \
    opencv-python \
    scipy \
    scikit-learn \
    pykeops==2.3.0

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Build CUDA extensions
# Build diff-gaussian-rasterization
RUN cd /app/submodules/diff-gaussian-rasterization && \
    pip install --no-cache-dir  --no-build-isolation .

# Build simple-knn
RUN cd /app/submodules/simple-knn && \
    pip install --no-cache-dir --no-build-isolation .

# Default command
CMD ["bash"]
