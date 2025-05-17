FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    git \
    libopenmpi-dev \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch and other dependencies
ENV CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"
RUN pip install --no-cache-dir \
    "torch==2.6.0" \
    wheel_stub \
    pybind11 \
    "Cython>=3.0.0"

# Clone and install NeMo
WORKDIR /workspace
COPY setup.py .
COPY nemo_export_deploy/__init__.py nemo_export_deploy/
COPY nemo_export_deploy/package_info.py nemo_export_deploy/
COPY requirements/ requirements/
RUN pip install --no-build-isolation --no-cache-dir '.[all]'

# Install development versions
RUN pip install --no-cache-dir \
    git+https://github.com/NVIDIA/Megatron-LM.git \
    git+https://github.com/NVIDIA/NeMo-Run.git


COPY . .