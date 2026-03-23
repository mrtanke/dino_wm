FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/opt/conda/bin:$PATH
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libglfw3 \
    libglfw3-dev \
    libglew2.2 \
    libosmesa6-dev \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy only env file first for build cache
COPY environment.yaml /tmp/environment.yaml

# Create conda env from repo definition
RUN conda env create -f /tmp/environment.yaml

SHELL ["/bin/bash", "-lc"]

# Upgrade wandb for modern API key support
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate dino_wm && \
    pip install --no-cache-dir --upgrade "wandb==0.22.3"

# Auto-activate conda env in interactive shell
RUN echo "source /opt/conda/etc/profile.d/conda.sh && conda activate dino_wm" >> /root/.bashrc

# Runtime env vars
ENV DATASET_DIR=/data
ENV WANDB_DIR=/workspace/wandb
ENV LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH}

CMD ["/bin/bash"]