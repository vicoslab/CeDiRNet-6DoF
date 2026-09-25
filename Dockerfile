# Public CeDiRNet-6DoF inference image.
# Blackwell-compatible runtime: CUDA 12.8 + PyTorch 2.7.
# Build from repository root:
#   docker build -t cedirnet-6dof:latest .
# Run with mounted observation/model folders; see README.md.
ARG UBUNTU_VERSION=22.04
ARG CUDA_VERSION=12.8.1-cudnn

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev build-essential git wget curl \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install --no-cache-dir -r /app/requirements.txt

COPY license.txt THIRD_PARTY_NOTICES.md /app/
COPY LICENSES /app/LICENSES
COPY src /app/src
COPY tools /app/tools
COPY models/model_definitions.json /app/models/model_definitions.json
COPY models/configs /app/models/configs

ENTRYPOINT ["python3", "/app/tools/infer_observation.py"]
