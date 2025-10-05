#!/bin/bash
# Setup script for RunPod environment

# Update system
apt-get update && apt-get install -y \
    wget \
    xvfb \
    x11-utils \
    python3-pip \
    python3-dev \
    fontconfig \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglu1-mesa \
    libgl1-mesa-glx \
    libosmesa6-dev \
    software-properties-common

# Download CARLA 0.9.15
cd /workspace
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.15.tar.gz
mkdir -p carla_simulator
tar -xzf CARLA_0.9.15.tar.gz -C carla_simulator
rm CARLA_0.9.15.tar.gz

# Install Python dependencies
pip install --upgrade pip
pip install -r /workspace/carla-rl-cv/requirements.txt

# Download CARLA Python API
cd /workspace/carla_simulator/PythonAPI/carla/dist
pip install carla-0.9.15-cp38-cp38-linux_x86_64.whl

echo "Setup complete! Start CARLA with: ./scripts/run_carla.sh"