#!/bin/bash

# Allow root user in Docker to access X server
xhost +si:localuser:root

# Run Docker container with bash shell
docker run -it --rm \
    --gpus all \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /home/junjieg2/sam_6d:/workspace \
    -w /workspace \
    lihualiu/sam-6d:1.0 \
    bash  # ← this is the missing part

# Revoke X access
xhost -si:localuser:root
