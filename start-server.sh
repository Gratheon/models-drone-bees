#!/bin/bash
set -e

echo "Starting Drone Bee Detection Server..."
echo "Model: project_beedetection/yolov10x_augm_pat30/weights/best.pt"
echo "Port: ${PORT:-8751}"
echo "Confidence threshold: ${CONF_THRESHOLD:-0.25}"
echo "IoU threshold: ${IOU_THRESHOLD:-0.7}"
echo "Image size: ${IMG_SIZE:-1280}"
echo ""

if [ ! -f "project_beedetection/yolov10x_augm_pat30/weights/best.pt" ]; then
    echo "ERROR: Model weights not found!"
    echo "Please ensure 'project_beedetection/yolov10x_augm_pat30/weights/best.pt' exists"
    exit 1
fi

exec python3 server.py

