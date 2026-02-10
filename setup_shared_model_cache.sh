#!/bin/bash
# Setup shared DepthAI model cache for all users
# Fixes "Permission denied" errors when multiple users run person_detector.py

echo "Setting up shared DepthAI model cache..."

# Create shared cache directory
sudo mkdir -p /opt/depthai-cache
sudo chmod 777 /opt/depthai-cache

# Set environment variable system-wide
echo "export DEPTHAI_ZOO_CACHE=/opt/depthai-cache" | sudo tee /etc/profile.d/depthai.sh
sudo chmod 644 /etc/profile.d/depthai.sh

# Clean up any existing cache in /tmp
echo "Cleaning up old cache files in /tmp..."
sudo rm -rf /tmp/yolov6n-r2-288x512.rvc2.tar.xz
sudo rm -rf /tmp/*.rvc2.tar.xz 2>/dev/null

echo ""
echo "✅ Done! Shared model cache configured at /opt/depthai-cache"
echo ""
echo "Users need to either:"
echo "  1. Log out and back in (to load new environment variable), OR"
echo "  2. Run: source /etc/profile.d/depthai.sh"
echo ""
echo "Then the model will be downloaded once and shared by everyone."
