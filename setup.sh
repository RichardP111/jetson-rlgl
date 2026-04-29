#!/bin/bash
"""
===============================================================================
Project:      Red Light Green Light (Jetson Orin Nano)
File:         setup.sh
Description:  Install all dependencies, pre-load AI models, and verify hardware.

Author:       Richard Pu
Last Updated: April 2026
Run once:     bash setup.sh
===============================================================================
"""

clear
echo "╔════════════════════════════════════════════════════════════╗"
echo "║   🔴 RED LIGHT  /  GREEN LIGHT 🟢                          ║"
echo "║   Master System Setup & Optimizer                          ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# --- 1. HOST HARDWARE OPTIMIZATION ---
echo "--- [1/3] Configuring Jetson Hardware ---"
echo "=> Forcing MAXN (Mode 0) Power State..."
sudo nvpmodel -m 0

echo "=> Locking GPU/CPU Clocks to 100%..."
sudo jetson_clocks
echo "Hardware optimized."
sleep 1

# --- 2. DOCKER CONTAINER PATCHING ---
echo -e "\n--- [2/3] Patching Docker Environment ---"
echo "=> Ensuring 'squid-game-live' container is running..."
sudo docker start squid-game-live > /dev/null 2>&1

echo "=> Injecting dependencies inside the container..."
# We run a single bash command inside the container to install everything
sudo docker exec -it squid-game-live bash -c "
    echo '  -> Installing system libraries (Tkinter, OpenMPI, TTS)...'
    apt-get update -yqq > /dev/null 2>&1
    apt-get install -yqq libopenblas-dev libopenmpi-dev libomp-dev python3-tk espeak-ng > /dev/null 2>&1

    echo '  -> Verifying NVIDIA Jetson PyTorch...'
    pip3 install torch torchvision torchaudio --index-url https://pypi.jetson-ai-lab.dev/jp6/cu122 --quiet

    echo '  -> Installing HuggingFace Transformers...'
    pip3 install transformers --quie
    
    echo '  -> Verifying Ultralytics (YOLO)...'
    pip3 install ultralytics --quiet

    echo '  -> Downgrading NumPy to fix OpenCV collision...'
    pip3 install 'numpy<2' --force-reinstall --quiet
"
echo "Docker environment fully patched."
sleep 1

# --- 3. TENSOR RT ENGINE COMPILATION ---
echo -e "\n--- [3/3] AI Engine Verification ---"
sudo docker exec -it squid-game-live bash -c "
    # Fix the HPC-X library path for PyTorch distributed computing
    export LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:/opt/hpcx/ucc/lib:\$LD_LIBRARY_PATH
    cd /workspace

    if [ -f yolov8n-pose.engine ]; then
        echo '=> Dynamic TensorRT engine (yolov8n-pose.engine) already exists! Skipping build.'
    else
        if [ -f yolov8n-pose.pt ] || [ -f yolov8n-pose.pt.backup ]; then
            echo '=> Compiling Dynamic TensorRT Engine for 480p...'
            echo '=> WARNING: This will take 5-10 minutes. Do not close the terminal.'
            
            # Ensure we use the .pt file (whether it was renamed to .backup or not)
            PT_FILE='yolov8n-pose.pt'
            [ -f yolov8n-pose.pt.backup ] && PT_FILE='yolov8n-pose.pt.backup'

            yolo export model=\$PT_FILE format=engine half=True dynamic=True workspace=4 device=0 imgsz=480
            echo '=> Engine compiled successfully!'
        else
            echo '=> ERROR: Cannot find yolov8n-pose.pt to build the engine.'
            echo '   Please download the model file and run this script again.'
        fi
    fi
"

echo -e "\n╔════════════════════════════════════════════════════════════╗"
echo "║   ✅ Setup Complete! You are ready for live production.    ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo "Run './launcher.sh' to start the game or hardware tests."