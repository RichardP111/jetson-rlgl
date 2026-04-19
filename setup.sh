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

set -e

# --- ANSI Color Codes ---
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}"
echo "  ╔════════════════════════════════════════════════╗"
echo "  ║  RLGL AI VISION GAME  •  Jetson Orin Nano      ║"
echo "  ║        Grade 7 STEM Day Production Build       ║"
echo "  ╚════════════════════════════════════════════════╝"
echo -e "${NC}"

# ── Step 1: System packages ─────────────────────────────────────────
echo -e "${CYAN}[1/7] Installing core system packages...${NC}"
sudo apt-get update -qq
sudo apt-get install -y \
    espeak espeak-ng \
    python3-pip \
    libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev \
    libportaudio2 \
    v4l-utils \
    i2c-tools \
    libgl1 libglib2.0-0 # Required for OpenCV on Jetson

# ── Step 2: Python packages ─────────────────────────────────────────
echo -e "${CYAN}[2/7] Installing Python AI and Hardware packages...${NC}"
pip install --break-system-packages --upgrade pip

pip install --break-system-packages \
    ultralytics \
    opencv-python \
    pygame \
    adafruit-circuitpython-pca9685 \
    adafruit-circuitpython-motor \
    numpy

# Jetson.GPIO should be native, but just in case:
# pip install --break-system-packages Jetson.GPIO

# ── Step 3: Asset directories ───────────────────────────────────────
echo -e "${CYAN}[3/7] Verifying asset directory structure...${NC}"
mkdir -p assets/sounds assets/captures assets/fonts

# ── Step 4: Font Check ──────────────────────────────────────────────
echo -e "${CYAN}[4/7] Checking for Google Sans font...${NC}"
if [ -f "assets/fonts/GoogleSans-Bold.ttf" ]; then
    echo -e "  ${GREEN}✓  GoogleSans-Bold.ttf found.${NC}"
else
    echo -e "  ${YELLOW}⚠  GoogleSans-Bold.ttf missing. Please drag it into assets/fonts/ to avoid PyGame fallback.${NC}"
fi

# ── Step 5: Pre-load YOLO Weights ───────────────────────────────────
echo -e "${CYAN}[5/7] Pre-loading YOLOv8 Pose Neural Network...${NC}"
if [ ! -f "yolov8n-pose.pt" ]; then
    echo "  Downloading yolov8n-pose.pt..."
    curl -L "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n-pose.pt" -o yolov8n-pose.pt 2>/dev/null
    echo -e "  ${GREEN}✓  AI Brain Downloaded.${NC}"
else
    echo -e "  ${GREEN}✓  yolov8n-pose.pt already exists locally.${NC}"
fi

# ── Step 6: Verify camera ────────────────────────────────────────────
echo -e "${CYAN}[6/7] Probing ISP for CSI Camera...${NC}"
if ls /dev/video* 1>/dev/null 2>&1; then
    v4l2-ctl --list-devices 2>/dev/null | head -3 || true
    echo -e "  ${GREEN}✓  Camera device(s) found.${NC}"
else
    echo -e "  ${RED}⚠  No /dev/video* found. Run 'sudo /opt/nvidia/jetson-io/jetson-io.py' to enable IMX219.${NC}"
fi

# ── Step 7: Verify I2C (PCA9685) ────────────────────────────────────
# Note: config.py targets I2C Bus 1. 
echo -e "${CYAN}[7/7] Probing I2C Bus 1 for PCA9685 Servo Controller...${NC}"
sudo i2cdetect -y 1 2>/dev/null | grep -q "40" \
    && echo -e "  ${GREEN}✓  PCA9685 found at 0x40 on I2C-1${NC}" \
    || echo -e "  ${RED}⚠  PCA9685 not detected on I2C-1. Check physical 5V wiring.${NC}"

echo ""
echo -e "${GREEN}  ════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  ENGINE SETUP COMPLETE.${NC}"
echo ""
echo "  Audio Checklist (Place in assets/sounds/):"
echo "    bgm.mp3          — Background ambient tension track"
echo "    elimination.wav  — Stinger/Buzzer for motion detection"
echo ""
echo "  To launch the console:"
echo -e "  ${YELLOW}python3 main.py${NC}"
echo -e "${GREEN}  ════════════════════════════════════════════════════════${NC}"
echo ""