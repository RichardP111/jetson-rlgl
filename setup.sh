#!/bin/bash
"""
===============================================================================
Project:      Red Light Green Light (Jetson Orin Nano)
File:         setup.sh
Description:  Install all dependencies for Red Light Green Light

Author:       Richard Pu
Last Updated: April 2026
Run once:  bash setup.sh
===============================================================================
"""

set -e

echo ""
echo "  ╔═════════════════════════════════════╗"
echo "  ║   RLGL Setup  •  Jetson Orin Nano   ║"
echo "  ║           Grade 7 STEM Day          ║"
echo "  ╚═════════════════════════════════════╝"
echo ""

# ── Step 1: System packages ─────────────────────────────────────────
echo "[1/6] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y \
    espeak espeak-ng \
    python3-pip \
    libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev \
    libportaudio2 \
    v4l-utils \
    i2c-tools

# ── Step 2: Python packages ─────────────────────────────────────────
echo "[2/6] Installing Python packages..."
pip install --break-system-packages --upgrade pip

pip install --break-system-packages \
    mediapipe \
    pygame \
    adafruit-circuitpython-pca9685 \
    adafruit-circuitpython-motor \
    numpy \
    pillow \
    qrcode[pil]

# Jetson.GPIO should already be installed via jetpack; if not:
# pip install --break-system-packages Jetson.GPIO

# ── Step 3: Asset directories ───────────────────────────────────────
echo "[3/6] Creating asset directories..."
mkdir -p assets/sounds assets/captures assets/fonts

# ── Step 4: Download Orbitron font ──────────────────────────────────
echo "[4/6] Downloading Orbitron font..."
FONT_URL="https://fonts.gstatic.com/s/orbitron/v29/yMJMMIlzdpvBhQQL_SC3X9yhF25-T1nysimBoWgz.woff2"
# Convert woff2 → ttf requires fonttools, easier to grab TTF directly
pip install --break-system-packages fonttools brotli 2>/dev/null || true

# Attempt direct TTF from GitHub mirror
TTF_URL="https://github.com/google/fonts/raw/main/ofl/orbitron/Orbitron%5Bwght%5D.ttf"
curl -L "$TTF_URL" -o assets/fonts/Orbitron-Bold.ttf 2>/dev/null \
    || echo "  ⚠  Could not download Orbitron — monospace fallback will be used."
cp assets/fonts/Orbitron-Bold.ttf assets/fonts/Orbitron-Regular.ttf 2>/dev/null || true

# ── Step 5: Verify camera ────────────────────────────────────────────
echo "[5/6] Checking camera..."
if ls /dev/video* 1>/dev/null 2>&1; then
    v4l2-ctl --list-devices 2>/dev/null | head -10 || true
    echo "  ✓  Camera device(s) found"
else
    echo "  ⚠  No /dev/video* found. Run 'sudo /opt/nvidia/jetson-io/jetson-io.py' to enable IMX219."
fi

# ── Step 6: Verify I2C (PCA9685) ────────────────────────────────────
echo "[6/6] Checking I2C bus 7 for PCA9685..."
sudo i2cdetect -y 7 2>/dev/null | grep -q "40" \
    && echo "  ✓  PCA9685 found at 0x40 on I2C-7" \
    || echo "  ⚠  PCA9685 not detected on I2C-7. Check wiring and run: i2cdetect -y 7"

echo ""
echo "  ════════════════════════════════════════════"
echo "  Setup complete!"
echo ""
echo "  Drop these files into  assets/sounds/ :"
echo "    green_light.wav   — 'Green Light' sound"
echo "    red_light.wav     — 'Red Light' sound / buzzer"
echo "    mugunghwa.wav     — The Korean phrase (find on YouTube → ffmpeg)"
echo "    eliminated.wav    — Dramatic sting"
echo "    winner.wav        — Victory fanfare"
echo "    countdown.wav     — Tick / blip sound"
echo "    bgm.mp3           — Background music loop"
echo ""
echo "  Run the game:   python3 main.py"
echo "  ════════════════════════════════════════════"
echo ""
