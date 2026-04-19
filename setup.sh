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
echo ""
echo "  ╔══════════════════════════════════════════╗"
echo "  ║   RLGL Setup  ·  Jetson Orin Nano        ║"
echo "  ╚══════════════════════════════════════════╝"
echo ""
 
# ── 1. System packages ──────────────────────────────────────────────
echo "[1/5] System packages…"
sudo apt-get update -qq
sudo apt-get install -y \
    espeak espeak-ng \
    python3-pip \
    v4l-utils \
    i2c-tools \
    libsdl2-dev libsdl2-mixer-dev libsdl2-ttf-dev
 
# ── 2. Python packages ──────────────────────────────────────────────
echo "[2/5] Python packages…"
pip install \
    ultralytics \
    pygame \
    adafruit-circuitpython-pca9685 \
    adafruit-circuitpython-motor \
    numpy
 
# Note: Jetson.GPIO should already be installed via JetPack.
# If not: pip install Jetson.GPIO
 
# ── 3. Asset directories ────────────────────────────────────────────
echo "[3/5] Asset directories…"
mkdir -p assets/sounds assets/fonts
 
# ── 4. Verify camera ─────────────────────────────────────────────────
echo "[4/5] Camera check…"
if ls /dev/video* 1>/dev/null 2>&1; then
    echo "  ✓  Camera found at $(ls /dev/video*)"
else
    echo "  ⚠  No /dev/video* — run Jetson-IO to enable IMX219"
fi
 
# ── 5. Verify I2C ─────────────────────────────────────────────────
echo "[5/5] I2C bus 7 check…"
if sudo i2cdetect -y 7 2>/dev/null | grep -q "40"; then
    echo "  ✓  PCA9685 detected at 0x40 on I2C-7"
else
    echo "  ⚠  PCA9685 not found on I2C-7. Check wiring. Run: sudo i2cdetect -y 7"
fi
 
echo ""
echo "  ════════════════════════════════════════════"
echo "  Setup complete!"
echo ""
echo "  Drop sound files into assets/sounds/ :"
echo "    bgm.mp3          — background music loop"
echo "    mugunghwa.wav    — Korean freeze phrase"
echo "    green_light.wav  — green light sound"
echo "    red_light.wav    — red light buzzer"
echo "    eliminated.wav   — elimination sting"
echo "    winner.wav       — victory fanfare"
echo "    tick.wav         — countdown beep"
echo ""
echo "  Drop fonts into assets/fonts/ :"
echo "    GoogleSans-Bold.ttf"
echo "    GoogleSans-Regular.ttf"
echo ""
echo "  Run the game:"
echo "    export DISPLAY=:1 && python3 main.py"
echo "  ════════════════════════════════════════════"
echo ""