#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Project:      Red Light Green Light (Jetson Orin Nano)
File:         config.py
Description:  Stores all global constants, tunable difficulty thresholds,
              and hardware pinout definitions.

Author:       Richard Pu
Last Updated: April 2026
===============================================================================
"""

import os

# =============================================================================
# --- PATHS & DIRECTORIES ---
# =============================================================================
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR   = os.path.join(BASE_DIR, "assets")
FONTS_DIR    = os.path.join(ASSETS_DIR, "fonts")
SOUNDS_DIR   = os.path.join(ASSETS_DIR, "sounds")

FONT_MAIN_BOLD = os.path.join(FONTS_DIR, "GoogleSans-Bold.ttf")
FONT_MAIN_REG  = os.path.join(FONTS_DIR, "GoogleSans-Bold.ttf")

# =============================================================================
# --- DISPLAY & UI COLORS ---
# =============================================================================
CAMERA_WIDTH:  int = 1280
CAMERA_HEIGHT: int = 720
FPS_CAP:       int = 30

C_BG      = (15, 15, 20)
C_PANEL   = (30, 30, 40)
C_RED     = (255, 40, 80)
C_GREEN   = (40, 255, 100)
C_WHITE   = (240, 240, 245)
C_GREY    = (100, 100, 110)
C_CYAN    = (0, 255, 255)
C_YELLOW  = (255, 200, 0)
C_PURPLE  = (180, 0, 255)
C_ACCENT  = C_CYAN

# =============================================================================
# --- GAMEPLAY TUNING ---
# =============================================================================
GREEN_LIGHT_MIN:  float = 3.0    # Minimum seconds Green Light stays on
GREEN_LIGHT_MAX:  float = 6.0    # Maximum seconds Green Light stays on
RED_LIGHT_MIN:    float = 2.0    # Minimum seconds Red Light stays on
RED_LIGHT_MAX:    float = 5.0    # Maximum seconds Red Light stays on

TURN_SETTLE_TIME: float = 0.5    # How long the servo takes to physically turn
GRACE_PERIOD:     float = 0.2    # Seconds after turning red before motion catches you
CAUGHT_HOLD:      float = 3.5    # How long the game pauses so the kid can walk back
MOTION_THRESHOLD: float = 15.0   # Pixels of movement required to trigger being caught

# =============================================================================
# --- HARDWARE CONSTANTS ---
# =============================================================================
GSTREAMER_PIPELINE = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw, width=1280, height=720, framerate=30/1 ! "
    "videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
)

I2C_BUS:              int = 1
PCA9685_ADDRESS:      int = 0x40
SERVO_CHANNEL:        int = 0
SERVO_FREQ:           int = 50
SERVO_MIN_PULSE:      int = 500
SERVO_MAX_PULSE:      int = 2500
SERVO_FACING_FORWARD: int = 180   # Red Light
SERVO_FACING_AWAY:    int = 0     # Green Light
SERVO_STEP_DEG:       int = 5     
SERVO_STEP_DELAY:   float = 0.01  
LASER_PIN:            int = 7

# =============================================================================
# --- AUDIO CONSTANTS ---
# =============================================================================
VOLUME_SFX:   float = 1.0
VOLUME_MUSIC: float = 0.5
TTS_SPEED:    int   = 150