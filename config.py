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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
FONTS_DIR = os.path.join(ASSETS_DIR, "fonts")
SOUNDS_DIR = os.path.join(ASSETS_DIR, "sounds")

# ── Fonts (drop GoogleSans-Bold.ttf + GoogleSans-Regular.ttf into assets/fonts/) ──
FONT_BOLD = os.path.join(FONTS_DIR, "GoogleSans-Bold.ttf")
FONT_REG = os.path.join(FONTS_DIR, "GoogleSans-Regular.ttf")

# ═══════════════════════════════════════════════════════
# DISPLAY
# ═══════════════════════════════════════════════════════
DISPLAY_W = 1920
DISPLAY_H = 1080
FPS_CAP = 30
FULLSCREEN = True  # set False for windowed dev testing

# ═══════════════════════════════════════════════════════
# CAMERA
# ═══════════════════════════════════════════════════════
CAM_W = 1280
CAM_H = 720
CAM_FPS = 30

# ═══════════════════════════════════════════════════════
# HARDWARE — GPIO (BOARD numbering)
# ═══════════════════════════════════════════════════════
LASER_PIN = 13  # Physical board pin 13 (GPIO pin with pull-up)
# Beam broken → LOW signal = player crossed finish line

# ═══════════════════════════════════════════════════════
# HARDWARE — PCA9685 Servo
# ═══════════════════════════════════════════════════════
I2C_BUS = 7  # Jetson Orin Nano primary header = bus 7
PCA9685_ADDR = 0x40
SERVO_CHANNEL = 0
SERVO_FREQ_HZ = 50
SERVO_MIN_US = 500  # pulse width for 0°
SERVO_MAX_US = 2500  # pulse width for 180°
SERVO_AWAY_DEG = 0  # GREEN LIGHT — head faces away
SERVO_FACE_DEG = 180  # RED LIGHT   — head faces players
SERVO_STEP_DEG = 4  # degrees moved per sweep tick
SERVO_TICK_S = 0.012  # seconds between sweep ticks

# ═══════════════════════════════════════════════════════
# GAMEPLAY
# ═══════════════════════════════════════════════════════
GREEN_MIN = 3.5  # s — shortest green-light phase
GREEN_MAX = 8.0  # s — longest  green-light phase
RED_MIN = 3.0  # s — shortest red-light phase
RED_MAX = 6.5  # s — longest  red-light phase

GRACE_S = 0.35  # s — delay after RED before motion triggers elim
SETTLE_S = 0.6  # s — wait for servo to settle before accepting elim
CAUGHT_PAUSE_S = 4.0  # s — freeze screen while caught player walks back
COUNTDOWN_N = 3  # start countdown from this number
PALM_HOLD_S = 2.0  # s — how long palm must be raised to start

# ═══════════════════════════════════════════════════════
# VISION / YOLO
# ═══════════════════════════════════════════════════════
YOLO_MODEL = "yolov8n-pose.pt"  # nano = fastest on Jetson
YOLO_CONF = 0.45
YOLO_IOU = 0.50
MOTION_PX = 14  # pixel movement delta to count as "moved"

# Wrist must be this many pixels ABOVE shoulder to count as palm-raise
PALM_WRIST_ABOVE_SHOULDER = 70

# ═══════════════════════════════════════════════════════
# FINISH LINE — coloured tape detection (HSV)
# ═══════════════════════════════════════════════════════
# Yellow tape (default).  Adjust for whatever colour you use:
#   Orange → (5, 120, 120) – (22, 255, 255)
#   Pink   → (140, 80, 100) – (170, 255, 255)
TAPE_HSV_LOW = (18, 100, 100)
TAPE_HSV_HIGH = (35, 255, 255)
TAPE_ZONE_X = 0.80  # rightmost fraction of frame = finish zone
TAPE_MIN_PX = 400  # min tape pixels to call it detected

# ═══════════════════════════════════════════════════════
# AUDIO
# ═══════════════════════════════════════════════════════
VOL_SFX = 1.0
VOL_MUSIC = 0.45
TTS_WPM = 145

# ═══════════════════════════════════════════════════════
# COLOUR PALETTE  (RGB — pygame order)
# ═══════════════════════════════════════════════════════
# Backgrounds
BG = (8, 10, 18)
PANEL = (18, 22, 38)
PANEL_LIGHT = (28, 33, 55)

# Accents
RED = (255, 38, 72)
RED_DIM = (90, 18, 30)
GREEN = (10, 235, 95)
GREEN_DIM = (8, 80, 38)
CYAN = (0, 210, 245)
YELLOW = (255, 195, 0)
PURPLE = (160, 60, 230)

# Text
WHITE = (240, 242, 248)
GREY = (90, 95, 115)
GREY_LIGHT = (150, 155, 175)

# State glow colours
STATE_COLS = {
    "GREEN": GREEN,
    "RED": RED,
    "TURNING": YELLOW,
    "WINNER": CYAN,
}
