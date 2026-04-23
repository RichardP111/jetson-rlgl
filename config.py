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

# ═══════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
FONTS_DIR = os.path.join(ASSETS_DIR, "fonts")
SOUNDS_DIR = os.path.join(ASSETS_DIR, "sounds")
STATS_FILE = os.path.join(ASSETS_DIR, "session_stats.json")

# Single font: GoogleSans.ttf (Regular weight)
FONT_PATH = os.path.join(FONTS_DIR, "GoogleSans.ttf")

# ═══════════════════════════════════════════════════════════════════
# DISPLAY — 1920×1080 Full HD
# ═══════════════════════════════════════════════════════════════════
DISPLAY_W = 1920
DISPLAY_H = 1080
FPS_CAP = 30
FULLSCREEN = True

# ═══════════════════════════════════════════════════════════════════
# CAMERA — Optimized for Docker (sharp + fast)
# ═══════════════════════════════════════════════════════════════════
CAM_W = 1280  # Internal resolution (sharp but efficient)
CAM_H = 720
CAM_FPS = 30
CAM_BRIGHTNESS = 1.15  # Slight brightness boost for gym lighting
CAM_CONTRAST = 1.10  # Contrast enhancement
CAM_SHARPNESS = True  # CLAHE sharpening filter enabled

# YOLO frame skip (Docker optimization)
YOLO_SKIP_FRAMES = 2  # Run YOLO every 3rd frame (saves 66% CPU)

# ═══════════════════════════════════════════════════════════════════
# MATERIAL DESIGN 3 COLOUR PALETTE
# ═══════════════════════════════════════════════════════════════════
# Background & Surfaces
MD3_BG = (16, 20, 24)  # Deep navy background
MD3_SURFACE = (28, 33, 38)  # Elevated surface
MD3_SURFACE_VAR = (38, 44, 50)  # Surface variant

# Primary (Teal/Cyan)
MD3_PRIMARY = (79, 195, 247)  # Soft teal primary
MD3_PRIMARY_DIM = (38, 97, 123)  # Dimmed teal

# Semantic Colors
MD3_ERROR = (244, 123, 123)  # Soft coral red (not harsh)
MD3_ERROR_DIM = (122, 61, 61)  # Dimmed coral
MD3_SUCCESS = (129, 199, 132)  # Soft green
MD3_SUCCESS_DIM = (64, 99, 66)  # Dimmed green
MD3_WARNING = (255, 183, 77)  # Warm amber
MD3_INFO = (144, 202, 249)  # Light blue

# Text
MD3_ON_BG = (245, 245, 250)  # Primary text
MD3_ON_BG_MED = (189, 189, 199)  # Medium emphasis
MD3_ON_BG_DIM = (117, 117, 128)  # Low emphasis

# State Colors (used in UI)
STATE_COLORS = {
    "GREEN": MD3_SUCCESS,
    "RED": MD3_ERROR,
    "TURNING": MD3_WARNING,
    "WINNER": MD3_PRIMARY,
}

# ═══════════════════════════════════════════════════════════════════
# GAMEPLAY TIMING — Difficulty Presets
# ═══════════════════════════════════════════════════════════════════
DIFFICULTY_PRESETS = {
    "Easy": {
        "green_min": 5.0,
        "green_max": 9.0,
        "red_min": 4.0,
        "red_max": 7.0,
        "motion_px": 20,
    },
    "Normal": {
        "green_min": 3.5,
        "green_max": 7.0,
        "red_min": 3.0,
        "red_max": 5.5,
        "motion_px": 15,
    },
    "Hard": {
        "green_min": 2.5,
        "green_max": 5.0,
        "red_min": 2.0,
        "red_max": 4.0,
        "motion_px": 10,
    },
}

# Default difficulty
DEFAULT_DIFFICULTY = "Normal"

# Fixed timing values (not affected by difficulty)
GRACE_PERIOD = 0.25  # Seconds after red before motion detection starts
SETTLE_TIME = 0.6  # Servo settle delay
CAUGHT_HOLD = 4.0  # Elimination screen duration
COUNTDOWN_N = 3  # Start countdown from 3
PALM_HOLD = 2.0  # Palm gesture hold duration
ALMOST_THRESHOLD = 0.5  # Fraction of motion_px to trigger "almost!" warning

# ═══════════════════════════════════════════════════════════════════
# HARDWARE
# ═══════════════════════════════════════════════════════════════════
# Servo (PCA9685)
I2C_BUS = 7
PCA9685_ADDR = 0x40
SERVO_CHANNEL = 0
SERVO_FREQ = 50
SERVO_MIN_US = 500
SERVO_MAX_US = 2500
SERVO_AWAY_DEG = 0  # Green light position
SERVO_FACE_DEG = 180  # Red light position
SERVO_STEP_DEG = 4
SERVO_TICK_S = 0.012

# Laser Break-Beam
LASER_PIN = 7  # BCM/BOARD depending on hardware.py mode

# ═══════════════════════════════════════════════════════════════════
# VISION / YOLO
# ═══════════════════════════════════════════════════════════════════
YOLO_MODEL = "yolov8n-pose.pt"
YOLO_CONF = 0.45
YOLO_IOU = 0.50

# Palm gesture detection
PALM_WRIST_ABOVE_SHOULDER = 70  # pixels

# Finish line (coloured tape HSV)
TAPE_HSV_LOW = (18, 100, 100)
TAPE_HSV_HIGH = (35, 255, 255)
TAPE_ZONE_X = 0.80
TAPE_MIN_PX = 400

# ═══════════════════════════════════════════════════════════════════
# AUDIO
# ═══════════════════════════════════════════════════════════════════
VOL_SFX = 1.0
VOL_MUSIC = 0.45
TTS_WPM = 145

# Sound file map
SOUNDS = {
    "bgm": "bgm.mp3",
    "green": "green_light.wav",
    "red": "red_light.wav",
    "mugunghwa": "mugunghwa.wav",
    "caught": "eliminated.wav",
    "winner": "winner.wav",
    "tick": "countdown.wav",
    "almost": "almost.wav",  # NEW: "almost!" warning sound
}

TTS_LINES = {
    "green": "Green light! Go go go!",
    "red": "Red light! Freeze!",
    "caught": "You moved! Return to the start!",
    "winner": "We have a winner! Congratulations!",
    "start": "Get ready! The game is about to begin!",
    "almost": "Careful! Almost moved!",
}

# ═══════════════════════════════════════════════════════════════════
# NEW FEATURES — Flags
# ═══════════════════════════════════════════════════════════════════
ENABLE_FPS_COUNTER = True  # Show FPS in top-right
ENABLE_PLAYER_COUNT = True  # Live player count badge
ENABLE_PHASE_TIMER = True  # Countdown timer during phases
ENABLE_SPEED_INDICATOR = True  # Movement speed gauge (green light)
ENABLE_ALMOST_WARNING = True  # Flash warning when near threshold
ENABLE_SESSION_STATS = True  # Track stats across rounds
ENABLE_INSTANT_REPLAY = True  # Slow-mo replay of eliminations
ENABLE_AUDIO_VIZ = True  # Pulse rings when sounds play
ENABLE_WIN_STREAK = True  # Track consecutive wins by color
ENABLE_LEADERBOARD = True  # Show top 3 fastest times
REPLAY_SLOW_FACTOR = 0.4  # Slow-mo speed (0.4 = 40% speed)
REPLAY_DURATION = 2.0  # Seconds of replay footage
