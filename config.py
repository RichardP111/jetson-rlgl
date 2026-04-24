#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Project:      Red Light Green Light (Jetson Orin Nano)
File:         config.py
Description:  Global constants, M3 Expressive palette, tunable difficulty,
              animation timings, and hardware pin definitions.

Author:       Richard Pu
Last Updated: April 2026
===============================================================================
"""

import os
import platform

# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------
IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
FONTS_DIR = os.path.join(ASSETS_DIR, "fonts")
SOUNDS_DIR = os.path.join(ASSETS_DIR, "sounds")
STATS_FILE = os.path.join(ASSETS_DIR, "session_stats.json")
FONT_PATH = os.path.join(FONTS_DIR, "GoogleSans.ttf")

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
DISPLAY_W = 1920
DISPLAY_H = 1080
FPS_CAP = 30
FULLSCREEN = True

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
CAM_W = 1280
CAM_H = 720
CAM_FPS = 30
CAM_BRIGHTNESS = 1.10
CAM_CONTRAST = 1.05
CAM_SHARPNESS = True
YOLO_SKIP_FRAMES = 2

# ---------------------------------------------------------------------------
# Material Design 3 "Expressive" palette
# Soft pastel containers on a deep surface. Each token has ON-variant for text.
# ---------------------------------------------------------------------------

# Surfaces
MD3_BG = (18, 16, 26)
MD3_SURFACE = (28, 26, 38)
MD3_SURFACE_HIGH = (40, 37, 54)
MD3_SURFACE_VAR = (50, 46, 68)
MD3_OUTLINE = (78, 74, 96)

# Primary - Expressive deep purple
MD3_PRIMARY = (188, 162, 255)
MD3_ON_PRIMARY = (28, 0, 90)
MD3_PRIMARY_CONTAINER = (64, 46, 122)
MD3_ON_PRIMARY_CONTAINER = (232, 221, 255)

# Secondary - Warm pink accent
MD3_SECONDARY = (255, 180, 200)
MD3_ON_SECONDARY = (80, 25, 50)

# Tertiary - Cyan pop
MD3_TERTIARY = (120, 220, 232)
MD3_ON_TERTIARY = (0, 55, 62)

# Semantic containers (soft pastels)
MD3_SUCCESS = (168, 232, 178)
MD3_SUCCESS_ON = (0, 74, 24)
MD3_SUCCESS_BG = (36, 70, 44)

MD3_ERROR = (255, 182, 180)
MD3_ERROR_ON = (94, 17, 24)
MD3_ERROR_BG = (92, 36, 40)

MD3_WARNING = (255, 220, 160)
MD3_WARNING_ON = (76, 48, 0)
MD3_WARNING_BG = (96, 64, 16)

MD3_INFO = (180, 210, 255)
MD3_INFO_ON = (8, 36, 92)

# Text
MD3_ON_BG = (235, 228, 245)
MD3_ON_BG_MED = (188, 182, 200)
MD3_ON_BG_DIM = (128, 122, 142)

# Per-state accent (used by the banner)
STATE_COLORS = {
    "START": MD3_PRIMARY,
    "COUNTDOWN": MD3_TERTIARY,
    "GREEN": MD3_SUCCESS,
    "TURNING": MD3_WARNING,
    "RED": MD3_ERROR,
    "CAUGHT": MD3_ERROR,
    "WINNER": MD3_PRIMARY,
}

# ---------------------------------------------------------------------------
# Motion - M3 Expressive uses strong overshoot easing on key changes
# ---------------------------------------------------------------------------
MOTION_FAST = 18.0
MOTION_MED = 10.0
MOTION_SLOW = 6.0
BANNER_PULSE_HZ = 1.4

# ---------------------------------------------------------------------------
# Gameplay timing
# ---------------------------------------------------------------------------
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
DEFAULT_DIFFICULTY = "Normal"

GRACE_PERIOD = 0.5
SETTLE_TIME = 0.6
CAUGHT_HOLD = 4.0
COUNTDOWN_N = 3
PALM_HOLD = 2.0
ALMOST_THRESHOLD = 0.6

# Highlights ring buffer (prevents memory growth)
HIGHLIGHT_MAX = 5
HIGHLIGHT_SCALE = 0.5

# Elimination log
ELIM_LOG_MAX = 6

# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------
I2C_BUS = 7
PCA9685_ADDR = 0x40
SERVO_CHANNEL = 0
SERVO_FREQ = 50
SERVO_MIN_US = 500
SERVO_MAX_US = 2500
SERVO_AWAY_DEG = 0
SERVO_FACE_DEG = 180
SERVO_STEP_DEG = 4
SERVO_TICK_S = 0.012

LASER_PIN = 7

# ---------------------------------------------------------------------------
# Vision / YOLO
# ---------------------------------------------------------------------------
YOLO_MODEL = "yolov8n-pose.pt"
YOLO_CONF = 0.45
YOLO_IOU = 0.50

PALM_WRIST_ABOVE_SHOULDER = 70

TAPE_HSV_LOW = (18, 100, 100)
TAPE_HSV_HIGH = (35, 255, 255)
TAPE_ZONE_X = 0.80
TAPE_MIN_PX = 400

# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
VOL_SFX = 1.0
VOL_MUSIC = 0.45
TTS_WPM = 145

# USB-C DAC friendly
AUDIO_FREQUENCY = 48000
AUDIO_BUFFER = 1024

SOUNDS = {
    "bgm": "bgm.mp3",
    "green": "green_light.wav",
    "red": "red_light.wav",
    "mugunghwa": "mugunghwa.wav",
    "caught": "eliminated.wav",
    "winner": "winner.wav",
    "tick": "countdown.wav",
    "almost": "almost.wav",
    "chime": "chime.wav",
}

TTS_LINES = {
    "green": "Green light! Go go go!",
    "red": "Red light! Freeze!",
    "caught": "You moved! Return to the start!",
    "winner": "We have a winner! Congratulations!",
    "start": "Get ready! The game is about to begin!",
    "almost": "Careful! Almost moved!",
}

# ---------------------------------------------------------------------------
# Dev mode
# ---------------------------------------------------------------------------
DEV_MODE_ENABLED_BY_DEFAULT = False
DEV_FPS_HISTORY = 90

# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------
ENABLE_FPS_COUNTER = True
ENABLE_PLAYER_COUNT = True
ENABLE_PHASE_TIMER = True
ENABLE_SPEED_INDICATOR = True
ENABLE_ALMOST_WARNING = True
ENABLE_SESSION_STATS = True
ENABLE_INSTANT_REPLAY = True
ENABLE_AUDIO_VIZ = True
ENABLE_WIN_STREAK = True
ENABLE_LEADERBOARD = True
REPLAY_SLOW_FACTOR = 0.4
REPLAY_DURATION = 2.0
