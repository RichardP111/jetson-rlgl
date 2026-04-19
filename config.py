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

# ─── Camera ──────────────────────────────────────────────────────────
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
CAMERA_SENSOR_ID = 0

GSTREAMER_PIPELINE = (
    f"nvarguscamerasrc sensor-id={CAMERA_SENSOR_ID} ! "
    f"video/x-raw(memory:NVMM), width={CAMERA_WIDTH}, height={CAMERA_HEIGHT}, "
    f"framerate={CAMERA_FPS}/1 ! nvvidconv ! "
    f"video/x-raw, format=BGRx ! videoconvert ! "
    f"video/x-raw, format=BGR ! appsink drop=1"
)

# ─── Display ─────────────────────────────────────────────────────────
DISPLAY_WIDTH = 1920
DISPLAY_HEIGHT = 1080
FPS_CAP = 60

# ─── Servo / PCA9685 ─────────────────────────────────────────────────
I2C_BUS = 7
PCA9685_ADDRESS = 0x40
SERVO_CHANNEL = 0  # Which PCA9685 channel drives the head motor
SERVO_FREQ = 50  # Hz

SERVO_MIN_PULSE = 500  # µs  ← Green Light (facing away)
SERVO_MAX_PULSE = 2500  # µs  ← Red Light   (facing players)

SERVO_FACING_FORWARD = 180  # degrees  (RED LIGHT  — looking at players)
SERVO_FACING_AWAY = 0  # degrees  (GREEN LIGHT — looking away)
SERVO_STEP_DEG = 3  # degrees per sweep step  (higher = faster turn)
SERVO_STEP_DELAY = 0.015  # seconds between steps

# ─── Game Timing ─────────────────────────────────────────────────────
GREEN_LIGHT_MIN = 4.0  # seconds
GREEN_LIGHT_MAX = 9.0  # seconds
RED_LIGHT_MIN = 3.5  # seconds
RED_LIGHT_MAX = 7.0  # seconds

GRACE_PERIOD = 0.6  # seconds after red-light before motion is penalised
TURN_SETTLE_TIME = 2.0  # sec to wait for head to settle before checking
ELIMINATION_HOLD = 3.5  # seconds to show elimination screen

# ─── Motion Detection ────────────────────────────────────────────────
MOTION_THRESHOLD = 22  # pixel-delta threshold (lower = more sensitive)
MOTION_AREA_MIN = 1000  # minimum contour area to count as motion
MOTION_BLUR_SIZE = 21  # Gaussian blur kernel (must be odd)
MOTION_HISTORY = 4  # frames in history – needs N/2 positives to trigger

# ─── Finish Line Detection ───────────────────────────────────────────
# Adjust hue range for the tape colour you use.
# Yellow tape:  (20, 100, 100) – (35, 255, 255)
# Orange tape:  (5,  120, 120) – (20, 255, 255)
# Pink tape:    (140, 80, 100) – (170, 255, 255)
FINISH_TAPE_HSV_LOW = (20, 100, 100)
FINISH_TAPE_HSV_HIGH = (35, 255, 255)
FINISH_LINE_X_RATIO = 0.82  # tape must be in the rightmost % of frame

# ─── GPIO  (Laser break-beam) ────────────────────────────────────────
LASER_PIN = 12  # BCM pin;  LOW = beam broken = someone crossed

# ─── Audio ───────────────────────────────────────────────────────────
SOUNDS_DIR = os.path.join(BASE_DIR, "assets", "sounds")
VOLUME_SFX = 0.90
VOLUME_MUSIC = 0.55
TTS_SPEED = 145  # espeak words-per-minute

# ─── Paths ───────────────────────────────────────────────────────────
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
CAPTURES_DIR = os.path.join(ASSETS_DIR, "captures")
FONTS_DIR = os.path.join(ASSETS_DIR, "fonts")

# ─── Colour Palette (RGB) ────────────────────────────────────────────
C_BG = (6, 6, 10)
C_PANEL = (14, 14, 22)
C_RED = (220, 28, 28)
C_RED_DIM = (100, 15, 15)
C_GREEN = (22, 210, 80)
C_GREEN_DIM = (12, 90, 38)
C_WHITE = (235, 235, 240)
C_GREY = (80, 80, 95)
C_CYAN = (0, 205, 240)
C_YELLOW = (255, 195, 0)
C_PURPLE = (130, 60, 220)
C_SKEL = (0, 255, 165)  # neon mint skeleton overlay
C_ACCENT = (0, 175, 215)

# ─── MediaPipe skeleton connections ──────────────────────────────────
SKELETON_CONNECTIONS = [
    (11, 12),  # shoulders
    (11, 13),
    (13, 15),  # left arm
    (12, 14),
    (14, 16),  # right arm
    (11, 23),
    (12, 24),  # torso sides
    (23, 24),  # hips
    (23, 25),
    (25, 27),  # left leg
    (24, 26),
    (26, 28),  # right leg
    (0, 11),
    (0, 12),  # neck-ish
]
