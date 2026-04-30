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
PROFILES_DIR = os.path.join(ASSETS_DIR, "player_profiles")
STATS_FILE = os.path.join(ASSETS_DIR, "session_stats.json")
LEADERBOARD_FILE = os.path.join(ASSETS_DIR, "leaderboard.json")
FONT_PATH = os.path.join(FONTS_DIR, "GoogleSans.ttf")

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
DISPLAY_W = 1920
DISPLAY_H = 1080
FPS_CAP = 60
FULLSCREEN = True
WINDOW_TITLE = "Red Light, Green Light"

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
CAM_W = 1280
CAM_H = 720
CAM_FPS = 60
YOLO_SKIP_FRAMES = 2  # How many camera frames between YOLO inferences
HOME_CAM_BOX_FRACTION = 0.6  # Camera occupies 60% of usable width on home screen

# ---------------------------------------------------------------------------
# Material Design 3 "Expressive" palette
# ---------------------------------------------------------------------------

# Surfaces
MD3_BG = (18, 16, 26)
MD3_SURFACE = (28, 26, 38)
MD3_SURFACE_HIGH = (40, 37, 54)
MD3_SURFACE_VAR = (50, 46, 68)
MD3_OUTLINE = (78, 74, 96)

# Primary
MD3_PRIMARY = (188, 162, 255)
MD3_ON_PRIMARY = (28, 0, 90)
MD3_PRIMARY_CONTAINER = (64, 46, 122)
MD3_ON_PRIMARY_CONTAINER = (232, 221, 255)

# Secondary
MD3_SECONDARY = (255, 180, 200)
MD3_ON_SECONDARY = (80, 25, 50)

# Tertiary
MD3_TERTIARY = (120, 220, 232)
MD3_ON_TERTIARY = (0, 55, 62)

# Semantic
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

# Podium
MD3_GOLD = (255, 215, 96)
MD3_SILVER = (200, 208, 220)
MD3_BRONZE = (217, 150, 100)

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
    "ELIMINATED": MD3_ERROR,  # walk-back banner colour (Apr 2026 rebrand)
    "WINNER": MD3_PRIMARY,
    "START_LINE": MD3_SUCCESS,
    "RETURN": MD3_ERROR,  # legacy alias kept for compatibility
    "LEADERBOARD": MD3_PRIMARY,
}

# ---------------------------------------------------------------------------
# Motion
# ---------------------------------------------------------------------------
MOTION_FAST = 18.0
MOTION_MED = 10.0
MOTION_SLOW = 6.0
BANNER_PULSE_HZ = 1.4

# ---------------------------------------------------------------------------
# Gameplay timing — slower-paced defaults per Apr 2026 design refresh
# ---------------------------------------------------------------------------
DIFFICULTY_PRESETS = {
    "Easy": {
        "green_min": 7.0,
        "green_max": 13.0,
        "red_min": 5.0,
        "red_max": 9.0,
        "motion_px": 28,
    },
    "Normal": {
        "green_min": 5.0,
        "green_max": 10.0,
        "red_min": 4.0,
        "red_max": 7.5,
        "motion_px": 20,
    },
    "Hard": {
        "green_min": 3.5,
        "green_max": 7.0,
        "red_min": 3.0,
        "red_max": 5.5,
        "motion_px": 14,
    },
}
DEFAULT_DIFFICULTY = "Normal"

# Phase-randomness flavour. The duration picker mixes a uniform draw with a
# triangular draw biased to the extremes, then has a small chance of a
# "fakeout" (a deliberately tiny phase). Tuned for surprise without being
# annoying.
PHASE_EXTREME_BIAS = 0.35  # 0 = pure uniform, 1 = always extreme
PHASE_FAKEOUT_PROB = 0.10  # 10% of phases use a fakeout duration
PHASE_FAKEOUT_MIN = 0.6  # seconds — minimum phase length even on fakeout
PHASE_FAKEOUT_MAX = 1.4

GRACE_PERIOD = 0.5
SETTLE_TIME = 0.6
COUNTDOWN_N = 3
PALM_HOLD = 1.6  # was 2.0 — start faster
PALM_HOLD_LEADERBOARD = 1.6  # palm-to-restart hold from the leaderboard screen
ALMOST_THRESHOLD = 0.6

# Caught logic — replaces the old simple "wait N seconds" pause.
CAUGHT_HOLD = 4.0  # legacy fallback; not used by the new FSM
CAUGHT_RETURN_MAX_S = 25.0  # hard timeout if return-to-start fails
CAUGHT_RETURN_GRACE_S = 1.0  # quiet period after caught before tracking starts
CAUGHT_RETURN_DWELL_S = 0.8  # must stay behind start line this long to count
CAUGHT_RETURN_RECHECK_HZ = 8  # how many times per second we test position

# ---------------------------------------------------------------------------
# Start-line check — gate the round on everyone being behind the line
# ---------------------------------------------------------------------------
START_LINE_REQUIRED = True  # If False, skip the gate entirely
START_LINE_MAX_WAIT_S = 12.0  # Force-start after this many seconds
START_LINE_DWELL_S = 0.6  # Must hold position for this long

# Pixel Y-row in the camera frame at which the start tape is laid.
#
# Apr 2026 orientation flip: the camera now sits next to the FINISH line on
# the wall, looking out at the runway. Players begin at the FAR start tape
# (top of the frame) and run TOWARD the camera, crossing the close finish
# tape (bottom of the frame). So:
#   - START line  → small Y (near top, far from camera)
#   - FINISH line → large Y (near bottom, just below camera)
#
# A player whose feet (bbox bottom) are AT or ABOVE the start row in the
# image is "behind" the start line — they haven't begun running yet.
START_LINE_Y_FRACTION = 0.32  # ~32% down (top of frame, far from camera)
START_LINE_Y_PX = int(CAM_H * START_LINE_Y_FRACTION)
START_LINE_TOLERANCE_PX = 16
START_LINE_HSV_LOW = (87, 21, 122)
START_LINE_HSV_HIGH = (97, 121, 228)
START_LINE_DETECT_FROM_TAPE = True  # If True, infer Y from tape blob. If False, use START_LINE_Y_PX.
START_LINE_DISPLAY_COLOR = MD3_SUCCESS

# ---------------------------------------------------------------------------
# Finish-line — laser is master, camera is fallback
# ---------------------------------------------------------------------------
USE_LASER = False  # Master flag: even if the laser inits, ignore it when False
USE_TAPE_FINISH = True  # Master flag for camera-based finish detection

# Color of the finish-line tape (default: bright red).
# Red wraps hue 0/180; we use both halves.
FINISH_LINE_HSV_LOW_1 = (172, 97, 171)
FINISH_LINE_HSV_HIGH_1 = (179, 224, 255)
FINISH_LINE_HSV_LOW_2 = (172, 97, 171)
FINISH_LINE_HSV_HIGH_2 = (179, 224, 255)
FINISH_LINE_DISPLAY_COLOR = MD3_ERROR

# Finish is now CLOSE to the camera (bottom of frame) — see start-line note.
FINISH_LINE_Y_FRACTION = 0.78  # ~78% down (bottom of frame, just below camera)
FINISH_LINE_Y_PX = int(CAM_H * FINISH_LINE_Y_FRACTION)
FINISH_LINE_TOLERANCE_PX = 22
FINISH_LINE_MIN_TAPE_PX = 350
FINISH_LINE_DETECT_FROM_TAPE = True

# How many tape pixels we need to see in a single row before we trust the
# colour-based detection. Below this we fall back to the configured Y.
# Used by both lines so the dev overlay can render a single "TAPE OK / FALLBACK"
# threshold pill per line.
LINE_TAPE_DETECTED_MIN_PX = 80

# ---------------------------------------------------------------------------
# Difficulty auto-easing
# ---------------------------------------------------------------------------
EASE_ENABLED = False
EASE_CHECK_EVERY_S = 120.0
EASE_TRIGGER_AFTER_S = 70.0
EASE_STEP_GREEN_S = 0.4
EASE_STEP_RED_S = 0.25
EASE_STEP_MOTION_PX = 1
EASE_MAX_GREEN_S = 10.0
EASE_MIN_RED_S = 1.6
EASE_MAX_MOTION_PX = 30.0

# ---------------------------------------------------------------------------
# Player identification
# Modes:
#   "vlm"   - Generative VLM (Moondream2). Rich descriptions. Heavier.
#   "clip"  - CLIP zero-shot scoring. Fast, still rich.
#   "color" - Multi-feature signature. Tiny, no model download.
# ---------------------------------------------------------------------------
IDENTIFICATION_MODE = "clip"

# When True, the engine takes one well-framed photo of each player during
# the FIRST red light when they're standing still — these photos drive the
# leaderboard at the end.
PROFILE_CAPTURE_ENABLED = True
PROFILE_CAPTURE_MIN_BBOX_PX = 60
PROFILE_CAPTURE_RETRY_PHASES = 3
PROFILE_THUMB_W = 220
PROFILE_THUMB_H = 280

# VLM specifics
VLM_MODEL_ID = "vikhyatk/moondream2"
VLM_REVISION = "2024-08-26"
VLM_PROMPT = "top clothing, bottom clothing, color"
VLM_MAX_NEW_TOKENS = 32
VLM_USE_CUDA_FP16 = True

# CLIP specifics
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
CLIP_TOP_GARMENTS = [
    "a t-shirt",
    "a hoodie",
    "a jacket",
    "a sweater",
    "a polo shirt",
    "a long-sleeve shirt",
    "a tank top",
    "a button-up shirt",
    "a jersey",
    "a vest",
    "a zip-up fleece",
    "short sleeves",
    "long sleeves",
]

CLIP_BOTTOM_GARMENTS = [
    "jeans",
    "shorts",
    "sweatpants",
    "leggings",
    "athletic shorts",
    "khaki pants",
    "cargo pants",
    "trousers",
]

CLIP_COLORS = [
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "pink",
    "white",
    "black",
    "grey",
    "brown",
    "navy blue",
    "dark grey",
    "light blue",
    "maroon",
    "olive green",
]

CLIP_FEATURES = [
    "wearing glasses",
    "with long hair",
    "with short hair",
    "no distinctive features",
]

# ---------------------------------------------------------------------------
# Leaderboard reveal sequence (Kahoot-style podium)
# ---------------------------------------------------------------------------
# Apr 2026 v3 — spotlight-cutout choreography. The 1st-place reveal
# now goes EXTRA LONG and dramatic: everything dims to black, a
# circular spotlight cuts a hole of brightness, the spotlight does a
# small back-and-forth sweep ("searching for the winner"), then the
# 1st-place podium rises inside the spotlight, the spotlight expands
# outward to flood the scene with light again, and confetti + winner
# music erupt. The dim-then-reveal is what makes the moment feel
# bigger than the 2nd/3rd reveals.
LEADERBOARD_PEDESTAL_RISE_S = 0.55  # empty pedestal grows up from baseline
LEADERBOARD_EMPTY_HOLD_S = 0.55  # the empty podium sits there for a beat
LEADERBOARD_PLAYER_POP_S = 0.7  # avatar/name/time pop in
LEADERBOARD_FILLED_HOLD_S = 0.9  # full card lingers in centre for a moment
LEADERBOARD_HERO_SLIDE_S = 0.85  # slide from centre to final pillar
LEADERBOARD_HERO_HOLD_S = 1.45  # legacy alias = empty + pop hold totals (kept for API compat)
LEADERBOARD_CARD_FALL_DURATION_S = 0.55  # legacy alias = pedestal rise
LEADERBOARD_TITLE_DELAY_S = 0.0
LEADERBOARD_PODIUM_FADE_S = 0.6  # empty pillars fade in over this time
LEADERBOARD_3RD_DELAY_S = 0.5  # 3rd hero pedestal starts rising
# 2nd appears the moment 3rd has slid into its right pillar.
LEADERBOARD_2ND_DELAY_S = (
    LEADERBOARD_3RD_DELAY_S
    + LEADERBOARD_PEDESTAL_RISE_S
    + LEADERBOARD_EMPTY_HOLD_S
    + LEADERBOARD_PLAYER_POP_S
    + LEADERBOARD_FILLED_HOLD_S
    + LEADERBOARD_HERO_SLIDE_S
    + 0.3  # small breath
)
# 1st-place dramatic reveal — broken into named phases so each is tunable.
LEADERBOARD_1ST_PAUSE_S = 1.5  # 2nd has slid left → moment of silence before drama
LEADERBOARD_DIM_FADE_IN_S = 0.6  # how fast the world dims to black
LEADERBOARD_SPOTLIGHT_SWEEP_S = 1.6  # spotlight scans back-and-forth ("who is it?")
LEADERBOARD_SPOTLIGHT_SETTLE_S = 0.4  # spotlight settles in centre before reveal
LEADERBOARD_1ST_REVEAL_S = 0.9  # podium rises inside the spotlight
LEADERBOARD_BLAST_OUT_S = 0.7  # spotlight grows outward, scene goes bright
LEADERBOARD_1ST_CELEBRATE_S = 3.0  # 1st card sits in glow for this long
# Derived total — when the 1st-place card ACTUALLY starts rising on
# screen.
LEADERBOARD_1ST_DELAY_S = (
    LEADERBOARD_2ND_DELAY_S
    + LEADERBOARD_PEDESTAL_RISE_S
    + LEADERBOARD_EMPTY_HOLD_S
    + LEADERBOARD_PLAYER_POP_S
    + LEADERBOARD_FILLED_HOLD_S
    + LEADERBOARD_HERO_SLIDE_S
    + LEADERBOARD_1ST_PAUSE_S
    + LEADERBOARD_DIM_FADE_IN_S
    + LEADERBOARD_SPOTLIGHT_SWEEP_S
    + LEADERBOARD_SPOTLIGHT_SETTLE_S
)

LEADERBOARD_LIST_DELAY_S = LEADERBOARD_1ST_DELAY_S + LEADERBOARD_1ST_REVEAL_S + LEADERBOARD_BLAST_OUT_S + 1.0
LEADERBOARD_PALM_ARMED_S = 1.4  # extra grace before palm-restart arms (after blast-out)
LEADERBOARD_CONFETTI_BURST_COUNT = 480  # confetti dumped when spotlight blasts out
LEADERBOARD_SPOTLIGHT_ALPHA = 70  # legacy — kept for back-compat with older skins
# Spotlight cutout effect — opacity of the dim layer everywhere outside
# the bright circle. 230 is near-pitch-black (the bright circle really
# stands out); 180 is more cinematic with detail still visible behind.
LEADERBOARD_SPOTLIGHT_DIM_ALPHA = 220
LEADERBOARD_SPOTLIGHT_RADIUS = 220  # radius of the bright circle during sweep
LEADERBOARD_SPOTLIGHT_SWEEP_AMPLITUDE = 220  # ± pixels of horizontal sweep

# ---------------------------------------------------------------------------
# Debug switches
# ---------------------------------------------------------------------------
# When True, the finish-line detector is bypassed entirely. Use this to
# test red/green light timing, the catch-and-walk-back loop, and the
# servo behaviour without needing to set up the finish tape or the
# laser break-beam. The round will run forever in green/red cycles
# until you hit `L` (jump to leaderboard) or `ESC` (quit).
#
# Toggle at runtime with the `F` key — there's a corner pill on the
# HUD when this is active so it's obvious you're in debug mode.
DEBUG_SKIP_FINISH = False

# ---------------------------------------------------------------------------
# Caught visual cue (Apr 2026 — "ELIMINATED" rebrand)
# ---------------------------------------------------------------------------
# Render a red bbox + ELIMINATED pill over each player who's currently
# walking back. Pill follows their head; pulses gently for visibility.
ELIMINATED_PULSE_HZ = 4
ELIMINATED_BOX_THICKNESS = 5

# ---------------------------------------------------------------------------
# Game Mode Settings
# ---------------------------------------------------------------------------
PERMANENT_ELIMINATION = True  # Set to True for "Battle Royale", False for "Re-entry"

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
SERVO_STEP_DEG = 6  # was 4 — faster turns
SERVO_TICK_S = 0.010  # was 0.012 — faster turns

LASER_PIN = 7

# ---------------------------------------------------------------------------
# Vision / YOLO
# ---------------------------------------------------------------------------
USE_SMALL_MODEL = True

if USE_SMALL_MODEL:
    YOLO_MODEL = "yolov8s-pose.engine"
    YOLO_FALLBACK_MODEL = "yolov8s-pose.pt"
else:
    YOLO_MODEL = "yolov8n-pose.engine"
    YOLO_FALLBACK_MODEL = "yolov8n-pose.pt"

YOLO_CONF = 0.35
YOLO_IOU = 0.50
YOLO_INFER_W = 960 
YOLO_INFER_H = 540
YOLO_IMGSZ = 960

PALM_WRIST_ABOVE_SHOULDER = 80

# Highlights ring buffer
HIGHLIGHT_MAX = 5
HIGHLIGHT_SCALE = 0.5

# Caught log (renamed from "elimination log" — players are NOT eliminated)
SENT_BACK_LOG_MAX = 6

# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
VOL_SFX = 0.8
VOL_MUSIC = 0.2
AUDIO_FREQUENCY = 48000
AUDIO_BUFFER = 1024

SOUNDS = {
    "bgm": "bgm.mp3",
    "leaderboard_bgm": "leaderboard.mp3",  

    "green": "green_light.wav",
    "red": "red_light.wav",
    "winner": "winner.wav",
    "chime": "chime.wav",
    "applause": "applause.wav",
    "podium_3": "podium_3.wav",
    "podium_2": "podium_2.wav",
    "podium_1": "podium_1.wav",
    "drumroll": "drumroll.wav",
    "tick": "countdown.wav",
    "wait_start": "wait_start.wav",  
    "num_3": "three.wav",
    "num_2": "two.wav",
    "num_1": "one.wav",
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
SHOW_PHASE_PROGRESS_BAR = False
ENABLE_SPEED_INDICATOR = True
ENABLE_ALMOST_WARNING = True
ENABLE_SESSION_STATS = True
ENABLE_INSTANT_REPLAY = True
ENABLE_AUDIO_VIZ = True
ENABLE_WIN_STREAK = True
ENABLE_LEADERBOARD = True
ENABLE_LINE_OVERLAY = True  # draw start/finish lines onto the camera feed for sanity

REPLAY_SLOW_FACTOR = 0.4
REPLAY_DURATION = 2.0
