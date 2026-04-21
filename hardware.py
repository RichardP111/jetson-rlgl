#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Project:      Red Light Green Light (Jetson Orin Nano)
File:         hardware.py
Description:  Manages the CSI camera thread, I2C PCA9685 servo
              movements, and GPIO laser break-beam inputs.

Author:       Richard Pu
Last Updated: April 2026
===============================================================================
"""

import threading
import time
from typing import Any, cast

import cv2
import numpy as np

_cv2 = cast(Any, cv2)

from config import (CAM_H, CAM_W, I2C_BUS, LASER_PIN, PCA9685_ADDR,
                    SERVO_AWAY_DEG, SERVO_CHANNEL, SERVO_FACE_DEG,
                    SERVO_FREQ_HZ, SERVO_MAX_US, SERVO_MIN_US, SERVO_STEP_DEG,
                    SERVO_TICK_S)

# ── Optional hardware imports ────────────────────────────────────────
try:
    import board
    import busio
    from adafruit_pca9685 import PCA9685 as _PCA9685
    from adafruit_motor import servo as adafruit_servo
    _ADAFRUIT = True
except Exception as e:
    print(f"[HW] SERVO CRASH REAL ERROR: {e}")
    _ADAFRUIT = False
    print("[HW] Adafruit libs missing — servo in sim mode")

try:
    import Jetson.GPIO as GPIO
    _GPIO = True
except (ImportError, AttributeError):
    _GPIO = False
    print("[HW] Jetson.GPIO missing — laser in sim mode")


# ════════════════════════════════════════════════════════════════════
#  CAMERA
# ════════════════════════════════════════════════════════════════════
class Camera:
    """
    Background thread keeps latest frame fresh.
    Uses direct V4L2 — works over NoMachine, no nvargus dependency.
    """

    def __init__(self):
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = True
        self._ok = False

        gstreamer_pipeline = (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), width=1920, height=1080, format=NV12, framerate=30/1 ! "
            "nvvidconv ! video/x-raw, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! "
            "appsink drop=true max-buffers=1"
        )

        print("[CAM] Booting NVIDIA Argus ISP...")
        # Force CAP_GSTREAMER
        self._cap = _cv2.VideoCapture(gstreamer_pipeline, _cv2.CAP_GSTREAMER)
        
        if self._cap.isOpened():
            # DO NOT use cap.set() here; the pipeline string handles it.
            self._ok = True
            print(f"[CAM] CSI Camera linked via GStreamer ✓")
        else:
            print("[CAM] FATAL: GStreamer pipeline failed. Is another app using the cam?")
            self._cap = None

        self._thread = threading.Thread(target=self._loop, daemon=True, name="cam")
        self._thread.start()

    def _loop(self):
        while self._running:
            if self._cap and self._ok:
                ret, frame = self._cap.read()
                if ret and frame is not None:
                    with self._lock:
                        self._frame = frame
            else:
                time.sleep(0.033)

    def read(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    @property
    def ok(self) -> bool:
        return self._ok

    def release(self):
        self._running = False
        self._thread.join(timeout=1.5)
        if self._cap:
            self._cap.release()


# ════════════════════════════════════════════════════════════════════
#  SERVO CONTROLLER  (PCA9685)
# ════════════════════════════════════════════════════════════════════
class ServoController:
    """
    Smooth servo sweep running in its own daemon thread.
    Falls back to simulation (just prints) if hardware isn't present.

    Public interface:
        servo.face_players()   → sweep to SERVO_FACE_DEG
        servo.face_away()      → sweep to SERVO_AWAY_DEG
        servo.set_angle(deg)   → sweep to arbitrary angle
        servo.angle            → current angle (float)
        servo.is_at_target     → True when within 3° of target
        servo.is_facing_players→ True when close to face position
    """

    def __init__(self):
        self._angle = float(SERVO_AWAY_DEG)
        self._target = float(SERVO_AWAY_DEG)
        self._lock = threading.Lock()
        self._running = True
        self._hw = False
        self._servo = None

        if _ADAFRUIT:
            try:
                i2c = busio.I2C(board.SCL, board.SDA)
                pca = _PCA9685(i2c, address=PCA9685_ADDR)
                pca.frequency = SERVO_FREQ_HZ
                self._servo = adafruit_servo.Servo(
                    pca.channels[SERVO_CHANNEL],  # type: ignore
                    min_pulse=SERVO_MIN_US,
                    max_pulse=SERVO_MAX_US,
                    actuation_range=180,
                )
                self._servo.angle = int(self._angle)
                self._hw = True
                print(f"[SRV] PCA9685 @ 0x{PCA9685_ADDR:02X} I2C-{I2C_BUS} ch{SERVO_CHANNEL} ✓")
            except Exception as exc:
                print(f"[SRV] PCA9685 init failed ({exc}) — sim mode")

        self._t = threading.Thread(target=self._sweep_loop, daemon=True, name="servo")
        self._t.start()
        print(f"[SRV] Servo ready ({'HW' if self._hw else 'SIM'})")

    def _sweep_loop(self):
        while self._running:
            with self._lock:
                diff = self._target - self._angle
                if abs(diff) > 0.5:
                    step = SERVO_STEP_DEG if diff > 0 else -SERVO_STEP_DEG
                    # Don't overshoot
                    if abs(step) > abs(diff):
                        step = diff
                    self._angle = round(max(0.0, min(180.0, self._angle + step)), 1)
                    if self._hw and self._servo:
                        try:
                            self._servo.angle = int(self._angle)
                        except Exception:
                            pass
            time.sleep(SERVO_TICK_S)

    # ── Public API ───────────────────────────────────────────────────

    def face_players(self):
        """Sweep head toward players (RED LIGHT)."""
        self.set_angle(SERVO_FACE_DEG)

    def face_away(self):
        """Sweep head away from players (GREEN LIGHT)."""
        self.set_angle(SERVO_AWAY_DEG)

    def set_angle(self, deg: float):
        with self._lock:
            self._target = float(max(0, min(180, deg)))

    @property
    def angle(self) -> float:
        with self._lock:
            return self._angle

    @property
    def is_at_target(self) -> bool:
        with self._lock:
            return abs(self._angle - self._target) <= 3

    @property
    def is_facing_players(self) -> bool:
        with self._lock:
            return abs(self._angle - SERVO_FACE_DEG) <= 10

    def cleanup(self):
        self._running = False
        self._t.join(timeout=1.5)
        if self._hw and self._servo:
            try:
                self._servo.angle = int(SERVO_AWAY_DEG)
            except Exception:
                pass


# ════════════════════════════════════════════════════════════════════
#  LASER BREAK-BEAM
# ════════════════════════════════════════════════════════════════════
class LaserBreakBeam:
    """
    Physical Pin 13 (BOARD mode).
    Input pulled HIGH; beam broken = signal goes LOW = player crossed.
    """

    def __init__(self):
        self._enabled = False
        if not _GPIO:
            print("[LAS] GPIO unavailable — laser disabled")
            return
        try:
            if GPIO.getmode() is None:
                GPIO.setmode(GPIO.BOARD)
            GPIO.setup(LASER_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            self._enabled = True
            print(f"[LAS] Laser on physical pin {LASER_PIN} ✓")
        except Exception as exc:
            print(f"[LAS] GPIO setup failed ({exc})")

    @property
    def broken(self) -> bool:
        if not self._enabled:
            return False
        try:
            return GPIO.input(LASER_PIN) == GPIO.LOW
        except Exception:
            return False

    def cleanup(self):
        if self._enabled and _GPIO:
            try:
                GPIO.cleanup()
            except Exception:
                pass
