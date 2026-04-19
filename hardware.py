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

import cv2
import time
import threading
import numpy as np

from config import (
    GSTREAMER_PIPELINE,
    PCA9685_ADDRESS,
    SERVO_FREQ,
    SERVO_CHANNEL,
    SERVO_MIN_PULSE,
    SERVO_MAX_PULSE,
    SERVO_STEP_DEG,
    SERVO_STEP_DELAY,
    SERVO_FACING_FORWARD,
    SERVO_FACING_AWAY,
    LASER_PIN,
    I2C_BUS,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
)


# ── Try to import hardware libs; fall back cleanly for dev machines ──
try:
    import board
    import busio
    from adafruit_pca9685 import PCA9685
    from adafruit_motor import servo as adafruit_servo

    HW_AVAILABLE = True
except ImportError:
    HW_AVAILABLE = False
    print("[HW]  Adafruit libs not found — servo running in sim mode")

try:
    import Jetson.GPIO as GPIO

    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("[HW]  Jetson.GPIO not found — laser beam disabled")


# ─────────────────────────────────────────────────────────────────────
class Camera:
    """
    Background-threaded camera reader so the main loop never blocks.
    Falls back from GStreamer → /dev/video0 automatically.
    """

    def __init__(self):
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = True
        self.moment_captures: list[dict] = []  # {frame, label, ts}

        # Try GStreamer first
        self._cap = cv2.VideoCapture(GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER)
        if not self._cap.isOpened():
            print("[CAM]  GStreamer failed – falling back to /dev/video0")
            self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            raise RuntimeError("[CAM]  Cannot open any camera source!")
        print("[CAM]  Camera open ✓")

        self._thread = threading.Thread(target=self._loop, daemon=True, name="cam-reader")
        self._thread.start()

    def _loop(self):
        while self._running:
            ret, frame = self._cap.read()
            if ret and frame is not None:
                with self._lock:
                    self._frame = frame

    def read(self) -> np.ndarray | None:
        """Return the most recent frame without blocking."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def capture_moment(self, label: str = ""):
        """Snapshot a highlight frame for the win-screen gallery."""
        frame = self.read()
        if frame is not None:
            self.moment_captures.append({"frame": frame, "label": label, "ts": time.time()})
            # Keep at most 12
            if len(self.moment_captures) > 12:
                self.moment_captures.pop(0)

    def release(self):
        self._running = False
        self._thread.join(timeout=2)
        self._cap.release()


# ─────────────────────────────────────────────────────────────────────
class ServoController:
    """
    Smooth, threaded servo control via PCA9685.
    Head rotates:
        0°   = GREEN LIGHT  (facing away from players)
        180° = RED LIGHT    (staring players down)
    """

    def __init__(self):
        self._angle = 0.0
        self._target = 0.0
        self._lock = threading.Lock()
        self._running = True
        self._hw = False

        if HW_AVAILABLE:
            try:
                i2c = busio.I2C(board.SCL, board.SDA)
                pca = PCA9685(i2c, address=PCA9685_ADDRESS)
                pca.frequency = SERVO_FREQ
                self._servo = adafruit_servo.Servo(
                    pca.channels[SERVO_CHANNEL],
                    min_pulse=SERVO_MIN_PULSE,
                    max_pulse=SERVO_MAX_PULSE,
                )
                self._hw = True
                print(f"[SRV]  PCA9685 @ 0x{PCA9685_ADDRESS:02X} on I2C-{I2C_BUS} ✓")
            except Exception as exc:
                print(f"[SRV]  Hardware init failed ({exc}) — sim mode")

        self._thread = threading.Thread(target=self._sweep, daemon=True, name="servo-sweep")
        self._thread.start()

    def _sweep(self):
        """Runs in background: smoothly closes gap between current and target angle."""
        while self._running:
            with self._lock:
                diff = self._target - self._angle
                if abs(diff) > 0.5:
                    step = SERVO_STEP_DEG if diff > 0 else -SERVO_STEP_DEG
                    self._angle = max(0.0, min(180.0, self._angle + step))
                    if self._hw:
                        try:
                            self._servo.angle = self._angle
                        except Exception:
                            pass
            time.sleep(SERVO_STEP_DELAY)

    def set_angle(self, deg: float):
        with self._lock:
            self._target = max(0.0, min(180.0, deg))

    def face_players(self):
        """RED LIGHT — head turns to face players."""
        self.set_angle(SERVO_FACING_FORWARD)

    def face_away(self):
        """GREEN LIGHT — head turns away."""
        self.set_angle(SERVO_FACING_AWAY)

    @property
    def angle(self) -> float:
        with self._lock:
            return self._angle

    @property
    def is_at_target(self) -> bool:
        with self._lock:
            return abs(self._angle - self._target) < 5

    @property
    def is_facing_players(self) -> bool:
        with self._lock:
            return abs(self._angle - SERVO_FACING_FORWARD) < 12

    def cleanup(self):
        self._running = False
        self._thread.join(timeout=2)
        if self._hw:
            try:
                self._servo.angle = SERVO_FACING_AWAY
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────
class LaserBreakBeam:
    """
    Optional laser break-beam finish-line detector.
    Pin goes LOW when beam is broken (someone crosses).
    """

    def __init__(self, pin: int = LASER_PIN):
        self._pin = pin
        self._enabled = False
        if GPIO_AVAILABLE:
            try:
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
                self._enabled = True
                print(f"[LAS]  Laser beam on GPIO-{pin} ✓")
            except Exception as exc:
                print(f"[LAS]  GPIO setup failed ({exc})")

    @property
    def broken(self) -> bool:
        """Returns True when the beam is broken (player crossed the line)."""
        if not self._enabled:
            return False
        return GPIO.input(self._pin) == GPIO.LOW

    def cleanup(self):
        if self._enabled and GPIO_AVAILABLE:
            GPIO.cleanup()
