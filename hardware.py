#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Project:      Red Light Green Light (Jetson Orin Nano)
File:         hardware.py
Description:  Platform-aware hardware abstraction.


Author:       Richard Pu
Last Updated: April 2026
===============================================================================
"""

from __future__ import annotations

import threading
import time
from typing import Any

import cv2  # type: ignore
import numpy as np

from config import (
    CAM_FPS,
    CAM_H,
    CAM_W,
    I2C_BUS,
    IS_LINUX,
    IS_WINDOWS,
    LASER_PIN,
    PCA9685_ADDR,
    SERVO_AWAY_DEG,
    SERVO_CHANNEL,
    SERVO_FACE_DEG,
    SERVO_FREQ,
    SERVO_MAX_US,
    SERVO_MIN_US,
    SERVO_STEP_DEG,
    SERVO_TICK_S,
    USE_LASER,
)

# ---------------------------------------------------------------------------
# Optional Jetson-only imports.
# ---------------------------------------------------------------------------
_ADAFRUIT_OK = False
_GPIO_OK = False

if IS_LINUX:
    try:
        import board  # type: ignore
        import busio  # type: ignore
        from adafruit_pca9685 import PCA9685 as _PCA9685  # type: ignore

        _ADAFRUIT_OK = True
    except (ImportError, AttributeError, NotImplementedError, ValueError):
        _ADAFRUIT_OK = False

    try:
        import Jetson.GPIO as GPIO  # type: ignore

        _GPIO_OK = True
    except (ImportError, AttributeError, RuntimeError):
        _GPIO_OK = False

HARDWARE_AVAILABLE = _ADAFRUIT_OK or _GPIO_OK


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
def _gstreamer_pipeline(
    capture_w: int = CAM_W,
    capture_h: int = CAM_H,
    output_w: int = 1920,
    output_h: int = 1080,
    fps: int = CAM_FPS,
) -> str:
    return (
        f"nvarguscamerasrc "
        f"wbmode=2 "
        f"saturation=1.01 "
        f'gainrange="1 8" '
        f'ispdigitalgainrange="1 1" '
        f'exposuretimerange="13000 16000000" '
        f"tnr-mode=2 tnr-strength=0.2 "
        f"exposurecompensation=1.14 "
        f"ee-mode=0 "
        f"aelock=false awblock=false "
        f"! video/x-raw(memory:NVMM), width={capture_w}, height={capture_h}, "
        f"format=NV12, framerate={fps}/1 ! "
        f"nvvidconv flip-method=2 ! "
        f"video/x-raw, width={output_w}, height={output_h}, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! "
        f"appsink drop=true max-buffers=1"
    )


class Camera:
    """Threaded camera capture with a single shared frame guarded by a lock.

    Producer: background daemon thread.
    Consumer: main loop calls read() / read_with_id() each frame.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._running = True
        self._ok = False
        self._backend = "none"
        self._fps_ts = time.time()
        self._fps_count = 0
        self._fps = 0.0
        self._frame_id = 0
        self._cap = self._open()

        self._thread = threading.Thread(target=self._loop, daemon=True, name="cam")
        self._thread.start()
        print(f"[CAM] Backend: {self._backend}  ok={self._ok}")

    def _open(self) -> Any | None:
        video_capture = getattr(cv2, "VideoCapture", None)
        if video_capture is None:
            self._backend = "none"
            self._ok = False
            return None

        if IS_WINDOWS:
            return self._open_default(video_capture)

        cap_gstreamer = int(getattr(cv2, "CAP_GSTREAMER", 0))
        cap = video_capture(_gstreamer_pipeline(), cap_gstreamer)
        if cap.isOpened():
            self._backend = "gstreamer"
            self._ok = True
            return cap

        cap_v4l2 = int(getattr(cv2, "CAP_V4L2", 0))
        cap = video_capture(0, cap_v4l2)
        if cap.isOpened():
            cap.set(int(getattr(cv2, "CAP_PROP_FRAME_WIDTH", 3)), CAM_W)
            cap.set(int(getattr(cv2, "CAP_PROP_FRAME_HEIGHT", 4)), CAM_H)
            cap.set(int(getattr(cv2, "CAP_PROP_FPS", 5)), CAM_FPS)
            cap.set(int(getattr(cv2, "CAP_PROP_BUFFERSIZE", 38)), 1)
            self._backend = "v4l2"
            self._ok = True
            return cap

        return self._open_default(video_capture)

    def _open_default(self, video_capture: Any | None = None) -> Any | None:
        if video_capture is None:
            video_capture = getattr(cv2, "VideoCapture", None)
            if video_capture is None:
                self._backend = "none"
                self._ok = False
                return None

        cap = video_capture(0)
        if cap.isOpened():
            cap.set(int(getattr(cv2, "CAP_PROP_FRAME_WIDTH", 3)), CAM_W)
            cap.set(int(getattr(cv2, "CAP_PROP_FRAME_HEIGHT", 4)), CAM_H)
            self._backend = "default"
            self._ok = True
            return cap
        self._backend = "none"
        self._ok = False
        return None

    def _loop(self) -> None:
        while self._running:
            if self._cap is not None and self._ok:
                ret, frame = self._cap.read()
                if ret and frame is not None:
                    with self._lock:
                        self._frame = frame
                        self._frame_id += 1
                    self._fps_count += 1
                    now = time.time()
                    if now - self._fps_ts >= 1.0:
                        self._fps = self._fps_count / (now - self._fps_ts)
                        self._fps_count = 0
                        self._fps_ts = now
                else:
                    time.sleep(0.01)
            else:
                time.sleep(0.033)
            time.sleep(0.001)

    def read(self) -> np.ndarray | None:
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def read_with_id(self) -> tuple[np.ndarray | None, int]:
        """Return a copy of the latest frame and a monotonic frame id.

        The id is incremented each time the camera thread stores a new frame.
        Callers can use it to skip expensive work when no new frame arrived.
        """
        with self._lock:
            if self._frame is None:
                return None, self._frame_id
            return self._frame.copy(), self._frame_id

    @property
    def ok(self) -> bool:
        return self._ok

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def fps(self) -> float:
        return self._fps

    def release(self) -> None:
        self._running = False
        self._thread.join(timeout=1.5)
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Servo
# ---------------------------------------------------------------------------
class ServoController:
    """PCA9685-driven pan servo with smooth background sweep."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._angle = float(SERVO_AWAY_DEG)
        self._target = float(SERVO_AWAY_DEG)
        self._running = True
        self._hw = False
        self._pca = None

        if _ADAFRUIT_OK:
            try:
                i2c = busio.I2C(board.SCL, board.SDA)
                self._pca = _PCA9685(i2c, address=PCA9685_ADDR)
                self._pca.frequency = SERVO_FREQ
                self._hw = True
                self._set_hardware_angle(self._angle)
                print(f"[SRV] PCA9685 Direct @ 0x{PCA9685_ADDR:02X} I2C-{I2C_BUS} ✓")
            except Exception as exc:
                print(f"[SRV] PCA9685 init failed ({exc}) - simulating")

        self._thread = threading.Thread(target=self._sweep_loop, daemon=True, name="servo")
        self._thread.start()
        print(f"[SRV] Ready ({'HW' if self._hw else 'SIM'})")

    def _set_hardware_angle(self, angle: float) -> None:
        if not self._hw or self._pca is None:
            return
        pulse_us = SERVO_MIN_US + (angle / 180.0) * (SERVO_MAX_US - SERVO_MIN_US)
        duty = int((pulse_us / (1000000.0 / SERVO_FREQ)) * 65535)
        self._pca.channels[SERVO_CHANNEL].duty_cycle = duty

    def _sweep_loop(self) -> None:
        while self._running:
            with self._lock:
                diff = self._target - self._angle
                if abs(diff) > 0.5:
                    step = SERVO_STEP_DEG if diff > 0 else -SERVO_STEP_DEG
                    if abs(step) > abs(diff):
                        step = diff
                    self._angle = round(max(0.0, min(180.0, self._angle + step)), 1)
                    self._set_hardware_angle(self._angle)
            time.sleep(SERVO_TICK_S)

    def face_players(self) -> None:
        self.set_angle(SERVO_FACE_DEG)

    def face_away(self) -> None:
        self.set_angle(SERVO_AWAY_DEG)

    def set_angle(self, deg: float) -> None:
        with self._lock:
            self._target = float(max(0.0, min(180.0, deg)))

    @property
    def angle(self) -> float:
        with self._lock:
            return self._angle

    @property
    def target(self) -> float:
        with self._lock:
            return self._target

    @property
    def is_at_target(self) -> bool:
        with self._lock:
            return abs(self._angle - self._target) <= 3

    @property
    def is_facing_players(self) -> bool:
        with self._lock:
            return abs(self._angle - SERVO_FACE_DEG) <= 10

    @property
    def is_hardware(self) -> bool:
        return self._hw

    def cleanup(self) -> None:
        self._running = False
        self._thread.join(timeout=1.5)
        if self._hw:
            self.set_angle(SERVO_AWAY_DEG)
            time.sleep(0.5)
            if self._pca:
                try:
                    self._pca.deinit()
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Laser break-beam
# ---------------------------------------------------------------------------
class LaserBreakBeam:
    """GPIO break-beam input. Returns broken=True when beam is interrupted.

    Honors the USE_LASER master flag from config.py: if False, ``broken``
    always reports False and the game falls back to camera-based finish
    detection. This lets you disable a flaky laser without re-flashing
    the board, just by toggling one config value.
    """

    def __init__(self) -> None:
        self._enabled = False
        self._gpio_ready = False
        self._master = bool(USE_LASER)

        if not self._master:
            print("[LAS] Disabled by config.USE_LASER=False - camera-only finish detection")
            return

        if not _GPIO_OK:
            print("[LAS] GPIO unavailable - laser disabled")
            return
        try:
            if GPIO.getmode() is None:
                GPIO.setmode(GPIO.BOARD)
            GPIO.setup(LASER_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            self._enabled = True
            self._gpio_ready = True
            print(f"[LAS] Laser break-beam on pin {LASER_PIN} ✓")
        except Exception as exc:
            print(f"[LAS] GPIO setup failed ({exc})")

    @property
    def enabled(self) -> bool:
        """True iff the laser is wired up AND the master flag allows use."""
        return self._enabled and self._master

    @property
    def in_use(self) -> bool:
        """Alias for enabled — clearer name for game logic."""
        return self.enabled

    @property
    def broken(self) -> bool:
        if not self.enabled:
            return False
        try:
            return GPIO.input(LASER_PIN) == GPIO.LOW
        except Exception:
            return False

    def cleanup(self) -> None:
        if self._gpio_ready and _GPIO_OK:
            try:
                GPIO.cleanup()
            except Exception:
                pass
