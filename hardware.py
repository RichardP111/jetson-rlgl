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

import cv2, threading, time, numpy as np
from config import (
    CAM_W,
    CAM_H,
    CAM_FPS,
    I2C_BUS,
    PCA9685_ADDR,
    SERVO_CHANNEL,
    SERVO_FREQ,
    SERVO_MIN_US,
    SERVO_MAX_US,
    SERVO_AWAY_DEG,
    SERVO_FACE_DEG,
    SERVO_STEP_DEG,
    SERVO_TICK_S,
    LASER_PIN,
)

try:
    import board, busio
    from adafruit_pca9685 import PCA9685 as _PCA9685
    from adafruit_motor import servo as adafruit_servo

    _ADAFRUIT = True
except (ImportError, AttributeError):
    _ADAFRUIT = False

try:
    import Jetson.GPIO as GPIO

    _GPIO = True
except (ImportError, AttributeError):
    _GPIO = False


class Camera:
    def __init__(self):
        self._frame, self._lock, self._running, self._ok = None, threading.Lock(), True, False
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
            cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._cap, self._ok = cap, True
            print(f"[CAM] V4L2 camera @ {CAM_W}×{CAM_H} ✓")
        else:
            print("[CAM] No camera — blank frames")
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

    def read(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    @property
    def ok(self):
        return self._ok

    def release(self):
        self._running = False
        self._thread.join(timeout=1.5)
        if self._cap:
            self._cap.release()


class ServoController:
    def __init__(self):
        self._angle, self._target, self._lock, self._running = float(SERVO_AWAY_DEG), float(SERVO_AWAY_DEG), threading.Lock(), True
        self._hw, self._servo = False, None
        if _ADAFRUIT:
            try:
                i2c = busio.I2C(board.SCL, board.SDA)
                pca = _PCA9685(i2c, address=PCA9685_ADDR)
                pca.frequency = SERVO_FREQ
                self._servo = adafruit_servo.Servo(pca.channels[SERVO_CHANNEL], min_pulse=SERVO_MIN_US, max_pulse=SERVO_MAX_US, actuation_range=180)
                self._servo.angle = self._angle
                self._hw = True
                print(f"[SRV] PCA9685 @ 0x{PCA9685_ADDR:02X} I2C-{I2C_BUS} ✓")
            except Exception as exc:
                print(f"[SRV] PCA9685 failed ({exc}) — sim mode")
        self._t = threading.Thread(target=self._sweep_loop, daemon=True, name="servo")
        self._t.start()
        print(f"[SRV] Servo ready ({'HW' if self._hw else 'SIM'})")

    def _sweep_loop(self):
        while self._running:
            with self._lock:
                diff = self._target - self._angle
                if abs(diff) > 0.5:
                    step = SERVO_STEP_DEG if diff > 0 else -SERVO_STEP_DEG
                    if abs(step) > abs(diff):
                        step = diff
                    self._angle = round(max(0.0, min(180.0, self._angle + step)), 1)
                    if self._hw and self._servo:
                        try:
                            self._servo.angle = self._angle
                        except Exception:
                            pass
            time.sleep(SERVO_TICK_S)

    def face_players(self):
        self.set_angle(SERVO_FACE_DEG)

    def face_away(self):
        self.set_angle(SERVO_AWAY_DEG)

    def set_angle(self, deg):
        with self._lock:
            self._target = float(max(0, min(180, deg)))

    @property
    def angle(self):
        with self._lock:
            return self._angle

    @property
    def is_at_target(self):
        with self._lock:
            return abs(self._angle - self._target) <= 3

    @property
    def is_facing_players(self):
        with self._lock:
            return abs(self._angle - SERVO_FACE_DEG) <= 10

    def cleanup(self):
        self._running = False
        self._t.join(timeout=1.5)
        if self._hw and self._servo:
            try:
                self._servo.angle = SERVO_AWAY_DEG
            except Exception:
                pass


class LaserBreakBeam:
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
            print(f"[LAS] Laser on pin {LASER_PIN} ✓")
        except Exception as exc:
            print(f"[LAS] GPIO setup failed ({exc})")

    @property
    def broken(self):
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
