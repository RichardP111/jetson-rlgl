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
import threading
import numpy as np
import time
from config import CAMERA_WIDTH, CAMERA_HEIGHT, LASER_PIN

try:
    import Jetson.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False

class Camera:
    def __init__(self):
        self._frame = None
        self._lock = threading.Lock()
        self._running = True
        
        print("[CAM] Opening Direct V4L2 Driver (NoMachine Compatible)...")
        # Direct V4L2 bypasses the Argus daemon which crashes in NoMachine
        self._cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        
        if self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            print("[CAM] Hardware link established ✓")
        else:
            print("[CAM] ERROR: Physical camera not detected on /dev/video0")
            
        # Point to the _loop method defined below
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        """Internal thread loop to keep the camera feed fresh."""
        while self._running:
            ret, frame = self._cap.read()
            if ret and frame is not None:
                with self._lock:
                    self._frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        """Safely fetch the latest frame."""
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def release(self):
        self._running = False
        if hasattr(self, '_thread'):
            self._thread.join(timeout=1.0)
        self._cap.release()

class ServoController:
    """Mock controller for NoMachine testing without board."""
    def __init__(self):
        self.is_facing_players = False
        print("[SRV] Servo initialized (Sim Mode)")

    def face_players(self): 
        self.is_facing_players = True
        print("[SRV] RED LIGHT -> Face Players")

    def face_away(self): 
        self.is_facing_players = False
        print("[SRV] GREEN LIGHT -> Face Away")

class LaserBreakBeam:
    def __init__(self):
        self._enabled = False
        if GPIO_AVAILABLE:
            try:
                # Use BOARD mode to match physical Pin 13
                if GPIO.getmode() is None:
                    GPIO.setmode(GPIO.BOARD)
                
                GPIO.setup(LASER_PIN, GPIO.IN)
                self._enabled = True
                print(f"[LAS] Laser initialized on Physical Pin {LASER_PIN} ✓")
            except Exception as e:
                print(f"[LAS] GPIO setup skipped/failed: {e}")

    @property
    def broken(self) -> bool:
        if self._enabled:
            return GPIO.input(LASER_PIN) == GPIO.LOW
        return False