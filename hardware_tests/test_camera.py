#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Project:      Red Light Green Light (Jetson Orin Nano)
File:         test_camera.py
Description:  Isolated hardware test for the IMX219 CSI Camera using GStreamer.

Author:       Richard Pu
Last Updated: April 2026
===============================================================================
"""

# --- Standard Library Imports ---
import sys
import os

# --- Third-Party Imports ---
import cv2

# --- Constants ---
ARGUS_PIPELINE = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, format=I420 ! videoconvert ! "
    "video/x-raw, format=BGR ! appsink drop=1"
)

# Pipeline 2: V4L2 via GStreamer (Compatibility layer)
V4L2_GSTREAMER_PIPELINE = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw, width=1280, height=720, framerate=30/1 ! "
    "videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
)

def test_pipeline(name, pipeline_str, api_preference=cv2.CAP_GSTREAMER):
    print(f"\n[{name}] Attempting...")
    cap = cv2.VideoCapture(pipeline_str, api_preference)
    if cap.isOpened():
        return cap
    cap.release()
    print(f"  ⚠ {name} failed.")
    return None

def main():
    print("--- JETSON CAMERA DIAGNOSTIC (NOMACHINE MODE) ---")
    print(f"Current DISPLAY: {os.environ.get('DISPLAY', 'Not Set')}")
    
    # 0. Hardware existence check
    if not os.path.exists("/dev/video0"):
        print("  ❌ FATAL ERROR: /dev/video0 does not exist.")
        print("  Check your ribbon cable (Blue side away from heatsink).")
        return

    # 1. Try Argus
    cap = test_pipeline("NVIDIA Argus", ARGUS_PIPELINE)
    
    # 2. Try V4L2 GStreamer
    if not cap:
        cap = test_pipeline("V4L2 GStreamer", V4L2_GSTREAMER_PIPELINE)
        
    # 3. Try Direct V4L2 (No GStreamer)
    if not cap:
        print("\n[V4L2 Direct] Attempting (Bypassing GStreamer)...")
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if cap.isOpened():
            print("  ✅ Success using Direct V4L2!")
        else:
            print("  ❌ Direct V4L2 failed.")
            cap.release()
            cap = None

    if not cap:
        print("\n--- DIAGNOSIS ---")
        print("1. Run: sudo systemctl restart nvargus-daemon")
        print("2. Run: v4l2-ctl --list-devices (Verify imx219 is listed)")
        print("3. Ensure no other apps (like a previous crash) are holding the camera.")
        return

    print("\n✅ Camera feed successfully opened!")
    print("Press 'q' to exit.")
    
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("  ⚠ Failed to receive frame.")
            break

        cv2.imshow("Jetson Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Closing camera...")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()