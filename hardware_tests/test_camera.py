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

# --- Third-Party Imports ---
import cv2

# --- Constants ---
GSTREAMER_PIPELINE: str = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! "
    "video/x-raw, format=BGR ! appsink drop=1"
)


def main() -> int:
    print("--- CSI CAMERA TEST ---")
    print("Initializing GStreamer pipeline...")

    cap = cv2.VideoCapture(GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("Error: Could not open CSI camera.")
        print(
            "Did you run 'sudo /opt/nvidia/jetson-io/jetson-io.py' and enable IMX219?"
        )
        return 1

    print("Camera feed live! Press 'Q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        cv2.imshow("Jetson CSI Camera Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
