#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Project:      Red Light Green Light (Jetson Orin Nano)
File:         vision_pro.py
Description:  Advanced Computer Vision Engine using YOLOv8 for multi-person
              pose estimation and tracking.

Author:       Richard Pu
Last Updated: April 2026
===============================================================================
"""

# --- Standard Library Imports ---
import sys
import math

# --- Third-Party Imports ---
import cv2
import numpy as np

try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[VIS-PRO] Ultralytics not found. Run: pip install ultralytics")


class ProPoseTracker:
    """
    State-of-the-art multi-person skeleton tracking using YOLOv8-Pose.
    Capable of tracking 20+ people simultaneously.
    """

    def __init__(self, model_size: str = "m"):
        """
        model_size: 'n'(nano), 's'(small), 'm'(medium - best)
        """
        if not YOLO_AVAILABLE:
            raise RuntimeError("Cannot initialize Pro tracker without YOLO.")

        print(f"[VIS-PRO] Loading YOLOv8{model_size}-pose model...")
        # This will automatically download the weights file on the first run
        self.model = YOLO(f"yolov8{model_size}-pose.pt")
        print("[VIS-PRO] YOLOv8 Model Loaded ✓")

    def process_frame(self, frame: np.ndarray):
        """
        Runs the neural network on a single frame.
        Returns the annotated frame and the raw coordinate data.
        """
        if frame is None:
            return None, frame

        # Run inference (verbose=False keeps your terminal clean)
        results = self.model(frame, verbose=False)

        # The .plot() method automatically draws all the glowing neon
        # skeletons and bounding boxes for every person it finds.
        annotated_frame = results[0].plot()

        # Extract how many people are currently alive/in-frame
        players_detected = len(results[0].boxes) if results[0].boxes else 0

        # For the engine, we return the annotated frame and the count
        pose_data = {"players_alive": players_detected, "raw_data": results[0]}

        return pose_data, annotated_frame


class ColorAnalyzer:
    """Extracts the dominant shirt color of a tracked player."""

    # Standard colors in BGR format (OpenCV uses BGR, not RGB)
    KNOWN_COLORS = {
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0),
        "yellow": (0, 255, 255),
        "purple": (128, 0, 128),
        "orange": (0, 165, 255),
        "white": (255, 255, 255),
        "black": (40, 40, 40),  # Slightly gray to catch dark shirts
        "grey": (128, 128, 128),
    }

    @staticmethod
    def get_shirt_color(frame: np.ndarray, box: list) -> str:
        """Takes a full frame and a YOLO bounding box, returns a color string."""
        if frame is None:
            return "unknown"

        x1, y1, x2, y2 = map(int, box)
        h, w = y2 - y1, x2 - x1

        # Crop to the "Torso" (middle 30% vertically, middle 40% horizontally)
        torso_y1 = int(y1 + (h * 0.3))
        torso_y2 = int(y1 + (h * 0.6))
        torso_x1 = int(x1 + (w * 0.3))
        torso_x2 = int(x1 + (w * 0.7))

        # Safety check to ensure we don't crop outside the frame
        if torso_x1 >= torso_x2 or torso_y1 >= torso_y2:
            return "unknown"

        roi = frame[torso_y1:torso_y2, torso_x1:torso_x2]

        if roi.size == 0:
            return "unknown"

        # Calculate average B, G, R values in the torso crop
        avg_color_per_row = np.average(roi, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        b, g, r = avg_color[:3]

        # Find the closest known color using 3D distance
        closest_color = "unknown"
        min_dist = float("inf")

        for name, (kb, kg, kr) in ColorAnalyzer.KNOWN_COLORS.items():
            dist = math.sqrt((b - kb) ** 2 + (g - kg) ** 2 + (r - kr) ** 2)
            if dist < min_dist:
                min_dist = dist
                closest_color = name

        return closest_color


# =============================================================================
# --- TEST MODE ---
# Run this file directly to test the webcam without launching the full game.
# =============================================================================
def main():
    print("--- YOLOv8 PRO VISION TEST ---")
    print("Press 'Q' to quit the camera feed.")

    tracker = ProPoseTracker(model_size="n")  # Using 'nano' for laptop webcam speed
    cap = cv2.VideoCapture(0)  # 0 is usually your laptop's built-in webcam

    if not cap.isOpened():
        print("Error: Could not open laptop webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run the AI
        pose_data, overlay = tracker.process_frame(frame)

        # Add a custom HUD
        cv2.putText(
            overlay,
            f"TARGETS: {pose_data['players_alive']}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3,
        )
        cv2.putText(
            overlay,
            "YOLOv8 PRO ENGINE ACTIVE",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        # Show the result
        cv2.imshow("Jetson Pro Vision Test", overlay)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
