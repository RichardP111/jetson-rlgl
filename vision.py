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

import cv2
import numpy as np
import math
from ultralytics import YOLO

class ColorAnalyzer:
    KNOWN_COLORS = {
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0),
        "yellow": (0, 255, 255),
        "purple": (128, 0, 128),
        "orange": (0, 165, 255),
        "white": (255, 255, 255),
        "black": (40, 40, 40),
        "grey": (128, 128, 128)
    }

    @staticmethod
    def get_shirt_color(frame: np.ndarray | None, box: list) -> str:
        if frame is None:
            return "unknown"
            
        x1, y1, x2, y2 = map(int, box)
        h, w = y2 - y1, x2 - x1
        
        torso_y1 = int(y1 + (h * 0.3))
        torso_y2 = int(y1 + (h * 0.6))
        torso_x1 = int(x1 + (w * 0.3))
        torso_x2 = int(x1 + (w * 0.7))
        
        if torso_x1 >= torso_x2 or torso_y1 >= torso_y2:
            return "unknown"
            
        roi = frame[torso_y1:torso_y2, torso_x1:torso_x2]
        
        if roi.size == 0:
            return "unknown"

        avg_color_per_row = np.average(roi, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        b, g, r = avg_color[:3]

        closest_color = "unknown"
        min_dist = float('inf')
        
        for name, (kb, kg, kr) in ColorAnalyzer.KNOWN_COLORS.items():
            dist = math.sqrt((b - kb)**2 + (g - kg)**2 + (r - kr)**2)
            if dist < min_dist:
                min_dist = dist
                closest_color = name
                
        return closest_color


class ProPoseTracker:
    def __init__(self, model_size='n'):
        self.model = YOLO(f"yolov8{model_size}-pose.pt")

    def process_frame(self, frame: np.ndarray | None):
        if frame is None:
            return None, None
            
        # persist=True enables ID tracking across frames
        results = self.model.track(frame, persist=True, verbose=False)
        annotated_frame = results[0].plot()

        pose_data = {
            "raw_data": results[0],
            "players_alive": len(results[0].boxes) if results[0].boxes else 0
        }
        
        return pose_data, annotated_frame