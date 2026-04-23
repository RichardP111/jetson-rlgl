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

# Import after checking CAM_SHARPNESS exists
try:
    from config import (
        YOLO_MODEL,
        YOLO_CONF,
        YOLO_IOU,
        PALM_WRIST_ABOVE_SHOULDER,
        TAPE_HSV_LOW,
        TAPE_HSV_HIGH,
        TAPE_ZONE_X,
        TAPE_MIN_PX,
        CAM_SHARPNESS,
        CAM_BRIGHTNESS,
        CAM_CONTRAST,
        CAM_W,
        CAM_H,
    )
except ImportError:
    CAM_SHARPNESS = False
    CAM_BRIGHTNESS = 1.0
    CAM_CONTRAST = 1.0

_KP = {
    "nose": 0,
    "l_eye": 1,
    "r_eye": 2,
    "l_ear": 3,
    "r_ear": 4,
    "l_shoulder": 5,
    "r_shoulder": 6,
    "l_elbow": 7,
    "r_elbow": 8,
    "l_wrist": 9,
    "r_wrist": 10,
    "l_hip": 11,
    "r_hip": 12,
    "l_knee": 13,
    "r_knee": 14,
    "l_ankle": 15,
    "r_ankle": 16,
}

_SKELETON_EDGES = [
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 5),
    (0, 6),
]

_SKEL_COL = (79, 195, 247)  # Material 3 teal
_JOINT_COL = (129, 199, 132)  # Material 3 green

_NAMED = [
    ("red", np.array([0, 50, 50]), np.array([10, 255, 255])),
    ("red", np.array([170, 50, 50]), np.array([180, 255, 255])),
    ("orange", np.array([11, 100, 100]), np.array([22, 255, 255])),
    ("yellow", np.array([23, 80, 80]), np.array([34, 255, 255])),
    ("green", np.array([35, 50, 50]), np.array([85, 255, 255])),
    ("cyan", np.array([86, 50, 50]), np.array([100, 255, 255])),
    ("blue", np.array([101, 50, 50]), np.array([130, 255, 255])),
    ("purple", np.array([131, 40, 40]), np.array([155, 255, 255])),
    ("pink", np.array([156, 40, 80]), np.array([169, 255, 255])),
    ("white", np.array([0, 0, 200]), np.array([180, 35, 255])),
    ("black", np.array([0, 0, 0]), np.array([180, 255, 50])),
    ("grey", np.array([0, 0, 50]), np.array([180, 35, 200])),
]


def _as_numpy(value):
    if value is None:
        return None
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def get_shirt_colour(frame: np.ndarray, box: np.ndarray) -> str:
    if frame is None:
        return "unknown"
    x1, y1, x2, y2 = map(int, box[:4])
    h = y2 - y1
    w = x2 - x1
    if h < 10 or w < 10:
        return "unknown"
    ty1 = y1 + int(h * 0.25)
    ty2 = y1 + int(h * 0.55)
    tx1 = x1 + int(w * 0.25)
    tx2 = x1 + int(w * 0.75)
    roi = frame[ty1:ty2, tx1:tx2]
    if roi.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    best_name = "unknown"
    best_count = 0
    for name, lo, hi in _NAMED:
        cnt = int(np.count_nonzero(cv2.inRange(hsv, lo, hi)))
        if cnt > best_count:
            best_count = cnt
            best_name = name
    return best_name


class ProPoseTracker:
    """YOLOv8-pose with CLAHE sharpening for crystal-clear tracking"""

    def __init__(self):
        self.model = YOLO(YOLO_MODEL)
        print(f"[VIS] YOLOv8-pose loaded: {YOLO_MODEL} ✓")

        # CLAHE for sharpening
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def process_frame(self, frame: np.ndarray | None):
        if frame is None:
            return None, None

        # CAMERA ENHANCEMENT (new feature)
        if CAM_SHARPNESS:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            l_channel = self._clahe.apply(l_channel)
            frame = cv2.merge([l_channel, a_channel, b_channel])
            frame = cv2.cvtColor(frame, cv2.COLOR_LAB2BGR)

        # Brightness & contrast boost
        frame = cv2.convertScaleAbs(frame, alpha=CAM_CONTRAST, beta=(CAM_BRIGHTNESS - 1.0) * 50)

        # YOLO tracking
        results = self.model.track(
            frame,
            persist=True,
            verbose=False,
            conf=YOLO_CONF,
            iou=YOLO_IOU,
        )
        r = results[0]

        # Draw skeleton
        overlay = frame.copy()
        self._draw_skeletons(overlay, r)

        if r.boxes is None or len(r.boxes) == 0:
            return {"players_alive": 0, "raw_data": r, "boxes": [], "track_ids": [], "keypoints": [], "shirt_colours": []}, overlay

        boxes = _as_numpy(r.boxes.xyxy)
        boxes = np.asarray(boxes) if boxes is not None else np.zeros((0, 4))
        ids_raw = r.boxes.id
        ids_numpy = _as_numpy(ids_raw)
        track_ids = ids_numpy.astype(int).tolist() if ids_numpy is not None else [None] * len(boxes)

        kpts = []
        if r.keypoints is not None and r.keypoints.xy is not None:
            xy = _as_numpy(r.keypoints.xy)
            xy = np.asarray(xy) if xy is not None else np.zeros((0, 17, 2))
            conf = _as_numpy(r.keypoints.conf) if r.keypoints.conf is not None else None
            conf = np.asarray(conf) if conf is not None else np.ones(xy.shape[:2], dtype=float)
            for i in range(len(boxes)):
                if i < len(xy):
                    kp = np.concatenate([xy[i], conf[i][:, None]], axis=-1)
                    kpts.append(kp)
                else:
                    kpts.append(np.zeros((17, 3)))
        else:
            kpts = [np.zeros((17, 3))] * len(boxes)

        shirt_colours = [get_shirt_colour(frame, b) for b in boxes]

        data = {
            "players_alive": len(boxes),
            "raw_data": r,
            "boxes": boxes,
            "track_ids": track_ids,
            "keypoints": kpts,
            "shirt_colours": shirt_colours,
        }
        return data, overlay

    def _draw_skeletons(self, frame: np.ndarray, r):
        if r.keypoints is None or r.keypoints.xy is None:
            return
        h, w = frame.shape[:2]
        xy = _as_numpy(r.keypoints.xy)
        xy = np.asarray(xy) if xy is not None else np.zeros((0, 17, 2))
        conf_all = None
        if r.keypoints.conf is not None:
            conf_all = _as_numpy(r.keypoints.conf)
            conf_all = np.asarray(conf_all) if conf_all is not None else None

        for pi, person in enumerate(xy):
            for a, b in _SKELETON_EDGES:
                if a >= len(person) or b >= len(person):
                    continue
                xa, ya = int(person[a][0]), int(person[a][1])
                xb, yb = int(person[b][0]), int(person[b][1])
                if xa == 0 and ya == 0:
                    continue
                if xb == 0 and yb == 0:
                    continue
                ca = conf_all[pi][a] if conf_all is not None else 1.0
                cb = conf_all[pi][b] if conf_all is not None else 1.0
                if ca < 0.3 or cb < 0.3:
                    continue
                cv2.line(frame, (xa, ya), (xb, yb), _SKEL_COL, 2, cv2.LINE_AA)

            for ki, (x, y) in enumerate(person):
                if x == 0 and y == 0:
                    continue
                if conf_all is not None and conf_all[pi][ki] < 0.3:
                    continue
                r_px = 5 if ki in (5, 6, 11, 12) else 3
                cv2.circle(frame, (int(x), int(y)), r_px, _JOINT_COL, -1, cv2.LINE_AA)


def detect_palm_raise(pose_data: dict | None) -> bool:
    if pose_data is None or not pose_data.get("keypoints"):
        return False

    for kp in pose_data["keypoints"]:
        l_sh = kp[_KP["l_shoulder"]]
        r_sh = kp[_KP["r_shoulder"]]
        l_wr = kp[_KP["l_wrist"]]
        r_wr = kp[_KP["r_wrist"]]

        def _raised(shoulder, wrist) -> bool:
            if shoulder[2] < 0.3 or wrist[2] < 0.3:
                return False
            return (shoulder[1] - wrist[1]) > PALM_WRIST_ABOVE_SHOULDER

        if _raised(l_sh, l_wr) or _raised(r_sh, r_wr):
            return True
    return False


def check_tape_finish(frame: np.ndarray | None, pose_data: dict | None) -> bool:
    if frame is None:
        return False

    fw = frame.shape[1]
    zone_x = int(fw * TAPE_ZONE_X)
    roi = frame[:, zone_x:]
    if roi.size == 0:
        return False

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(TAPE_HSV_LOW), np.array(TAPE_HSV_HIGH))
    if int(np.count_nonzero(mask)) < TAPE_MIN_PX:
        return False

    if pose_data and pose_data.get("boxes") is not None:
        for box in pose_data["boxes"]:
            cx = (box[0] + box[2]) / 2
            if cx >= zone_x - 50:
                return True
    return False


# Backwards compatibility
class ColorAnalyzer:
    @staticmethod
    def get_shirt_color(frame, box):
        return get_shirt_colour(frame, box)
