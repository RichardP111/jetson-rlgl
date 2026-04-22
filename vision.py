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

import math
from typing import Any, cast

import cv2 as _cv2
import numpy as np
from ultralytics import YOLO

cv2 = cast(Any, _cv2)


from config import (CYAN, GREEN, MOTION_PX, PALM_WRIST_ABOVE_SHOULDER, RED,
                    TAPE_HSV_HIGH, TAPE_HSV_LOW, TAPE_MIN_PX, TAPE_ZONE_X,
                    WHITE, YELLOW, YOLO_CONF, YOLO_IOU, YOLO_MODEL)

# ── YOLO keypoint indices (COCO 17-point skeleton) ──────────────────
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
    (8, 10),  # arms
    (5, 11),
    (6, 12),
    (11, 12),  # torso
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),  # legs
    (0, 5),
    (0, 6),  # neck
]

# BGR colour for the skeleton overlay
_SKEL_COL = (0, 220, 160)  # neon teal
_JOINT_COL = (0, 255, 190)


def _tensor_to_numpy(tensor):
    """Convert PyTorch tensor or NumPy array to NumPy array."""
    if hasattr(tensor, "cpu"):
        return tensor.cpu().numpy()
    elif hasattr(tensor, "numpy"):
        return tensor.numpy()
    else:
        return np.array(tensor)


# ════════════════════════════════════════════════════════════════════
#  COLOUR ANALYSER  — identify a person by their shirt colour name
# ════════════════════════════════════════════════════════════════════
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


def get_shirt_colour(frame: np.ndarray, box: np.ndarray) -> str:
    """
    Sample the torso region of a bounding box and return the
    closest named colour string for the shirt.
    """
    if frame is None:
        return "unknown"
    x1, y1, x2, y2 = map(int, box[:4])
    h = y2 - y1
    w = x2 - x1
    if h < 10 or w < 10:
        return "unknown"
    # Sample the upper-middle torso (between 25% and 55% down the box)
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


# ════════════════════════════════════════════════════════════════════
#  POSE TRACKER
# ════════════════════════════════════════════════════════════════════
class PoseTracker:
    """
    Wraps YOLOv8-pose with persistent tracking.

    process_frame() returns:
        result_data : dict  (see below) or None
        overlay     : annotated BGR frame

    result_data keys:
        players_alive  : int
        raw_results    : ultralytics Results object
        boxes          : list of [x1,y1,x2,y2] (numpy float32)
        track_ids      : list of int track IDs  (or None per box)
        keypoints      : list of (17,3) arrays  (x, y, conf per kp)
        shirt_colours  : list of str colour names
    """

    def __init__(self):
        self.model = YOLO(YOLO_MODEL)
        print(f"[VIS] YOLOv8-pose loaded: {YOLO_MODEL} ✓")

    def process_frame(self, frame: np.ndarray | None):
        if frame is None:
            return None, None

        results = self.model.track(
            frame,
            persist=True,
            verbose=False,
            conf=YOLO_CONF,
            iou=YOLO_IOU,
        )
        r = results[0]

        # Build annotated overlay with our own skeleton (cleaner than .plot())
        overlay = frame.copy()
        self._draw_skeletons(overlay, r)

        if r.boxes is None or len(r.boxes) == 0:
            return {"players_alive": 0, "raw_results": r, "boxes": [], "track_ids": [], "keypoints": [], "shirt_colours": []}, overlay

        boxes = _tensor_to_numpy(r.boxes.xyxy)  # (N,4)
        ids_raw = r.boxes.id
        track_ids = _tensor_to_numpy(ids_raw).astype(int).tolist() if ids_raw is not None else [None] * len(boxes)

        kpts = []
        if r.keypoints is not None and r.keypoints.xy is not None:
            xy = _tensor_to_numpy(r.keypoints.xy)  # (N,17,2)
            conf = _tensor_to_numpy(r.keypoints.conf) if r.keypoints.conf is not None else np.ones(xy.shape[:2] + (1,))
            for i in range(len(boxes)):
                if i < len(xy):
                    kp = np.concatenate([xy[i], conf[i : i + 1].T if conf[i].ndim == 1 else conf[i]], axis=-1)  # (17,3)
                    kpts.append(kp)
                else:
                    kpts.append(np.zeros((17, 3)))
        else:
            kpts = [np.zeros((17, 3))] * len(boxes)

       for i, box in enumerate(boxes):
            tid = track_ids[i]
            
            # Only recalculate color if it's a new player OR every 30 frames
            if tid not in self.color_cache or self.frame_count % 30 == 0:
                self.color_cache[tid] = get_shirt_colour(frame, box)
            
            shirt_colours.append(self.color_cache[tid])

        data = {
            "players_alive": len(boxes),
            "raw_results": r,
            "boxes": boxes,
            "track_ids": track_ids,
            "keypoints": kpts,
            "shirt_colours": shirt_colours,
        }
        return data, overlay

    # ── Skeleton drawing ─────────────────────────────────────────────

    def _draw_skeletons(self, frame: np.ndarray, r):
        """Draw neon skeleton overlays on frame in-place."""
        if r.keypoints is None or r.keypoints.xy is None:
            return
        h, w = frame.shape[:2]
        xy = _tensor_to_numpy(r.keypoints.xy)  # (N,17,2)
        conf_all = None
        if r.keypoints.conf is not None:
            conf_all = _tensor_to_numpy(r.keypoints.conf)  # (N,17)

        for pi, person in enumerate(xy):
            # Draw edges
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

            # Draw joints
            for ki, (x, y) in enumerate(person):
                if x == 0 and y == 0:
                    continue
                if conf_all is not None and conf_all[pi][ki] < 0.3:
                    continue
                r_px = 5 if ki in (5, 6, 11, 12) else 3
                cv2.circle(frame, (int(x), int(y)), r_px, _JOINT_COL, -1, cv2.LINE_AA)


# ════════════════════════════════════════════════════════════════════
#  PALM-RAISE GESTURE DETECTION
# ════════════════════════════════════════════════════════════════════
def detect_palm_raise(pose_data: dict | None) -> bool:
    """
    Returns True if ANY detected person has at least one wrist raised
    clearly above their shoulder — the game-start gesture.
    Uses YOLO keypoints; no MediaPipe dependency.
    """
    if pose_data is None or not pose_data.get("keypoints"):
        return False

    for kp in pose_data["keypoints"]:
        # kp shape: (17, 3) — x, y, conf
        l_sh = kp[_KP["l_shoulder"]]
        r_sh = kp[_KP["r_shoulder"]]
        l_wr = kp[_KP["l_wrist"]]
        r_wr = kp[_KP["r_wrist"]]

        def _raised(shoulder, wrist) -> bool:
            if shoulder[2] < 0.3 or wrist[2] < 0.3:
                return False
            # In image coords, Y increases downward → wrist ABOVE = smaller y
            return (shoulder[1] - wrist[1]) > PALM_WRIST_ABOVE_SHOULDER

        if _raised(l_sh, l_wr) or _raised(r_sh, r_wr):
            return True
    return False


# ════════════════════════════════════════════════════════════════════
#  FINISH-LINE TAPE DETECTOR
# ════════════════════════════════════════════════════════════════════
def check_tape_finish(frame: np.ndarray | None, pose_data: dict | None) -> bool:
    """
    Returns True if the tape colour is visible in the finish zone AND
    at least one player centroid is near/past that zone.
    """
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

    # Tape is in frame — now check if any player centroid is in finish zone
    if pose_data and pose_data.get("boxes") is not None:
        for box in pose_data["boxes"]:
            cx = (box[0] + box[2]) / 2
            if cx >= zone_x - 50:  # small tolerance
                return True
    return False
