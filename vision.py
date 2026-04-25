#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Project:      Red Light Green Light (Jetson Orin Nano)
File:         vision.py
Description:  YOLOv8-pose tracking, shirt-colour classification, palm-raise
              detection, and finish-tape detection.

              This module returns data only. All rendering (skeletons, chips,
              overlays) lives in ui.py so pipeline stages stay decoupled.
===============================================================================
"""

from __future__ import annotations

import threading
import time
from typing import Any

import cv2  # noqa: F401
import numpy as np

from config import (
    CAM_BRIGHTNESS,
    CAM_CONTRAST,
    CAM_SHARPNESS,
    PALM_WRIST_ABOVE_SHOULDER,
    TAPE_HSV_HIGH,
    TAPE_HSV_LOW,
    TAPE_MIN_PX,
    TAPE_ZONE_X,
    YOLO_CONF,
    YOLO_IOU,
    YOLO_MODEL,
)

# ---------------------------------------------------------------------------
# Keypoint index map (YOLOv8-pose / COCO 17)
# ---------------------------------------------------------------------------
KP = {
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

SKELETON_EDGES = [
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

# Two-range red handled separately below.
_COLOUR_RANGES = [
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

_RED_LO_1 = np.array([0, 50, 50])
_RED_HI_1 = np.array([10, 255, 255])
_RED_LO_2 = np.array([170, 50, 50])
_RED_HI_2 = np.array([180, 255, 255])


def _as_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def get_shirt_colour(frame: np.ndarray | None, box: np.ndarray) -> str:
    """Sample the torso ROI and pick the dominant colour label."""
    if frame is None:
        return "unknown"
    x1, y1, x2, y2 = map(int, box[:4])
    h = y2 - y1
    w = x2 - x1
    if h < 10 or w < 10:
        return "unknown"
    ty1 = max(0, y1 + int(h * 0.25))
    ty2 = max(0, y1 + int(h * 0.55))
    tx1 = max(0, x1 + int(w * 0.25))
    tx2 = max(0, x1 + int(w * 0.75))
    roi = frame[ty1:ty2, tx1:tx2]
    if roi.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)  # type: ignore[attr-defined]

    # Red spans both ends of hue; combine masks.
    red_mask = cv2.bitwise_or(  # type: ignore[attr-defined]
        cv2.inRange(hsv, _RED_LO_1, _RED_HI_1),  # type: ignore[attr-defined]
        cv2.inRange(hsv, _RED_LO_2, _RED_HI_2),  # type: ignore[attr-defined]
    )
    best_name = "red"
    best_count = int(np.count_nonzero(red_mask))

    for name, lo, hi in _COLOUR_RANGES:
        cnt = int(np.count_nonzero(cv2.inRange(hsv, lo, hi)))  # type: ignore[attr-defined]
        if cnt > best_count:
            best_count = cnt
            best_name = name

    if best_count < 40:
        return "unknown"
    return best_name


class ProPoseTracker:
    """YOLOv8-pose tracker with CUDA-aware FP16 inference.

    Emits pure data: boxes, track ids, keypoints, shirt colours. Timing for
    the dev panel is exposed via ``last_inference_ms``.
    """

    def __init__(self) -> None:
        # Late imports so unit-test contexts without ultralytics installed
        # can still import this module's helpers.
        from ultralytics import YOLO  # noqa: WPS433

        self._use_half = False
        self._device = "cpu"
        try:
            import torch  # type: ignore # noqa: WPS433

            if torch.cuda.is_available():
                self._device = "cuda"
                self._use_half = True
        except Exception:
            pass

        self.model = YOLO(YOLO_MODEL)
        try:
            self.model.to(self._device)
        except Exception:
            pass

        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # type: ignore[attr-defined]
        self.last_inference_ms: float = 0.0
        print(f"[VIS] YOLOv8-pose on {self._device} " f"(half={self._use_half})")

    def _enhance(self, frame: np.ndarray) -> np.ndarray:
        if CAM_SHARPNESS:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)  # type: ignore[attr-defined]
            lc, ac, bc = cv2.split(lab)  # type: ignore[attr-defined]
            lc = self._clahe.apply(lc)  # type: ignore[attr-defined]
            frame = cv2.cvtColor(cv2.merge([lc, ac, bc]), cv2.COLOR_LAB2BGR)  # type: ignore[attr-defined]
        if CAM_BRIGHTNESS != 1.0 or CAM_CONTRAST != 1.0:
            frame = cv2.convertScaleAbs(  # type: ignore[attr-defined]
                frame,
                alpha=CAM_CONTRAST,
                beta=(CAM_BRIGHTNESS - 1.0) * 50,
            )
        return frame

    def process_frame(self, frame: np.ndarray | None) -> tuple[dict | None, np.ndarray | None]:
        if frame is None:
            return None, None

        frame = self._enhance(frame)

        t0 = time.perf_counter()
        try:
            results = self.model.track(
                frame,
                persist=True,
                verbose=False,
                conf=YOLO_CONF,
                iou=YOLO_IOU,
                half=self._use_half,
                device=self._device,
            )
        except TypeError:
            # Older ultralytics versions reject half/device kwargs in track().
            results = self.model.track(frame, persist=True, verbose=False, conf=YOLO_CONF, iou=YOLO_IOU)
        self.last_inference_ms = (time.perf_counter() - t0) * 1000.0

        if not results:
            return self._empty(frame), frame
        r = results[0]

        if r.boxes is None or len(r.boxes) == 0:
            return self._empty(frame), frame

        boxes_np = _as_numpy(r.boxes.xyxy)
        boxes = boxes_np if boxes_np is not None else np.zeros((0, 4))

        ids_np = _as_numpy(r.boxes.id) if r.boxes.id is not None else None
        if ids_np is not None:
            track_ids = ids_np.astype(int).tolist()
        else:
            track_ids = [None] * len(boxes)

        kpts: list[np.ndarray] = []
        if r.keypoints is not None and r.keypoints.xy is not None:
            xy = _as_numpy(r.keypoints.xy)
            xy = xy if xy is not None else np.zeros((0, 17, 2))
            conf = _as_numpy(r.keypoints.conf) if r.keypoints.conf is not None else None
            if conf is None:
                conf = np.ones(xy.shape[:2], dtype=float)
            for i in range(len(boxes)):
                if i < len(xy):
                    kpts.append(np.concatenate([xy[i], conf[i][:, None]], axis=-1))
                else:
                    kpts.append(np.zeros((17, 3)))
        else:
            kpts = [np.zeros((17, 3))] * len(boxes)

        shirt_colours = [get_shirt_colour(frame, b) for b in boxes]

        return (
            {
                "players_alive": len(boxes),
                "boxes": boxes,
                "track_ids": track_ids,
                "keypoints": kpts,
                "shirt_colours": shirt_colours,
                "frame_w": int(frame.shape[1]),
                "frame_h": int(frame.shape[0]),
            },
            frame,
        )

    def _empty(self, frame: np.ndarray) -> dict:
        return {
            "players_alive": 0,
            "boxes": np.zeros((0, 4)),
            "track_ids": [],
            "keypoints": [],
            "shirt_colours": [],
            "frame_w": int(frame.shape[1]),
            "frame_h": int(frame.shape[0]),
        }


def detect_palm_raise(pose_data: dict | None) -> bool:
    """True if any detected person has a wrist above their shoulder."""
    if pose_data is None or not pose_data.get("keypoints"):
        return False

    threshold = PALM_WRIST_ABOVE_SHOULDER

    def _raised(shoulder: np.ndarray, wrist: np.ndarray) -> bool:
        if shoulder[2] < 0.3 or wrist[2] < 0.3:
            return False
        return (shoulder[1] - wrist[1]) > threshold

    for kp in pose_data["keypoints"]:
        if _raised(kp[KP["l_shoulder"]], kp[KP["l_wrist"]]):
            return True
        if _raised(kp[KP["r_shoulder"]], kp[KP["r_wrist"]]):
            return True
    return False


def check_tape_finish(frame: np.ndarray | None, pose_data: dict | None) -> bool:
    """True if finish-tape colour is present near the right edge AND a
    tracked person is close enough to it to count as crossing."""
    if frame is None:
        return False

    fw = frame.shape[1]
    zone_x = int(fw * TAPE_ZONE_X)
    roi = frame[:, zone_x:]
    if roi.size == 0:
        return False

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)  # type: ignore[attr-defined]
    mask = cv2.inRange(hsv, np.array(TAPE_HSV_LOW), np.array(TAPE_HSV_HIGH))  # type: ignore[attr-defined]
    if int(np.count_nonzero(mask)) < TAPE_MIN_PX:
        return False

    if pose_data is None or pose_data.get("boxes") is None:
        return False
    for box in pose_data["boxes"]:
        cx = (box[0] + box[2]) / 2
        if cx >= zone_x - 50:
            return True
    return False


# ===========================================================================
# PoseWorker — threaded inference loop
# ===========================================================================
class PoseWorker:
    """Run YOLO inference on its own daemon thread.

    Why this exists
    ---------------
    Previously ``GameEngine.run`` called ``tracker.process_frame(frame)``
    every Nth iteration on the *main* Pygame thread. On the Orin Nano,
    pose inference takes 30–80 ms per call, which blocks the entire UI
    loop. The dev dashboard reported ~10 FPS as a result, even though
    the camera daemon was producing frames at 30 FPS. Decoupling
    inference from rendering removes the bottleneck — the main loop can
    now hold a steady 60 FPS while the worker chews through frames at
    whatever rate the GPU permits.

    Contract
    --------
    * ``latest()`` is non-blocking and returns ``(frame, pose, frame_id)``
      where ``frame`` is the BGR ndarray the pose was computed on (or
      None until the first inference completes), ``pose`` is the
      ProPoseTracker dict (or None), and ``frame_id`` increments each
      time a new pair is published.
    * The returned arrays are *shared, read-only*. Don't mutate them.
    * ``stop()`` joins the thread cleanly.
    """

    def __init__(self, camera, tracker: ProPoseTracker) -> None:
        self._cam = camera
        self._trk = tracker

        self._frame: np.ndarray | None = None
        self._pose: dict | None = None
        self._frame_id: int = 0
        self._lock = threading.Lock()

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="pose-worker")
        self._thread.start()
        print("[POSE] Worker thread started")

    # ----- producer side ------------------------------------------------
    def _loop(self) -> None:
        while self._running:
            f = self._cam.read()
            if f is None:
                # Camera not ready yet — back off briefly.
                time.sleep(0.01)
                continue
            try:
                pose, _ = self._trk.process_frame(f)
            except Exception as exc:
                print(f"[POSE] inference error: {exc}")
                pose = None
            with self._lock:
                self._frame = f
                self._pose = pose
                self._frame_id += 1
            # Yield so we don't pin a CPU when YOLO is unusually fast
            # (e.g. when no people are in frame, the tracker can spin).
            time.sleep(0.001)

    # ----- consumer side ------------------------------------------------
    def latest(self) -> tuple[np.ndarray | None, dict | None, int]:
        """Return the most recent (frame, pose, frame_id) without blocking."""
        with self._lock:
            return self._frame, self._pose, self._frame_id

    def stop(self) -> None:
        self._running = False
        try:
            self._thread.join(timeout=1.5)
        except Exception:
            pass
