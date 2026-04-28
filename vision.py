#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Project:      Red Light Green Light (Jetson Orin Nano)
File:         vision.py
Description:  YOLOv8-pose tracking, shirt-colour classification, palm-raise
              detection, and finish-tape detection.

Author:       Richard Pu
Last Updated: April 2026
===============================================================================
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

import cv2  # type: ignore
import numpy as np

from config import (
    CAM_H,
    CAM_W,
    CLIP_BOTTOM_GARMENTS,
    CLIP_COLORS,
    CLIP_FEATURES,
    CLIP_MODEL_ID,
    CLIP_TOP_GARMENTS,
    FINISH_LINE_DETECT_FROM_TAPE,
    FINISH_LINE_HSV_HIGH_1,
    FINISH_LINE_HSV_HIGH_2,
    FINISH_LINE_HSV_LOW_1,
    FINISH_LINE_HSV_LOW_2,
    FINISH_LINE_MIN_TAPE_PX,
    FINISH_LINE_TOLERANCE_PX,
    FINISH_LINE_Y_PX,
    IDENTIFICATION_MODE,
    PALM_WRIST_ABOVE_SHOULDER,
    PROFILE_THUMB_H,
    PROFILE_THUMB_W,
    START_LINE_DETECT_FROM_TAPE,
    START_LINE_HSV_HIGH,
    START_LINE_HSV_LOW,
    START_LINE_TOLERANCE_PX,
    START_LINE_Y_PX,
    USE_TAPE_FINISH,
    VLM_MAX_NEW_TOKENS,
    VLM_MODEL_ID,
    VLM_PROMPT,
    VLM_REVISION,
    VLM_USE_CUDA_FP16,
    YOLO_CONF,
    YOLO_FALLBACK_MODEL,
    YOLO_IMGSZ,
    YOLO_INFER_H,
    YOLO_INFER_W,
    YOLO_IOU,
    YOLO_MODEL,
)

# ---------------------------------------------------------------------------
# Keypoint index map (YOLOv8-pose / COCO 17)
# Re-exported so ui.py can draw skeletons without re-defining indices.
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

# ---------------------------------------------------------------------------
# Color labelling — tiny multi-feature signature.
# Used directly when IDENTIFICATION_MODE == "color", and as a fast initial
# label for the other modes while the heavier model warms up.
# ---------------------------------------------------------------------------
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


def _dominant_colour(roi: np.ndarray) -> str:
    if roi is None or roi.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)  # type: ignore
    red_mask = cv2.bitwise_or(  # type: ignore
        cv2.inRange(hsv, _RED_LO_1, _RED_HI_1),  # type: ignore
        cv2.inRange(hsv, _RED_LO_2, _RED_HI_2),  # type: ignore
    )
    best_name = "red"
    best_count = int(np.count_nonzero(red_mask))
    for name, lo, hi in _COLOUR_RANGES:
        cnt = int(np.count_nonzero(cv2.inRange(hsv, lo, hi)))  # type: ignore
        if cnt > best_count:
            best_count = cnt
            best_name = name
    if best_count < 40:
        return "unknown"
    return best_name


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
    return _dominant_colour(roi)


def get_lower_colour(frame: np.ndarray | None, box: np.ndarray) -> str:
    """Sample the lower-body ROI (pants/shorts) for a complementary tag."""
    if frame is None:
        return "unknown"
    x1, y1, x2, y2 = map(int, box[:4])
    h = y2 - y1
    w = x2 - x1
    if h < 20 or w < 10:
        return "unknown"
    ly1 = max(0, y1 + int(h * 0.62))
    ly2 = max(0, y1 + int(h * 0.88))
    lx1 = max(0, x1 + int(w * 0.30))
    lx2 = max(0, x1 + int(w * 0.70))
    roi = frame[ly1:ly2, lx1:lx2]
    return _dominant_colour(roi)


# ===========================================================================
# YOLO Pose Tracker
# ===========================================================================
class ProPoseTracker:
    """YOLOv8-pose tracker with CUDA-aware FP16 inference."""

    def __init__(self) -> None:
        from ultralytics import YOLO  # noqa: WPS433

        self._use_half = False
        self._device = "cpu"
        try:
            import torch  # type: ignore  # noqa: WPS433

            if torch.cuda.is_available():
                self._device = "cuda"
                self._use_half = True
        except Exception:
            pass

        try:
            self.model = YOLO(YOLO_MODEL)
        except Exception as exc:
            print(f"[VIS] YOLO engine load failed ({exc}); falling back to {YOLO_FALLBACK_MODEL}")
            self.model = YOLO(YOLO_FALLBACK_MODEL)

        try:
            self.model.to(self._device)
        except Exception:
            pass

        self.last_inference_ms: float = 0.0
        print(f"[VIS] YOLOv8-pose on {self._device} (half={self._use_half})")

    def process_frame(
        self,
        frame: np.ndarray | None,
        want_shirts: bool = False,
    ) -> tuple[dict | None, np.ndarray | None]:
        if frame is None:
            return None, None

        fh, fw = frame.shape[:2]
        if (fw, fh) != (YOLO_INFER_W, YOLO_INFER_H):
            infer_frame = cv2.resize(frame, (YOLO_INFER_W, YOLO_INFER_H), interpolation=cv2.INTER_LINEAR)  # type: ignore
        else:
            infer_frame = frame
        sx, sy = fw / YOLO_INFER_W, fh / YOLO_INFER_H

        t0 = time.perf_counter()
        try:
            results = self.model.track(
                infer_frame,
                persist=True,
                verbose=False,
                conf=YOLO_CONF,
                iou=YOLO_IOU,
                half=self._use_half,
                device=self._device,
                imgsz=YOLO_IMGSZ,
                tracker="bytetrack.yaml",
            )
        except TypeError:
            results = self.model.track(
                infer_frame,
                persist=True,
                verbose=False,
                conf=YOLO_CONF,
                iou=YOLO_IOU,
                imgsz=YOLO_IMGSZ,
                tracker="bytetrack.yaml",
            )
        self.last_inference_ms = (time.perf_counter() - t0) * 1000.0

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return self._empty(frame), frame

        r = results[0]
        boxes_np = _as_numpy(r.boxes.xyxy)  # type: ignore
        boxes = boxes_np if boxes_np is not None else np.zeros((0, 4))
        if boxes.size:
            boxes = boxes.copy()
            boxes[:, 0] *= sx
            boxes[:, 2] *= sx
            boxes[:, 1] *= sy
            boxes[:, 3] *= sy

        ids_np = _as_numpy(r.boxes.id) if r.boxes.id is not None else None  # type: ignore
        track_ids = ids_np.astype(int).tolist() if ids_np is not None else [None] * len(boxes)

        kpts: list[np.ndarray] = []
        if r.keypoints is not None and r.keypoints.xy is not None:
            xy = _as_numpy(r.keypoints.xy)
            xy = xy if xy is not None else np.zeros((0, 17, 2))
            if xy.size:
                xy = xy.copy()
                xy[..., 0] *= sx
                xy[..., 1] *= sy
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

        if want_shirts:
            shirt_colours = [get_shirt_colour(frame, b) for b in boxes]
        else:
            shirt_colours = ["unknown"] * len(boxes)

        return (
            {
                "players_alive": len(boxes),
                "boxes": boxes,
                "track_ids": track_ids,
                "keypoints": kpts,
                "shirt_colours": shirt_colours,
                "frame_w": int(fw),
                "frame_h": int(fh),
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


# ===========================================================================
# Line detector — finds the start tape and the finish tape from colour
# ===========================================================================
class LineDetector:
    """Detect bright tape lines on the floor by HSV colour.

    Each ``detect_*`` returns a Y-row (in the camera frame's coordinate
    system) where the tape has the highest pixel mass, or the configured
    fallback Y if not enough tape is visible. Results are cached for a
    short window so we don't re-mask the whole frame every tick.
    """

    def __init__(self) -> None:
        self._cache: dict[str, tuple[float, int, int]] = {}
        self._cache_ttl_s = 1.5

    def _cached(self, key: str) -> tuple[int, int] | None:
        entry = self._cache.get(key)
        if not entry:
            return None
        ts, y, count = entry
        if time.time() - ts > self._cache_ttl_s:
            return None
        return y, count

    def _store(self, key: str, y: int, count: int) -> None:
        self._cache[key] = (time.time(), y, count)

    def detect_start(self, frame: np.ndarray | None) -> tuple[int, int]:
        """Return (y_px, tape_pixel_count) for the start line.

        Apr 2026 orientation flip: the start line is on the FAR side of the
        gym (small Y, top of frame). We search the TOP two-thirds of the
        frame and ignore the bottom third (which is where players' feet
        and the close finish tape live)."""
        if frame is None:
            return START_LINE_Y_PX, 0
        cached = self._cached("start")
        if cached:
            return cached
        if not START_LINE_DETECT_FROM_TAPE:
            self._store("start", START_LINE_Y_PX, 0)
            return START_LINE_Y_PX, 0
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # type: ignore
        mask = cv2.inRange(hsv, np.array(START_LINE_HSV_LOW), np.array(START_LINE_HSV_HIGH))  # type: ignore
        # Start line lives in the top 2/3 of the frame (far from camera).
        h = mask.shape[0]
        mask[2 * h // 3 :, :] = 0
        row_sums = np.sum(mask > 0, axis=1)
        peak = int(np.argmax(row_sums))
        peak_count = int(row_sums[peak])
        if peak_count < 60:  # not enough tape — fall back
            self._store("start", START_LINE_Y_PX, peak_count)
            return START_LINE_Y_PX, peak_count
        self._store("start", peak, peak_count)
        return peak, peak_count

    def detect_finish(self, frame: np.ndarray | None) -> tuple[int, int]:
        """Return (y_px, tape_pixel_count) for the finish line.

        Apr 2026 orientation flip: the finish line is CLOSE to the camera
        (large Y, bottom of frame). We search the BOTTOM two-thirds of the
        frame."""
        if frame is None:
            return FINISH_LINE_Y_PX, 0
        cached = self._cached("finish")
        if cached:
            return cached
        if not FINISH_LINE_DETECT_FROM_TAPE:
            self._store("finish", FINISH_LINE_Y_PX, 0)
            return FINISH_LINE_Y_PX, 0
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # type: ignore
        mask1 = cv2.inRange(hsv, np.array(FINISH_LINE_HSV_LOW_1), np.array(FINISH_LINE_HSV_HIGH_1))  # type: ignore
        mask2 = cv2.inRange(hsv, np.array(FINISH_LINE_HSV_LOW_2), np.array(FINISH_LINE_HSV_HIGH_2))  # type: ignore
        mask = cv2.bitwise_or(mask1, mask2)  # type: ignore
        h = mask.shape[0]
        # Finish line lives in the bottom 2/3 of the frame (close to camera).
        mask[: h // 3, :] = 0
        row_sums = np.sum(mask > 0, axis=1)
        peak = int(np.argmax(row_sums))
        peak_count = int(row_sums[peak])
        if peak_count < FINISH_LINE_MIN_TAPE_PX // 4:
            self._store("finish", FINISH_LINE_Y_PX, peak_count)
            return FINISH_LINE_Y_PX, peak_count
        self._store("finish", peak, peak_count)
        return peak, peak_count

    @staticmethod
    def player_foot_y(box: np.ndarray) -> float:
        """Y of the player's feet — bottom of the bounding box."""
        return float(box[3])

    @staticmethod
    def is_behind_start(foot_y: float, start_y: int) -> bool:
        """A player is behind the start line when their feet are at or
        ABOVE it in the image (smaller Y → further from camera).

        Apr 2026 orientation flip: with the camera at the finish, players
        begin at the FAR start tape (top of frame) and run toward the
        camera. A small tolerance lets a foot poke just past the line
        without disqualifying them from the "ready" check."""
        return foot_y <= (start_y + START_LINE_TOLERANCE_PX)

    @staticmethod
    def has_crossed_finish(foot_y: float, finish_y: int) -> bool:
        """A player has crossed the finish when their feet are at or
        BELOW it in the image (larger Y → closer to camera).

        Apr 2026 orientation flip: the finish tape is now near the bottom
        of the frame, just below the camera mount."""
        return foot_y >= (finish_y - FINISH_LINE_TOLERANCE_PX)


def check_tape_finish(frame: np.ndarray | None, pose_data: dict | None) -> bool:
    """Legacy helper — kept for any external callers. Uses the new
    LineDetector under the hood, but returns just a yes/no."""
    if not USE_TAPE_FINISH or frame is None:
        return False
    if pose_data is None or pose_data.get("boxes") is None:
        return False
    detector = LineDetector()
    finish_y, _ = detector.detect_finish(frame)
    for box in pose_data["boxes"]:
        if LineDetector.has_crossed_finish(LineDetector.player_foot_y(box), finish_y):
            return True
    return False


# ===========================================================================
# Photo capture — grab the highest-quality crop of each player during RED
# ===========================================================================
class PhotoCapture:
    """Crop a player from a frame for the leaderboard.

    Quality is approximated by the variance-of-Laplacian (a simple
    sharpness measure). We also reject crops that are too small.
    """

    @staticmethod
    def quality_score(crop: np.ndarray) -> float:
        if crop is None or crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)  # type: ignore
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())  # type: ignore

    @staticmethod
    def crop_player(frame: np.ndarray, box: np.ndarray, pad: float = 0.08) -> np.ndarray | None:
        if frame is None:
            return None
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (float(v) for v in box[:4])
        bw, bh = x2 - x1, y2 - y1
        x1 = max(0, int(x1 - bw * pad))
        y1 = max(0, int(y1 - bh * pad))
        x2 = min(w, int(x2 + bw * pad))
        y2 = min(h, int(y2 + bh * pad))
        if x2 - x1 < 30 or y2 - y1 < 60:
            return None
        return frame[y1:y2, x1:x2].copy()

    @staticmethod
    def make_thumbnail(crop: np.ndarray) -> np.ndarray:
        """Resize while preserving aspect ratio + letterbox onto a fixed canvas."""
        canvas = np.full((PROFILE_THUMB_H, PROFILE_THUMB_W, 3), 30, dtype=np.uint8)
        ch, cw = crop.shape[:2]
        scale = min(PROFILE_THUMB_W / cw, PROFILE_THUMB_H / ch)
        new_w = max(1, int(cw * scale))
        new_h = max(1, int(ch * scale))
        resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)  # type: ignore
        ox = (PROFILE_THUMB_W - new_w) // 2
        oy = (PROFILE_THUMB_H - new_h) // 2
        canvas[oy : oy + new_h, ox : ox + new_w] = resized
        return canvas


# ===========================================================================
# Player descriptor — VLM / CLIP / colour
# ===========================================================================
class PlayerDescriber:
    """Produce a short natural-language description of a player from a crop.

    The active backend is chosen by ``IDENTIFICATION_MODE`` in config.
    Heavier backends (VLM, CLIP) are loaded lazily on the first call so
    importing this module is cheap on machines without torch.

    All ``describe`` calls are thread-safe — the heavy backends are
    serialised by ``self._lock`` so a Jetson Orin Nano isn't asked to run
    two CLIP forward passes simultaneously.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._mode = IDENTIFICATION_MODE
        self._fallback_only = False
        self._vlm = None
        self._vlm_tokenizer = None
        self._clip_model = None
        self._clip_processor = None
        print(f"[VIS] Identification mode: {self._mode}")

    @property
    def mode(self) -> str:
        return self._mode

    def describe(self, crop: np.ndarray | None, frame_box: np.ndarray | None = None) -> str:
        if crop is None or crop.size == 0:
            return "Unknown player"
        try:
            if self._mode == "vlm" and not self._fallback_only:
                return self._describe_vlm(crop) or self._describe_color(crop)
            if self._mode == "clip" and not self._fallback_only:
                return self._describe_clip(crop) or self._describe_color(crop)
            return self._describe_color(crop)
        except Exception as exc:
            print(f"[VIS] Describer error ({self._mode}): {exc} — using colour fallback")
            self._fallback_only = True
            return self._describe_color(crop)

    # -------------------- Colour signature (always available) -------------
    def _describe_color(self, crop: np.ndarray) -> str:
        h, w = crop.shape[:2]
        torso = crop[int(h * 0.20) : int(h * 0.55), int(w * 0.20) : int(w * 0.80)]
        legs = crop[int(h * 0.60) : int(h * 0.92), int(w * 0.25) : int(w * 0.75)]
        shirt = _dominant_colour(torso)
        bottom = _dominant_colour(legs)
        if shirt == "unknown" and bottom == "unknown":
            return "Player"
        if shirt == "unknown":
            return f"Player in {bottom} pants"
        if bottom == "unknown":
            return f"Player in {shirt} top"
        return f"{shirt.capitalize()} top, {bottom} bottoms"

    # -------------------- CLIP zero-shot ----------------------------------
    def _ensure_clip(self) -> None:
        if self._clip_model is not None:
            return
        from transformers import CLIPModel, CLIPProcessor  # type: ignore  # noqa: WPS433
        import torch  # type: ignore  # noqa: WPS433

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._clip_device = device
        self._clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
        model.to(device).eval()
        if device == "cuda":
            try:
                model = model.half()
                self._clip_half = True
            except Exception:
                self._clip_half = False
        else:
            self._clip_half = False
        self._clip_model = model
        self._clip_torch = torch
        print(f"[VIS] CLIP loaded on {device}")

    def _clip_pick(self, image: Any, prompts: list[str], prefix: str) -> str:
        """Run CLIP and pick the best-scoring prompt."""
        torch = self._clip_torch
        proc = self._clip_processor  # type: ignore
        model = self._clip_model  # type: ignore
        full_prompts = [f"a photo of a person wearing {prefix}{p}" for p in prompts]
        inputs = proc(text=full_prompts, images=image, return_tensors="pt", padding=True)  # type: ignore
        inputs = {k: v.to(self._clip_device) for k, v in inputs.items()}
        if self._clip_half:
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].half()
        with torch.no_grad():
            out = model(**inputs)  # type: ignore
            logits = out.logits_per_image.softmax(dim=-1).cpu().numpy().flatten()
        best = int(np.argmax(logits))
        return prompts[best]

    def _clip_pick_feature(self, image: Any) -> str:
        torch = self._clip_torch
        proc = self._clip_processor  # type: ignore
        model = self._clip_model  # type: ignore
        full_prompts = [f"a photo of a person {p}" for p in CLIP_FEATURES]
        inputs = proc(text=full_prompts, images=image, return_tensors="pt", padding=True)  # type: ignore
        inputs = {k: v.to(self._clip_device) for k, v in inputs.items()}
        if self._clip_half and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].half()
        with torch.no_grad():
            out = model(**inputs)  # type: ignore
            logits = out.logits_per_image.softmax(dim=-1).cpu().numpy().flatten()
        best = int(np.argmax(logits))
        return CLIP_FEATURES[best]

    def _describe_clip(self, crop: np.ndarray) -> str:
        with self._lock:
            self._ensure_clip()
            from PIL import Image  # type: ignore  # noqa: WPS433

            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)  # type: ignore
            pil = Image.fromarray(rgb)

            top_colour = self._clip_pick(pil, CLIP_COLORS, "a ")
            top_garment = self._clip_pick(pil, CLIP_TOP_GARMENTS, "")
            bottom = self._clip_pick(pil, CLIP_BOTTOM_GARMENTS, "")
            feature = self._clip_pick_feature(pil)

            base = f"{top_colour.capitalize()} {top_garment.replace('a ', '').strip()}, {bottom}"
            if feature != "no distinctive features":
                return f"{base}, {feature}"
            return base

    # -------------------- VLM (Moondream2) --------------------------------
    def _ensure_vlm(self) -> None:
        if self._vlm is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore  # noqa: WPS433
        import torch  # type: ignore  # noqa: WPS433

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if (VLM_USE_CUDA_FP16 and device == "cuda") else torch.float32
        tok = AutoTokenizer.from_pretrained(VLM_MODEL_ID, revision=VLM_REVISION)
        model = (
            AutoModelForCausalLM.from_pretrained(
                VLM_MODEL_ID,
                revision=VLM_REVISION,
                trust_remote_code=True,
                torch_dtype=dtype,
            )
            .to(device)
            .eval()
        )
        self._vlm = model
        self._vlm_tokenizer = tok
        self._vlm_device = device
        self._vlm_torch = torch
        print(f"[VIS] VLM loaded ({VLM_MODEL_ID}) on {device}")

    def _describe_vlm(self, crop: np.ndarray) -> str:
        with self._lock:
            self._ensure_vlm()
            from PIL import Image  # type: ignore  # noqa: WPS433

            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)  # type: ignore
            pil = Image.fromarray(rgb)
            try:
                # Moondream2 has its own answer_question() helper.
                enc = self._vlm.encode_image(pil)  # type: ignore
                answer = self._vlm.answer_question(  # type: ignore
                    enc,
                    VLM_PROMPT,
                    self._vlm_tokenizer,
                    max_new_tokens=VLM_MAX_NEW_TOKENS,
                )
                answer = answer.strip().rstrip(".")
                # Trim runaway answers
                if len(answer) > 90:
                    answer = answer[:87] + "…"
                return answer or "Player"
            except Exception as exc:
                raise RuntimeError(f"VLM inference failed: {exc}") from exc


# ===========================================================================
# PoseWorker — threaded inference loop
# ===========================================================================
class PoseWorker:
    def __init__(self, camera, tracker: ProPoseTracker) -> None:
        self._cam = camera
        self._trk = tracker
        self._pose: dict | None = None
        self._frame: np.ndarray | None = None
        self._frame_id: int = 0
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="pose-worker")
        self._thread.start()
        print("[POSE] Worker thread started")

    def _loop(self) -> None:
        while self._running:
            f = self._cam.read()
            if f is None:
                time.sleep(0.005)
                continue
            try:
                pose, _ = self._trk.process_frame(f, want_shirts=False)
            except Exception as exc:
                print(f"[POSE] inference error: {exc}")
                pose = None
            with self._lock:
                self._pose = pose
                self._frame = f
                self._frame_id += 1

    def latest(self) -> tuple[dict | None, int]:
        with self._lock:
            return self._pose, self._frame_id

    def latest_frame_and_pose(self) -> tuple[np.ndarray | None, dict | None, int]:
        """Get the frame that was actually used for the most recent inference,
        along with the pose results. Used by the photo-capture pipeline so
        that bbox coordinates line up with pixel data."""
        with self._lock:
            f = None if self._frame is None else self._frame.copy()
            return f, self._pose, self._frame_id

    def stop(self) -> None:
        self._running = False
        try:
            self._thread.join(timeout=1.5)
        except Exception:
            pass