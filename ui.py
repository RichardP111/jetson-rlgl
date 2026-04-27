#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Project:      Red Light Green Light (Jetson Orin Nano)
File:         ui.py
Description:  Material Design 3 "Expressive" rendering engine for Pygame.

              Merged Master Edition:
                * Smooth animated pill chips, banners, and palm glyphs (OLD)
                * Graphical Dev Dashboard with FPS chart (OLD) + New Metrics
                * Kahoot-style Leaderboard & Confetti (NEW)
                * Amber "Sent Back" elimination terminology (NEW)
                * Wait at Start Line & Caught Return screens (NEW)
                * Hidden Phase Progress Bar for surprise timing (NEW)

Author:       Richard Pu
Last Updated: April 2026
===============================================================================
"""

from __future__ import annotations

import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

import cv2
import numpy as np
import pygame

from config import (
    BANNER_PULSE_HZ,
    DEV_FPS_HISTORY,
    DISPLAY_H,
    DISPLAY_W,
    ENABLE_LINE_OVERLAY,
    FINISH_LINE_DISPLAY_COLOR,
    START_LINE_DISPLAY_COLOR,
    FONT_PATH,
    HOME_CAM_BOX_FRACTION,
    MD3_BG,
    MD3_ERROR,
    MD3_ERROR_BG,
    MD3_ON_BG,
    MD3_ON_BG_DIM,
    MD3_ON_BG_MED,
    MD3_ON_PRIMARY,
    MD3_OUTLINE,
    MD3_PRIMARY,
    MD3_PRIMARY_CONTAINER,
    MD3_SECONDARY,
    MD3_SUCCESS,
    MD3_SURFACE,
    MD3_SURFACE_HIGH,
    MD3_SURFACE_VAR,
    MD3_TERTIARY,
    MD3_WARNING,
    MOTION_FAST,
    MOTION_MED,
    MOTION_SLOW,
    STATE_COLORS,
)
from vision import KP, SKELETON_EDGES

# New Leaderboard Colors
MD3_GOLD = (255, 215, 0)
MD3_SILVER = (192, 192, 192)
MD3_BRONZE = (205, 127, 50)
MD3_SUCCESS_BG_DARK = (28, 56, 32)
SENT_BACK_LOG_MAX = 6


# ===========================================================================
# Easing + interpolation
# ===========================================================================
def clamp(x: float, a: float = 0.0, b: float = 1.0) -> float:
    return max(a, min(b, x))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def ease_out_cubic(t: float) -> float:
    t = clamp(t)
    return 1.0 - (1.0 - t) ** 3


def ease_in_out_cubic(t: float) -> float:
    t = clamp(t)
    if t < 0.5:
        return 4.0 * t * t * t
    return 1.0 - ((-2.0 * t + 2.0) ** 3) / 2.0


def ease_out_back(t: float, overshoot: float = 1.70158) -> float:
    t = clamp(t)
    c1 = overshoot
    c3 = c1 + 1.0
    return 1.0 + c3 * (t - 1.0) ** 3 + c1 * (t - 1.0) ** 2


def pulse(seconds: float, hz: float = 1.0) -> float:
    return 0.5 + 0.5 * math.sin(seconds * 2.0 * math.pi * hz)


# ===========================================================================
# Color helpers
# ===========================================================================
Color = tuple[int, int, int]
ColorA = tuple[int, int, int, int]


def with_alpha(color: Color, alpha: int) -> ColorA:
    return (color[0], color[1], color[2], int(clamp(alpha, 0, 255)))


def lerp_color(a: Color, b: Color, t: float) -> Color:
    t = clamp(t)
    return (
        int(lerp(a[0], b[0], t)),
        int(lerp(a[1], b[1], t)),
        int(lerp(a[2], b[2], t)),
    )


SHIRT_DISPLAY_COLOR: dict[str, Color] = {
    "red": (244, 108, 108),
    "orange": (255, 163, 86),
    "yellow": (255, 224, 130),
    "green": (129, 230, 150),
    "cyan": (120, 220, 232),
    "blue": (130, 180, 255),
    "purple": (188, 162, 255),
    "pink": (255, 180, 210),
    "white": (245, 245, 250),
    "black": (60, 58, 70),
    "grey": (170, 168, 180),
    "unknown": (128, 122, 142),
}


# ===========================================================================
# Motion value
# ===========================================================================
class MotionValue:
    __slots__ = ("current", "target", "speed")

    def __init__(self, initial: float, speed: float = MOTION_MED) -> None:
        self.current = float(initial)
        self.target = float(initial)
        self.speed = float(speed)

    def set(self, target: float) -> None:
        self.target = float(target)

    def snap(self, value: float) -> None:
        self.current = float(value)
        self.target = float(value)

    def update(self, dt: float) -> None:
        k = 1.0 - math.exp(-self.speed * dt)
        self.current += (self.target - self.current) * k

    def __float__(self) -> float:
        return self.current


# ===========================================================================
# Font cache
# ===========================================================================
class FontCache:
    def __init__(self, font_path: str) -> None:
        self._path = font_path if os.path.exists(font_path) else None
        self._cache: dict[tuple[int, bool], pygame.font.Font] = {}

    def get(self, size: int, bold: bool = False) -> pygame.font.Font:
        key = (size, bold)
        if key not in self._cache:
            if self._path:
                font = pygame.font.Font(self._path, size)
            else:
                font = pygame.font.SysFont("arial", size, bold=bold)
            font.set_bold(bold)
            self._cache[key] = font
        return self._cache[key]

    def render(self, text: str, size: int, color: Color, bold: bool = False, antialias: bool = True) -> pygame.Surface:
        return self.get(size, bold).render(text, antialias, color)


# ===========================================================================
# Drawing primitives
# ===========================================================================
_SHADOW_CACHE: "dict[tuple[int, int, int, int, int], pygame.Surface]" = {}
_SHADOW_CACHE_LIMIT = 64


def _build_shadow_surface(w: int, h: int, radius: int, spread: int, alpha: int) -> pygame.Surface:
    sw = w + spread * 2
    sh = h + spread * 2
    surf = pygame.Surface((sw, sh), pygame.SRCALPHA)
    pygame.draw.rect(surf, (0, 0, 0, alpha), pygame.Rect(spread, spread, w, h), border_radius=radius)
    try:
        alpha_view = pygame.surfarray.pixels_alpha(surf)
        sigma = max(1.0, spread / 2.0)
        cv2.GaussianBlur(alpha_view, (0, 0), sigmaX=sigma, sigmaY=sigma, dst=alpha_view)  # type: ignore
        del alpha_view
    except Exception:
        pass
    return surf


def draw_shadow_rrect(
    target: pygame.Surface, rect: pygame.Rect, radius: int, offset: tuple[int, int] = (0, 8), spread: int = 14, alpha: int = 130
) -> None:
    key = (rect.w, rect.h, radius, spread, alpha)
    cached = _SHADOW_CACHE.get(key)
    if cached is None:
        cached = _build_shadow_surface(rect.w, rect.h, radius, spread, alpha)
        if len(_SHADOW_CACHE) >= _SHADOW_CACHE_LIMIT:
            _SHADOW_CACHE.pop(next(iter(_SHADOW_CACHE)))
        _SHADOW_CACHE[key] = cached
    target.blit(cached, (rect.x - spread + offset[0], rect.y - spread + offset[1]))


def draw_rrect(
    target: pygame.Surface, rect: pygame.Rect, color: Color, radius: int, alpha: int = 255, outline: Color | None = None, outline_w: int = 2
) -> None:
    surf = pygame.Surface(rect.size, pygame.SRCALPHA)
    pygame.draw.rect(surf, (color[0], color[1], color[2], alpha), surf.get_rect(), border_radius=radius)
    if outline is not None:
        pygame.draw.rect(surf, (outline[0], outline[1], outline[2], alpha), surf.get_rect(), width=outline_w, border_radius=radius)
    target.blit(surf, rect.topleft)


def draw_panel(
    target: pygame.Surface, rect: pygame.Rect, color: Color = MD3_SURFACE_HIGH, radius: int = 24, alpha: int = 235, shadow: bool = True
) -> None:
    if shadow:
        draw_shadow_rrect(target, rect, radius)
    draw_rrect(target, rect, color, radius, alpha=alpha)


def draw_pill(target: pygame.Surface, rect: pygame.Rect, color: Color, alpha: int = 255) -> None:
    draw_rrect(target, rect, color, radius=rect.height // 2, alpha=alpha)


def draw_arc_thick(
    target: pygame.Surface,
    center: tuple[int, int],
    radius: int,
    thickness: int,
    start_rad: float,
    end_rad: float,
    color: Color,
    alpha: int = 255,
    segments: int = 96,
) -> None:
    if end_rad <= start_rad:
        return
    cx, cy = center
    inner = max(1, radius - thickness)
    sweep = end_rad - start_rad
    n = max(6, int(segments * sweep / (2.0 * math.pi)))
    points: list[tuple[float, float]] = []
    for i in range(n + 1):
        t = start_rad + sweep * i / n
        points.append((cx + radius * math.cos(t), cy + radius * math.sin(t)))
    for i in range(n, -1, -1):
        t = start_rad + sweep * i / n
        points.append((cx + inner * math.cos(t), cy + inner * math.sin(t)))

    if len(points) >= 3:
        surf_rect = pygame.Rect(int(cx - radius - 2), int(cy - radius - 2), radius * 2 + 4, radius * 2 + 4)
        surf = pygame.Surface(surf_rect.size, pygame.SRCALPHA)
        local = [(p[0] - surf_rect.x, p[1] - surf_rect.y) for p in points]
        pygame.draw.polygon(surf, (color[0], color[1], color[2], alpha), local)
        target.blit(surf, surf_rect.topleft)


def draw_progress_ring(
    target: pygame.Surface, center: tuple[int, int], radius: int, thickness: int, progress: float, fg: Color, bg: Color = MD3_SURFACE_VAR
) -> None:
    draw_arc_thick(target, center, radius, thickness, 0.0, 2.0 * math.pi, bg, alpha=200)
    progress = clamp(progress)
    if progress <= 0.0:
        return
    start = -math.pi / 2.0
    end = start + 2.0 * math.pi * progress
    draw_arc_thick(target, center, radius, thickness, start, end, fg)


def draw_text_centered(
    target: pygame.Surface, text: str, center: tuple[int, int], font_cache: FontCache, size: int, color: Color, bold: bool = False
) -> pygame.Rect:
    surf = font_cache.render(text, size, color, bold=bold)
    r = surf.get_rect(center=center)
    target.blit(surf, r)
    return r


def draw_text_left(
    target: pygame.Surface, text: str, pos: tuple[int, int], font_cache: FontCache, size: int, color: Color, bold: bool = False
) -> pygame.Rect:
    surf = font_cache.render(text, size, color, bold=bold)
    r = surf.get_rect(topleft=pos)
    target.blit(surf, r)
    return r


# ===========================================================================
# Transient UI state
# ===========================================================================
@dataclass
class ChipState:
    x: MotionValue = field(default_factory=lambda: MotionValue(0.0, MOTION_FAST))
    y: MotionValue = field(default_factory=lambda: MotionValue(0.0, MOTION_FAST))
    alpha: MotionValue = field(default_factory=lambda: MotionValue(0.0, MOTION_MED))
    last_seen: float = 0.0
    label: str = ""
    shirt: str = "unknown"
    finished: bool = False


@dataclass
class LogEntry:
    text: str
    colour: Color
    ts: float
    alpha: MotionValue = field(default_factory=lambda: MotionValue(0.0, MOTION_MED))
    slide: MotionValue = field(default_factory=lambda: MotionValue(24.0, MOTION_MED))


@dataclass
class Flash:
    x: int
    y: int
    ts: float = field(default_factory=time.time)


# ===========================================================================
# Main UI renderer
# ===========================================================================
class UIRenderer:
    def __init__(self, screen: pygame.Surface) -> None:
        self._screen = screen
        self._font = FontCache(FONT_PATH)

        self._banner_color: Color = MD3_PRIMARY
        self._banner_target: Color = MD3_PRIMARY
        self._banner_blend = MotionValue(1.0, MOTION_SLOW)
        self._prev_banner_label = ""
        self._banner_label = ""
        self._banner_entry_ts = time.time()

        self._chips: dict[int, ChipState] = {}
        self._log: deque[LogEntry] = deque(maxlen=SENT_BACK_LOG_MAX)
        self._flashes: list[Flash] = []

        self._decor_surfaces: list[pygame.Surface] = []
        self._decor_radii: list[int] = []
        for base in (MD3_PRIMARY_CONTAINER, MD3_SURFACE_VAR, MD3_PRIMARY):
            r = 480
            s = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*base, 40), (r, r), r)
            self._decor_surfaces.append(s)
            self._decor_radii.append(r)

        self._countdown_last_n = -1
        self._countdown_trigger_ts = 0.0
        self._confetti: list[tuple[float, float, float, float, Color]] = []

        self._dev_mode = False
        self._dev_fps_history: deque[float] = deque(maxlen=DEV_FPS_HISTORY)

        self._cam_full_scaled: pygame.Surface | None = None
        self._cam_full_size: tuple[int, int] = (0, 0)
        self._cam_box_scaled: pygame.Surface | None = None
        self._cam_box_size: tuple[int, int] = (0, 0)

        self._last_tick = time.time()
        self._dt = 0.0

        # Calibration overlay state
        self._line_y_start = 0
        self._line_y_finish = 0
        self._line_count_start = 0
        self._line_count_finish = 0
        self._line_frame_h = 0

    # ------------------------------------------------------------------
    # Frame lifecycle
    # ------------------------------------------------------------------
    def begin_frame(self) -> None:
        now = time.time()
        self._dt = min(0.05, max(0.0, now - self._last_tick))
        self._last_tick = now
        self._screen.fill(MD3_BG)

    def present(self) -> None:
        pygame.display.flip()

    def toggle_dev_mode(self) -> bool:
        self._dev_mode = not self._dev_mode
        print(f"[UI ] Dev mode: {'ON' if self._dev_mode else 'OFF'}")
        return self._dev_mode

    def is_dev_mode(self) -> bool:
        return self._dev_mode

    def set_line_calibration(self, frame_h: int, start_y: int, start_count: int, finish_y: int, finish_count: int) -> None:
        self._line_frame_h = frame_h
        self._line_y_start = start_y
        self._line_count_start = start_count
        self._line_y_finish = finish_y
        self._line_count_finish = finish_count

    # ------------------------------------------------------------------
    # Camera layer
    # ------------------------------------------------------------------
    def draw_camera(self, frame: np.ndarray | None) -> None:
        if frame is None:
            self._screen.fill(MD3_BG)
            self._draw_decorative_bg()
            return
        try:
            h, w = frame.shape[:2]
            if (w, h) == (DISPLAY_W, DISPLAY_H):
                pygame.surfarray.blit_array(self._screen, frame.swapaxes(0, 1)[:, :, ::-1])
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # type: ignore
                surf = pygame.image.frombuffer(rgb.tobytes(), (w, h), "RGB")
                if self._cam_full_size != (DISPLAY_W, DISPLAY_H):
                    self._cam_full_scaled = pygame.Surface((DISPLAY_W, DISPLAY_H))
                    self._cam_full_size = (DISPLAY_W, DISPLAY_H)
                pygame.transform.scale(surf, (DISPLAY_W, DISPLAY_H), self._cam_full_scaled)
                assert self._cam_full_scaled is not None
                self._screen.blit(self._cam_full_scaled, (0, 0))
        except Exception as exc:
            print(f"[UI ] draw_camera: {exc}")
            self._screen.fill(MD3_BG)

        if ENABLE_LINE_OVERLAY:
            self._draw_line_overlay()

    def draw_camera_in_rect(self, frame: np.ndarray | None, dest: pygame.Rect, radius: int = 36) -> None:
        draw_shadow_rrect(self._screen, dest, radius, offset=(0, 14), spread=22, alpha=110)

        if frame is None:
            draw_rrect(self._screen, dest, MD3_SURFACE_HIGH, radius, alpha=235)
            placeholder = self._font.render("Camera warming up…", 28, MD3_ON_BG_DIM, bold=False)
            self._screen.blit(placeholder, placeholder.get_rect(center=dest.center))
            return

        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # type: ignore
            fh, fw = rgb.shape[:2]
            surf = pygame.image.frombuffer(rgb.tobytes(), (fw, fh), "RGB").convert()
            target_size = (dest.w, dest.h)

            self._cam_box_scaled = pygame.transform.scale(surf, target_size)
            mask = pygame.Surface(target_size, pygame.SRCALPHA).convert_alpha()
            pygame.draw.rect(mask, (255, 255, 255, 255), mask.get_rect(), border_radius=radius)
            clipped = pygame.Surface(target_size, pygame.SRCALPHA).convert_alpha()
            clipped.fill((0, 0, 0, 0))

            assert self._cam_box_scaled is not None
            clipped.blit(self._cam_box_scaled, (0, 0))
            clipped.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
            self._screen.blit(clipped, dest.topleft)

            outline_surf = pygame.Surface(target_size, pygame.SRCALPHA).convert_alpha()
            pygame.draw.rect(outline_surf, (*MD3_OUTLINE, 200), outline_surf.get_rect(), width=2, border_radius=radius)
            self._screen.blit(outline_surf, dest.topleft)

        except Exception as exc:
            print(f"[UI ] draw_camera_in_rect: {exc}")
            draw_rrect(self._screen, dest, MD3_SURFACE_HIGH, radius, alpha=235)

    def _draw_line_overlay(self) -> None:
        if self._line_frame_h <= 0:
            return
        scale = DISPLAY_H / float(self._line_frame_h)
        if self._line_y_start > 0:
            y = int(self._line_y_start * scale)
            self._draw_line_band(y, START_LINE_DISPLAY_COLOR, "START", self._line_count_start)
        if self._line_y_finish > 0:
            y = int(self._line_y_finish * scale)
            self._draw_line_band(y, FINISH_LINE_DISPLAY_COLOR, "FINISH", self._line_count_finish)

    def _draw_line_band(self, y: int, color: Color, label: str, pixel_count: int) -> None:
        band = pygame.Surface((DISPLAY_W, 6), pygame.SRCALPHA)
        band.fill((*color, 180))
        self._screen.blit(band, (0, y - 3))
        tag_text = f"{label} ({pixel_count}px)" if pixel_count else f"{label} (fallback)"
        surf = self._font.render(tag_text, 18, color, bold=True)
        bg_rect = pygame.Rect(20, y - 22, surf.get_width() + 16, 26)
        draw_pill(self._screen, bg_rect, MD3_SURFACE_HIGH, alpha=210)
        self._screen.blit(surf, (bg_rect.x + 8, bg_rect.y + 4))

    def _draw_decorative_bg(self) -> None:
        t = time.time()
        for i, surf in enumerate(self._decor_surfaces):
            phase = t * 0.2 + i * 2.1
            cx = int(DISPLAY_W * (0.3 + 0.4 * math.sin(phase)))
            cy = int(DISPLAY_H * (0.3 + 0.4 * math.cos(phase * 0.8)))
            r = self._decor_radii[i]
            self._screen.blit(surf, (cx - r, cy - r))

    # ------------------------------------------------------------------
    # Pose overlay
    # ------------------------------------------------------------------
    def draw_pose(self, pose: dict | None, finished_ids: set[int] | None = None, labels_by_id: dict[int, str] | None = None) -> None:
        if pose is None:
            return
        finished_ids = finished_ids or set()
        labels_by_id = labels_by_id or {}
        self._draw_skeletons(pose, finished_ids)
        self._update_chips(pose, finished_ids, labels_by_id)
        self._draw_chips()

    def _draw_skeletons(self, pose: dict, finished_ids: set[int]) -> None:
        kpts = pose.get("keypoints", [])
        ids = pose.get("track_ids", [])
        fw = pose.get("frame_w", DISPLAY_W) or DISPLAY_W
        fh = pose.get("frame_h", DISPLAY_H) or DISPLAY_H
        sx = DISPLAY_W / fw
        sy = DISPLAY_H / fh

        for idx, kp in enumerate(kpts):
            tid = ids[idx] if idx < len(ids) else None
            faded = tid in finished_ids
            line_color = (160, 160, 170) if faded else (240, 240, 250)

            for a, b in SKELETON_EDGES:
                if kp[a, 2] < 0.3 or kp[b, 2] < 0.3:
                    continue
                p1 = (int(kp[a, 0] * sx), int(kp[a, 1] * sy))
                p2 = (int(kp[b, 0] * sx), int(kp[b, 1] * sy))
                pygame.draw.line(self._screen, line_color, p1, p2, 4 if not faded else 2)

            for j in range(17):
                if kp[j, 2] < 0.3:
                    continue
                p = (int(kp[j, 0] * sx), int(kp[j, 1] * sy))
                r = 7 if j in (5, 6, 11, 12) else 5
                if faded:
                    r -= 2
                pygame.draw.circle(self._screen, line_color, p, r)

    def _update_chips(self, pose: dict, finished_ids: set[int], labels_by_id: dict[int, str]) -> None:
        now = time.time()
        kpts = pose.get("keypoints", [])
        boxes = pose.get("boxes", [])
        ids = pose.get("track_ids", [])
        shirts = pose.get("shirt_colours", [])

        if boxes is None or len(boxes) == 0:
            for chip in self._chips.values():
                chip.alpha.set(0.0)
                chip.alpha.update(self._dt)
            return

        fw = pose.get("frame_w", DISPLAY_W) or DISPLAY_W
        fh = pose.get("frame_h", DISPLAY_H) or DISPLAY_H
        sx = DISPLAY_W / fw
        sy = DISPLAY_H / fh
        seen: set[int] = set()

        for i, tid in enumerate(ids):
            if tid is None:
                continue
            seen.add(tid)
            kp = kpts[i] if i < len(kpts) else None

            if kp is not None and len(kp) >= 17 and kp[KP["nose"]][2] > 0.3:
                nx = kp[KP["nose"]][0] * sx
                ny = kp[KP["nose"]][1] * sy - 80
            elif i < len(boxes):
                b = boxes[i]
                nx = (b[0] + b[2]) / 2.0 * sx
                ny = b[1] * sy - 40
            else:
                continue

            nx = float(clamp(nx, 80, DISPLAY_W - 80))
            ny = float(clamp(ny, 40, DISPLAY_H - 40))

            chip = self._chips.get(tid)
            if chip is None:
                chip = ChipState()
                chip.x.snap(nx)
                chip.y.snap(ny)
                self._chips[tid] = chip
            chip.x.set(nx)
            chip.y.set(ny)
            chip.alpha.set(255.0)
            chip.last_seen = now
            chip.label = labels_by_id.get(tid, f"Player {tid}")
            chip.shirt = shirts[i] if i < len(shirts) else "unknown"
            chip.finished = tid in finished_ids

        # Fade out stale chips
        stale = []
        for tid, chip in self._chips.items():
            if tid not in seen:
                chip.alpha.set(0.0)
            if now - chip.last_seen > 2.0 and chip.alpha.current < 4.0:
                stale.append(tid)
            chip.x.update(self._dt)
            chip.y.update(self._dt)
            chip.alpha.update(self._dt)
        for tid in stale:
            del self._chips[tid]

    def _draw_chips(self) -> None:
        for chip in self._chips.values():
            alpha = int(chip.alpha.current)
            if alpha < 8:
                continue

            col = SHIRT_DISPLAY_COLOR.get(chip.shirt, MD3_PRIMARY) if not chip.finished else MD3_SUCCESS
            text = self._font.render(chip.label, 22, MD3_ON_PRIMARY if not chip.finished else MD3_SUCCESS_BG_DARK, bold=True)

            pad_x = 18
            pad_y = 8
            dot_r = 9
            w = text.get_width() + pad_x * 2 + dot_r * 2 + 8
            h = text.get_height() + pad_y * 2
            rect = pygame.Rect(int(chip.x.current - w / 2), int(chip.y.current - h / 2), w, h)

            draw_shadow_rrect(self._screen, rect, h // 2, offset=(0, 6), spread=10, alpha=int(alpha * 0.4))

            surf = pygame.Surface(rect.size, pygame.SRCALPHA)
            pygame.draw.rect(surf, (col[0], col[1], col[2], alpha), surf.get_rect(), border_radius=h // 2)

            dot_center = (pad_x + dot_r, h // 2)
            pygame.draw.circle(surf, (255, 255, 255, alpha), dot_center, dot_r)
            pygame.draw.circle(surf, (col[0], col[1], col[2], alpha), dot_center, dot_r - 3)

            self._screen.blit(surf, rect.topleft)
            text.set_alpha(alpha)
            self._screen.blit(text, (rect.x + pad_x + dot_r * 2 + 8, rect.y + (h - text.get_height()) // 2))

    # ------------------------------------------------------------------
    # Sent-back log (Amber warnings)
    # ------------------------------------------------------------------
    def log_sent_back(self, descriptor: str) -> None:
        pretty = descriptor if descriptor != "unknown" else "?"
        entry = LogEntry(
            text=f"{pretty.upper()} — back to start",
            colour=MD3_WARNING,
            ts=time.time(),
        )
        entry.alpha.set(255.0)
        entry.slide.set(0.0)
        self._log.appendleft(entry)

    def log_elimination(self, descriptor: str) -> None:
        self.log_sent_back(descriptor)

    def _draw_sent_back_log(self) -> None:
        if not self._log:
            return
        x = 36
        y = DISPLAY_H - 40
        for entry in self._log:
            entry.alpha.update(self._dt)
            entry.slide.update(self._dt)
            alpha = int(clamp(entry.alpha.current, 0, 255))
            if alpha < 6:
                continue
            label_surf = self._font.render(entry.text, 22, MD3_ON_BG, bold=True)
            pad_x, pad_y, dot_r = 14, 8, 7
            w = label_surf.get_width() + pad_x * 2 + dot_r * 2 + 10
            h = label_surf.get_height() + pad_y * 2
            rect = pygame.Rect(x, int(y - h + entry.slide.current), w, h)

            surf = pygame.Surface(rect.size, pygame.SRCALPHA)
            pygame.draw.rect(surf, (*MD3_SURFACE_HIGH, int(alpha * 0.9)), surf.get_rect(), border_radius=h // 2)
            pygame.draw.circle(surf, (*entry.colour, alpha), (pad_x + dot_r, h // 2), dot_r)

            self._screen.blit(surf, rect.topleft)
            label_surf.set_alpha(alpha)
            self._screen.blit(label_surf, (rect.x + pad_x + dot_r * 2 + 10, rect.y + pad_y))
            y = rect.y - 8

    # ------------------------------------------------------------------
    # Banner
    # ------------------------------------------------------------------
    def _update_banner(self, label: str) -> None:
        if label != self._banner_label:
            self._prev_banner_label = self._banner_label
            self._banner_label = label
            self._banner_target = STATE_COLORS.get(label, MD3_PRIMARY)
            self._banner_blend.snap(0.0)
            self._banner_blend.set(1.0)
            self._banner_entry_ts = time.time()
        self._banner_blend.update(self._dt)
        self._banner_color = lerp_color(self._banner_color, self._banner_target, self._banner_blend.current)

    def _draw_banner(self, label: str) -> None:
        self._update_banner(label)
        elapsed = time.time() - self._banner_entry_ts
        scale = 0.92 + 0.08 * ease_out_back(clamp(elapsed * 3.5))
        pulse_amt = pulse(time.time(), BANNER_PULSE_HZ) * 0.08
        cx = DISPLAY_W // 2
        base_w, base_h = 560, 128
        w = int(base_w * scale * (1.0 + pulse_amt))
        h = int(base_h * scale)
        rect = pygame.Rect(cx - w // 2, 48, w, h)

        draw_shadow_rrect(self._screen, rect, h // 2, offset=(0, 14), spread=22, alpha=150)

        surf = pygame.Surface(rect.size, pygame.SRCALPHA)
        col = self._banner_color
        pygame.draw.rect(surf, (col[0], col[1], col[2], 250), surf.get_rect(), border_radius=h // 2)

        sheen = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(sheen, (255, 255, 255, 50), pygame.Rect(0, 0, rect.w, rect.h // 2), border_radius=h // 2)
        surf.blit(sheen, (0, 0))
        self._screen.blit(surf, rect.topleft)

        txt_col = self._readable_on(col)
        display_text = self._banner_display_text(label)
        label_surf = self._font.render(display_text, 56, txt_col, bold=True)
        self._screen.blit(label_surf, label_surf.get_rect(center=rect.center))

    @staticmethod
    def _banner_display_text(label: str) -> str:
        return {
            "GREEN": "GREEN LIGHT",
            "RED": "RED LIGHT",
            "TURNING": "TURNING...",
            "COUNTDOWN": "GET READY",
            "WINNER": "FINISHED!",
            "CAUGHT": "CAUGHT!",
            "START": "RED LIGHT, GREEN LIGHT",
            "START_LINE": "GET TO THE START LINE",
            "RETURN": "WAITING — PLAYERS WALKING BACK",
            "LEADERBOARD": "RESULTS",
        }.get(label, label)

    @staticmethod
    def _readable_on(color: Color) -> Color:
        r, g, b = (c / 255.0 for c in color)
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return (20, 20, 24) if lum > 0.55 else (245, 245, 250)

    # ------------------------------------------------------------------
    # Top-right status cluster
    # ------------------------------------------------------------------
    def _draw_status_cluster(self, in_play: int, finished: int, total: int, elapsed: float, clock: pygame.time.Clock, ease_steps: int = 0) -> None:
        y = 60
        pad_x, pad_y = 18, 10

        items: list[tuple[str, Color]] = [
            (f"{int(clock.get_fps())} FPS", MD3_ON_BG_MED),
            (f"{in_play} IN PLAY", MD3_TERTIARY),
            (f"{finished}/{total} FINISHED", MD3_SUCCESS),
            (self._fmt_time(elapsed), MD3_ON_BG),
        ]
        if ease_steps > 0:
            items.append((f"EASED ×{ease_steps}", MD3_WARNING))

        for text, color in items:
            surf = self._font.render(text, 22, MD3_ON_BG, bold=True)
            w = surf.get_width() + pad_x * 2 + 14
            h = surf.get_height() + pad_y * 2
            rect = pygame.Rect(DISPLAY_W - 36 - w, y, w, h)
            bg = pygame.Surface(rect.size, pygame.SRCALPHA)
            pygame.draw.rect(bg, (*MD3_SURFACE_HIGH, 215), bg.get_rect(), border_radius=h // 2)
            pygame.draw.circle(bg, color, (pad_x, h // 2), 6)
            self._screen.blit(bg, rect.topleft)
            self._screen.blit(surf, (rect.x + pad_x + 14, rect.y + pad_y))
            y += h + 10

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"

    # ------------------------------------------------------------------
    # Motion meter
    # ------------------------------------------------------------------
    def _draw_motion_meter(self, motion_score: float) -> None:
        bar_w = 520
        bar_h = 18
        cx = DISPLAY_W // 2
        y = DISPLAY_H - 80
        rect = pygame.Rect(cx - bar_w // 2, y, bar_w, bar_h)
        draw_rrect(self._screen, rect, MD3_SURFACE_HIGH, radius=bar_h // 2, alpha=215)
        frac = clamp(motion_score)
        fill_w = int(bar_w * frac)
        if fill_w > 6:
            col = lerp_color(MD3_SUCCESS, MD3_ERROR, frac)
            draw_rrect(self._screen, pygame.Rect(rect.x, rect.y, fill_w, bar_h), col, radius=bar_h // 2)
        surf = self._font.render("MOTION", 16, MD3_ON_BG_MED, bold=True)
        self._screen.blit(surf, surf.get_rect(midbottom=(cx, y - 6)))

    def flash_caught(self, x: int, y: int) -> None:
        self._flashes.append(Flash(x, y))

    def _draw_flashes(self) -> None:
        now = time.time()
        survivors: list[Flash] = []
        for fl in self._flashes:
            age = now - fl.ts
            if age > 0.9:
                continue
            survivors.append(fl)
            t = age / 0.9
            r = int(40 + 260 * ease_out_cubic(t))
            alpha = int(200 * (1.0 - t))
            surf_size = r * 2 + 8
            surf = pygame.Surface((surf_size, surf_size), pygame.SRCALPHA)
            pygame.draw.circle(surf, (*MD3_ERROR, alpha), (surf_size // 2, surf_size // 2), r, width=8)
            self._screen.blit(surf, (fl.x - surf_size // 2, fl.y - surf_size // 2))
        self._flashes = survivors

    # ==================================================================
    # PUBLIC SCREENS
    # ==================================================================
    def _home_layout(self) -> tuple[pygame.Rect, pygame.Rect]:
        margin = 64
        gap = 48
        usable_w = DISPLAY_W - margin * 2 - gap
        cam_w = int(usable_w * HOME_CAM_BOX_FRACTION)
        anim_w = usable_w - cam_w
        cam_h = int(cam_w * 9 / 16)
        cam_h = min(cam_h, DISPLAY_H - 320)
        cam_y = (DISPLAY_H - cam_h) // 2 + 20
        cam_rect = pygame.Rect(margin, cam_y, cam_w, cam_h)
        anim_rect = pygame.Rect(margin + cam_w + gap, cam_y, anim_w, cam_h)
        return cam_rect, anim_rect

    # ---- Start Screen -----------------------------------------
    def draw_start_screen(self, frame: np.ndarray | None, palm_progress: float, clock: pygame.time.Clock) -> None:
        self.begin_frame()
        self._draw_decorative_bg()

        cam_rect, anim_rect = self._home_layout()

        title_surf = self._font.render("RED LIGHT, GREEN LIGHT", 78, MD3_ON_BG, bold=True)
        self._screen.blit(title_surf, title_surf.get_rect(center=(DISPLAY_W // 2, 90)))
        sub_surf = self._font.render("STEM Day · Version 1.0.0", 24, MD3_PRIMARY, bold=False)
        self._screen.blit(sub_surf, sub_surf.get_rect(center=(DISPLAY_W // 2, 138)))

        self.draw_camera_in_rect(frame, cam_rect, radius=36)

        live_pill = pygame.Rect(cam_rect.x + 20, cam_rect.y + 20, 86, 32)
        live_surf = pygame.Surface(live_pill.size, pygame.SRCALPHA)
        pygame.draw.rect(live_surf, (*MD3_ERROR, 230), live_surf.get_rect(), border_radius=16)
        pygame.draw.circle(live_surf, (255, 255, 255, 230), (16, 16), 5)
        self._screen.blit(live_surf, live_pill.topleft)
        live_label = self._font.render("LIVE", 18, (255, 255, 255), bold=True)
        self._screen.blit(live_label, live_label.get_rect(midleft=(live_pill.x + 30, live_pill.y + 16)))

        draw_shadow_rrect(self._screen, anim_rect, 36, offset=(0, 14), spread=22, alpha=110)
        draw_rrect(self._screen, anim_rect, MD3_SURFACE_HIGH, radius=36, alpha=215)

        header = self._font.render("RAISE YOUR HAND", 32, MD3_ON_BG, bold=True)
        self._screen.blit(header, header.get_rect(center=(anim_rect.centerx, anim_rect.y + 60)))
        sub = self._font.render("to begin", 22, MD3_ON_BG_MED, bold=False)
        self._screen.blit(sub, sub.get_rect(center=(anim_rect.centerx, anim_rect.y + 96)))

        ring_cx = anim_rect.centerx
        ring_cy = anim_rect.centery + 10
        ring_radius = min(140, anim_rect.w // 3)

        halo_t = pulse(time.time(), 1.6) * 0.6 + 0.4
        halo_r = int(ring_radius + 20 + 12 * halo_t)
        halo_surf = pygame.Surface((halo_r * 2, halo_r * 2), pygame.SRCALPHA)
        pygame.draw.circle(halo_surf, (*MD3_PRIMARY, int(50 + 60 * palm_progress)), (halo_r, halo_r), halo_r)
        self._screen.blit(halo_surf, (ring_cx - halo_r, ring_cy - halo_r))

        draw_progress_ring(self._screen, (ring_cx, ring_cy), ring_radius, 16, palm_progress, MD3_PRIMARY)
        self._draw_palm_glyph((ring_cx, ring_cy), scale=ring_radius / 140.0, tint=MD3_PRIMARY, wave_t=time.time())

        inner_label = "HOLD" if palm_progress < 0.99 else "STARTING"
        pct_label = f"{int(palm_progress * 100)}%"
        self._draw_text_center(inner_label, (ring_cx, ring_cy + ring_radius + 36), 22, MD3_ON_BG, bold=True)
        self._draw_text_center(pct_label, (ring_cx, ring_cy + ring_radius + 64), 20, MD3_ON_BG_MED)

        self._draw_text_center("Raise your hand above your shoulder to start", (DISPLAY_W // 2, DISPLAY_H - 70), 26, MD3_ON_BG_MED)
        self._draw_text_center("ESC to quit · CTRL+D for dev mode · SPACE to skip", (DISPLAY_W // 2, DISPLAY_H - 36), 18, MD3_ON_BG_DIM)

    def _draw_palm_glyph(self, center: tuple[int, int], scale: float = 1.0, tint: Color = MD3_PRIMARY, wave_t: float = 0.0) -> None:
        cx, cy = center
        sway = math.sin(wave_t * 2.0) * 6.0 * scale
        palm_w = int(60 * scale)
        palm_h = int(72 * scale)
        palm = pygame.Rect(0, 0, palm_w, palm_h)
        palm.center = (int(cx + sway), int(cy + 8 * scale))
        draw_rrect(self._screen, palm, tint, radius=int(20 * scale), alpha=240)

        wrist = pygame.Rect(0, 0, int(palm_w * 0.7), int(14 * scale))
        wrist.midtop = (palm.centerx, palm.bottom - int(6 * scale))
        draw_rrect(self._screen, wrist, MD3_PRIMARY_CONTAINER, radius=int(7 * scale), alpha=240)

        finger_w = int(12 * scale)
        finger_h = int(46 * scale)
        thumb_h = int(34 * scale)
        gap = int(3 * scale)
        total_w = finger_w * 4 + gap * 3
        start_x = palm.centerx - total_w // 2
        for i in range(4):
            f = pygame.Rect(0, 0, finger_w, finger_h)
            f.midbottom = (start_x + i * (finger_w + gap) + finger_w // 2, palm.top + int(4 * scale))
            draw_rrect(self._screen, f, tint, radius=int(6 * scale), alpha=240)

        thumb = pygame.Rect(0, 0, finger_w, thumb_h)
        thumb.midright = (palm.left + int(6 * scale), palm.centery - int(8 * scale))
        draw_rrect(self._screen, thumb, tint, radius=int(6 * scale), alpha=240)

    # ---- Countdown ---------------------------------------------------
    def draw_countdown(self, frame: np.ndarray | None, n: int, clock: pygame.time.Clock) -> None:
        self.begin_frame()
        self.draw_camera(frame)
        self._draw_vignette(strength=160)
        self._draw_banner("COUNTDOWN")

        if n != self._countdown_last_n:
            self._countdown_last_n = n
            self._countdown_trigger_ts = time.time()
        age = time.time() - self._countdown_trigger_ts
        scale = 0.6 + 0.4 * ease_out_back(clamp(age * 3.0))
        alpha = int(255 * clamp(1.0 - age * 0.6))

        big = self._font.render(str(n), int(400 * scale), MD3_PRIMARY, bold=True)
        big.set_alpha(alpha)
        self._screen.blit(big, big.get_rect(center=(DISPLAY_W // 2, DISPLAY_H // 2 + 40)))

    # ---- Wait at start line -----------------------------------------
    def draw_wait_start_line(
        self, frame: np.ndarray | None, behind_count: int, total_visible: int, time_left: float, clock: pygame.time.Clock
    ) -> None:
        self.begin_frame()
        self.draw_camera(frame)
        self._draw_vignette(strength=110)
        self._draw_banner("START_LINE")

        cw, ch = 760, 240
        card = pygame.Rect(DISPLAY_W // 2 - cw // 2, DISPLAY_H // 2 - ch // 2 + 40, cw, ch)
        draw_panel(self._screen, card, MD3_SURFACE_HIGH, radius=32, alpha=235)

        ratio_surf = self._font.render(f"{behind_count} / {max(behind_count, total_visible)} behind line", 46, MD3_SUCCESS, bold=True)
        self._screen.blit(ratio_surf, ratio_surf.get_rect(center=(card.centerx, card.y + 80)))

        msg = "Step behind the GREEN tape on the floor" if behind_count < max(1, total_visible) else "Hold position..."
        msg_surf = self._font.render(msg, 26, MD3_ON_BG_MED)
        self._screen.blit(msg_surf, msg_surf.get_rect(center=(card.centerx, card.y + 140)))

        # Caller pushes the actual bar rendering
        pass

    def draw_progress_bar_centered(self, y: int, ratio: float, color: Color = MD3_WARNING, width: int = 560) -> None:
        bar = pygame.Rect(DISPLAY_W // 2 - width // 2, y, width, 10)
        draw_rrect(self._screen, bar, MD3_SURFACE_VAR, 5, alpha=220)
        fill_w = int(width * clamp(ratio))
        if fill_w > 0:
            draw_rrect(self._screen, pygame.Rect(bar.x, bar.y, fill_w, bar.h), color, 5)

    # ---- Live game HUD ----------------------------------------------
    def draw_game_hud(
        self,
        frame: np.ndarray | None,
        pose: dict | None,
        state_label: str,
        in_play: int,
        finished: int,
        total: int,
        elapsed: float,
        motion_score: float,
        finished_ids: set[int],
        labels_by_id: dict[int, str],
        ease_steps: int,
        clock: pygame.time.Clock,
    ) -> None:
        self.begin_frame()
        self.draw_camera(frame)
        self.draw_pose(pose, finished_ids=finished_ids, labels_by_id=labels_by_id)
        self._draw_banner(state_label)
        self._draw_status_cluster(in_play, finished, total, elapsed, clock, ease_steps)
        if state_label == "RED":
            self._draw_motion_meter(motion_score)
        self._draw_sent_back_log()
        self._draw_flashes()

    # ---- Caught return screen ---------------------------------------
    def draw_caught_return(
        self, frame: np.ndarray | None, returning: list[tuple[str, bool]], time_left_ratio: float, total_time: float, clock: pygame.time.Clock
    ) -> None:
        self.begin_frame()
        self.draw_camera(frame)
        tint = pygame.Surface((DISPLAY_W, DISPLAY_H), pygame.SRCALPHA)
        tint.fill((*MD3_WARNING, 80))
        self._screen.blit(tint, (0, 0))
        self._draw_banner("RETURN")

        rows = max(1, len(returning))
        cw = 820
        ch = min(560, 200 + 56 * rows)
        card = pygame.Rect(DISPLAY_W // 2 - cw // 2, DISPLAY_H // 2 - ch // 2 + 30, cw, ch)
        draw_panel(self._screen, card, MD3_SURFACE_HIGH, radius=32, alpha=235)

        title = self._font.render("Walk back to the start line", 30, MD3_WARNING, bold=True)
        self._screen.blit(title, title.get_rect(center=(card.centerx, card.y + 50)))
        sub = self._font.render("The game resumes when everyone is back", 22, MD3_ON_BG_MED)
        self._screen.blit(sub, sub.get_rect(center=(card.centerx, card.y + 90)))

        list_y = card.y + 140
        for descriptor, is_back in returning:
            row_h = 44
            row = pygame.Rect(card.x + 40, list_y, card.w - 80, row_h)
            row_color = MD3_SUCCESS if is_back else MD3_SURFACE_VAR
            draw_rrect(self._screen, row, row_color, 14, alpha=240 if is_back else 200)
            mark_x = row.x + 24
            if is_back:
                pygame.draw.circle(self._screen, MD3_SUCCESS_BG_DARK, (mark_x, row.centery), 10)
                pygame.draw.line(self._screen, MD3_ON_BG, (mark_x - 6, row.centery), (mark_x - 1, row.centery + 5), 3)
                pygame.draw.line(self._screen, MD3_ON_BG, (mark_x - 1, row.centery + 5), (mark_x + 7, row.centery - 4), 3)
            else:
                pygame.draw.circle(self._screen, MD3_OUTLINE, (mark_x, row.centery), 10, width=2)

            text = self._font.render(descriptor, 22, MD3_ON_BG if is_back else MD3_ON_BG_MED, bold=True)
            self._screen.blit(text, (mark_x + 22, row.y + (row_h - text.get_height()) // 2))
            list_y += row_h + 10

        self.draw_progress_bar_centered(card.bottom - 30, time_left_ratio, color=MD3_WARNING, width=cw - 80)
        bar_label = f"Auto-resume in {max(0.0, total_time * time_left_ratio):.1f}s"
        lab = self._font.render(bar_label, 18, MD3_ON_BG_DIM)
        self._screen.blit(lab, lab.get_rect(center=(card.centerx, card.bottom - 50)))

    # ---- Leaderboard ------------------------------------------------
    def draw_leaderboard(self, results: list[dict], replay_progress: float, clock: pygame.time.Clock) -> None:
        self.begin_frame()
        self._draw_decorative_bg()
        self._update_confetti()
        self._draw_confetti()

        title = self._font.render("RESULTS", 88, MD3_PRIMARY, bold=True)
        self._screen.blit(title, title.get_rect(center=(DISPLAY_W // 2, 90)))
        sub = self._font.render("Everyone made it across!", 26, MD3_ON_BG_MED)
        self._screen.blit(sub, sub.get_rect(center=(DISPLAY_W // 2, 144)))

        podium_y = 240
        podium_h = 480
        podium_layout = []
        if len(results) >= 1:
            podium_layout.append((results[0], DISPLAY_W // 2, podium_h, MD3_GOLD))
        if len(results) >= 2:
            podium_layout.append((results[1], DISPLAY_W // 2 - 360, int(podium_h * 0.85), MD3_SILVER))
        if len(results) >= 3:
            podium_layout.append((results[2], DISPLAY_W // 2 + 360, int(podium_h * 0.72), MD3_BRONZE))

        for entry, cx, h, badge_color in podium_layout:
            self._draw_podium_card(entry, cx, podium_y, h, badge_color)

        if len(results) > 3:
            list_y = podium_y + podium_h + 60
            list_w = 900
            list_x = DISPLAY_W // 2 - list_w // 2
            for i, entry in enumerate(results[3:], start=4):
                row = pygame.Rect(list_x, list_y, list_w, 56)
                draw_panel(self._screen, row, MD3_SURFACE_HIGH, 18, 230, shadow=False)
                rank_lbl = self._font.render(f"#{i}", 26, MD3_ON_BG, bold=True)
                self._screen.blit(rank_lbl, (row.x + 24, row.y + 14))
                name_lbl = self._font.render(entry["descriptor"], 22, MD3_ON_BG)
                self._screen.blit(name_lbl, (row.x + 96, row.y + 16))
                time_lbl = self._font.render(self._fmt_time(entry["time_s"]), 22, MD3_ON_BG_MED, bold=True)
                self._screen.blit(time_lbl, time_lbl.get_rect(midright=(row.right - 24, row.centery)))
                list_y += 64

        footer_y = DISPLAY_H - 60
        self._draw_text_center("Press SPACE to play again · ESC to quit", (DISPLAY_W // 2, footer_y), 22, MD3_ON_BG_MED)

    def _draw_podium_card(self, entry: dict, cx: int, base_y: int, height: int, badge_color: Color) -> None:
        photo: pygame.Surface | None = entry.get("photo_surface")
        descriptor = entry.get("descriptor", "Player")
        rank = entry.get("rank", 0)
        time_s = entry.get("time_s", 0.0)

        card_w = 320
        card = pygame.Rect(cx - card_w // 2, base_y + (480 - height), card_w, height)
        draw_panel(self._screen, card, MD3_SURFACE_HIGH, 28, 240)

        photo_h = 280
        photo_rect = pygame.Rect(card.x + 20, card.y + 20, card.w - 40, photo_h)
        if photo is not None:
            scaled = pygame.transform.smoothscale(photo, photo_rect.size)
            mask = pygame.Surface(photo_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(mask, (255, 255, 255, 255), mask.get_rect(), border_radius=20)
            clipped = pygame.Surface(photo_rect.size, pygame.SRCALPHA)
            clipped.blit(scaled, (0, 0))
            clipped.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
            self._screen.blit(clipped, photo_rect.topleft)
        else:
            draw_rrect(self._screen, photo_rect, MD3_SURFACE_VAR, 20, alpha=240)
            placeholder = self._font.render("?", 96, MD3_ON_BG_DIM, bold=True)
            self._screen.blit(placeholder, placeholder.get_rect(center=photo_rect.center))

        badge = pygame.Rect(photo_rect.x - 6, photo_rect.y - 6, 64, 64)
        pygame.draw.circle(self._screen, badge_color, badge.center, 32)
        pygame.draw.circle(self._screen, MD3_SURFACE_HIGH, badge.center, 32, width=3)
        rank_txt = self._font.render(str(rank), 32, (32, 24, 8), bold=True)
        self._screen.blit(rank_txt, rank_txt.get_rect(center=badge.center))

        text_y = photo_rect.bottom + 14
        words = descriptor.split()
        line = ""
        max_chars = 22
        lines = []
        for word in words:
            if len(line) + len(word) + 1 > max_chars:
                lines.append(line.strip())
                line = word
            else:
                line += " " + word
        if line.strip():
            lines.append(line.strip())
        for ln in lines[:2]:
            surf = self._font.render(ln, 18, MD3_ON_BG, bold=True)
            self._screen.blit(surf, surf.get_rect(center=(card.centerx, text_y)))
            text_y += 22

        time_lbl = self._font.render(self._fmt_time(time_s), 24, badge_color, bold=True)
        self._screen.blit(time_lbl, time_lbl.get_rect(center=(card.centerx, card.bottom - 30)))

    def _update_confetti(self) -> None:
        if len(self._confetti) < 180:
            for _ in range(8):
                x = float(np.random.randint(0, DISPLAY_W))
                y = -20.0 - float(np.random.randint(0, DISPLAY_H))
                vx = float(np.random.uniform(-30.0, 30.0))
                vy = float(np.random.uniform(120.0, 260.0))
                palette = [MD3_PRIMARY, MD3_SECONDARY, MD3_TERTIARY, MD3_SUCCESS, MD3_WARNING, MD3_GOLD, MD3_SILVER]
                col = palette[np.random.randint(0, len(palette))]
                self._confetti.append((x, y, vx, vy, col))
        new_list = []
        for x, y, vx, vy, col in self._confetti:
            y += vy * self._dt
            x += vx * self._dt
            vy += 180.0 * self._dt
            if y < DISPLAY_H + 40:
                new_list.append((x, y, vx, vy, col))
        self._confetti = new_list

    def _draw_confetti(self) -> None:
        for x, y, _vx, _vy, col in self._confetti:
            rect = pygame.Rect(int(x), int(y), 8, 14)
            pygame.draw.rect(self._screen, col, rect, border_radius=2)

    def _draw_vignette(self, strength: int = 120) -> None:
        overlay = pygame.Surface((DISPLAY_W, DISPLAY_H), pygame.SRCALPHA)
        for i in range(8):
            a = int(strength * (i + 1) / 8)
            pad = (8 - i) * 14
            pygame.draw.rect(overlay, (0, 0, 0, a // 8), pygame.Rect(pad, pad, DISPLAY_W - pad * 2, DISPLAY_H - pad * 2), width=2, border_radius=24)
        band = pygame.Surface((DISPLAY_W, 180), pygame.SRCALPHA)
        for i in range(180):
            a = int(strength * (1.0 - i / 180.0))
            pygame.draw.rect(band, (0, 0, 0, a), (0, i, DISPLAY_W, 1))
        self._screen.blit(band, (0, 0))
        band2 = pygame.transform.flip(band, False, True)
        self._screen.blit(band2, (0, DISPLAY_H - 180))
        self._screen.blit(overlay, (0, 0))

    # ------------------------------------------------------------------
    # Dev panel (Master Merged Version)
    # ------------------------------------------------------------------
    def draw_dev_panel(self, metrics: dict) -> None:
        if not self._dev_mode:
            return
        self._dev_fps_history.append(float(metrics.get("fps", 0)))

        panel_w = 400
        panel = pygame.Rect(DISPLAY_W - panel_w - 24, 24, panel_w, DISPLAY_H - 48)
        draw_panel(self._screen, panel, MD3_SURFACE, radius=28, alpha=235)

        header_y = panel.y + 24
        self._draw_text_left("DEV · DEBUG", (panel.x + 24, header_y), 26, MD3_PRIMARY, bold=True)
        self._draw_text_left("CTRL+D to close", (panel.x + 24, header_y + 32), 16, MD3_ON_BG_DIM)

        y = header_y + 80

        # --- Performance ---
        y = self._dev_section(panel, "PERFORMANCE", y)
        y = self._dev_kv(panel, "Overall FPS", f"{metrics.get('fps', 0):.1f}", y)
        y = self._dev_kv(panel, "Camera FPS", f"{metrics.get('cam_fps', 0):.1f}", y)
        y = self._dev_kv(panel, "YOLO inference", f"{metrics.get('inference_ms', 0):.1f} ms", y)

        y += 8
        graph = pygame.Rect(panel.x + 24, y, panel_w - 48, 60)
        draw_rrect(self._screen, graph, MD3_SURFACE_HIGH, radius=12, alpha=220)
        if self._dev_fps_history:
            hist = list(self._dev_fps_history)
            mx = max(1.0, max(hist))
            bar_w = max(2, (graph.w - 8) // max(1, len(hist)))
            for i, v in enumerate(hist):
                bh = int((graph.h - 8) * clamp(v / max(60.0, mx)))
                bx = graph.x + 4 + i * bar_w
                by = graph.bottom - 4 - bh
                col = MD3_SUCCESS if v >= 50 else MD3_WARNING if v >= 30 else MD3_ERROR
                pygame.draw.rect(self._screen, col, pygame.Rect(bx, by, bar_w - 1, bh), border_radius=2)
        y = graph.bottom + 20

        # --- State ---
        y = self._dev_section(panel, "STATE", y)
        y = self._dev_kv(panel, "Current", str(metrics.get("state", "?")), y)
        y = self._dev_kv(panel, "Time in state", f"{metrics.get('state_time', 0):.2f} s", y)
        y = self._dev_kv(panel, "Players in-play", str(metrics.get("players", 0)), y)
        y = self._dev_kv(panel, "Finishers", str(metrics.get("finishers", 0)), y)
        if metrics.get("ease_steps", 0) > 0:
            y = self._dev_kv(panel, "Ease step", f"×{metrics.get('ease_steps', 0)}", y, value_color=MD3_WARNING)

        # --- Hardware & Vision ---
        y = self._dev_section(panel, "HARDWARE & VISION", y)
        y = self._dev_kv(panel, "Servo", f"{metrics.get('servo_angle', 0):.0f}° → {metrics.get('servo_target', 0):.0f}°", y)

        laser = "BROKEN" if metrics.get("laser_broken") else ("ON" if metrics.get("laser_in_use") else "intact")
        laser_col = MD3_ERROR if metrics.get("laser_broken") else MD3_SUCCESS
        y = self._dev_kv(panel, "Laser beam", laser, y, value_color=laser_col)

        id_mode = metrics.get("id_mode", "?")
        y = self._dev_kv(panel, "ID Mode", id_mode, y)

        # --- Hardware Test Menu ---
        y = self._dev_section(panel, "HARDWARE TEST", y)
        tests = [
            ("1", "Servo -> face players"),
            ("2", "Servo -> face away"),
            ("3", "Refresh laser read"),
            ("4", "Play test chime"),
        ]
        for key, desc in tests:
            y = self._dev_test_row(panel, key, desc, y)

        # Footer hints
        hints = metrics.get("dev_hints", [])
        fy = panel.bottom - 24 - 18 * len(hints)
        for h in hints:
            self._draw_text_left(h, (panel.x + 24, fy), 14, MD3_ON_BG_DIM)
            fy += 18

    # ------------------------------------------------------------------
    # Dev panel helpers
    # ------------------------------------------------------------------
    def _dev_section(self, panel: pygame.Rect, title: str, y: int) -> int:
        self._draw_text_left(title, (panel.x + 24, y), 14, MD3_SECONDARY, bold=True)
        pygame.draw.line(self._screen, MD3_OUTLINE, (panel.x + 24, y + 22), (panel.right - 24, y + 22), 1)
        return y + 34

    def _dev_kv(self, panel: pygame.Rect, key: str, value: str, y: int, value_color: Color = MD3_ON_BG) -> int:
        self._draw_text_left(key, (panel.x + 24, y), 18, MD3_ON_BG_MED)
        surf = self._font.render(value, 20, value_color, bold=True)
        self._screen.blit(surf, surf.get_rect(topright=(panel.right - 24, y - 2)))
        return y + 28

    def _dev_test_row(self, panel: pygame.Rect, key: str, desc: str, y: int) -> int:
        badge = pygame.Rect(panel.x + 24, y, 34, 28)
        draw_rrect(self._screen, badge, MD3_PRIMARY_CONTAINER, radius=8)
        self._draw_text_center(key, badge.center, 20, MD3_PRIMARY, bold=True)
        self._draw_text_left(desc, (badge.right + 12, y + 4), 18, MD3_ON_BG)
        return y + 36

    # ------------------------------------------------------------------
    # Tiny helpers
    # ------------------------------------------------------------------
    def _draw_text_center(self, text: str, center: tuple[int, int], size: int, color: Color, bold: bool = False) -> pygame.Rect:
        return draw_text_centered(self._screen, text, center, self._font, size, color, bold)

    def _draw_text_left(self, text: str, pos: tuple[int, int], size: int, color: Color, bold: bool = False) -> pygame.Rect:
        return draw_text_left(self._screen, text, pos, self._font, size, color, bold)
