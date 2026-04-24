#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Project:      Red Light Green Light (Jetson Orin Nano)
File:         ui.py
Description:  Material Design 3 "Expressive" rendering engine for Pygame.

              Draws the full-screen camera feed as the base layer and
              composites translucent, rounded UI elements on top:

                  * State banner with pulse + colour transition
                  * Player chips (pill-shaped, colour-coded, animated)
                  * Pose skeletons rendered with anti-aliased Pygame lines
                  * Palm-raise progress ring
                  * Countdown, caught, winner overlays with eased entrances
                  * Scrolling elimination log
                  * Developer dashboard (Ctrl+D) with FPS graph, inference
                    timings, state timers, and a hardware test menu

              Animation is built on a small MotionValue helper that lerps a
              current value toward its target each frame. Static visuals are
              rendered on-demand, then blitted once per frame.
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
    ELIM_LOG_MAX,
    FONT_PATH,
    MD3_BG,
    MD3_ERROR,
    MD3_ERROR_BG,
    MD3_ERROR_ON,
    MD3_INFO,
    MD3_ON_BG,
    MD3_ON_BG_DIM,
    MD3_ON_BG_MED,
    MD3_ON_PRIMARY,
    MD3_ON_SECONDARY,
    MD3_ON_TERTIARY,
    MD3_OUTLINE,
    MD3_PRIMARY,
    MD3_PRIMARY_CONTAINER,
    MD3_SECONDARY,
    MD3_SUCCESS,
    MD3_SUCCESS_BG,
    MD3_SUCCESS_ON,
    MD3_SURFACE,
    MD3_SURFACE_HIGH,
    MD3_SURFACE_VAR,
    MD3_TERTIARY,
    MD3_WARNING,
    MD3_WARNING_BG,
    MD3_WARNING_ON,
    MOTION_FAST,
    MOTION_MED,
    MOTION_SLOW,
    STATE_COLORS,
)
from vision import KP, SKELETON_EDGES


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
    """Returns 0.0 .. 1.0 .. 0.0 oscillation."""
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


def darken(color: Color, amount: float = 0.2) -> Color:
    k = clamp(1.0 - amount)
    return (int(color[0] * k), int(color[1] * k), int(color[2] * k))


def lighten(color: Color, amount: float = 0.2) -> Color:
    k = clamp(amount)
    return (
        int(lerp(color[0], 255, k)),
        int(lerp(color[1], 255, k)),
        int(lerp(color[2], 255, k)),
    )


# Map vision.py colour labels -> display chip colours
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
# Motion value - a smooth, frame-rate-aware chase for animated scalars
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
        if self._path is None:
            print(f"[UI ] Font not found at {font_path} - using default")

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

    def render(
        self,
        text: str,
        size: int,
        color: Color,
        bold: bool = False,
        antialias: bool = True,
    ) -> pygame.Surface:
        return self.get(size, bold).render(text, antialias, color)


# ===========================================================================
# Drawing primitives
# ===========================================================================
def draw_shadow_rrect(
    target: pygame.Surface,
    rect: pygame.Rect,
    radius: int,
    offset: tuple[int, int] = (0, 6),
    spread: int = 10,
    alpha: int = 90,
) -> None:
    """Soft drop shadow approximated by concentric alpha rects."""
    passes = 4
    for i in range(passes):
        extra = spread * (i + 1) // passes
        a = max(1, alpha // (i + 1))
        sr = rect.inflate(extra * 2, extra * 2).move(offset[0], offset[1])
        surf = pygame.Surface(sr.size, pygame.SRCALPHA)
        pygame.draw.rect(
            surf,
            (0, 0, 0, a),
            surf.get_rect(),
            border_radius=radius + extra,
        )
        target.blit(surf, sr.topleft)


def draw_rrect(
    target: pygame.Surface,
    rect: pygame.Rect,
    color: Color,
    radius: int,
    alpha: int = 255,
    outline: Color | None = None,
    outline_w: int = 2,
) -> None:
    surf = pygame.Surface(rect.size, pygame.SRCALPHA)
    pygame.draw.rect(
        surf,
        (color[0], color[1], color[2], alpha),
        surf.get_rect(),
        border_radius=radius,
    )
    if outline is not None:
        pygame.draw.rect(
            surf,
            (outline[0], outline[1], outline[2], alpha),
            surf.get_rect(),
            width=outline_w,
            border_radius=radius,
        )
    target.blit(surf, rect.topleft)


def draw_panel(
    target: pygame.Surface,
    rect: pygame.Rect,
    color: Color = MD3_SURFACE_HIGH,
    radius: int = 24,
    alpha: int = 235,
    shadow: bool = True,
) -> None:
    if shadow:
        draw_shadow_rrect(target, rect, radius)
    draw_rrect(target, rect, color, radius, alpha=alpha)


def draw_pill(
    target: pygame.Surface,
    rect: pygame.Rect,
    color: Color,
    alpha: int = 255,
) -> None:
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
    """Polygon-approximated thick arc; smoother than pygame.draw.arc."""
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
        surf_rect = pygame.Rect(
            int(cx - radius - 2),
            int(cy - radius - 2),
            radius * 2 + 4,
            radius * 2 + 4,
        )
        surf = pygame.Surface(surf_rect.size, pygame.SRCALPHA)
        local = [(p[0] - surf_rect.x, p[1] - surf_rect.y) for p in points]
        pygame.draw.polygon(surf, (color[0], color[1], color[2], alpha), local)
        target.blit(surf, surf_rect.topleft)


def draw_progress_ring(
    target: pygame.Surface,
    center: tuple[int, int],
    radius: int,
    thickness: int,
    progress: float,
    fg: Color,
    bg: Color = MD3_SURFACE_VAR,
) -> None:
    """Circular progress ring. Progress starts at 12 o'clock, clockwise."""
    # Background full circle
    draw_arc_thick(
        target,
        center,
        radius,
        thickness,
        0.0,
        2.0 * math.pi,
        bg,
        alpha=200,
    )
    progress = clamp(progress)
    if progress <= 0.0:
        return
    start = -math.pi / 2.0
    end = start + 2.0 * math.pi * progress
    draw_arc_thick(target, center, radius, thickness, start, end, fg)


def draw_text_centered(
    target: pygame.Surface,
    text: str,
    center: tuple[int, int],
    font_cache: FontCache,
    size: int,
    color: Color,
    bold: bool = False,
) -> pygame.Rect:
    surf = font_cache.render(text, size, color, bold=bold)
    r = surf.get_rect(center=center)
    target.blit(surf, r)
    return r


def draw_text_left(
    target: pygame.Surface,
    text: str,
    pos: tuple[int, int],
    font_cache: FontCache,
    size: int,
    color: Color,
    bold: bool = False,
) -> pygame.Rect:
    surf = font_cache.render(text, size, color, bold=bold)
    r = surf.get_rect(topleft=pos)
    target.blit(surf, r)
    return r


# ===========================================================================
# Chip tracking (per-player chip with smoothed position)
# ===========================================================================
@dataclass
class ChipState:
    x: MotionValue = field(default_factory=lambda: MotionValue(0.0, MOTION_FAST))
    y: MotionValue = field(default_factory=lambda: MotionValue(0.0, MOTION_FAST))
    alpha: MotionValue = field(default_factory=lambda: MotionValue(0.0, MOTION_MED))
    last_seen: float = 0.0
    shirt: str = "unknown"


# ===========================================================================
# Elimination log entry
# ===========================================================================
@dataclass
class LogEntry:
    text: str
    colour: Color
    ts: float
    alpha: MotionValue = field(default_factory=lambda: MotionValue(0.0, MOTION_MED))
    slide: MotionValue = field(default_factory=lambda: MotionValue(24.0, MOTION_MED))


# ===========================================================================
# Flash (caught pulse overlay)
# ===========================================================================
@dataclass
class Flash:
    x: int
    y: int
    ts: float = field(default_factory=time.time)


# ===========================================================================
# Main UI renderer
# ===========================================================================
class UIRenderer:
    """High-level rendering facade used by game.py.

    Usage per frame:

        ui.begin_frame(dt)
        ui.draw_camera(frame)
        ui.draw_pose(pose_data)
        ui.draw_game_hud(...)   # or whichever screen is active
        ui.draw_dev_panel(metrics)  # if dev mode
        ui.present()
    """

    def __init__(self, screen: pygame.Surface) -> None:
        self._screen = screen
        self._font = FontCache(FONT_PATH)

        # Animation state
        self._state_color = MotionValue(0.0, MOTION_SLOW)  # unused: color lerps manually
        self._banner_color: Color = MD3_PRIMARY
        self._banner_target: Color = MD3_PRIMARY
        self._banner_blend = MotionValue(1.0, MOTION_SLOW)
        self._prev_banner_label = ""
        self._banner_label = ""
        self._banner_entry_ts = time.time()

        self._chips: dict[int, ChipState] = {}
        self._log: deque[LogEntry] = deque(maxlen=ELIM_LOG_MAX)
        self._flashes: list[Flash] = []

        # Countdown animation
        self._countdown_last_n = -1
        self._countdown_trigger_ts = 0.0

        # Confetti for winner
        self._confetti: list[tuple[float, float, float, float, Color]] = []

        # Dev mode
        self._dev_mode = False
        self._dev_fps_history: deque[float] = deque(maxlen=DEV_FPS_HISTORY)

        # Camera feed cache: avoid recreating the scale buffer each frame
        self._cam_scaled: pygame.Surface | None = None
        self._cam_scaled_size: tuple[int, int] = (0, 0)

        # Frame timing
        self._last_tick = time.time()
        self._dt = 0.0

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

    # ------------------------------------------------------------------
    # Dev mode
    # ------------------------------------------------------------------
    def toggle_dev_mode(self) -> bool:
        self._dev_mode = not self._dev_mode
        print(f"[UI ] Dev mode: {'ON' if self._dev_mode else 'OFF'}")
        return self._dev_mode

    def is_dev_mode(self) -> bool:
        return self._dev_mode

    # ------------------------------------------------------------------
    # Camera layer
    # ------------------------------------------------------------------
    def draw_camera(self, frame: np.ndarray | None) -> None:
        if frame is None:
            self._screen.fill(MD3_BG)
            # Subtle background blobs for menus when no camera
            self._draw_decorative_bg()
            return
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            surf = pygame.image.frombuffer(rgb.tobytes(), (w, h), "RGB")
            if (w, h) != (DISPLAY_W, DISPLAY_H):
                if self._cam_scaled_size != (DISPLAY_W, DISPLAY_H):
                    self._cam_scaled = pygame.Surface((DISPLAY_W, DISPLAY_H))
                    self._cam_scaled_size = (DISPLAY_W, DISPLAY_H)
                if self._cam_scaled is None:
                    self._cam_scaled = pygame.Surface((DISPLAY_W, DISPLAY_H))
                pygame.transform.scale(surf, (DISPLAY_W, DISPLAY_H), self._cam_scaled)
                self._screen.blit(self._cam_scaled, (0, 0))
            else:
                self._screen.blit(surf, (0, 0))
        except Exception as exc:
            print(f"[UI ] draw_camera: {exc}")
            self._screen.fill(MD3_BG)

    def _draw_decorative_bg(self) -> None:
        """Animated gradient blobs for menus when no camera feed is present."""
        t = time.time()
        for i, base in enumerate((MD3_PRIMARY_CONTAINER, MD3_SURFACE_VAR, MD3_PRIMARY)):
            phase = t * 0.2 + i * 2.1
            cx = int(DISPLAY_W * (0.3 + 0.4 * math.sin(phase)))
            cy = int(DISPLAY_H * (0.3 + 0.4 * math.cos(phase * 0.8)))
            r = int(420 + 60 * math.sin(phase * 1.5))
            surf = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(surf, (*base, 40), (r, r), r)
            self._screen.blit(surf, (cx - r, cy - r))

    # ------------------------------------------------------------------
    # Pose overlay: skeletons + player chips
    # ------------------------------------------------------------------
    def draw_pose(self, pose: dict | None) -> None:
        if pose is None:
            return

        fw = pose.get("frame_w", DISPLAY_W) or DISPLAY_W
        fh = pose.get("frame_h", DISPLAY_H) or DISPLAY_H
        sx = DISPLAY_W / fw
        sy = DISPLAY_H / fh

        self._draw_skeletons(pose, sx, sy)
        self._update_chips(pose, sx, sy)
        self._draw_chips()

    def _draw_skeletons(self, pose: dict, sx: float, sy: float) -> None:
        keypoints = pose.get("keypoints", [])
        shirts = pose.get("shirt_colours", [])
        for idx, kp in enumerate(keypoints):
            if kp is None or len(kp) < 17:
                continue
            shirt = shirts[idx] if idx < len(shirts) else "unknown"
            col = SHIRT_DISPLAY_COLOR.get(shirt, MD3_PRIMARY)

            # Bones
            for a, b in SKELETON_EDGES:
                pa = kp[a]
                pb = kp[b]
                if pa[2] < 0.3 or pb[2] < 0.3:
                    continue
                if pa[0] == 0 and pa[1] == 0:
                    continue
                if pb[0] == 0 and pb[1] == 0:
                    continue
                xa, ya = int(pa[0] * sx), int(pa[1] * sy)
                xb, yb = int(pb[0] * sx), int(pb[1] * sy)
                pygame.draw.line(self._screen, col, (xa, ya), (xb, yb), 4)

            # Joints
            for ki, joint in enumerate(kp):
                if joint[2] < 0.3:
                    continue
                if joint[0] == 0 and joint[1] == 0:
                    continue
                jx, jy = int(joint[0] * sx), int(joint[1] * sy)
                r = 7 if ki in (5, 6, 11, 12) else 5
                pygame.draw.circle(self._screen, MD3_ON_BG, (jx, jy), r)
                pygame.draw.circle(self._screen, col, (jx, jy), r - 2)

    def _update_chips(self, pose: dict, sx: float, sy: float) -> None:
        now = time.time()
        keypoints = pose.get("keypoints", [])
        boxes = pose.get("boxes", [])
        ids = pose.get("track_ids", [])
        shirts = pose.get("shirt_colours", [])

        for i, tid in enumerate(ids):
            if tid is None:
                continue
            kp = keypoints[i] if i < len(keypoints) else None
            if kp is None or len(kp) < 17:
                continue
            # Anchor above nose; fall back to box top center.
            if kp[KP["nose"]][2] > 0.3:
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
            chip.shirt = shirts[i] if i < len(shirts) else "unknown"

        # Fade out stale chips
        stale = []
        for tid, chip in self._chips.items():
            if now - chip.last_seen > 0.3:
                chip.alpha.set(0.0)
            if now - chip.last_seen > 2.0 and chip.alpha.current < 4.0:
                stale.append(tid)
            chip.x.update(self._dt)
            chip.y.update(self._dt)
            chip.alpha.update(self._dt)
        for tid in stale:
            del self._chips[tid]

    def _draw_chips(self) -> None:
        for tid, chip in self._chips.items():
            alpha = int(chip.alpha.current)
            if alpha < 8:
                continue
            col = SHIRT_DISPLAY_COLOR.get(chip.shirt, MD3_PRIMARY)
            label = f"Player {tid}"
            text = self._font.render(label, 22, MD3_ON_PRIMARY, bold=True)
            pad_x = 18
            pad_y = 8
            dot_r = 9
            w = text.get_width() + pad_x * 2 + dot_r * 2 + 8
            h = text.get_height() + pad_y * 2
            rect = pygame.Rect(
                int(chip.x.current - w / 2),
                int(chip.y.current - h / 2),
                w,
                h,
            )
            draw_shadow_rrect(self._screen, rect, h // 2, offset=(0, 4), alpha=int(alpha * 0.4))
            surf = pygame.Surface(rect.size, pygame.SRCALPHA)
            pygame.draw.rect(
                surf,
                (col[0], col[1], col[2], alpha),
                surf.get_rect(),
                border_radius=h // 2,
            )
            # Inner dot indicating shirt colour (white ring + colour fill)
            dot_center = (pad_x + dot_r, h // 2)
            pygame.draw.circle(surf, (255, 255, 255, alpha), dot_center, dot_r)
            pygame.draw.circle(
                surf,
                (col[0], col[1], col[2], alpha),
                dot_center,
                dot_r - 3,
            )
            self._screen.blit(surf, rect.topleft)
            text_pos = (
                rect.x + pad_x + dot_r * 2 + 8,
                rect.y + (h - text.get_height()) // 2,
            )
            text.set_alpha(alpha)
            self._screen.blit(text, text_pos)

    # ------------------------------------------------------------------
    # Elimination log
    # ------------------------------------------------------------------
    def log_elimination(self, colour_label: str) -> None:
        pretty = colour_label if colour_label != "unknown" else "?"
        entry = LogEntry(
            text=f"{pretty.upper()} eliminated",
            colour=SHIRT_DISPLAY_COLOR.get(colour_label, MD3_ERROR),
            ts=time.time(),
        )
        entry.alpha.set(255.0)
        entry.slide.set(0.0)
        self._log.appendleft(entry)

    def _draw_elimination_log(self) -> None:
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
            pygame.draw.rect(
                surf,
                (*MD3_SURFACE_HIGH, int(alpha * 0.9)),
                surf.get_rect(),
                border_radius=h // 2,
            )
            pygame.draw.circle(
                surf,
                (*entry.colour, alpha),
                (pad_x + dot_r, h // 2),
                dot_r,
            )
            self._screen.blit(surf, rect.topleft)
            label_surf.set_alpha(alpha)
            self._screen.blit(
                label_surf,
                (rect.x + pad_x + dot_r * 2 + 10, rect.y + pad_y),
            )
            y = rect.y - 8

    # ------------------------------------------------------------------
    # State banner (GREEN LIGHT / RED LIGHT / TURNING)
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

    def _draw_banner(self, label: str, time_left: float | None = None) -> None:
        self._update_banner(label)
        elapsed = time.time() - self._banner_entry_ts
        scale = 0.92 + 0.08 * ease_out_back(clamp(elapsed * 3.5))

        pulse_amt = pulse(time.time(), BANNER_PULSE_HZ) * 0.08
        cx = DISPLAY_W // 2
        base_w = 560
        base_h = 128
        w = int(base_w * scale * (1.0 + pulse_amt))
        h = int(base_h * scale)
        rect = pygame.Rect(cx - w // 2, 48, w, h)

        draw_shadow_rrect(self._screen, rect, h // 2, offset=(0, 12), spread=16, alpha=140)
        surf = pygame.Surface(rect.size, pygame.SRCALPHA)
        col = self._banner_color
        pygame.draw.rect(
            surf,
            (col[0], col[1], col[2], 250),
            surf.get_rect(),
            border_radius=h // 2,
        )
        # Highlight sheen on top third
        sheen = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(
            sheen,
            (255, 255, 255, 50),
            pygame.Rect(0, 0, rect.w, rect.h // 2),
            border_radius=h // 2,
        )
        surf.blit(sheen, (0, 0))
        self._screen.blit(surf, rect.topleft)

        # Pick legible text colour for chosen container
        txt_col = self._readable_on(col)
        display_text = self._banner_display_text(label)
        label_surf = self._font.render(display_text, 56, txt_col, bold=True)
        self._screen.blit(label_surf, label_surf.get_rect(center=rect.center))

        # Thin progress bar underneath banner for phase timer
        if time_left is not None:
            bar_w = int(base_w * 0.75)
            bar = pygame.Rect(cx - bar_w // 2, rect.bottom + 14, bar_w, 8)
            pygame.draw.rect(self._screen, MD3_SURFACE_HIGH, bar, border_radius=4)
            frac = clamp(time_left / 10.0)
            fill_w = int(bar_w * frac)
            if fill_w > 0:
                pygame.draw.rect(
                    self._screen,
                    col,
                    pygame.Rect(bar.x, bar.y, fill_w, bar.h),
                    border_radius=4,
                )

    @staticmethod
    def _banner_display_text(label: str) -> str:
        return {
            "GREEN": "GREEN LIGHT",
            "RED": "RED LIGHT",
            "TURNING": "TURNING...",
            "COUNTDOWN": "GET READY",
            "WINNER": "WINNER!",
            "CAUGHT": "CAUGHT!",
            "START": "RED LIGHT, GREEN LIGHT",
        }.get(label, label)

    @staticmethod
    def _readable_on(color: Color) -> Color:
        # Relative luminance (0..1)
        r, g, b = (c / 255.0 for c in color)
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return (20, 20, 24) if lum > 0.55 else (245, 245, 250)

    # ------------------------------------------------------------------
    # Top-right status cluster
    # ------------------------------------------------------------------
    def _draw_status_cluster(
        self,
        players_alive: int,
        elapsed: float,
        round_n: int,
        clock: pygame.time.Clock,
    ) -> None:
        x = DISPLAY_W - 260
        y = 60
        pad_x, pad_y = 18, 10

        items: list[tuple[str, Color]] = [
            (f"{int(clock.get_fps())} FPS", MD3_ON_BG_MED),
            (f"{players_alive} PLAYERS", MD3_TERTIARY),
            (f"ROUND {round_n}", MD3_SECONDARY),
            (self._fmt_time(elapsed), MD3_ON_BG),
        ]
        for text, color in items:
            surf = self._font.render(text, 22, MD3_ON_BG, bold=True)
            w = surf.get_width() + pad_x * 2 + 14
            h = surf.get_height() + pad_y * 2
            rect = pygame.Rect(DISPLAY_W - 36 - w, y, w, h)
            bg = pygame.Surface(rect.size, pygame.SRCALPHA)
            pygame.draw.rect(
                bg,
                (*MD3_SURFACE_HIGH, 215),
                bg.get_rect(),
                border_radius=h // 2,
            )
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
    # Motion meter (bottom center during RED)
    # ------------------------------------------------------------------
    def _draw_motion_meter(self, motion_score: float) -> None:
        bar_w = 520
        bar_h = 18
        cx = DISPLAY_W // 2
        y = DISPLAY_H - 80
        rect = pygame.Rect(cx - bar_w // 2, y, bar_w, bar_h)
        draw_rrect(
            self._screen,
            rect,
            MD3_SURFACE_HIGH,
            radius=bar_h // 2,
            alpha=215,
        )
        frac = clamp(motion_score)
        fill_w = int(bar_w * frac)
        if fill_w > 6:
            col = lerp_color(MD3_SUCCESS, MD3_ERROR, frac)
            draw_rrect(
                self._screen,
                pygame.Rect(rect.x, rect.y, fill_w, bar_h),
                col,
                radius=bar_h // 2,
            )
        label = "MOTION"
        surf = self._font.render(label, 16, MD3_ON_BG_MED, bold=True)
        self._screen.blit(
            surf,
            surf.get_rect(midbottom=(cx, y - 6)),
        )

    # ------------------------------------------------------------------
    # Caught flash (center-out ring pulse at person's location)
    # ------------------------------------------------------------------
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
            pygame.draw.circle(
                surf,
                (*MD3_ERROR, alpha),
                (surf_size // 2, surf_size // 2),
                r,
                width=8,
            )
            self._screen.blit(surf, (fl.x - surf_size // 2, fl.y - surf_size // 2))
        self._flashes = survivors

    # ==================================================================
    # Public draw methods called by GameEngine
    # ==================================================================
    def draw_start_screen(
        self,
        frame: np.ndarray | None,
        palm_progress: float,
        clock: pygame.time.Clock,
    ) -> None:
        self.begin_frame()
        self.draw_camera(frame)
        self._draw_vignette(strength=140)

        # Title
        title = "RED LIGHT, GREEN LIGHT"
        title_surf = self._font.render(title, 84, MD3_ON_BG, bold=True)
        self._screen.blit(
            title_surf,
            title_surf.get_rect(center=(DISPLAY_W // 2, 220)),
        )

        sub = "STEM Day · Version 1.0.0"
        sub_surf = self._font.render(sub, 26, MD3_PRIMARY, bold=False)
        self._screen.blit(
            sub_surf,
            sub_surf.get_rect(center=(DISPLAY_W // 2, 280)),
        )

        # Palm raise ring
        cx, cy = DISPLAY_W // 2, DISPLAY_H // 2 + 80
        draw_progress_ring(self._screen, (cx, cy), 140, 18, palm_progress, MD3_PRIMARY)
        inner_label = "HOLD PALM" if palm_progress < 0.99 else "STARTING"
        self._draw_text_center(inner_label, (cx, cy), 28, MD3_ON_BG, bold=True)
        pct = int(palm_progress * 100)
        self._draw_text_center(f"{pct}%", (cx, cy + 36), 22, MD3_ON_BG_MED)

        self._draw_text_center(
            "Raise your hand above your shoulder to start",
            (DISPLAY_W // 2, DISPLAY_H - 140),
            30,
            MD3_ON_BG_MED,
        )
        self._draw_text_center(
            "ESC to quit · CTRL+D for dev mode",
            (DISPLAY_W // 2, DISPLAY_H - 80),
            20,
            MD3_ON_BG_DIM,
        )

    def draw_countdown(
        self,
        frame: np.ndarray | None,
        n: int,
        clock: pygame.time.Clock,
    ) -> None:
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
        self._screen.blit(
            big,
            big.get_rect(center=(DISPLAY_W // 2, DISPLAY_H // 2 + 40)),
        )

    def draw_game_hud(
        self,
        frame: np.ndarray | None,
        pose: dict | None,
        state_label: str,
        alive: int,
        elapsed: float,
        round_n: int,
        caught_log: Iterable[str],
        motion_score: float,
        time_left: float,
        clock: pygame.time.Clock,
    ) -> None:
        self.begin_frame()
        self.draw_camera(frame)
        self.draw_pose(pose)
        self._draw_banner(state_label, time_left)
        self._draw_status_cluster(alive, elapsed, round_n, clock)
        if state_label == "RED":
            self._draw_motion_meter(motion_score)
        self._draw_elimination_log()
        self._draw_flashes()

    def draw_caught_screen(
        self,
        frame: np.ndarray | None,
        colour: str,
        progress: float,
        clock: pygame.time.Clock,
    ) -> None:
        self.begin_frame()
        self.draw_camera(frame)
        # Red tint overlay
        tint = pygame.Surface((DISPLAY_W, DISPLAY_H), pygame.SRCALPHA)
        tint.fill((*MD3_ERROR_BG, 170))
        self._screen.blit(tint, (0, 0))

        self._draw_banner("CAUGHT")

        cx, cy = DISPLAY_W // 2, DISPLAY_H // 2
        # Big rounded card
        card_w, card_h = 720, 360
        card = pygame.Rect(cx - card_w // 2, cy - card_h // 2, card_w, card_h)
        draw_panel(self._screen, card, MD3_SURFACE_HIGH, radius=36, alpha=240)

        # Shirt-color chip
        col = SHIRT_DISPLAY_COLOR.get(colour, MD3_ERROR)
        chip_rect = pygame.Rect(0, 0, 320, 70)
        chip_rect.center = (cx, cy - 60)
        draw_pill(self._screen, chip_rect, col, alpha=255)
        label = f"{colour.upper()} SHIRT" if colour else "PLAYER"
        self._draw_text_center(label, chip_rect.center, 28, self._readable_on(col), bold=True)

        self._draw_text_center("You moved!", (cx, cy + 30), 44, MD3_ON_BG, bold=True)
        self._draw_text_center("Return to the start line", (cx, cy + 80), 26, MD3_ON_BG_MED)

        # Progress bar
        bar_w = 560
        bar = pygame.Rect(cx - bar_w // 2, cy + 130, bar_w, 10)
        draw_rrect(self._screen, bar, MD3_SURFACE_VAR, 5, alpha=220)
        fill_w = int(bar_w * clamp(progress))
        if fill_w > 0:
            draw_rrect(
                self._screen,
                pygame.Rect(bar.x, bar.y, fill_w, bar.h),
                MD3_ERROR,
                5,
            )

    def draw_winner_screen(
        self,
        frame: np.ndarray | None,
        colour: str,
        palm_progress: float,
        clock: pygame.time.Clock,
    ) -> None:
        self.begin_frame()
        if frame is not None:
            self.draw_camera(frame)
            dim = pygame.Surface((DISPLAY_W, DISPLAY_H), pygame.SRCALPHA)
            dim.fill((0, 0, 0, 140))
            self._screen.blit(dim, (0, 0))
        else:
            self._draw_decorative_bg()

        self._update_confetti()
        self._draw_confetti()

        self._draw_text_center("WINNER!", (DISPLAY_W // 2, 260), 160, MD3_PRIMARY, bold=True)

        col = SHIRT_DISPLAY_COLOR.get(colour, MD3_PRIMARY)
        chip_rect = pygame.Rect(0, 0, 420, 88)
        chip_rect.center = (DISPLAY_W // 2, 420)
        draw_pill(self._screen, chip_rect, col)
        self._draw_text_center(
            f"{colour.upper()} SHIRT",
            chip_rect.center,
            36,
            self._readable_on(col),
            bold=True,
        )

        # Restart ring
        cx, cy = DISPLAY_W // 2, 720
        draw_progress_ring(self._screen, (cx, cy), 120, 16, palm_progress, MD3_PRIMARY)
        self._draw_text_center("HOLD PALM", (cx, cy), 24, MD3_ON_BG, bold=True)
        self._draw_text_center(
            "Raise your hand to play again",
            (DISPLAY_W // 2, DISPLAY_H - 90),
            26,
            MD3_ON_BG_MED,
        )

    # ------------------------------------------------------------------
    # Confetti (winner screen)
    # ------------------------------------------------------------------
    def _update_confetti(self) -> None:
        if len(self._confetti) < 180:
            for _ in range(8):
                x = float(np.random.randint(0, DISPLAY_W))
                y = -20.0 - float(np.random.randint(0, DISPLAY_H))
                vx = float(np.random.uniform(-30.0, 30.0))
                vy = float(np.random.uniform(120.0, 260.0))
                palette = [
                    MD3_PRIMARY,
                    MD3_SECONDARY,
                    MD3_TERTIARY,
                    MD3_SUCCESS,
                    MD3_WARNING,
                ]
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

    # ------------------------------------------------------------------
    # Vignette
    # ------------------------------------------------------------------
    def _draw_vignette(self, strength: int = 120) -> None:
        overlay = pygame.Surface((DISPLAY_W, DISPLAY_H), pygame.SRCALPHA)
        # Darken edges: four gradients
        for i in range(8):
            a = int(strength * (i + 1) / 8)
            pad = (8 - i) * 14
            pygame.draw.rect(
                overlay,
                (0, 0, 0, a // 8),
                pygame.Rect(pad, pad, DISPLAY_W - pad * 2, DISPLAY_H - pad * 2),
                width=2,
                border_radius=24,
            )
        # Top + bottom soft bands
        band = pygame.Surface((DISPLAY_W, 180), pygame.SRCALPHA)
        for i in range(180):
            a = int(strength * (1.0 - i / 180.0))
            pygame.draw.rect(band, (0, 0, 0, a), (0, i, DISPLAY_W, 1))
        self._screen.blit(band, (0, 0))
        band2 = pygame.transform.flip(band, False, True)
        self._screen.blit(band2, (0, DISPLAY_H - 180))
        self._screen.blit(overlay, (0, 0))

    # ------------------------------------------------------------------
    # Dev panel
    # ------------------------------------------------------------------
    def draw_dev_panel(self, metrics: dict) -> None:
        """Metrics expected keys: fps, cam_fps, inference_ms, state,
        state_time, servo_angle, servo_target, laser_broken, players,
        dev_hints (list[str])."""
        if not self._dev_mode:
            return
        self._dev_fps_history.append(float(metrics.get("fps", 0)))

        panel_w = 400
        panel = pygame.Rect(DISPLAY_W - panel_w - 24, 24, panel_w, DISPLAY_H - 48)
        draw_panel(self._screen, panel, MD3_SURFACE, radius=28, alpha=235)

        # Header
        header_y = panel.y + 24
        self._draw_text_left("DEV · DEBUG", (panel.x + 24, header_y), 26, MD3_PRIMARY, bold=True)
        self._draw_text_left(
            "CTRL+D to close",
            (panel.x + 24, header_y + 32),
            16,
            MD3_ON_BG_DIM,
        )

        y = header_y + 80

        # --- Performance ---
        y = self._dev_section(panel, "PERFORMANCE", y)
        y = self._dev_kv(panel, "Overall FPS", f"{metrics.get('fps', 0):.1f}", y)
        y = self._dev_kv(panel, "Camera FPS", f"{metrics.get('cam_fps', 0):.1f}", y)
        y = self._dev_kv(
            panel,
            "YOLO inference",
            f"{metrics.get('inference_ms', 0):.1f} ms",
            y,
        )

        # FPS mini graph
        y += 8
        graph = pygame.Rect(panel.x + 24, y, panel_w - 48, 60)
        draw_rrect(self._screen, graph, MD3_SURFACE_HIGH, radius=12, alpha=220)
        if self._dev_fps_history:
            hist = list(self._dev_fps_history)
            mx = max(1.0, max(hist))
            bar_w = max(2, (graph.w - 8) // max(1, len(hist)))
            for i, v in enumerate(hist):
                bh = int((graph.h - 8) * clamp(v / max(30.0, mx)))
                bx = graph.x + 4 + i * bar_w
                by = graph.bottom - 4 - bh
                col = MD3_SUCCESS if v >= 24 else MD3_WARNING if v >= 16 else MD3_ERROR
                pygame.draw.rect(
                    self._screen,
                    col,
                    pygame.Rect(bx, by, bar_w - 1, bh),
                    border_radius=2,
                )
        y = graph.bottom + 20

        # --- State ---
        y = self._dev_section(panel, "STATE", y)
        y = self._dev_kv(panel, "Current", str(metrics.get("state", "?")), y)
        y = self._dev_kv(panel, "Time in state", f"{metrics.get('state_time', 0):.2f} s", y)
        y = self._dev_kv(panel, "Players", str(metrics.get("players", 0)), y)

        # --- Hardware ---
        y = self._dev_section(panel, "HARDWARE", y)
        y = self._dev_kv(
            panel,
            "Servo angle",
            f"{metrics.get('servo_angle', 0):.1f}°",
            y,
        )
        y = self._dev_kv(
            panel,
            "Servo target",
            f"{metrics.get('servo_target', 0):.1f}°",
            y,
        )
        laser = "BROKEN" if metrics.get("laser_broken") else "intact"
        laser_col = MD3_ERROR if metrics.get("laser_broken") else MD3_SUCCESS
        y = self._dev_kv(panel, "Laser beam", laser, y, value_color=laser_col)

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
        pygame.draw.line(
            self._screen,
            MD3_OUTLINE,
            (panel.x + 24, y + 22),
            (panel.right - 24, y + 22),
            1,
        )
        return y + 34

    def _dev_kv(
        self,
        panel: pygame.Rect,
        key: str,
        value: str,
        y: int,
        value_color: Color = MD3_ON_BG,
    ) -> int:
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
    def _draw_text_center(
        self,
        text: str,
        center: tuple[int, int],
        size: int,
        color: Color,
        bold: bool = False,
    ) -> pygame.Rect:
        return draw_text_centered(self._screen, text, center, self._font, size, color, bold)

    def _draw_text_left(
        self,
        text: str,
        pos: tuple[int, int],
        size: int,
        color: Color,
        bold: bool = False,
    ) -> pygame.Rect:
        return draw_text_left(self._screen, text, pos, self._font, size, color, bold)

    # Backwards compat with earlier signature used in game.py
    @property
    def dev_mode(self) -> bool:
        return self._dev_mode
