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

# Module-level cache for soft drop shadows. Keyed by geometry so we only
# pay the Gaussian-blur cost once per unique pill size. The cache is bounded
# (see _SHADOW_CACHE_LIMIT) to prevent unbounded growth if many distinct
# rect sizes are drawn.
_SHADOW_CACHE: "dict[tuple[int, int, int, int, int, int, int], pygame.Surface]" = {}
_SHADOW_CACHE_LIMIT = 64


def _build_shadow_surface(
    w: int,
    h: int,
    radius: int,
    spread: int,
    alpha: int,
) -> pygame.Surface:
    """Return an SRCALPHA surface containing a soft, blurred pill shadow.

    The pill itself sits at (spread, spread) inside the surface; the
    surrounding ``spread`` pixels of padding give the Gaussian blur room
    to fade out cleanly.
    """
    sw = w + spread * 2
    sh = h + spread * 2
    surf = pygame.Surface((sw, sh), pygame.SRCALPHA)
    pygame.draw.rect(
        surf,
        (0, 0, 0, alpha),
        pygame.Rect(spread, spread, w, h),
        border_radius=radius,
    )
    # Blur the alpha channel only — colour is solid black, so we only need
    # the alpha plane to soften. surfarray.pixels_alpha gives a mutable
    # WxH (transposed) view; cv2.GaussianBlur is in-place safe via dst.
    try:
        alpha_view = pygame.surfarray.pixels_alpha(surf)  # shape (sw, sh)
        sigma = max(1.0, spread / 2.0)
        cv2.GaussianBlur(alpha_view, (0, 0), sigmaX=sigma, sigmaY=sigma, dst=alpha_view)
        del alpha_view  # release surface lock
    except Exception:
        # If pixels_alpha isn't available (some pygame builds), fall back
        # to the original multi-pass approach by leaving the rect crisp.
        pass
    return surf


def draw_shadow_rrect(
    target: pygame.Surface,
    rect: pygame.Rect,
    radius: int,
    offset: tuple[int, int] = (0, 8),
    spread: int = 14,
    alpha: int = 130,
) -> None:
    """Soft drop shadow built from a single Gaussian-blurred alpha pill.

    The shadow is rendered ONCE per unique geometry and cached, so calling
    this every frame on a fixed-size banner is effectively free after the
    first frame. The whole shadow is shifted by ``offset`` from ``rect``,
    so by giving a positive ``offset[1]`` we guarantee the shadow sits
    below the rect rather than haloing around it (which is what the old
    multi-pass inflate did, causing the visible misalignment on the
    GREEN/RED LIGHT banner).
    """
    key = (rect.w, rect.h, radius, spread, alpha, 0, 0)
    cached = _SHADOW_CACHE.get(key)
    if cached is None:
        cached = _build_shadow_surface(rect.w, rect.h, radius, spread, alpha)
        if len(_SHADOW_CACHE) >= _SHADOW_CACHE_LIMIT:
            # Evict an arbitrary entry — we don't need true LRU here.
            _SHADOW_CACHE.pop(next(iter(_SHADOW_CACHE)))
        _SHADOW_CACHE[key] = cached
    # Surface contains spread-pixel padding around the pill, so blit at
    # rect.topleft minus spread, plus the requested offset.
    target.blit(cached, (rect.x - spread + offset[0], rect.y - spread + offset[1]))


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
    """High-level rendering facade used by game.py."""

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

        # Camera feed cache: avoid recreating the scale buffer each frame.
        # _cam_full_scaled is for full-screen rendering (game HUD).
        # _cam_box_scaled is for the boxed render on the home / winner screens.
        self._cam_full_scaled: pygame.Surface | None = None
        self._cam_full_size: tuple[int, int] = (0, 0)
        self._cam_box_scaled: pygame.Surface | None = None
        self._cam_box_size: tuple[int, int] = (0, 0)

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
        """Full-screen camera rendering used by the in-game HUD."""
        if frame is None:
            self._screen.fill(MD3_BG)
            self._draw_decorative_bg()
            return
        try:
            # ── PERF (Apr 2026) ───────────────────────────────────────
            # Convert BGR→RGB once, then hand the contiguous numpy buffer
            # straight to pygame via buffer protocol after tobytes().
            # This ensures proper type compatibility with pygame's frombuffer.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            surf = pygame.image.frombuffer(rgb.tobytes(), (w, h), "RGB")
            if (w, h) != (DISPLAY_W, DISPLAY_H):
                if self._cam_full_size != (DISPLAY_W, DISPLAY_H):
                    self._cam_full_scaled = pygame.Surface((DISPLAY_W, DISPLAY_H))
                    self._cam_full_size = (DISPLAY_W, DISPLAY_H)
                pygame.transform.scale(surf, (DISPLAY_W, DISPLAY_H), self._cam_full_scaled)
                assert self._cam_full_scaled is not None
                self._screen.blit(self._cam_full_scaled, (0, 0))
            else:
                self._screen.blit(surf, (0, 0))
        except Exception as exc:
            print(f"[UI ] draw_camera: {exc}")
            self._screen.fill(MD3_BG)

    def draw_camera_in_rect(
        self,
        frame: np.ndarray | None,
        dest: pygame.Rect,
        radius: int = 36,
    ) -> None:
        """Render the camera feed inside a rounded "window" of size ``dest``.

        Used by the home and winner screens, where the camera is no longer
        full-screen but contained on the left side of a 60/40 split.
        Empty / no-frame state shows a placeholder card so the layout
        doesn't collapse.
        """
        # Soft drop shadow + base card so something is visible even if the
        # camera is still warming up.
        draw_shadow_rrect(self._screen, dest, radius, offset=(0, 14), spread=22, alpha=110)

        if frame is None:
            draw_rrect(self._screen, dest, MD3_SURFACE_HIGH, radius, alpha=235)
            placeholder = self._font.render("Camera warming up…", 28, MD3_ON_BG_DIM, bold=False)
            self._screen.blit(
                placeholder,
                placeholder.get_rect(center=dest.center),
            )
            return

        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            fh, fw = rgb.shape[:2]
            surf = pygame.image.frombuffer(rgb.tobytes(), (fw, fh), "RGB")
            target_size = (dest.w, dest.h)
            if self._cam_box_size != target_size:
                self._cam_box_scaled = pygame.Surface(target_size)
                self._cam_box_size = target_size
            pygame.transform.scale(surf, target_size, self._cam_box_scaled)

            # Clip the scaled feed into a rounded rect by using a mask.
            # Build a mask once per dest size and reuse via the cache key
            # (radius + size).
            mask = pygame.Surface(target_size, pygame.SRCALPHA)
            pygame.draw.rect(
                mask,
                (255, 255, 255, 255),
                mask.get_rect(),
                border_radius=radius,
            )
            # Apply mask: copy camera onto a SRCALPHA surface, then BLEND_RGBA_MIN
            # the mask in to clip the corners.
            clipped = pygame.Surface(target_size, pygame.SRCALPHA)
            assert self._cam_box_scaled is not None
            clipped.blit(self._cam_box_scaled, (0, 0))
            clipped.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
            self._screen.blit(clipped, dest.topleft)

            # Subtle outline to make the window read as a "card".
            outline_surf = pygame.Surface(target_size, pygame.SRCALPHA)
            pygame.draw.rect(
                outline_surf,
                (*MD3_OUTLINE, 200),
                outline_surf.get_rect(),
                width=2,
                border_radius=radius,
            )
            self._screen.blit(outline_surf, dest.topleft)
        except Exception as exc:
            print(f"[UI ] draw_camera_in_rect: {exc}")
            draw_rrect(self._screen, dest, MD3_SURFACE_HIGH, radius, alpha=235)

    def _draw_decorative_bg(self) -> None:
        """Animated gradient blobs — the purple-circle background the user
        loves. Renders across the whole screen and is now the BASE layer of
        both the home and winner screens (with the camera box composited on
        top), not just a fallback for "no camera"."""
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
    def draw_pose(
        self,
        pose: dict | None,
        target_rect: pygame.Rect | None = None,
    ) -> None:
        """Render skeleton + chips. If ``target_rect`` is given, pose
        coordinates are mapped into that rectangle (for the boxed home
        screen camera). Default is full-screen mapping (game HUD)."""
        if pose is None:
            return

        fw = pose.get("frame_w", DISPLAY_W) or DISPLAY_W
        fh = pose.get("frame_h", DISPLAY_H) or DISPLAY_H

        if target_rect is None:
            sx = DISPLAY_W / fw
            sy = DISPLAY_H / fh
            offx = 0
            offy = 0
        else:
            sx = target_rect.w / fw
            sy = target_rect.h / fh
            offx = target_rect.x
            offy = target_rect.y

        self._draw_skeletons(pose, sx, sy, offx, offy)
        self._update_chips(pose, sx, sy, offx, offy)
        self._draw_chips()

    def _draw_skeletons(
        self,
        pose: dict,
        sx: float,
        sy: float,
        offx: int = 0,
        offy: int = 0,
    ) -> None:
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
                xa, ya = int(pa[0] * sx) + offx, int(pa[1] * sy) + offy
                xb, yb = int(pb[0] * sx) + offx, int(pb[1] * sy) + offy
                pygame.draw.line(self._screen, col, (xa, ya), (xb, yb), 4)

            # Joints
            for ki, joint in enumerate(kp):
                if joint[2] < 0.3:
                    continue
                if joint[0] == 0 and joint[1] == 0:
                    continue
                jx, jy = int(joint[0] * sx) + offx, int(joint[1] * sy) + offy
                r = 7 if ki in (5, 6, 11, 12) else 5
                pygame.draw.circle(self._screen, MD3_ON_BG, (jx, jy), r)
                pygame.draw.circle(self._screen, col, (jx, jy), r - 2)

    def _update_chips(
        self,
        pose: dict,
        sx: float,
        sy: float,
        offx: int = 0,
        offy: int = 0,
    ) -> None:
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
                nx = kp[KP["nose"]][0] * sx + offx
                ny = kp[KP["nose"]][1] * sy + offy - 80
            elif i < len(boxes):
                b = boxes[i]
                nx = (b[0] + b[2]) / 2.0 * sx + offx
                ny = b[1] * sy + offy - 40
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
            draw_shadow_rrect(self._screen, rect, h // 2, offset=(0, 6), spread=10, alpha=int(alpha * 0.4))
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

    def _draw_banner(
        self,
        label: str,
        time_left: float | None = None,
        time_total: float | None = None,
    ) -> None:
        """Draw the GREEN/RED LIGHT banner with optional phase progress bar.

        ``time_left`` and ``time_total`` together render the under-banner
        bar:
          * frac = clamp(time_left / time_total)
          * fill is anchored to the RIGHT edge of the bar and depletes
            leftward, so it visibly starts completely full at the right.
        If ``time_total`` is omitted (legacy callers), the bar is hidden
        rather than rendered with bogus geometry.
        """
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

        # Soft drop shadow — note offset[1] >= spread/3 so it sits below
        # the pill rather than haloing around it.
        draw_shadow_rrect(self._screen, rect, h // 2, offset=(0, 14), spread=22, alpha=150)

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

        # ── Phase progress bar ────────────────────────────────────────
        # Right-anchored, depletes left. Uses the actual phase duration
        # rather than the old hard-coded 10 s scale.
        if time_left is not None and time_total is not None and time_total > 0.0:
            bar_w = int(base_w * 0.75)
            bar = pygame.Rect(cx - bar_w // 2, rect.bottom + 14, bar_w, 8)

            # Track (full-width unfilled bar)
            pygame.draw.rect(self._screen, MD3_SURFACE_HIGH, bar, border_radius=4)

            frac = clamp(time_left / time_total)
            fill_w = int(bar_w * frac)
            if fill_w > 0:
                # Anchor the fill to the RIGHT edge: as time_left → 0,
                # fill_w shrinks from full width to 0 toward the left.
                fill_rect = pygame.Rect(
                    bar.right - fill_w,
                    bar.y,
                    fill_w,
                    bar.h,
                )
                pygame.draw.rect(
                    self._screen,
                    col,
                    fill_rect,
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
    def _home_layout(self) -> tuple[pygame.Rect, pygame.Rect]:
        """Return (camera_rect, anim_rect) for the 60/40 home layout."""
        margin = 64
        gap = 48
        usable_w = DISPLAY_W - margin * 2 - gap
        cam_w = int(usable_w * HOME_CAM_BOX_FRACTION)
        anim_w = usable_w - cam_w
        # Camera box: keep a roughly 16:9 aspect so the feed isn't stretched.
        cam_h = int(cam_w * 9 / 16)
        cam_h = min(cam_h, DISPLAY_H - 320)
        cam_y = (DISPLAY_H - cam_h) // 2 + 20
        cam_rect = pygame.Rect(margin, cam_y, cam_w, cam_h)
        anim_rect = pygame.Rect(margin + cam_w + gap, cam_y, anim_w, cam_h)
        return cam_rect, anim_rect

    def draw_start_screen(
        self,
        frame: np.ndarray | None,
        palm_progress: float,
        clock: pygame.time.Clock,
    ) -> None:
        """Home / start screen.

        Layout (Apr 2026 redesign):
          * Animated purple decorative background spans the whole screen.
          * Camera feed lives in a rounded "window" on the LEFT (60%).
          * Palm-raise progress ring + instructions on the RIGHT (40%).
        """
        self.begin_frame()

        # Base layer: the purple animated blobs the user wants visible
        # everywhere — drawn unconditionally so it shows behind both the
        # camera box and the right-hand prompt.
        self._draw_decorative_bg()

        cam_rect, anim_rect = self._home_layout()

        # Title across the top, spanning both columns.
        title = "RED LIGHT, GREEN LIGHT"
        title_surf = self._font.render(title, 78, MD3_ON_BG, bold=True)
        self._screen.blit(
            title_surf,
            title_surf.get_rect(center=(DISPLAY_W // 2, 90)),
        )
        sub_surf = self._font.render("STEM Day · Version 1.0.0", 24, MD3_PRIMARY, bold=False)
        self._screen.blit(
            sub_surf,
            sub_surf.get_rect(center=(DISPLAY_W // 2, 138)),
        )

        # ── Left: contained camera window ────────────────────────────
        self.draw_camera_in_rect(frame, cam_rect, radius=36)
        # Tiny "LIVE" pill in the corner of the camera card so the player
        # can tell it's active.
        live_pill = pygame.Rect(cam_rect.x + 20, cam_rect.y + 20, 86, 32)
        live_surf = pygame.Surface(live_pill.size, pygame.SRCALPHA)
        pygame.draw.rect(
            live_surf,
            (*MD3_ERROR, 230),
            live_surf.get_rect(),
            border_radius=16,
        )
        pygame.draw.circle(live_surf, (255, 255, 255, 230), (16, 16), 5)
        self._screen.blit(live_surf, live_pill.topleft)
        live_label = self._font.render("LIVE", 18, (255, 255, 255), bold=True)
        self._screen.blit(
            live_label,
            live_label.get_rect(midleft=(live_pill.x + 30, live_pill.y + 16)),
        )

        # ── Right: palm-raise animation panel ────────────────────────
        # Soft card so the panel sits cleanly on the purple bg.
        draw_shadow_rrect(self._screen, anim_rect, 36, offset=(0, 14), spread=22, alpha=110)
        draw_rrect(self._screen, anim_rect, MD3_SURFACE_HIGH, radius=36, alpha=215)

        # Header
        header = self._font.render("RAISE YOUR HAND", 32, MD3_ON_BG, bold=True)
        self._screen.blit(
            header,
            header.get_rect(center=(anim_rect.centerx, anim_rect.y + 60)),
        )
        sub = self._font.render("to begin", 22, MD3_ON_BG_MED, bold=False)
        self._screen.blit(
            sub,
            sub.get_rect(center=(anim_rect.centerx, anim_rect.y + 96)),
        )

        # Animated palm icon: a simple stylised hand circle that pulses
        # while the user is mid-progress, plus the existing progress ring.
        ring_cx = anim_rect.centerx
        ring_cy = anim_rect.centery + 10
        ring_radius = min(140, anim_rect.w // 3)

        # Soft halo behind the ring that pulses with palm_progress
        halo_t = pulse(time.time(), 1.6) * 0.6 + 0.4
        halo_r = int(ring_radius + 20 + 12 * halo_t)
        halo_surf = pygame.Surface((halo_r * 2, halo_r * 2), pygame.SRCALPHA)
        pygame.draw.circle(
            halo_surf,
            (*MD3_PRIMARY, int(50 + 60 * palm_progress)),
            (halo_r, halo_r),
            halo_r,
        )
        self._screen.blit(halo_surf, (ring_cx - halo_r, ring_cy - halo_r))

        # Progress ring
        draw_progress_ring(
            self._screen,
            (ring_cx, ring_cy),
            ring_radius,
            16,
            palm_progress,
            MD3_PRIMARY,
        )

        # Hand glyph in the centre of the ring (drawn as a few rounded
        # rects — the emoji-free, font-independent version)
        self._draw_palm_glyph(
            (ring_cx, ring_cy),
            scale=ring_radius / 140.0,
            tint=MD3_PRIMARY,
            wave_t=time.time(),
        )

        # Status text under the ring
        inner_label = "HOLD" if palm_progress < 0.99 else "STARTING"
        pct_label = f"{int(palm_progress * 100)}%"
        self._draw_text_center(
            inner_label,
            (ring_cx, ring_cy + ring_radius + 36),
            22,
            MD3_ON_BG,
            bold=True,
        )
        self._draw_text_center(
            pct_label,
            (ring_cx, ring_cy + ring_radius + 64),
            20,
            MD3_ON_BG_MED,
        )

        # Footer hints across the bottom
        self._draw_text_center(
            "Raise your hand above your shoulder to start",
            (DISPLAY_W // 2, DISPLAY_H - 70),
            26,
            MD3_ON_BG_MED,
        )
        self._draw_text_center(
            "ESC to quit · CTRL+D for dev mode",
            (DISPLAY_W // 2, DISPLAY_H - 36),
            18,
            MD3_ON_BG_DIM,
        )

    def _draw_palm_glyph(
        self,
        center: tuple[int, int],
        scale: float = 1.0,
        tint: Color = MD3_PRIMARY,
        wave_t: float = 0.0,
    ) -> None:
        """A simple raised-hand glyph (palm + 5 fingers) that gently waves.

        Drawn programmatically so we don't depend on any specific font's
        emoji glyph being available inside the Docker image.
        """
        cx, cy = center
        # Wave: small horizontal sway
        sway = math.sin(wave_t * 2.0) * 6.0 * scale
        # Palm
        palm_w = int(60 * scale)
        palm_h = int(72 * scale)
        palm = pygame.Rect(0, 0, palm_w, palm_h)
        palm.center = (int(cx + sway), int(cy + 8 * scale))
        draw_rrect(self._screen, palm, tint, radius=int(20 * scale), alpha=240)
        # Wrist band
        wrist = pygame.Rect(0, 0, int(palm_w * 0.7), int(14 * scale))
        wrist.midtop = (palm.centerx, palm.bottom - int(6 * scale))
        draw_rrect(self._screen, wrist, MD3_PRIMARY_CONTAINER, radius=int(7 * scale), alpha=240)
        # Fingers (5)
        finger_w = int(12 * scale)
        finger_h = int(46 * scale)
        thumb_h = int(34 * scale)
        gap = int(3 * scale)
        # Centred fingers row above the palm
        total_w = finger_w * 4 + gap * 3
        start_x = palm.centerx - total_w // 2
        for i in range(4):
            f = pygame.Rect(0, 0, finger_w, finger_h)
            f.midbottom = (start_x + i * (finger_w + gap) + finger_w // 2, palm.top + int(4 * scale))
            draw_rrect(self._screen, f, tint, radius=int(6 * scale), alpha=240)
        # Thumb sticks out to the left
        thumb = pygame.Rect(0, 0, finger_w, thumb_h)
        thumb.midright = (palm.left + int(6 * scale), palm.centery - int(8 * scale))
        draw_rrect(self._screen, thumb, tint, radius=int(6 * scale), alpha=240)

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
        time_total: float | None = None,
    ) -> None:
        self.begin_frame()
        self.draw_camera(frame)
        self.draw_pose(pose)
        self._draw_banner(state_label, time_left, time_total)
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

        # Progress bar (caught-pause countdown). This bar is intentionally
        # left-anchored — it represents elapsed-pause progress, not
        # remaining-time, so growing left→right is correct here.
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
        """Winner screen: same 60/40 layout idiom as the home screen so
        the camera feed of the celebrating winner is contained, with the
        palm-raise restart prompt in the right panel."""
        self.begin_frame()
        self._draw_decorative_bg()

        cam_rect, anim_rect = self._home_layout()
        self._update_confetti()
        self._draw_confetti()

        self.draw_camera_in_rect(frame, cam_rect, radius=36)

        # Winner card on the right
        draw_shadow_rrect(self._screen, anim_rect, 36, offset=(0, 14), spread=22, alpha=110)
        draw_rrect(self._screen, anim_rect, MD3_SURFACE_HIGH, radius=36, alpha=220)

        self._draw_text_center(
            "WINNER!",
            (anim_rect.centerx, anim_rect.y + 90),
            96,
            MD3_PRIMARY,
            bold=True,
        )

        col = SHIRT_DISPLAY_COLOR.get(colour, MD3_PRIMARY)
        chip_rect = pygame.Rect(0, 0, min(360, anim_rect.w - 80), 76)
        chip_rect.center = (anim_rect.centerx, anim_rect.y + 200)
        draw_pill(self._screen, chip_rect, col)
        self._draw_text_center(
            f"{colour.upper()} SHIRT",
            chip_rect.center,
            30,
            self._readable_on(col),
            bold=True,
        )

        # Restart ring
        ring_cx = anim_rect.centerx
        ring_cy = anim_rect.centery + 120
        ring_radius = min(110, anim_rect.w // 4)
        draw_progress_ring(
            self._screen,
            (ring_cx, ring_cy),
            ring_radius,
            14,
            palm_progress,
            MD3_PRIMARY,
        )
        self._draw_text_center("HOLD PALM", (ring_cx, ring_cy), 22, MD3_ON_BG, bold=True)
        self._draw_text_center(
            "Raise your hand to play again",
            (anim_rect.centerx, anim_rect.bottom - 50),
            22,
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

        # FPS mini graph — green threshold raised to 50 since target is 60.
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
