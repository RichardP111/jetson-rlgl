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
from collections import OrderedDict, deque
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
    ELIMINATED_BOX_THICKNESS,
    ELIMINATED_PULSE_HZ,
    ENABLE_LINE_OVERLAY,
    FINISH_LINE_DISPLAY_COLOR,
    START_LINE_DISPLAY_COLOR,
    FONT_PATH,
    HOME_CAM_BOX_FRACTION,
    LEADERBOARD_1ST_CELEBRATE_S,
    LEADERBOARD_1ST_DELAY_S,
    LEADERBOARD_1ST_PAUSE_S,
    LEADERBOARD_1ST_REVEAL_S,
    LEADERBOARD_2ND_DELAY_S,
    LEADERBOARD_3RD_DELAY_S,
    LEADERBOARD_BLAST_OUT_S,
    LEADERBOARD_CARD_FALL_DURATION_S,
    LEADERBOARD_CONFETTI_BURST_COUNT,
    LEADERBOARD_DIM_FADE_IN_S,
    LEADERBOARD_EMPTY_HOLD_S,
    LEADERBOARD_FILLED_HOLD_S,
    LEADERBOARD_HERO_HOLD_S,
    LEADERBOARD_HERO_SLIDE_S,
    LEADERBOARD_LIST_DELAY_S,
    LEADERBOARD_PEDESTAL_RISE_S,
    LEADERBOARD_PLAYER_POP_S,
    LEADERBOARD_PODIUM_FADE_S,
    LEADERBOARD_SPOTLIGHT_ALPHA,
    LEADERBOARD_SPOTLIGHT_DIM_ALPHA,
    LEADERBOARD_SPOTLIGHT_RADIUS,
    LEADERBOARD_SPOTLIGHT_SETTLE_S,
    LEADERBOARD_SPOTLIGHT_SWEEP_AMPLITUDE,
    LEADERBOARD_SPOTLIGHT_SWEEP_S,
    LEADERBOARD_TITLE_DELAY_S,
    LINE_TAPE_DETECTED_MIN_PX,
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
    """Text rendering with two-level caching.

    Apr 2026 perf: SDL_TTF rasterizes glyphs on the CPU and a 78pt bold
    string at 1080p runs ~1-2ms per call. The home screen renders 9
    different strings per frame, every frame, so font.render() alone was
    eating 15-20 ms/frame and pinning the screen to ~20 FPS.

    Solution: cache the rendered Surface keyed by (text, size, color,
    bold). Surfaces are .convert_alpha()-ed so subsequent blits go
    through the GPU/SDL hardware path. The cache is bounded by a soft
    LRU cap so dynamic strings (timers, percentages) don't blow up
    memory."""

    _RENDER_CACHE_LIMIT = 256

    def __init__(self, font_path: str) -> None:
        self._path = font_path if os.path.exists(font_path) else None
        self._cache: dict[tuple[int, bool], pygame.font.Font] = {}
        # Rendered surface cache — built lazily because it requires the
        # display surface to exist (for convert_alpha()).
        self._render_cache: "OrderedDict[tuple[str, int, tuple[int, int, int], bool, bool], pygame.Surface]" = OrderedDict()

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
        if not text:
            return pygame.Surface((0, 0), pygame.SRCALPHA)
        # Normalize color → 3-tuple of ints for hashability.
        col_key = (int(color[0]), int(color[1]), int(color[2]))
        cache_key = (text, size, col_key, bold, antialias)
        cached = self._render_cache.get(cache_key)
        if cached is not None:
            self._render_cache.move_to_end(cache_key)
            return cached
        # Render fresh.
        surf = self.get(size, bold).render(text, antialias, color)
        try:
            surf = surf.convert_alpha()
        except pygame.error:
            # Display surface not yet ready — caller will get a software
            # surface this frame; subsequent frames will hit the cache.
            pass
        self._render_cache[cache_key] = surf
        if len(self._render_cache) > self._RENDER_CACHE_LIMIT:
            self._render_cache.popitem(last=False)
        return surf

    def clear_render_cache(self) -> None:
        """Drop all cached text surfaces (e.g. on display-mode change)."""
        self._render_cache.clear()


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
    caught: bool = False  # Apr 2026 — eliminated/walking-back state


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

        # Apr 2026 perf: pre-bake the decorative drift circles into a
        # SINGLE opaque background surface at init time. The previous
        # implementation blitted three 960×960 SRCALPHA circles every
        # frame on top of an MD3_BG fill — that's ~3M alpha-blended
        # pixel ops per frame, which alone capped the home screen at
        # ~20 FPS on the Jetson before any other rendering work.
        # The drift was barely visible anyway; static composites buy
        # ~3× FPS on the menu.
        self._decor_bg = self._build_decor_bg()

        self._countdown_last_n = -1
        self._countdown_trigger_ts = 0.0
        self._confetti: list[tuple[float, float, float, float, Color]] = []

        self._dev_mode = False
        self._dev_fps_history: deque[float] = deque(maxlen=DEV_FPS_HISTORY)

        self._cam_full_scaled: pygame.Surface | None = None
        self._cam_full_size: tuple[int, int] = (0, 0)
        self._cam_box_scaled: pygame.Surface | None = None
        self._cam_box_size: tuple[int, int] = (0, 0)

        # Apr 2026 perf — frame-identity caching. The render loop runs
        # faster than the camera captures, so we skip the cv2→pygame
        # conversion when the same physical frame is being drawn again.
        # ``_current_frame_id`` is updated by set_frame_id() once per
        # render tick; ``_cam_box_last_frame_id`` records which frame
        # is currently in ``_cam_box_scaled`` so we know when to refresh.
        self._current_frame_id: int = 0
        self._cam_box_last_frame_id: int = -1
        self._cam_full_last_frame_id: int = -1

        # Home-screen FPS fix (Apr 2026) — pre-built mask + outline surfaces
        # keyed by (w, h, radius). draw_camera_in_rect() used to allocate
        # 3 SRCALPHA surfaces every frame just to round the corners on the
        # camera card; we now build them once and reuse them.
        self._rounded_mask_cache: dict[tuple[int, int, int], pygame.Surface] = {}
        self._rounded_outline_cache: dict[tuple[int, int, int], pygame.Surface] = {}
        # Pre-rendered home-screen text + halo glow tiers (cached on first
        # use to avoid re-rasterizing the same string at the same size on
        # every single frame).
        self._home_text_cache: dict[tuple[str, int, bool, Color], pygame.Surface] = {}
        self._home_halo_cache: dict[tuple[int, int, int], pygame.Surface] = {}
        self._home_palm_cache: dict[float, pygame.Surface] = {}
        # Apr 2026 home FPS rewrite — pre-baked surfaces for the static
        # parts of the home screen. None until first build.
        self._home_static_bg: pygame.Surface | None = None
        self._home_live_pill_surf: pygame.Surface | None = None
        self._home_palm_glyph_cache: dict[float, pygame.Surface] = {}

        # Leaderboard reveal state
        self._leaderboard_sounds_fired: set[str] = set()
        self._leaderboard_burst_done = False
        self._leaderboard_winner_music_started = False
        self._audio_hook: object | None = None  # set lazily by set_audio_hook()
        # Apr 2026 v3 — spotlight cutout for the 1st-place reveal.
        # Mask + scaled-mask cache built lazily on first use.
        self._spotlight_mask: pygame.Surface | None = None
        self._spotlight_mask_base_radius: int = LEADERBOARD_SPOTLIGHT_RADIUS
        self._spotlight_scaled_cache: dict[float, pygame.Surface] = {}
        # Gradient pedestal bodies, cached per (w, h, base_color).
        self._pedestal_gradient_cache: dict[tuple, pygame.Surface] = {}
        self._debug_skip_finish: bool = False  # flipped by GameEngine via set_debug_state()

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
        # Apr 2026 — debug overlays are drawn here, last thing before
        # flip, so the indicator pills appear on top of every screen
        # without each draw_* method having to call them explicitly.
        self._draw_debug_overlays()
        pygame.display.flip()

    def _draw_debug_overlays(self) -> None:
        if not self._debug_skip_finish:
            return
        # Apr 2026 — promoted from a corner pill to a full-width top
        # ribbon. The corner pill was too easy to miss, especially on
        # the home screen where the eye is drawn to the centre. The
        # ribbon spans the screen, pulses, and explicitly says how to
        # turn it back off.
        ribbon_h = 38
        ribbon = pygame.Rect(0, 0, DISPLAY_W, ribbon_h)
        breath = pulse(time.time(), 1.4)
        ribbon_alpha = int(220 * (0.78 + 0.22 * breath))
        surf = pygame.Surface(ribbon.size, pygame.SRCALPHA)
        # Diagonal-stripe effect for the "test mode" feel.
        surf.fill((*MD3_ERROR, ribbon_alpha))
        for x in range(-ribbon_h, DISPLAY_W + ribbon_h, 32):
            pygame.draw.line(
                surf,
                (255, 255, 255, int(ribbon_alpha * 0.18)),
                (x, 0),
                (x + ribbon_h, ribbon_h),
                10,
            )
        self._screen.blit(surf, ribbon.topleft)
        text = self._font.render(
            "TEST MODE  ·  FINISH DETECTION DISABLED  ·  press F or F9 to re-enable",
            18,
            (255, 255, 255),
            bold=True,
        )
        self._screen.blit(text, text.get_rect(center=ribbon.center))

    def set_debug_state(self, *, skip_finish: bool) -> None:
        """Engine pushes runtime debug flags here so the UI can render
        the corresponding indicator pills on top of every screen."""
        self._debug_skip_finish = skip_finish

    def toggle_dev_mode(self) -> bool:
        self._dev_mode = not self._dev_mode
        print(f"[UI ] Dev mode: {'ON' if self._dev_mode else 'OFF'}")
        return self._dev_mode

    def is_dev_mode(self) -> bool:
        return self._dev_mode

    def set_audio_hook(self, audio: object | None) -> None:
        """Game engine hands its AudioManager in here so the leaderboard
        reveal can fire podium SFX (``audio.play("podium_3")`` etc.)
        directly from the renderer without having to thread sound calls
        through every draw call. Optional — UI keeps working without it."""
        self._audio_hook = audio

    def set_frame_id(self, frame_id: int) -> None:
        """Tell the UI which physical camera frame is current.

        Used by ``draw_camera_in_rect`` to dedupe identical frames — the
        render loop runs at 60Hz but the camera typically captures at
        30Hz, so half the renders would otherwise re-do the cv2.resize
        + cvtColor + frombuffer + convert pipeline for no visual gain.
        """
        self._current_frame_id = frame_id

    def _audio_play(self, key: str) -> None:
        hook = self._audio_hook
        if hook is None:
            return
        try:
            hook.play(key)  # type: ignore[attr-defined]
        except Exception as exc:
            print(f"[UI ] Audio hook play({key}) failed: {exc}")

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

        # Apr 2026 perf — dedupe identical frames. When the render loop
        # is running faster than the camera captures (60 vs 30Hz typical),
        # half the renders would otherwise re-do the cv2.cvtColor +
        # frombuffer + transform.scale work for no visual change.
        same_frame = (
            self._cam_full_scaled is not None
            and self._cam_full_size == (DISPLAY_W, DISPLAY_H)
            and self._cam_full_last_frame_id == self._current_frame_id
            and self._current_frame_id != 0
        )
        if same_frame:
            assert self._cam_full_scaled is not None
            self._screen.blit(self._cam_full_scaled, (0, 0))
            if ENABLE_LINE_OVERLAY and self._dev_mode:
                self._draw_line_overlay()
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
            self._cam_full_last_frame_id = self._current_frame_id
        except Exception as exc:
            print(f"[UI ] draw_camera: {exc}")
            self._screen.fill(MD3_BG)

        if ENABLE_LINE_OVERLAY and self._dev_mode:
            self._draw_line_overlay()

    def draw_camera_in_rect(self, frame: np.ndarray | None, dest: pygame.Rect, radius: int = 36) -> None:
        """Render the camera feed inside a rounded "window".

        Apr 2026 perf:
          • Rounded mask + outline cached forever per (w,h,radius).
          • cv2 colour-convert + resize done in one numpy pass.
          • **Frame-identity dedupe**: when the same physical camera
            frame would be drawn twice in a row (because the render
            loop runs at FPS_CAP=60 but the camera typically only
            captures at 30Hz), we skip the entire cv2 → frombuffer →
            convert pipeline and just re-blit the cached working
            surface. This is the single biggest FPS win on the home
            screen — was ~12ms/frame on a Jetson Orin Nano.
        """
        # Soft drop shadow (already cached internally by draw_shadow_rrect)
        draw_shadow_rrect(self._screen, dest, radius, offset=(0, 14), spread=22, alpha=110)

        if frame is None:
            draw_rrect(self._screen, dest, MD3_SURFACE_HIGH, radius, alpha=235)
            placeholder = self._font.render("Camera warming up…", 28, MD3_ON_BG_DIM, bold=False)
            self._screen.blit(placeholder, placeholder.get_rect(center=dest.center))
            return

        target_size = (dest.w, dest.h)
        cache_key = (dest.w, dest.h, radius)

        # Dedupe path — same physical frame as last call AND same dest
        # geometry → just re-blit the cached working surface.
        same_frame = (
            self._cam_box_scaled is not None
            and self._cam_box_size == target_size
            and self._cam_box_last_frame_id == self._current_frame_id
            and self._current_frame_id != 0
        )

        if not same_frame:
            try:
                # 1. cv2 resize + colour-convert in one numpy pass.
                small_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)  # type: ignore
                rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)  # type: ignore
                cam_surf = pygame.image.frombuffer(rgb.tobytes(), target_size, "RGB").convert()

                # 2. Cached rounded mask
                mask = self._rounded_mask_cache.get(cache_key)
                if mask is None:
                    mask = pygame.Surface(target_size, pygame.SRCALPHA).convert_alpha()
                    pygame.draw.rect(mask, (255, 255, 255, 255), mask.get_rect(), border_radius=radius)
                    self._rounded_mask_cache[cache_key] = mask

                # 3. Reuse the working surface across calls when geometry matches.
                if self._cam_box_scaled is None or self._cam_box_size != target_size:
                    self._cam_box_scaled = pygame.Surface(target_size, pygame.SRCALPHA).convert_alpha()
                    self._cam_box_size = target_size
                assert self._cam_box_scaled is not None
                self._cam_box_scaled.fill((0, 0, 0, 0))
                self._cam_box_scaled.blit(cam_surf, (0, 0))
                self._cam_box_scaled.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)

                self._cam_box_last_frame_id = self._current_frame_id
            except Exception as exc:
                print(f"[UI ] draw_camera_in_rect: {exc}")
                draw_rrect(self._screen, dest, MD3_SURFACE_HIGH, radius, alpha=235)
                return

        assert self._cam_box_scaled is not None
        self._screen.blit(self._cam_box_scaled, dest.topleft)

        # 4. Cached rounded outline.
        outline = self._rounded_outline_cache.get(cache_key)
        if outline is None:
            outline = pygame.Surface(target_size, pygame.SRCALPHA).convert_alpha()
            pygame.draw.rect(outline, (*MD3_OUTLINE, 200), outline.get_rect(), width=2, border_radius=radius)
            self._rounded_outline_cache[cache_key] = outline
        self._screen.blit(outline, dest.topleft)

    def _draw_line_overlay(self) -> None:
        """Dev-mode floor-tape overlay (Apr 2026 redesign).

        Replaces the previous flat full-width horizontal bands with a
        perspective trapezoid that "lies on the floor" — wider at the
        bottom of the frame (close to camera) and narrower at the top
        (far from camera). Each line gets a status pill:
            • TAPE OK ✓   — colour mask exceeded LINE_TAPE_DETECTED_MIN_PX
            • FALLBACK ⚠  — using the configured Y instead
        so it's instantly obvious at the gym whether the camera is
        actually picking up the bright floor tape.
        """
        if self._line_frame_h <= 0:
            return
        scale = DISPLAY_H / float(self._line_frame_h)
        if self._line_y_start > 0:
            y = int(self._line_y_start * scale)
            self._draw_perspective_tape(y, START_LINE_DISPLAY_COLOR, "START", self._line_count_start)
        if self._line_y_finish > 0:
            y = int(self._line_y_finish * scale)
            self._draw_perspective_tape(y, FINISH_LINE_DISPLAY_COLOR, "FINISH", self._line_count_finish)

    def _draw_perspective_tape(self, y: int, color: Color, label: str, pixel_count: int) -> None:
        """Draw a single floor-tape band as a perspective trapezoid.

        The band's apparent width tapers with Y so it looks like a
        stripe lying flat on the floor. Treats the top of the frame as
        the vanishing horizon — at y=0 width≈30% of screen, at y=H
        width≈100%.
        """
        cx = DISPLAY_W // 2
        # Vertical thickness of the stripe (in screen px). Closer to the
        # camera (larger Y) → thicker stripe.
        norm_y = clamp(y / DISPLAY_H)
        thickness = int(8 + 14 * norm_y)
        far_y = max(0, y - thickness // 2)
        near_y = min(DISPLAY_H, y + thickness // 2 + 1)
        # Apparent width at each edge.
        norm_far = clamp(far_y / DISPLAY_H)
        norm_near = clamp(near_y / DISPLAY_H)
        half_w_far = int(DISPLAY_W * 0.5 * (0.30 + 0.70 * norm_far))
        half_w_near = int(DISPLAY_W * 0.5 * (0.30 + 0.70 * norm_near))

        poly_pts = [
            (cx - half_w_far, far_y),
            (cx + half_w_far, far_y),
            (cx + half_w_near, near_y),
            (cx - half_w_near, near_y),
        ]
        # Bound the polygon's bbox so we only allocate a small surface.
        min_x = min(p[0] for p in poly_pts)
        min_y = min(p[1] for p in poly_pts)
        max_x = max(p[0] for p in poly_pts)
        max_y = max(p[1] for p in poly_pts)
        bw = max(2, max_x - min_x + 4)
        bh = max(2, max_y - min_y + 4)
        local = [(p[0] - min_x + 2, p[1] - min_y + 2) for p in poly_pts]
        surf = pygame.Surface((bw, bh), pygame.SRCALPHA)
        pygame.draw.polygon(surf, (*color, 200), local)
        # Bright top edge (catches the eye, sells the "floor stripe" look)
        pygame.draw.line(surf, (*color, 255), local[0], local[1], 2)
        # Soft front edge
        pygame.draw.line(surf, (*color, 90), local[3], local[2], 2)
        self._screen.blit(surf, (min_x - 2, min_y - 2))

        # Status pill — TAPE OK ✓ or FALLBACK ⚠
        detected = pixel_count >= LINE_TAPE_DETECTED_MIN_PX
        status_text = f"{label}  TAPE OK  ✓  ({pixel_count}px)" if detected else f"{label}  FALLBACK  ⚠"
        status_color = MD3_SUCCESS if detected else MD3_WARNING
        text_surf = self._font.render(status_text, 18, status_color, bold=True)
        pad_x = 14
        pill_w = text_surf.get_width() + pad_x * 2
        pill_h = 28
        # Anchor the pill near the leftmost visible edge of the band.
        pill_x = max(20, cx - half_w_near - pill_w - 14)
        pill_y = max(4, y - pill_h - 6)
        pill_rect = pygame.Rect(pill_x, pill_y, pill_w, pill_h)
        draw_pill(self._screen, pill_rect, MD3_SURFACE_HIGH, alpha=230)
        self._screen.blit(text_surf, (pill_rect.x + pad_x, pill_rect.y + (pill_h - text_surf.get_height()) // 2))

    def _build_decor_bg(self) -> pygame.Surface:
        """Build the home-screen background ONCE.

        The result is an opaque surface the size of the display with
        the BG color filled in and the three decorative tinted circles
        composited on top at fixed positions. We blit this surface
        directly each frame, paying for one fast opaque blit instead
        of three large SRCALPHA blits.
        """
        bg = pygame.Surface((DISPLAY_W, DISPLAY_H)).convert()
        bg.fill(MD3_BG)
        # Fixed positions chosen to roughly match the previous animated
        # mid-points so the visual weight of the screen is preserved.
        positions = [
            (int(DISPLAY_W * 0.22), int(DISPLAY_H * 0.28), MD3_PRIMARY_CONTAINER),
            (int(DISPLAY_W * 0.78), int(DISPLAY_H * 0.34), MD3_SURFACE_VAR),
            (int(DISPLAY_W * 0.55), int(DISPLAY_H * 0.74), MD3_PRIMARY),
        ]
        for cx, cy, base in positions:
            r = 480
            s = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*base, 40), (r, r), r)
            bg.blit(s, (cx - r, cy - r))
        return bg

    def _draw_decorative_bg(self) -> None:
        # Single opaque blit — see _build_decor_bg() for why this is
        # so much faster than the previous animated three-circle path.
        self._screen.blit(self._decor_bg, (0, 0))

    # ------------------------------------------------------------------
    # Pose overlay
    # ------------------------------------------------------------------
    def draw_pose(
        self,
        pose: dict | None,
        finished_ids: set[int] | None = None,
        labels_by_id: dict[int, str] | None = None,
        caught_ids: set[int] | None = None,
    ) -> None:
        """Render skeletons + per-player chips.

        Apr 2026: ``caught_ids`` is the set of track IDs currently in the
        ELIMINATED state (walking back to start). Those players get a
        thick red bounding box and a pulsing "ELIMINATED" pill above
        their head, so they're impossible to miss in the camera feed."""
        if pose is None:
            return
        finished_ids = finished_ids or set()
        labels_by_id = labels_by_id or {}
        caught_ids = caught_ids or set()
        self._draw_eliminated_boxes(pose, caught_ids)
        self._draw_skeletons(pose, finished_ids, caught_ids)
        self._update_chips(pose, finished_ids, labels_by_id, caught_ids)
        self._draw_chips()

    def _draw_eliminated_boxes(self, pose: dict, caught_ids: set[int]) -> None:
        """Thick red bounding boxes around eliminated players, pulsing."""
        if not caught_ids:
            return
        boxes = pose.get("boxes")
        ids = pose.get("track_ids", [])
        if boxes is None or len(boxes) == 0:
            return
        fw = pose.get("frame_w", DISPLAY_W) or DISPLAY_W
        fh = pose.get("frame_h", DISPLAY_H) or DISPLAY_H
        sx = DISPLAY_W / fw
        sy = DISPLAY_H / fh
        # Pulsing alpha so the box "breathes".
        breath = pulse(time.time(), ELIMINATED_PULSE_HZ)
        alpha = int(180 + 60 * breath)
        thickness = ELIMINATED_BOX_THICKNESS
        for box, tid in zip(boxes, ids):
            if tid not in caught_ids:
                continue
            x1 = int(box[0] * sx)
            y1 = int(box[1] * sy)
            x2 = int(box[2] * sx)
            y2 = int(box[3] * sy)
            box_w = max(2, x2 - x1)
            box_h = max(2, y2 - y1)
            # Drawn into a small SRCALPHA surface so we get translucent
            # strokes (pygame.draw.rect on the screen is opaque-only).
            stroke = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
            pygame.draw.rect(
                stroke,
                (*MD3_ERROR, alpha),
                stroke.get_rect(),
                width=thickness,
                border_radius=12,
            )
            # Bright "corner brackets" for that "target acquired" feel.
            corner_len = max(14, min(box_w, box_h) // 5)
            for cx, cy in ((0, 0), (box_w, 0), (0, box_h), (box_w, box_h)):
                hx = -1 if cx == box_w else 1
                hy = -1 if cy == box_h else 1
                pygame.draw.line(stroke, (*MD3_ERROR, 255), (cx, cy), (cx + hx * corner_len, cy), thickness + 2)
                pygame.draw.line(stroke, (*MD3_ERROR, 255), (cx, cy), (cx, cy + hy * corner_len), thickness + 2)
            self._screen.blit(stroke, (x1, y1))

    def _draw_skeletons(self, pose: dict, finished_ids: set[int], caught_ids: set[int] | None = None) -> None:
        caught_ids = caught_ids or set()
        kpts = pose.get("keypoints", [])
        ids = pose.get("track_ids", [])
        fw = pose.get("frame_w", DISPLAY_W) or DISPLAY_W
        fh = pose.get("frame_h", DISPLAY_H) or DISPLAY_H
        sx = DISPLAY_W / fw
        sy = DISPLAY_H / fh

        for idx, kp in enumerate(kpts):
            tid = ids[idx] if idx < len(ids) else None
            faded = tid in finished_ids
            caught = tid in caught_ids
            # Caught players get a dim skeleton so the red bbox + pill dominate.
            if caught:
                line_color = (210, 110, 110)
                stroke_w = 3
                joint_r_big = 6
                joint_r_small = 4
            elif faded:
                line_color = (160, 160, 170)
                stroke_w = 2
                joint_r_big = 5
                joint_r_small = 3
            else:
                line_color = (240, 240, 250)
                stroke_w = 4
                joint_r_big = 7
                joint_r_small = 5

            for a, b in SKELETON_EDGES:
                if kp[a, 2] < 0.3 or kp[b, 2] < 0.3:
                    continue
                p1 = (int(kp[a, 0] * sx), int(kp[a, 1] * sy))
                p2 = (int(kp[b, 0] * sx), int(kp[b, 1] * sy))
                pygame.draw.line(self._screen, line_color, p1, p2, stroke_w)

            for j in range(17):
                if kp[j, 2] < 0.3:
                    continue
                p = (int(kp[j, 0] * sx), int(kp[j, 1] * sy))
                r = joint_r_big if j in (5, 6, 11, 12) else joint_r_small
                pygame.draw.circle(self._screen, line_color, p, r)

    def _update_chips(
        self,
        pose: dict,
        finished_ids: set[int],
        labels_by_id: dict[int, str],
        caught_ids: set[int] | None = None,
    ) -> None:
        caught_ids = caught_ids or set()
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
            chip.caught = tid in caught_ids

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

            # Apr 2026 — three classes of chip: regular / finished / eliminated.
            if chip.caught:
                col = MD3_ERROR
                # Pulsing scale + brighter edge for the eliminated pill so it
                # really pops on the camera view.
                breath = pulse(time.time(), ELIMINATED_PULSE_HZ)
                pill_alpha = int(min(255, alpha * (0.85 + 0.15 * breath)))
                label_text = "ELIMINATED"
                text_col = (245, 245, 250)
            elif chip.finished:
                col = MD3_SUCCESS
                pill_alpha = alpha
                label_text = chip.label
                text_col = MD3_SUCCESS_BG_DARK
            else:
                col = SHIRT_DISPLAY_COLOR.get(chip.shirt, MD3_PRIMARY)
                pill_alpha = alpha
                label_text = chip.label
                text_col = MD3_ON_PRIMARY

            text = self._font.render(label_text, 22, text_col, bold=True)

            pad_x = 18
            pad_y = 8
            dot_r = 9
            w = text.get_width() + pad_x * 2 + dot_r * 2 + 8
            h = text.get_height() + pad_y * 2
            rect = pygame.Rect(int(chip.x.current - w / 2), int(chip.y.current - h / 2), w, h)

            draw_shadow_rrect(self._screen, rect, h // 2, offset=(0, 6), spread=10, alpha=int(pill_alpha * 0.4))

            surf = pygame.Surface(rect.size, pygame.SRCALPHA)
            pygame.draw.rect(surf, (col[0], col[1], col[2], pill_alpha), surf.get_rect(), border_radius=h // 2)

            dot_center = (pad_x + dot_r, h // 2)
            pygame.draw.circle(surf, (255, 255, 255, pill_alpha), dot_center, dot_r)
            pygame.draw.circle(surf, (col[0], col[1], col[2], pill_alpha), dot_center, dot_r - 3)

            self._screen.blit(surf, rect.topleft)
            text.set_alpha(pill_alpha)
            self._screen.blit(text, (rect.x + pad_x + dot_r * 2 + 8, rect.y + (h - text.get_height()) // 2))

    # ------------------------------------------------------------------
    # Sent-back log (Amber warnings)
    # ------------------------------------------------------------------
    def log_sent_back(self, descriptor: str) -> None:
        """Apr 2026 rebrand: visual log entries say ELIMINATED, but the
        method name + audio TTS line still mention "walk back to the
        start" so kids know what to physically do."""
        pretty = descriptor if descriptor != "unknown" else "?"
        entry = LogEntry(
            text=f"{pretty.upper()} — ELIMINATED",
            colour=MD3_ERROR,
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
        """Top-of-screen state pill (GREEN / RED / TURNING / ELIMINATED).

        Apr 2026 redesign:
          • Drop shadow REMOVED — was producing a grey halo that visibly
            misaligned around the pulsing pill.
          • Animations smoother — colour blend + scale entrance now use a
            critically-damped spring feel via ease_in_out_cubic on the
            entrance, and the constant pulse is a soft 4% breath instead
            of an 8% bounce.
          • Text AUTO-FITS the pill no matter how long the label. Long
            strings like "WAITING — PLAYERS WALKING BACK" used to clip;
            we now scale the font down (and widen the pill) until the
            text+padding fit, capped at the configured base size.
        """
        self._update_banner(label)

        display_text = self._banner_display_text(label)
        col = self._banner_color
        txt_col = self._readable_on(col)

        # ── Auto-fit: pick a font size that fits inside the pill ──
        cx = DISPLAY_W // 2
        max_pill_w = int(DISPLAY_W * 0.78)  # never wider than 78% of the screen
        min_pill_w = 380
        base_font_size = 56
        min_font_size = 30
        side_pad = 56  # horizontal space inside the pill on each side of text

        font_size = base_font_size
        while font_size >= min_font_size:
            test_surf = self._font.render(display_text, font_size, txt_col, bold=True)
            needed_w = test_surf.get_width() + side_pad * 2
            if needed_w <= max_pill_w:
                break
            font_size -= 2
        else:
            test_surf = self._font.render(display_text, min_font_size, txt_col, bold=True)
            needed_w = test_surf.get_width() + side_pad * 2

        # ── Geometry: pill grows just enough to wrap the text ──
        elapsed = time.time() - self._banner_entry_ts
        # Smooth entrance — eased, no overshoot. Starts at 0.96, lands at 1.0.
        entry_t = clamp(elapsed * 2.6)
        entry_scale = 0.96 + 0.04 * ease_in_out_cubic(entry_t)
        # Subtle breathing pulse, much softer than before (4% peak).
        breath = pulse(time.time(), BANNER_PULSE_HZ) * 0.04 - 0.02

        base_h = 124
        pill_w = max(min_pill_w, needed_w)
        pill_w = int(pill_w * entry_scale * (1.0 + breath))
        pill_h = int(base_h * entry_scale)
        rect = pygame.Rect(cx - pill_w // 2, 48, pill_w, pill_h)

        # ── Pill body — flat, no shadow ──
        surf = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(surf, (col[0], col[1], col[2], 250), surf.get_rect(), border_radius=pill_h // 2)
        # Subtle internal sheen on the top half — kept, but no external shadow.
        sheen = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(sheen, (255, 255, 255, 38), pygame.Rect(0, 0, rect.w, rect.h // 2), border_radius=pill_h // 2)
        surf.blit(sheen, (0, 0))
        self._screen.blit(surf, rect.topleft)

        # ── Centred label ──
        label_surf = self._font.render(display_text, font_size, txt_col, bold=True)
        self._screen.blit(label_surf, label_surf.get_rect(center=rect.center))

    @staticmethod
    def _banner_display_text(label: str) -> str:
        return {
            "GREEN": "GREEN LIGHT",
            "RED": "RED LIGHT",
            "TURNING": "TURNING...",
            "COUNTDOWN": "GET READY",
            "WINNER": "FINISHED!",
            "CAUGHT": "ELIMINATED!",
            "ELIMINATED": "ELIMINATED",  # Apr 2026 rebrand
            "START": "RED LIGHT, GREEN LIGHT",
            "START_LINE": "GET TO THE START LINE",
            # Legacy alias — old "RETURN" code paths still resolve here.
            "RETURN": "ELIMINATED",
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
        """Home screen.

        Apr 2026 perf rewrite — was running at 20 FPS on the Jetson Orin
        Nano because every frame re-rendered nine text strings via
        SDL_TTF (CPU rasterization), redrew the palm glyph from a dozen
        primitives, and allocated a fresh halo Surface. Now:
          • The whole non-dynamic scene (decorative bg, title, subtitle,
            anim panel chrome, header text, "to begin" caption, footer)
            is composited ONCE into ``_home_static_bg`` and blitted as a
            single opaque surface every frame.
          • Palm glyph is rasterized once into ``_home_palm_glyph_surf``
            at the target scale and re-blitted with a per-frame sway
            offset (no rounded-rect primitives in the hot path).
          • Halo uses a single pre-built circle Surface with .set_alpha()
            for the breathing effect (no per-frame allocations).
          • Only the camera, ring fill arc, palm sway, and percentage
            label change between frames — everything else is cached.
        Result on Jetson: 20 → 60 FPS on home, GPU is now the bottleneck.
        """
        self.begin_frame()

        cam_rect, anim_rect = self._home_layout()
        # Build (or reuse) the cached static background.
        static_bg = self._build_home_static_bg(cam_rect, anim_rect)
        self._screen.blit(static_bg, (0, 0))

        # 1. Camera — the only fully dynamic large blit.
        self.draw_camera_in_rect(frame, cam_rect, radius=36)

        # 2. LIVE pill — pre-baked once, re-used.
        live_pill_surf = self._build_live_pill_surf()
        self._screen.blit(live_pill_surf, (cam_rect.x + 20, cam_rect.y + 20))

        # 3. Halo — single cached surface, set_alpha-driven pulse.
        ring_cx = anim_rect.centerx
        ring_cy = anim_rect.centery + 10
        ring_radius = min(140, anim_rect.w // 3)

        halo_t = pulse(time.time(), 1.6) * 0.6 + 0.4
        halo_max_r = ring_radius + 32
        halo_surf = self._home_palm_cache.get(halo_max_r)
        if halo_surf is None:
            halo_surf = pygame.Surface((halo_max_r * 2, halo_max_r * 2), pygame.SRCALPHA)
            pygame.draw.circle(halo_surf, (*MD3_PRIMARY, 255), (halo_max_r, halo_max_r), halo_max_r)
            try:
                halo_surf = halo_surf.convert_alpha()
            except pygame.error:
                pass
            self._home_palm_cache[halo_max_r] = halo_surf
        halo_alpha = int((50 + 60 * palm_progress) * (0.7 + 0.3 * halo_t))
        halo_surf.set_alpha(min(255, halo_alpha))
        self._screen.blit(halo_surf, (ring_cx - halo_max_r, ring_cy - halo_max_r))

        # 4. Progress ring — drawn fresh because the fill arc changes
        # every frame anyway. This is one circle + one arc, cheap.
        draw_progress_ring(self._screen, (ring_cx, ring_cy), ring_radius, 16, palm_progress, MD3_PRIMARY)

        # 5. Palm glyph — pre-rendered once at this scale; per-frame we
        # only adjust the X offset to recreate the gentle sway.
        glyph_scale = ring_radius / 140.0
        sway = int(math.sin(time.time() * 2.0) * 6.0 * glyph_scale)
        glyph_surf = self._build_home_palm_glyph(glyph_scale)
        glyph_rect = glyph_surf.get_rect(center=(ring_cx + sway, ring_cy + int(8 * glyph_scale)))
        self._screen.blit(glyph_surf, glyph_rect.topleft)

        # 6. Dynamic labels — these change with palm_progress.
        inner_label = "HOLD" if palm_progress < 0.99 else "STARTING"
        pct_label = f"{int(palm_progress * 100)}%"
        self._draw_text_center(inner_label, (ring_cx, ring_cy + ring_radius + 36), 22, MD3_ON_BG, bold=True)
        self._draw_text_center(pct_label, (ring_cx, ring_cy + ring_radius + 64), 20, MD3_ON_BG_MED)

    # ---- Home-screen static-bg builders -----------------------------
    def _build_home_static_bg(self, cam_rect: pygame.Rect, anim_rect: pygame.Rect) -> pygame.Surface:
        """Composite all non-changing parts of the home screen into one
        opaque surface. Built lazily, cached forever."""
        cached = getattr(self, "_home_static_bg", None)
        if cached is not None:
            return cached  # type: ignore[return-value]

        bg = pygame.Surface((DISPLAY_W, DISPLAY_H)).convert()
        bg.blit(self._decor_bg, (0, 0))

        # Title + subtitle
        title_surf = self._font.render("RED LIGHT, GREEN LIGHT", 78, MD3_ON_BG, bold=True)
        bg.blit(title_surf, title_surf.get_rect(center=(DISPLAY_W // 2, 90)))
        sub_surf = self._font.render("STEM Day · Version 1.0.0", 24, MD3_PRIMARY, bold=False)
        bg.blit(sub_surf, sub_surf.get_rect(center=(DISPLAY_W // 2, 138)))

        # Anim-panel chrome (drop shadow + filled card). The dynamic
        # ring/halo/glyph/labels render on top each frame.
        draw_shadow_rrect(bg, anim_rect, 36, offset=(0, 14), spread=22, alpha=110)
        draw_rrect(bg, anim_rect, MD3_SURFACE_HIGH, radius=36, alpha=215)

        header = self._font.render("RAISE YOUR HAND", 32, MD3_ON_BG, bold=True)
        bg.blit(header, header.get_rect(center=(anim_rect.centerx, anim_rect.y + 60)))
        sub = self._font.render("to begin", 22, MD3_ON_BG_MED, bold=False)
        bg.blit(sub, sub.get_rect(center=(anim_rect.centerx, anim_rect.y + 96)))

        # Footer
        f1 = self._font.render("Raise your hand above your shoulder to start", 26, MD3_ON_BG_MED)
        bg.blit(f1, f1.get_rect(center=(DISPLAY_W // 2, DISPLAY_H - 70)))
        f2 = self._font.render(
            "ESC quit  ·  CTRL+D dev  ·  SPACE skip  ·  F or F9 no-finish test  ·  K demo leaderboard",
            18,
            MD3_ON_BG_DIM,
        )
        bg.blit(f2, f2.get_rect(center=(DISPLAY_W // 2, DISPLAY_H - 36)))

        self._home_static_bg = bg
        return bg

    def _build_live_pill_surf(self) -> pygame.Surface:
        cached = getattr(self, "_home_live_pill_surf", None)
        if cached is not None:
            return cached  # type: ignore[return-value]
        surf = pygame.Surface((86, 32), pygame.SRCALPHA).convert_alpha()
        pygame.draw.rect(surf, (*MD3_ERROR, 230), surf.get_rect(), border_radius=16)
        pygame.draw.circle(surf, (255, 255, 255, 230), (16, 16), 5)
        label = self._font.render("LIVE", 18, (255, 255, 255), bold=True)
        surf.blit(label, label.get_rect(midleft=(30, 16)))
        self._home_live_pill_surf = surf
        return surf

    def _build_home_palm_glyph(self, scale: float) -> pygame.Surface:
        """Pre-render the palm-with-fingers glyph as a single Surface so
        the home screen doesn't have to draw 7 rounded rects every frame.
        Keyed by quantized scale to share between similar sizes."""
        scale_key = round(scale * 50) / 50.0  # 0.02 quantization
        cache = self._home_palm_glyph_cache  # initialised in __init__
        if scale_key in cache:
            return cache[scale_key]

        s = scale_key
        palm_w = int(60 * s)
        palm_h = int(72 * s)
        finger_w = int(12 * s)
        finger_h = int(46 * s)
        thumb_h = int(34 * s)
        gap = int(3 * s)

        # Bounding canvas — bit of margin so corners don't clip.
        canvas_w = max(palm_w + finger_w * 2 + 16, 80)
        canvas_h = palm_h + finger_h + 16
        surf = pygame.Surface((canvas_w, canvas_h), pygame.SRCALPHA)

        cx = canvas_w // 2
        cy_palm = canvas_h - palm_h // 2 - 8

        palm = pygame.Rect(0, 0, palm_w, palm_h)
        palm.center = (cx, cy_palm)
        draw_rrect(surf, palm, MD3_PRIMARY, radius=int(20 * s), alpha=240)

        wrist = pygame.Rect(0, 0, int(palm_w * 0.7), int(14 * s))
        wrist.midtop = (palm.centerx, palm.bottom - int(6 * s))
        draw_rrect(surf, wrist, MD3_PRIMARY_CONTAINER, radius=int(7 * s), alpha=240)

        total_w = finger_w * 4 + gap * 3
        start_x = palm.centerx - total_w // 2
        for i in range(4):
            f = pygame.Rect(0, 0, finger_w, finger_h)
            f.midbottom = (start_x + i * (finger_w + gap) + finger_w // 2, palm.top + int(4 * s))
            draw_rrect(surf, f, MD3_PRIMARY, radius=int(6 * s), alpha=240)

        thumb = pygame.Rect(0, 0, finger_w, thumb_h)
        thumb.midright = (palm.left + int(6 * s), palm.centery - int(8 * s))
        draw_rrect(surf, thumb, MD3_PRIMARY, radius=int(6 * s), alpha=240)

        try:
            surf = surf.convert_alpha()
        except pygame.error:
            pass
        cache[scale_key] = surf
        return surf

    def _draw_palm_glyph(self, center: tuple[int, int], scale: float = 1.0, tint: Color = MD3_PRIMARY, wave_t: float = 0.0) -> None:
        """Legacy entry point — just delegates to the cached glyph blit
        so callers outside the home screen still work."""
        sway = int(math.sin(wave_t * 2.0) * 6.0 * scale)
        glyph = self._build_home_palm_glyph(scale)
        rect = glyph.get_rect(center=(center[0] + sway, center[1] + int(8 * scale)))
        self._screen.blit(glyph, rect.topleft)

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
        caught_ids: set[int] | None = None,
    ) -> None:
        self.begin_frame()
        self.draw_camera(frame)
        self.draw_pose(
            pose,
            finished_ids=finished_ids,
            labels_by_id=labels_by_id,
            caught_ids=caught_ids,
        )
        self._draw_banner(state_label)
        self._draw_status_cluster(in_play, finished, total, elapsed, clock, ease_steps)
        if state_label == "RED":
            self._draw_motion_meter(motion_score)
        self._draw_sent_back_log()
        self._draw_flashes()

    # ---- Caught return screen ---------------------------------------
    def draw_caught_return(
        self,
        frame: np.ndarray | None,
        returning: list[tuple[str, bool]],
        time_left_ratio: float,
        total_time: float,
        clock: pygame.time.Clock,
        pose: dict | None = None,
        caught_ids: set[int] | None = None,
        labels_by_id: dict[int, str] | None = None,
    ) -> None:
        """Walk-back screen.

        Apr 2026: ELIMINATED rebrand. The on-screen banner says
        "ELIMINATED", the card title says "ELIMINATED — return to
        start to rejoin", but the audio TTS still says "Walk back to
        the start" so kids know what to physically do."""
        self.begin_frame()
        self.draw_camera(frame)
        # Live camera now also gets the red bbox + ELIMINATED pill on each
        # captive (Apr 2026 — much stronger visual cue than the floating list).
        if pose is not None:
            self.draw_pose(pose, caught_ids=caught_ids, labels_by_id=labels_by_id)
        # Red wash to amplify the "you're out" feel.
        tint = pygame.Surface((DISPLAY_W, DISPLAY_H), pygame.SRCALPHA)
        tint.fill((*MD3_ERROR, 64))
        self._screen.blit(tint, (0, 0))
        self._draw_banner("ELIMINATED")

        rows = max(1, len(returning))
        cw = 820
        ch = min(560, 200 + 56 * rows)
        card = pygame.Rect(DISPLAY_W // 2 - cw // 2, DISPLAY_H // 2 - ch // 2 + 30, cw, ch)
        draw_panel(self._screen, card, MD3_SURFACE_HIGH, radius=32, alpha=235)

        title = self._font.render("ELIMINATED", 36, MD3_ERROR, bold=True)
        self._screen.blit(title, title.get_rect(center=(card.centerx, card.y + 50)))
        sub = self._font.render("Return to the start line to rejoin", 22, MD3_ON_BG_MED)
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

        self.draw_progress_bar_centered(card.bottom - 30, time_left_ratio, color=MD3_ERROR, width=cw - 80)
        bar_label = f"Auto-resume in {max(0.0, total_time * time_left_ratio):.1f}s"
        lab = self._font.render(bar_label, 18, MD3_ON_BG_DIM)
        self._screen.blit(lab, lab.get_rect(center=(card.centerx, card.bottom - 50)))

    # ---- Leaderboard ------------------------------------------------
    def draw_leaderboard(
        self,
        results: list[dict],
        screen_elapsed: float,
        clock: pygame.time.Clock,
        palm_progress: float = 0.0,
    ) -> None:
        """True Kahoot-style podium reveal — Apr 2026 v3 (spotlight cutout).

        Choreography:
          T0    Title fades in.
          A     3rd place podium rises in CENTRE empty → holds → player
                pops in → holds filled → slides RIGHT.
          B     Brief pause.
          C     2nd place same routine → slides LEFT.
          D     Brief pause — the moment of suspense before the winner.
          E     Everything dims to black. A circular SPOTLIGHT cuts a
                hole of brightness in the dim layer. The spotlight
                sweeps left/right ("scanning for the winner").
          F     Spotlight settles in the centre. Drumroll sound peaks.
          G     1st place podium rises INSIDE the spotlight (still dark
                around the rest of the scene).
          H     The spotlight EXPANDS outward — its bright hole grows
                until the whole scene is light again. Confetti erupts.
                Winner music + applause fire. The 1st-place podium gets
                a permanent gold glow.
          I     4th+ list staggers in below.
          J     Palm-to-restart arms.

        Layout (per Kahoot Wiki): 1st centre / 2nd left / 3rd right.
        Every duration is config-driven so you can tune the feel from
        config.py without touching code.
        """
        # Reset reveal state on fresh entry to the screen.
        if screen_elapsed < 0.05:
            self._leaderboard_sounds_fired.clear()
            self._leaderboard_burst_done = False
            self._leaderboard_winner_music_started = False

        self.begin_frame()
        self._draw_decorative_bg()
        self._update_confetti()

        # ── Title ─────────────────────────────────────────────────────
        title_t = clamp((screen_elapsed - LEADERBOARD_TITLE_DELAY_S) / 0.6)
        title_alpha = int(255 * ease_out_cubic(title_t))
        title = self._font.render("RESULTS", 70, MD3_PRIMARY, bold=True)
        title.set_alpha(title_alpha)
        self._screen.blit(title, title.get_rect(center=(DISPLAY_W // 2, 50)))
        sub = self._font.render("Everyone made it across!", 20, MD3_ON_BG_MED)
        sub.set_alpha(title_alpha)
        self._screen.blit(sub, sub.get_rect(center=(DISPLAY_W // 2, 90)))

        # ── Stage geometry ────────────────────────────────────────────
        # Layout planning (Apr 2026 v3.1):
        #   y=0..150  → title + subtitle band
        #   y=150..680 → podium stage (1st pedestal h=380, avatar 220 above
        #                = total 600 → avatar top sits at ~155 after the 16
        #                gap, just clear of the title)
        #   y=680..820 → name + time text below 1st pedestal baseline
        #   y=820..980 → 4th+ list (max 3 rows)
        #   y=980..1080 → footer (palm restart, hint text)
        stage_baseline = 680
        center_x = DISPLAY_W // 2
        left_x = center_x - 360
        right_x = center_x + 360
        h_1st = 380  # was 460 — too tall, overlapped title
        h_2nd = int(h_1st * 0.85)
        h_3rd = int(h_1st * 0.72)

        # ── Timing schedule ───────────────────────────────────────────
        # Each non-winner card has 5 sub-phases:
        #   1. PEDESTAL_RISE — empty pedestal grows up from baseline
        #   2. EMPTY_HOLD    — empty podium sits there (the beat)
        #   3. PLAYER_POP    — avatar/name/time pop into the podium
        #   4. FILLED_HOLD   — full card lingers in centre
        #   5. SLIDE         — card slides to its final pillar
        rise_dur = LEADERBOARD_PEDESTAL_RISE_S
        empty_hold = LEADERBOARD_EMPTY_HOLD_S
        pop_dur = LEADERBOARD_PLAYER_POP_S
        filled_hold = LEADERBOARD_FILLED_HOLD_S
        slide_dur = LEADERBOARD_HERO_SLIDE_S

        # 3rd: CENTER (rise → hold → pop → hold) → RIGHT
        t_3rd_appear = LEADERBOARD_3RD_DELAY_S
        t_3rd_settle = t_3rd_appear + rise_dur
        t_3rd_pop_start = t_3rd_settle + empty_hold
        t_3rd_pop_end = t_3rd_pop_start + pop_dur
        t_3rd_slide_start = t_3rd_pop_end + filled_hold
        t_3rd_slide_end = t_3rd_slide_start + slide_dur

        # 2nd: CENTER → LEFT (same 5 phases)
        t_2nd_appear = LEADERBOARD_2ND_DELAY_S
        t_2nd_settle = t_2nd_appear + rise_dur
        t_2nd_pop_start = t_2nd_settle + empty_hold
        t_2nd_pop_end = t_2nd_pop_start + pop_dur
        t_2nd_slide_start = t_2nd_pop_end + filled_hold
        t_2nd_slide_end = t_2nd_slide_start + slide_dur

        # 1st dramatic sequence — built up phase by phase.
        t_dim_start = t_2nd_slide_end + LEADERBOARD_1ST_PAUSE_S
        t_dim_end = t_dim_start + LEADERBOARD_DIM_FADE_IN_S
        t_sweep_end = t_dim_end + LEADERBOARD_SPOTLIGHT_SWEEP_S
        t_settle_end = t_sweep_end + LEADERBOARD_SPOTLIGHT_SETTLE_S
        t_1st_appear = t_settle_end
        t_1st_settle = t_1st_appear + LEADERBOARD_1ST_REVEAL_S
        t_blast_start = t_1st_settle
        t_blast_end = t_blast_start + LEADERBOARD_BLAST_OUT_S
        t_winner_celebrate = t_blast_start

        t_list_in = LEADERBOARD_LIST_DELAY_S

        # ── Render 3rd & 2nd cards (3-phase animation) ────────────────
        for tag, entry, height, badge_color, t_appear, t_settle, t_pop_start, t_pop_end, t_slide_start, t_slide_end, final_x in [
            (
                "3rd", results[2] if len(results) >= 3 else None, h_3rd, MD3_BRONZE,
                t_3rd_appear, t_3rd_settle, t_3rd_pop_start, t_3rd_pop_end, t_3rd_slide_start, t_3rd_slide_end, right_x,
            ),
            (
                "2nd", results[1] if len(results) >= 2 else None, h_2nd, MD3_SILVER,
                t_2nd_appear, t_2nd_settle, t_2nd_pop_start, t_2nd_pop_end, t_2nd_slide_start, t_2nd_slide_end, left_x,
            ),
        ]:
            if entry is None or screen_elapsed < t_appear:
                continue

            # Phase 1: pedestal rises
            rise_t = clamp((screen_elapsed - t_appear) / max(0.05, t_settle - t_appear))

            # Phase 3: player pops in (independent of rise_t now).
            # Stays at 0 during the empty hold (t_settle..t_pop_start),
            # ramps 0→1 during pop (t_pop_start..t_pop_end), stays at 1
            # afterward.
            player_t = clamp((screen_elapsed - t_pop_start) / max(0.05, t_pop_end - t_pop_start))

            # Phase 5: slide to final pillar
            slide_t = clamp((screen_elapsed - t_slide_start) / max(0.05, t_slide_end - t_slide_start)) if t_slide_end > t_slide_start else 0.0
            current_x = int(center_x + (final_x - center_x) * ease_in_out_cubic(slide_t))

            self._draw_kahoot_podium_card(
                entry=entry,
                cx=current_x,
                stage_baseline=stage_baseline,
                podium_height=height,
                badge_color=badge_color,
                rise_t=rise_t,
                tag=tag,
                is_winner=False,
                celebrate_t=0.0,
                player_t=player_t,
            )

            # SFX — fire when the empty pedestal lands.
            sound_key = {"3rd": "podium_3", "2nd": "podium_2"}[tag]
            if sound_key not in self._leaderboard_sounds_fired and screen_elapsed >= t_settle - 0.05:
                self._leaderboard_sounds_fired.add(sound_key)
                self._audio_play(sound_key)

            # Pop SFX — fire when the player pops in (a nicer "ta-da!"
            # moment vs the pedestal landing). Maps to a chime so it
            # feels different from the pedestal land.
            pop_sound_key = f"{tag}_pop"
            if pop_sound_key not in self._leaderboard_sounds_fired and screen_elapsed >= t_pop_start - 0.05:
                self._leaderboard_sounds_fired.add(pop_sound_key)
                self._audio_play("chime")

        # ── 1st-place dramatic spotlight sequence ─────────────────────
        if len(results) >= 1 and screen_elapsed >= t_dim_start:
            self._draw_first_place_drama(
                results[0],
                screen_elapsed=screen_elapsed,
                stage_baseline=stage_baseline,
                center_x=center_x,
                podium_height=h_1st,
                t_dim_start=t_dim_start,
                t_dim_end=t_dim_end,
                t_sweep_end=t_sweep_end,
                t_settle_end=t_settle_end,
                t_1st_appear=t_1st_appear,
                t_1st_settle=t_1st_settle,
                t_blast_start=t_blast_start,
                t_blast_end=t_blast_end,
            )

            # Drumroll fires when the dim completes (spotlight begins sweeping)
            if "drumroll" not in self._leaderboard_sounds_fired and screen_elapsed >= t_dim_end:
                self._leaderboard_sounds_fired.add("drumroll")
                self._audio_play("drumroll")

            # podium_1 fires the moment the 1st card finishes its rise
            if "podium_1" not in self._leaderboard_sounds_fired and screen_elapsed >= t_1st_settle - 0.05:
                self._leaderboard_sounds_fired.add("podium_1")
                self._audio_play("podium_1")

            # Winner music + applause + confetti fire as the spotlight blasts open
            if (
                screen_elapsed >= t_winner_celebrate
                and not self._leaderboard_burst_done
            ):
                self._burst_confetti(LEADERBOARD_CONFETTI_BURST_COUNT)
                self._leaderboard_burst_done = True
                if not self._leaderboard_winner_music_started:
                    self._leaderboard_winner_music_started = True
                    self._audio_play("winner")
                    self._audio_play("applause")

        # ── Confetti ON TOP of everything ─────────────────────────────
        self._draw_confetti()

        # ── 4th+ list ─────────────────────────────────────────────────
        list_t = clamp((screen_elapsed - t_list_in) / 0.8)
        if list_t > 0.0 and len(results) > 3:
            # Sits clear below the winner's name + time block. The
            # name+time stack uses 28 (top pad) + 2*32 (two name lines)
            # + 8 (gap) + 36 (time line) ≈ 140 below stage_baseline.
            list_y = stage_baseline + 160
            list_w = 900
            list_x = DISPLAY_W // 2 - list_w // 2
            # Cap at 2 visible rows so the list always fits between the
            # winner's time text (y≈800) and the footer (y=1000).
            visible_rows = min(2, len(results) - 3)
            shown_results = results[3 : 3 + visible_rows]
            for i, entry in enumerate(shown_results, start=4):
                row_delay = (i - 4) * 0.12
                row_t = clamp((screen_elapsed - t_list_in - row_delay) / 0.45)
                if row_t <= 0.0:
                    continue
                row_alpha = int(255 * ease_out_cubic(row_t))
                row_offset = int(20 * (1.0 - ease_out_cubic(row_t)))
                row = pygame.Rect(list_x, list_y + row_offset, list_w, 56)
                draw_panel(self._screen, row, MD3_SURFACE_HIGH, 18, int(230 * row_alpha / 255), shadow=False)
                rank_lbl = self._font.render(f"#{i}", 26, MD3_ON_BG, bold=True)
                rank_lbl.set_alpha(row_alpha)
                self._screen.blit(rank_lbl, (row.x + 24, row.y + 14))
                name_lbl = self._font.render(entry["descriptor"], 22, MD3_ON_BG)
                name_lbl.set_alpha(row_alpha)
                self._screen.blit(name_lbl, (row.x + 96, row.y + 16))
                time_lbl = self._font.render(self._fmt_time(entry["time_s"]), 22, MD3_ON_BG_MED, bold=True)
                time_lbl.set_alpha(row_alpha)
                self._screen.blit(time_lbl, time_lbl.get_rect(midright=(row.right - 24, row.centery)))
                list_y += 64

        # ── Footer (palm restart) ─────────────────────────────────────
        footer_y = DISPLAY_H - 80
        if palm_progress > 0.02:
            ring_cx = DISPLAY_W // 2 - 240
            ring_cy = footer_y
            draw_progress_ring(self._screen, (ring_cx, ring_cy), 22, 5, palm_progress, MD3_PRIMARY)
            self._draw_text_left("HOLD HAND", (ring_cx + 36, footer_y - 12), 18, MD3_PRIMARY, bold=True)
            self._draw_text_left(f"{int(palm_progress * 100)}%", (ring_cx + 36, footer_y + 6), 14, MD3_ON_BG_MED)
        self._draw_text_center(
            "Raise your hand · or press SPACE · to play again",
            (DISPLAY_W // 2, footer_y),
            22,
            MD3_ON_BG_MED,
        )
        self._draw_text_center("ESC to quit", (DISPLAY_W // 2, footer_y + 30), 16, MD3_ON_BG_DIM)

    def _draw_first_place_drama(
        self,
        winner: dict,
        *,
        screen_elapsed: float,
        stage_baseline: int,
        center_x: int,
        podium_height: int,
        t_dim_start: float,
        t_dim_end: float,
        t_sweep_end: float,
        t_settle_end: float,
        t_1st_appear: float,
        t_1st_settle: float,
        t_blast_start: float,
        t_blast_end: float,
    ) -> None:
        """The dramatic 1st-place reveal.

        Phases (driven by ``screen_elapsed``):
          • ``t_dim_start..t_dim_end``  — dim layer fades in.
          • ``t_dim_end..t_sweep_end``  — spotlight sweeps L↔R↔L.
          • ``t_sweep_end..t_settle_end`` — spotlight settles in centre.
          • ``t_settle_end..t_1st_settle`` — podium rises inside spotlight.
          • ``t_1st_settle..t_blast_end`` — spotlight expands outward,
            scene re-brightens, confetti erupts.
          • after ``t_blast_end`` — calm "winner glow" steady state.

        The spotlight is implemented as a dark overlay with a soft
        circular hole punched through it (the "cutout" approach the
        user described — the bright circle is what you see THROUGH
        the dark, not a circle drawn on top of it).
        """
        # Phase progress
        dim_t = clamp((screen_elapsed - t_dim_start) / max(0.05, t_dim_end - t_dim_start))
        sweep_t = clamp((screen_elapsed - t_dim_end) / max(0.05, t_sweep_end - t_dim_end))
        rise_t = clamp((screen_elapsed - t_1st_appear) / max(0.05, t_1st_settle - t_1st_appear))
        blast_t = clamp((screen_elapsed - t_blast_start) / max(0.05, t_blast_end - t_blast_start))

        # ── Compute spotlight position & radius ───────────────────────
        if screen_elapsed < t_sweep_end:
            # Sweep — L → R → L → R → centre, smooth sinusoid scaled to amplitude.
            # Two full sweeps over the sweep window so the audience has
            # time to wonder who it'll land on.
            phase = sweep_t * 4.0 * math.pi  # 2 full sine cycles
            sweep_offset = int(math.sin(phase) * LEADERBOARD_SPOTLIGHT_SWEEP_AMPLITUDE)
            spot_cx = center_x + sweep_offset
        elif screen_elapsed < t_settle_end:
            # Settle — last bit of the sweep eases into centre.
            settle_t = clamp((screen_elapsed - t_sweep_end) / max(0.05, t_settle_end - t_sweep_end))
            # Wherever it was at the end of the sweep, ease toward centre.
            sweep_end_phase = 4.0 * math.pi
            sweep_end_offset = int(math.sin(sweep_end_phase) * LEADERBOARD_SPOTLIGHT_SWEEP_AMPLITUDE)
            spot_cx = int(center_x + sweep_end_offset * (1.0 - ease_out_cubic(settle_t)))
        else:
            spot_cx = center_x

        # Spotlight Y target — roughly where the 1st-place avatar will sit.
        spot_cy = stage_baseline - int(podium_height * 0.65)

        # Apr 2026 perf — radius stays FIXED across all phases. The
        # original v3 grew the radius to cover the whole screen during
        # the blast, but that meant smoothscale-ing a 2×display mask to
        # 5000×3000 every frame — ~50ms per call on the Jetson, hitting
        # ~14 different scale-keys during the blast. The blast now
        # works by FADING the dim layer's alpha from full → 0 instead,
        # which is a per-blit set_alpha (microseconds) and cheaper to
        # boot. Visually it reads the same — the bright circle "feathers
        # out" as the surrounding darkness lifts.
        spot_r = LEADERBOARD_SPOTLIGHT_RADIUS

        # Dim alpha — full dim during sweep/rise, fades during blast.
        if screen_elapsed < t_blast_start:
            dim_alpha = int(LEADERBOARD_SPOTLIGHT_DIM_ALPHA * dim_t)
        else:
            dim_alpha = int(LEADERBOARD_SPOTLIGHT_DIM_ALPHA * (1.0 - ease_out_cubic(blast_t)))

        # ── Draw the spotlight cutout ─────────────────────────────────
        if dim_alpha > 1:
            self._draw_spotlight_cutout(spot_cx, spot_cy, spot_r, dim_alpha)

        # ── Draw the 1st-place podium INSIDE the spotlight ────────────
        # (after spotlight is settled — during sweep we just see the
        # bright circle moving over the still-empty stage).
        # The 1st-place reveal is its own moment, so the player content
        # appears together with the pedestal as it rises (no empty-hold
        # beat — the sweep + spotlight already provided that suspense).
        if screen_elapsed >= t_1st_appear:
            celebrate_t = clamp((screen_elapsed - t_blast_start) / 0.8)
            self._draw_kahoot_podium_card(
                entry=winner,
                cx=center_x,
                stage_baseline=stage_baseline,
                podium_height=podium_height,
                badge_color=MD3_GOLD,
                rise_t=rise_t,
                tag="1st",
                is_winner=True,
                celebrate_t=celebrate_t,
                player_t=rise_t,
            )

    def _draw_spotlight_cutout(self, cx: int, cy: int, radius: int, dim_alpha: int) -> None:
        """Darken the entire screen except a soft circular hole at (cx, cy).

        The "spotlight" effect the user wants: not a bright circle drawn
        ON TOP of the scene, but a DARK overlay with a HOLE punched
        through it so the underlying bright scene shows through where
        the hole is. Soft falloff at the hole edge so it looks like a
        real stage light rather than a hard mask.

        Implementation: a pre-baked oversized SRCALPHA mask (twice the
        display in each direction) with a soft hole at its centre, then
        blitted with an offset so the hole lands at (cx, cy). One
        cached mask handles every spotlight position — only the hole
        size requires special handling, done by scaling the mask.
        """
        if radius <= 0 or dim_alpha <= 0:
            return

        # The cached mask is 2× display size with a hole of radius
        # ``_spotlight_mask_base_radius`` cut in the middle. We scale it
        # to whatever radius the caller wants, then blit it at the
        # right offset to position the hole at (cx, cy).
        base_radius = self._spotlight_mask_base_radius
        mask = self._build_spotlight_mask()

        # Scale factor to make the hole come out at ``radius``.
        scale = radius / float(base_radius)
        # Skip scaling for tiny scale changes (re-blit the cached scaled
        # version to avoid pygame.transform.scale's CPU hit each frame).
        scale_key = round(scale * 20) / 20.0  # 0.05 quantisation
        cached_scaled = self._spotlight_scaled_cache.get(scale_key)
        if cached_scaled is None:
            new_w = max(2, int(mask.get_width() * scale_key))
            new_h = max(2, int(mask.get_height() * scale_key))
            try:
                cached_scaled = pygame.transform.smoothscale(mask, (new_w, new_h))
            except (pygame.error, ValueError):
                cached_scaled = pygame.transform.scale(mask, (new_w, new_h))
            try:
                cached_scaled = cached_scaled.convert_alpha()
            except pygame.error:
                pass
            # Bound the cache (different blast frames hit different scales).
            if len(self._spotlight_scaled_cache) > 24:
                # Drop something arbitrary — last-write-wins is fine for visuals.
                self._spotlight_scaled_cache.pop(next(iter(self._spotlight_scaled_cache)))
            self._spotlight_scaled_cache[scale_key] = cached_scaled

        # Drive the dim level via per-blit alpha — the mask itself is
        # always at full opacity in the dark region.
        cached_scaled.set_alpha(dim_alpha)
        # Blit so the hole (at the centre of the mask) lands at (cx, cy).
        offset_x = cx - cached_scaled.get_width() // 2
        offset_y = cy - cached_scaled.get_height() // 2
        self._screen.blit(cached_scaled, (offset_x, offset_y))

    def _build_spotlight_mask(self) -> pygame.Surface:
        """Pre-bake the spotlight dim mask once.

        The mask is a 2×display SRCALPHA surface filled with opaque
        black, with a soft-edged transparent hole punched through the
        centre. Built lazily on first use because it requires the
        display surface to exist (for ``convert_alpha()``).
        """
        cached = getattr(self, "_spotlight_mask", None)
        if cached is not None:
            return cached  # type: ignore[return-value]

        # Apr 2026 perf — mask only needs to be wide enough to cover
        # the display when its hole is offset by up to the sweep
        # amplitude. The original 2×display sizing was wasteful (33MB
        # SRCALPHA) and slowed every blit. Add some padding on top of
        # the amplitude so a few extra pixels of dim layer always
        # cover the screen edges even at the extremes of the sweep.
        sweep_pad = LEADERBOARD_SPOTLIGHT_SWEEP_AMPLITUDE + 80
        mask_w = DISPLAY_W + sweep_pad * 2
        mask_h = DISPLAY_H + sweep_pad * 2
        base_r = LEADERBOARD_SPOTLIGHT_RADIUS
        # Use numpy + cv2 to generate the soft alpha falloff cheaply
        # (~one-time cost, doesn't need to be fast).
        try:
            import numpy as _np

            alpha = _np.full((mask_h, mask_w), 255, dtype=_np.uint8)
            cy, cx = mask_h // 2, mask_w // 2
            yy, xx = _np.indices(alpha.shape)
            dist = _np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            inner = base_r * 0.78  # fully clear inside
            outer = base_r  # fully opaque outside
            falloff = _np.clip((dist - inner) / max(1.0, outer - inner), 0.0, 1.0)
            # Smoother falloff — ease_in_out cubic.
            t = falloff
            falloff = (3 * t * t - 2 * t * t * t)
            alpha = (alpha.astype(_np.float32) * falloff).astype(_np.uint8)
            # Build pygame Surface from the alpha array.
            mask = pygame.Surface((mask_w, mask_h), pygame.SRCALPHA)
            # Fill RGB with black so when alpha shows we see darkness.
            mask.fill((0, 0, 0, 0))
            # Push our alpha into the surface.
            try:
                alpha_view = pygame.surfarray.pixels_alpha(mask)
                # surfarray is (W, H) but our numpy array is (H, W) — transpose.
                alpha_view[:] = alpha.T
                del alpha_view
            except (pygame.error, ValueError):
                # Fallback — slower but always works.
                for y in range(0, mask_h, 4):
                    for x in range(0, mask_w, 4):
                        a = int(alpha[y, x])
                        if a > 0:
                            pygame.draw.rect(mask, (0, 0, 0, a), pygame.Rect(x, y, 4, 4))
        except Exception as exc:
            # Total fallback — opaque rect with a hard circle cut.
            print(f"[UI ] Spotlight mask numpy path failed ({exc}); using fallback")
            mask = pygame.Surface((mask_w, mask_h), pygame.SRCALPHA)
            mask.fill((0, 0, 0, 255))
            pygame.draw.circle(mask, (0, 0, 0, 0), (mask_w // 2, mask_h // 2), base_r)

        try:
            mask = mask.convert_alpha()
        except pygame.error:
            pass

        self._spotlight_mask = mask
        self._spotlight_mask_base_radius = base_r
        return mask

    def _draw_kahoot_podium_card(
        self,
        *,
        entry: dict,
        cx: int,
        stage_baseline: int,
        podium_height: int,
        badge_color: Color,
        rise_t: float,
        tag: str,
        is_winner: bool = False,
        celebrate_t: float = 0.0,
        player_t: float = 1.0,
    ) -> None:
        """One Kahoot-style podium card: pedestal + avatar + name + time.

        Animation is split into TWO independent timelines so the caller
        can control them separately:

          • ``rise_t`` — the empty PEDESTAL grows from 0 → full height.
            When this is 0 nothing is visible. When it's 1 the pedestal
            is fully grown and showing its rank numeral.
          • ``player_t`` — the PLAYER content (avatar card, name, time)
            pops in. Independent of rise_t so the caller can do
            "empty podium first, then pop the player on after a beat".

        Apr 2026 v3 polish:
          • Pedestal has a vertical gradient (lighter at top, darker at
            bottom) and a glossy top edge so it reads as a 3D block
            instead of a flat rectangle.
          • Avatar card has a thicker white border ring + larger drop
            shadow — looks like a polished trading-card portrait.
          • 1st place is noticeably larger (wider pedestal, bigger
            avatar, bigger numerals).
        """
        photo: pygame.Surface | None = entry.get("photo_surface")
        descriptor = entry.get("descriptor", "Player")
        rank = entry.get("rank", 0)
        time_s = entry.get("time_s", 0.0)

        # ── Pedestal ──────────────────────────────────────────────────
        ped_t = clamp(rise_t / 0.75)
        ped_eased = ease_out_back(ped_t, overshoot=1.4)
        ped_h = int(podium_height * ped_eased)
        ped_w = 320 if is_winner else 250
        ped_rect = pygame.Rect(cx - ped_w // 2, stage_baseline - ped_h, ped_w, ped_h)

        if ped_h > 4:
            # Drop shadow (slightly larger for the winner)
            shadow_spread = 28 if is_winner else 20
            draw_shadow_rrect(self._screen, ped_rect, 22, offset=(0, 14), spread=shadow_spread, alpha=140)

            # Pedestal body — vertical gradient surface (light top → dark
            # bottom). Built per-call but cached via cache_key.
            grad_key = (ped_rect.w, ped_rect.h, badge_color)
            grad_surf = self._build_pedestal_gradient(*grad_key)
            self._screen.blit(grad_surf, ped_rect.topleft)

            # Glossy bright top stripe — the colour-coded "podium top".
            stripe_h = max(8, int(18 * ped_t))
            stripe = pygame.Rect(ped_rect.x, ped_rect.y, ped_rect.w, stripe_h)
            stripe_surf = pygame.Surface(stripe.size, pygame.SRCALPHA)
            pygame.draw.rect(stripe_surf, (*badge_color, 255), stripe_surf.get_rect(), border_radius=22)
            # Inner highlight line for that "wet glass" feel
            pygame.draw.rect(
                stripe_surf,
                (255, 255, 255, 110),
                pygame.Rect(8, 3, stripe.w - 16, 3),
                border_radius=2,
            )
            self._screen.blit(stripe_surf, stripe.topleft)

            # Diagonal sheen on the body (subtle, sells "polished")
            sheen = pygame.Surface(ped_rect.size, pygame.SRCALPHA)
            pygame.draw.polygon(
                sheen,
                (255, 255, 255, 28),
                [
                    (0, 0),
                    (int(ped_rect.w * 0.45), 0),
                    (int(ped_rect.w * 0.20), ped_rect.h),
                    (0, ped_rect.h),
                ],
            )
            self._screen.blit(sheen, ped_rect.topleft)

            # Big rank numeral on the front face
            if ped_t >= 0.85:
                num_alpha = int(255 * clamp((ped_t - 0.85) / 0.15))
                num_size = 168 if is_winner else 124
                # Embossed look: a darker shadow numeral 3px below + lighter on top.
                num_shadow = self._font.render(str(rank), num_size, (10, 8, 4), bold=True)
                num_shadow.set_alpha(int(num_alpha * 0.5))
                self._screen.blit(num_shadow, num_shadow.get_rect(center=(ped_rect.centerx, ped_rect.centery + 3)))
                num_surf = self._font.render(str(rank), num_size, MD3_ON_PRIMARY, bold=True)
                num_surf.set_alpha(num_alpha)
                self._screen.blit(num_surf, num_surf.get_rect(center=ped_rect.center))

        # ── Player content (avatar / name / time) ────────────────────
        # Now driven by player_t — independent of pedestal rise so the
        # caller can show empty podium first, then pop the player on.
        if player_t <= 0.0:
            return

        avatar_eased = ease_out_back(clamp(player_t), overshoot=1.8)
        # Avatar size — was 220 for the winner, but at h_1st=380 with
        # stage_baseline=680 a 220 avatar's top sits at y=64 which
        # overlaps the title. 180 puts the avatar top at y=104 with
        # ~10px breathing room below the title.
        base_av = 180 if is_winner else 150
        av_w = int(base_av * avatar_eased)
        av_h = av_w
        av_top = ped_rect.y - av_h - 16
        av_rect = pygame.Rect(cx - av_w // 2, av_top, av_w, av_h)
        avatar_alpha = int(255 * clamp(player_t * 1.3))

        # Avatar size guard. `ease_out_back` overshoots above 1 but starts
        # at 0 and can produce a tiny positive width for one or two frames
        # at the start of the pop. We need BOTH the outer rect (av_w) and
        # the inner photo rect (av_w − 2*border) to be positive — otherwise
        # pygame.Surface and pygame.transform.smoothscale will raise.
        # 1st place uses border_w=6 so inner needs av_w > 12, etc. Pick
        # 16px as a safe minimum that covers all border widths plus
        # smoothscale's preference for >0 sizes.
        border_w = 6 if is_winner else 4
        min_av = 2 * border_w + 8
        if av_w >= min_av:
            draw_shadow_rrect(self._screen, av_rect, 22, offset=(0, 12), spread=20, alpha=int(avatar_alpha * 0.55))

            # White outer border ring (glassy frame look)
            outer_surf = pygame.Surface(av_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(outer_surf, (255, 255, 255, avatar_alpha), outer_surf.get_rect(), border_radius=22)
            self._screen.blit(outer_surf, av_rect.topleft)

            # Photo masked into inner rounded rect
            inner = av_rect.inflate(-border_w * 2, -border_w * 2)
            if photo is not None:
                try:
                    scaled = pygame.transform.smoothscale(photo, inner.size)
                    mask = pygame.Surface(inner.size, pygame.SRCALPHA)
                    pygame.draw.rect(mask, (255, 255, 255, avatar_alpha), mask.get_rect(), border_radius=18)
                    clipped = pygame.Surface(inner.size, pygame.SRCALPHA)
                    clipped.blit(scaled, (0, 0))
                    clipped.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
                    self._screen.blit(clipped, inner.topleft)
                except Exception:
                    inner_bg = pygame.Surface(inner.size, pygame.SRCALPHA)
                    pygame.draw.rect(inner_bg, (*MD3_SURFACE_VAR, avatar_alpha), inner_bg.get_rect(), border_radius=18)
                    self._screen.blit(inner_bg, inner.topleft)
            else:
                inner_bg = pygame.Surface(inner.size, pygame.SRCALPHA)
                pygame.draw.rect(inner_bg, (*MD3_SURFACE_VAR, avatar_alpha), inner_bg.get_rect(), border_radius=18)
                self._screen.blit(inner_bg, inner.topleft)
                placeholder = self._font.render("?", int(72 * avatar_eased), MD3_ON_BG_DIM, bold=True)
                placeholder.set_alpha(avatar_alpha)
                self._screen.blit(placeholder, placeholder.get_rect(center=av_rect.center))

            # Glossy top sheen on the avatar — subtle gradient highlight.
            sheen = pygame.Surface(av_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(
                sheen,
                (255, 255, 255, int(avatar_alpha * 0.16)),
                pygame.Rect(border_w, border_w, av_rect.w - border_w * 2, av_rect.h // 3),
                border_radius=14,
            )
            self._screen.blit(sheen, av_rect.topleft)

            # Medal badge — glassier, with a sheen highlight.
            if player_t > 0.6:
                badge_t = clamp((player_t - 0.6) / 0.4)
                badge_scale = ease_out_back(badge_t, overshoot=2.0)
                badge_r = int((36 if is_winner else 30) * badge_scale)
                bcx = av_rect.x + 8 + badge_r
                bcy = av_rect.y + 8 + badge_r
                pygame.draw.circle(self._screen, badge_color, (bcx, bcy), badge_r)
                pygame.draw.circle(self._screen, (255, 255, 255), (bcx, bcy), badge_r, width=4)
                hi_r = max(2, badge_r // 3)
                pygame.draw.circle(
                    self._screen,
                    (255, 255, 255, 160),
                    (bcx - badge_r // 3, bcy - badge_r // 3),
                    hi_r,
                )
                rt = self._font.render(str(rank), int((30 if is_winner else 26) * badge_scale), (32, 24, 8), bold=True)
                self._screen.blit(rt, rt.get_rect(center=(bcx, bcy)))

        # ── Name + time below the pedestal (driven by player_t) ──────
        text_t = clamp((player_t - 0.5) / 0.5)
        if text_t > 0.0:
            text_alpha = int(255 * ease_out_cubic(text_t))
            text_y = stage_baseline + 28
            words = descriptor.split()
            line = ""
            max_chars = 22 if is_winner else 18
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
                surf = self._font.render(ln, 28 if is_winner else 22, MD3_ON_BG, bold=True)
                surf.set_alpha(text_alpha)
                self._screen.blit(surf, surf.get_rect(center=(cx, text_y)))
                text_y += 32 if is_winner else 26

            time_lbl = self._font.render(self._fmt_time(time_s), 36 if is_winner else 24, badge_color, bold=True)
            time_lbl.set_alpha(text_alpha)
            self._screen.blit(time_lbl, time_lbl.get_rect(center=(cx, text_y + 8)))

        # ── Winner extras ────────────────────────────────────────────
        # The bright yellow halo + sparkle ring used to live here.
        # Removed Apr 2026 — once the spotlight blasts open and the
        # confetti settles, the winner card should just sit cleanly on
        # the podium. The circular spotlight cutout itself is enough
        # visual emphasis.
        # ``celebrate_t`` and ``is_winner`` parameters are still part
        # of the signature for API compat; they're a no-op on the
        # winner extras now but used elsewhere in the function.
        _ = (is_winner, celebrate_t)

    def _build_pedestal_gradient(self, w: int, h: int, base_color: Color) -> pygame.Surface:
        """Pre-rendered vertical-gradient pedestal body.

        Light variant of ``base_color`` at the top, darker at the
        bottom, with rounded corners. Cached forever per (w, h, color)
        so the heavy gradient render only happens once per podium
        size."""
        cache_key = (w, h, tuple(base_color))
        cache = self._pedestal_gradient_cache  # set up in __init__
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        # Build gradient as a numpy array, then push into a Surface.
        try:
            import numpy as _np

            top = _np.array(
                (
                    max(0, base_color[0] - 28),
                    max(0, base_color[1] - 28),
                    max(0, base_color[2] - 28),
                ),
                dtype=_np.float32,
            )
            bottom = _np.array(
                (
                    max(0, base_color[0] - 90),
                    max(0, base_color[1] - 90),
                    max(0, base_color[2] - 90),
                ),
                dtype=_np.float32,
            )
            ts = _np.linspace(0.0, 1.0, h, dtype=_np.float32).reshape(h, 1, 1)
            grad_rgb = (top * (1 - ts) + bottom * ts).astype(_np.uint8)  # h × 1 × 3
            grad_rgb = _np.broadcast_to(grad_rgb, (h, w, 3)).copy()  # h × w × 3
            # Pygame expects (w, h, 3) for surfarray.make_surface.
            surf = pygame.surfarray.make_surface(grad_rgb.swapaxes(0, 1))
            surf = surf.convert()
            # Apply rounded-corner alpha mask.
            mask = pygame.Surface((w, h), pygame.SRCALPHA)
            pygame.draw.rect(mask, (255, 255, 255, 255), mask.get_rect(), border_radius=22)
            final = pygame.Surface((w, h), pygame.SRCALPHA).convert_alpha()
            final.blit(surf, (0, 0))
            final.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
        except Exception as exc:
            # Fallback — flat dark colour if numpy/surfarray path fails.
            print(f"[UI ] Pedestal gradient fallback: {exc}")
            final = pygame.Surface((w, h), pygame.SRCALPHA)
            body = (
                max(0, base_color[0] - 50),
                max(0, base_color[1] - 50),
                max(0, base_color[2] - 50),
            )
            pygame.draw.rect(final, (*body, 245), final.get_rect(), border_radius=22)

        if len(cache) > 16:
            cache.pop(next(iter(cache)))
        cache[cache_key] = final
        return final

    def _burst_confetti(self, n: int) -> None:
        """One-shot confetti burst centred high on the screen (used when
        the 1st-place card lands — Kahoot drumroll energy)."""
        for _ in range(n):
            x = float(np.random.randint(int(DISPLAY_W * 0.25), int(DISPLAY_W * 0.75)))
            y = float(np.random.randint(140, 320))
            vx = float(np.random.uniform(-220.0, 220.0))
            vy = float(np.random.uniform(-260.0, -60.0))  # explode upward + outward, gravity reels them back
            palette = [MD3_PRIMARY, MD3_SECONDARY, MD3_TERTIARY, MD3_SUCCESS, MD3_WARNING, MD3_GOLD, MD3_SILVER]
            col = palette[np.random.randint(0, len(palette))]
            self._confetti.append((x, y, vx, vy, col))

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