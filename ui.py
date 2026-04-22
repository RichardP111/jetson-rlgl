#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Project:      Red Light Green Light (Jetson Orin Nano)
File:         ui.py
Description:  Renders the PyGame graphical interface, including menus,
              live camera feeds, and on-screen animations.

Author:       Richard Pu
Last Updated: April 2026
===============================================================================
"""

import math
import os
import random
import time
from typing import Any, cast

import cv2
import numpy as np
import pygame

from config import (BG, CAM_H, CAM_W, CYAN, DISPLAY_H, DISPLAY_W, FONT_BOLD,
                    FONT_REG, GREEN, GREEN_DIM, GREY, GREY_LIGHT, PANEL,
                    PANEL_LIGHT, PURPLE, RED, RED_DIM, STATE_COLS, WHITE,
                    YELLOW)

# OpenCV attribute shims for strict type checkers
_cv2_resize: Any = getattr(cv2, "resize", None)
_cv2_cvtColor: Any = getattr(cv2, "cvtColor", None)
_cv2_INTER_LINEAR: int = int(getattr(cv2, "INTER_LINEAR", 1))
_cv2_COLOR_BGR2RGB: Any = getattr(cv2, "COLOR_BGR2RGB", None)

# ════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════


def _load_font(path: str, size: int, bold: bool = False) -> pygame.font.Font:
    if os.path.exists(path):
        return pygame.font.Font(path, size)
    return pygame.font.SysFont("sans-serif", size, bold=bold)


def cv2surf(frame: np.ndarray | None, size: tuple | None = None) -> pygame.Surface | None:
    """Convert OpenCV BGR frame → pygame Surface."""
    if frame is None:
        return None
    if size:
        if callable(_cv2_resize):
            frame = cast(np.ndarray, _cv2_resize(frame, size, interpolation=_cv2_INTER_LINEAR))
        else:
            frame = np.ascontiguousarray(frame)
    if callable(_cv2_cvtColor) and _cv2_COLOR_BGR2RGB is not None:
        rgb = cast(np.ndarray, _cv2_cvtColor(frame, _cv2_COLOR_BGR2RGB))
    else:
        rgb = frame[:, :, ::-1].copy()
    surf = pygame.surfarray.make_surface(cast(np.ndarray, rgb).swapaxes(0, 1))
    return surf


def alpha_surf(w: int, h: int, color: tuple, alpha: int) -> pygame.Surface:
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    s.fill((*color[:3], alpha))
    return s


def rounded_panel(
    surface: pygame.Surface, rect: pygame.Rect, color: tuple, alpha: int = 210, radius: int = 18, border: tuple | None = None, border_w: int = 2
):
    """Draw a frosted-glass rounded panel."""
    s = pygame.Surface(rect.size, pygame.SRCALPHA)
    pygame.draw.rect(s, (*color[:3], alpha), (0, 0, *rect.size), border_radius=radius)
    surface.blit(s, rect.topleft)
    if border:
        pygame.draw.rect(surface, border, rect, border_w, border_radius=radius)


def text_shadow(surface, text, font, color, pos, center=True, shadow_col=(0, 0, 0), shadow_off=(2, 2)):
    """Render text with a subtle shadow for legibility on camera feed."""
    shadow = font.render(text, True, shadow_col)
    s_rect = shadow.get_rect()
    if center:
        s_rect.center = (pos[0] + shadow_off[0], pos[1] + shadow_off[1])
    else:
        s_rect.topleft = (pos[0] + shadow_off[0], pos[1] + shadow_off[1])
    surface.blit(shadow, s_rect)

    surf = font.render(text, True, color)
    rect = surf.get_rect()
    if center:
        rect.center = pos
    else:
        rect.topleft = pos
    surface.blit(surf, rect)
    return rect


def glow_text(surface, text, font, color, pos, glow_r: int = 3, center=True):
    """Text with coloured glow bloom."""
    gc = tuple(min(255, c + 80) for c in color[:3])
    for dx in range(-glow_r, glow_r + 1):
        for dy in range(-glow_r, glow_r + 1):
            if dx == dy == 0:
                continue
            d = math.sqrt(dx * dx + dy * dy)
            if d > glow_r:
                continue
            a = int(max(0, 45 - d * 12))
            g = font.render(text, True, gc)
            g.set_alpha(a)
            r = g.get_rect()
            if center:
                r.center = (pos[0] + dx * 2, pos[1] + dy * 2)
            else:
                r.topleft = (pos[0] + dx * 2, pos[1] + dy * 2)
            surface.blit(g, r)
    return text_shadow(surface, text, font, color, pos, center=center)


# ════════════════════════════════════════════════════════════════════
#  PARTICLE / CONFETTI SYSTEM
# ════════════════════════════════════════════════════════════════════


class _Particle:
    __slots__ = ("x", "y", "vx", "vy", "r", "alpha", "color", "dead")

    def __init__(self, x, y, color):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1.5, 5.0)
        self.x = x
        self.y = y
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.r = random.uniform(3, 7)
        self.alpha = 255
        self.color = color
        self.dead = False

    def tick(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.12  # gravity
        self.alpha -= 5
        self.r -= 0.07
        if self.alpha <= 0 or self.r <= 0:
            self.dead = True

    def draw(self, surface):
        if self.dead:
            return
        s = pygame.Surface((int(self.r * 2 + 2), int(self.r * 2 + 2)), pygame.SRCALPHA)
        pygame.draw.circle(s, (*self.color[:3], int(self.alpha)), (int(self.r + 1), int(self.r + 1)), int(self.r))
        surface.blit(s, (int(self.x - self.r), int(self.y - self.r)))


class ParticleSystem:
    def __init__(self):
        self._particles: list[_Particle] = []

    def burst(self, x, y, color, n: int = 25):
        for _ in range(n):
            self._particles.append(_Particle(x, y, color))

    def tick_draw(self, surface):
        self._particles = [p for p in self._particles if not p.dead]
        for p in self._particles:
            p.tick()
            p.draw(surface)

    def clear(self):
        self._particles.clear()


class _Confetti:
    __slots__ = ("x", "y", "vx", "vy", "w", "h", "rot", "drot", "color", "dead")

    def __init__(self, W):
        self.x = random.uniform(0, W)
        self.y = random.uniform(-200, 0)
        self.vx = random.uniform(-1.5, 1.5)
        self.vy = random.uniform(2.5, 5.5)
        self.w = random.randint(8, 20)
        self.h = random.randint(4, 10)
        self.rot = random.uniform(0, 360)
        self.drot = random.uniform(-5, 5)
        colors = [RED, GREEN, CYAN, YELLOW, PURPLE, (255, 100, 50), (255, 220, 50)]
        self.color = random.choice(colors)
        self.dead = False

    def tick(self, H):
        self.x += self.vx
        self.y += self.vy
        self.rot = (self.rot + self.drot) % 360
        if self.y > H + 30:
            self.dead = True

    def draw(self, surface):
        if self.dead:
            return
        s = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        s.fill((*self.color[:3], 220))
        rot = pygame.transform.rotate(s, self.rot)
        surface.blit(rot, (int(self.x), int(self.y)))


# ════════════════════════════════════════════════════════════════════
#  MAIN RENDERER
# ════════════════════════════════════════════════════════════════════


class UIRenderer:

    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.W = screen.get_width()
        self.H = screen.get_height()
        self._t0 = time.time()

        # Fonts
        self.f_hero = _load_font(FONT_BOLD, 96, True)
        self.f_title = _load_font(FONT_BOLD, 64, True)
        self.f_big = _load_font(FONT_BOLD, 48, True)
        self.f_med = _load_font(FONT_BOLD, 32, True)
        self.f_body = _load_font(FONT_REG, 24)
        self.f_small = _load_font(FONT_REG, 18)
        self.f_tiny = _load_font(FONT_REG, 14)

        # Particle / confetti
        self.particles = ParticleSystem()
        self._confetti: list[_Confetti] = []

        # Static layers
        self._bg_dots = self._bake_dots()

        # Camera feed display region (left 66% of screen)
        self._cam_rect = pygame.Rect(24, 24, int(self.W * 0.635), int(self.W * 0.635 * CAM_H / CAM_W))
        # Right panel
        self._rp_x = self._cam_rect.right + 20
        self._rp_w = self.W - self._rp_x - 24

    # ── Timer ────────────────────────────────────────────────────────

    def _t(self) -> float:
        return time.time() - self._t0

    # ── Background layer ─────────────────────────────────────────────

    def _bake_dots(self) -> pygame.Surface:
        s = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        for _ in range(180):
            x = random.randint(0, self.W)
            y = random.randint(0, self.H)
            r = random.choice([1, 1, 1, 2])
            a = random.randint(20, 70)
            pygame.draw.circle(s, (180, 200, 255, a), (x, y), r)
        return s

    def _draw_bg(self, state: str = ""):
        self.screen.fill(BG)
        self.screen.blit(self._bg_dots, (0, 0))

        # Subtle state-colour ambient glow in corners
        col = STATE_COLS.get(state, CYAN)
        for corner, pos in [(0, (0, 0)), (1, (self.W, 0)), (2, (0, self.H)), (3, (self.W, self.H))]:
            s = pygame.Surface((500, 500), pygame.SRCALPHA)
            pygame.draw.circle(s, (*col, 18), (250, 250), 250)
            self.screen.blit(s, (pos[0] - 250, pos[1] - 250))

    # ── Common drawing primitives ────────────────────────────────────

    def _rp(self, r: pygame.Rect, col=PANEL, alpha=215, rad=18, border=None, bw=2):
        rounded_panel(self.screen, r, col, alpha, rad, border, bw)

    def _txt(self, text, font, color, pos, center=True):
        return text_shadow(self.screen, text, font, color, pos, center=center)

    def _glow(self, text, font, color, pos, gr=3, center=True):
        return glow_text(self.screen, text, font, color, pos, glow_r=gr, center=center)

    def _scanlines(self, alpha: int = 10):
        sl = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        for y in range(0, self.H, 4):
            pygame.draw.line(sl, (0, 0, 0, alpha), (0, y), (self.W, y))
        self.screen.blit(sl, (0, 0))

    def _progress_ring(self, cx, cy, radius, progress, color, bg=(40, 40, 60), width=6):
        """Draw a circular progress arc (0.0 – 1.0)."""
        # Background ring
        pygame.draw.circle(self.screen, bg, (cx, cy), radius, width)
        if progress <= 0:
            return
        # Arc via small line segments
        steps = 120
        end = int(steps * progress)
        for i in range(end):
            a1 = math.radians(-90 + (360 / steps) * i)
            a2 = math.radians(-90 + (360 / steps) * (i + 1))
            x1 = int(cx + math.cos(a1) * radius)
            y1 = int(cy + math.sin(a1) * radius)
            x2 = int(cx + math.cos(a2) * radius)
            y2 = int(cy + math.sin(a2) * radius)
            pygame.draw.line(self.screen, color, (x1, y1), (x2, y2), width + 1)

    # ── Camera feed ──────────────────────────────────────────────────

    def _draw_cam(self, frame: np.ndarray | None, border_col=CYAN, label: str = ""):
        r = self._cam_rect
        # Panel behind
        self._rp(r.inflate(8, 8), col=PANEL, alpha=255, rad=22, border=border_col)

        if frame is not None:
            surf = cv2surf(frame, (r.w, r.h))
            if surf:
                # Clip to rounded rect by drawing on a surface with mask
                clip = pygame.Surface((r.w, r.h), pygame.SRCALPHA)
                pygame.draw.rect(clip, (255, 255, 255, 255), (0, 0, r.w, r.h), border_radius=14)
                surf.blit(clip, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
                self.screen.blit(surf, r.topleft)
        else:
            # Placeholder
            self._rp(r, col=(12, 14, 24), alpha=255, rad=14)
            self._txt("NO CAMERA SIGNAL", self.f_med, GREY, r.center)

        # Border re-draw on top
        pygame.draw.rect(self.screen, border_col, r.inflate(8, 8), 2, border_radius=22)

        # Label badge (top-left of feed)
        if label:
            lw = self.f_small.size(label)[0] + 20
            lr = pygame.Rect(r.x + 14, r.y + 14, lw, 30)
            self._rp(lr, col=(10, 10, 18), alpha=220, rad=8, border=border_col)
            self._txt(label, self.f_small, border_col, lr.center)

    # ── Stat tile ────────────────────────────────────────────────────

    def _stat_tile(self, x, y, w, h, label, value, accent=CYAN, rad=16):
        r = pygame.Rect(x, y, w, h)
        self._rp(r, col=PANEL, alpha=210, rad=rad, border=accent, bw=1)
        # Label small on top
        self._txt(label, self.f_tiny, GREY_LIGHT, (x + w // 2, y + 16))
        # Value big
        self._glow(str(value), self.f_big, accent, (x + w // 2, y + h // 2 + 8))

    # ════════════════════════════════════════════════════════════════
    #  SCREEN: START
    # ════════════════════════════════════════════════════════════════

    def draw_start_screen(self, frame: np.ndarray | None, palm_progress: float = 0.0):
        """
        palm_progress: 0.0 – 1.0  (how long palm has been held up)
        """
        self._draw_bg("GREEN")
        t = self._t()

        # ── Left: camera thumbnail ────────────────────────────────
        thumb_w = int(self.W * 0.42)
        thumb_h = int(thumb_w * CAM_H / CAM_W)
        thumb_r = pygame.Rect(40, (self.H - thumb_h) // 2, thumb_w, thumb_h)

        col = GREEN if palm_progress > 0 else CYAN
        self._rp(thumb_r.inflate(10, 10), col=PANEL, alpha=220, rad=24, border=col)
        if frame is not None:
            surf = cv2surf(frame, (thumb_w, thumb_h))
            if surf:
                self.screen.blit(surf, thumb_r.topleft)
        pygame.draw.rect(self.screen, col, thumb_r.inflate(10, 10), 2, border_radius=24)

        # Camera label
        lbl = "✋  PALM DETECTED" if palm_progress > 0 else "CAMERA LIVE"
        self._txt(lbl, self.f_small, col, (thumb_r.centerx, thumb_r.bottom + 18))

        # ── Right: title + instructions ───────────────────────────
        rx = thumb_r.right + 40
        rw = self.W - rx - 40
        cy = self.H // 2

        # Title
        bob = int(7 * math.sin(t * 1.6))
        self._glow("RED LIGHT", self.f_hero, RED, (rx + rw // 2, cy - 175 + bob), gr=6)
        self._glow("GREEN LIGHT", self.f_hero, GREEN, (rx + rw // 2, cy - 70 + bob), gr=6)

        # Divider
        dw = int(rw * 0.8)
        dx = rx + (rw - dw) // 2
        pygame.draw.line(self.screen, CYAN, (dx, cy - 2), (dx + dw, cy - 2), 1)

        # Instructions card
        card = pygame.Rect(rx, cy + 10, rw, 230)
        self._rp(card, col=PANEL, alpha=200, rad=20, border=GREY)

        lines = [
            (GREEN, "🟢  GREEN LIGHT  →  Move toward the finish line"),
            (RED, "🔴  RED LIGHT    →  FREEZE — any movement = OUT"),
            (YELLOW, "🏆  First to cross the finish line WINS!"),
            (CYAN, "✋  Raise your palm to start the game"),
        ]
        for i, (c, txt) in enumerate(lines):
            self._txt(txt, self.f_body, c, (card.centerx, card.y + 35 + i * 50))

        # ── Palm progress ring ────────────────────────────────────
        prx = rx + rw // 2
        pry = card.bottom + 55
        ring_r = 38
        self._progress_ring(prx, pry, ring_r, palm_progress, GREEN, width=7)

        if palm_progress > 0:
            pct = int(palm_progress * 100)
            self._txt(f"{pct}%", self.f_small, GREEN, (prx, pry))
            self._txt("Hold steady…", self.f_small, GREEN, (prx, pry + ring_r + 16))
        else:
            self._txt("✋ RAISE PALM", self.f_small, GREY_LIGHT, (prx, pry))
            self._txt("to start", self.f_small, GREY, (prx, pry + ring_r + 16))

        # Footer
        self._txt("STEM DAY  •  Press ESC to exit  •  G/R/W for debug", self.f_tiny, GREY, (self.W // 2, self.H - 18))

        self._scanlines()
        pygame.display.flip()

    # ════════════════════════════════════════════════════════════════
    #  SCREEN: COUNTDOWN  (3 – 2 – 1)
    # ════════════════════════════════════════════════════════════════

    def draw_countdown(self, frame: np.ndarray | None, n: int):
        self._draw_bg()
        # Dim camera behind
        self._draw_cam(frame, border_col=GREY, label="CAMERA")
        # Big number
        col = [GREEN, YELLOW, RED][max(0, min(2, n - 1))]
        dim = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 120))
        self.screen.blit(dim, (0, 0))
        self._glow(str(n), self.f_hero, col, (self.W // 2, self.H // 2 - 20), gr=12)
        self._txt("GET READY", self.f_med, WHITE, (self.W // 2, self.H // 2 + 90))
        self._scanlines()
        pygame.display.flip()

    # ════════════════════════════════════════════════════════════════
    #  SCREEN: GAME HUD  (Green / Turning / Red)
    # ════════════════════════════════════════════════════════════════

    def draw_game_hud(
        self,
        frame: np.ndarray | None,
        state: str,  # "GREEN" | "RED" | "TURNING"
        players_alive: int,
        elapsed: float,
        round_num: int,
        caught_list: list[str],  # list of colour strings
        motion_score: float = 0.0,
    ):

        col = STATE_COLS.get(state, CYAN)
        self._draw_bg(state)

        # ── Camera feed (main view) ───────────────────────────────
        self._draw_cam(frame, border_col=col, label={"GREEN": "● GREEN LIGHT", "RED": "● RED LIGHT", "TURNING": "● TURNING…"}.get(state, state))

        # State pulsing glow on feed border during RED
        if state == "RED":
            t = self._t()
            pulse_alpha = int(50 + 50 * abs(math.sin(t * 4)))
            pulse = pygame.Surface(self._cam_rect.inflate(8, 8).size, pygame.SRCALPHA)
            pygame.draw.rect(pulse, (*RED, pulse_alpha), (0, 0, *pulse.get_size()), 8, border_radius=22)
            self.screen.blit(pulse, self._cam_rect.inflate(8, 8).topleft)

        # ── Right panel ───────────────────────────────────────────
        rx, rw = self._rp_x, self._rp_w

        # State badge
        badge = pygame.Rect(rx, 24, rw, 72)
        badge_col = {"GREEN": GREEN_DIM, "RED": RED_DIM, "TURNING": (40, 35, 10)}.get(state, PANEL)
        self._rp(badge, col=badge_col, alpha=240, rad=18, border=col, bw=2)
        self._glow({"GREEN": "GREEN LIGHT 🟢", "RED": "RED LIGHT 🔴", "TURNING": "TURNING…"}.get(state, state), self.f_med, col, badge.center, gr=2)

        # ── Stat tiles ────────────────────────────────────────────
        th = 88
        tw = (rw - 12) // 2
        tx = rx
        ty = badge.bottom + 12

        elapsed_str = f"{int(elapsed//60):02d}:{int(elapsed%60):02d}"
        tiles = [
            ("ROUND", str(round_num), CYAN),
            ("ALIVE", str(max(0, players_alive)), GREEN),
            ("TIME", elapsed_str, YELLOW),
            ("CAUGHT", str(len(caught_list)), RED),
        ]
        for i, (lbl, val, ac) in enumerate(tiles):
            col_i = i % 2
            row_i = i // 2
            self._stat_tile(tx + col_i * (tw + 12), ty + row_i * (th + 8), tw, th, lbl, val, ac)

        # ── Motion bar (visible during RED) ──────────────────────
        ty2 = ty + 2 * (th + 8) + 12
        mbar_r = pygame.Rect(rx, ty2, rw, 54)
        self._rp(mbar_r, col=PANEL, alpha=200, rad=14, border=GREY, bw=1)
        self._txt("MOTION DETECTOR", self.f_tiny, GREY_LIGHT, (rx + rw // 2, mbar_r.y + 11))
        bx = rx + 12
        bw = rw - 24
        bh = 12
        by = mbar_r.y + 30
        pygame.draw.rect(self.screen, (30, 30, 45), (bx, by, bw, bh), border_radius=6)
        fill = int(bw * motion_score)
        if fill > 0:
            bar_col = RED if motion_score > 0.4 else YELLOW if motion_score > 0.15 else GREEN
            pygame.draw.rect(self.screen, bar_col, (bx, by, fill, bh), border_radius=6)

        # ── Caught list ───────────────────────────────────────────
        ty3 = mbar_r.bottom + 12
        list_h = max(60, min(self.H - ty3 - 24, len(caught_list) * 34 + 50))
        list_r = pygame.Rect(rx, ty3, rw, list_h)
        self._rp(list_r, col=PANEL, alpha=200, rad=16, border=RED_DIM, bw=1)
        self._txt("❌  ELIMINATED", self.f_small, RED, (rx + rw // 2, list_r.y + 16))
        for i, c in enumerate(caught_list[-8:]):
            row_y = list_r.y + 42 + i * 30
            # Colour dot
            dot_col = _colour_name_to_rgb(c)
            pygame.draw.circle(self.screen, dot_col, (rx + 20, row_y + 8), 7)
            pygame.draw.circle(self.screen, WHITE, (rx + 20, row_y + 8), 7, 1)
            self._txt(f"Player in {c} shirt", self.f_small, (160, 60, 60), (rx + 24, row_y), center=False)

        #self.particles.tick_draw(self.screen)
        self._scanlines()
        pygame.display.flip()

    # ════════════════════════════════════════════════════════════════
    #  SCREEN: CAUGHT / PAUSE
    # ════════════════════════════════════════════════════════════════

    def draw_caught_screen(self, frame: np.ndarray | None, colour: str, progress: float):
        """
        progress: 0.0 – 1.0  (how far through the pause we are)
        Shows a big "CAUGHT!" overlay with a countdown back to the game.
        """
        self._draw_bg("RED")

        # Camera behind (dimmed)
        if frame is not None:
            surf = cv2surf(frame, (self.W, self.H))
            if surf:
                surf.set_alpha(130)
                self.screen.blit(surf, (0, 0))

        # Dark tint
        tint = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        tint.fill((0, 0, 0, 155))
        self.screen.blit(tint, (0, 0))

        # Red flash at 100% progress (just caught)
        if progress < 0.15:
            fl = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
            fl.fill((200, 0, 0, int(140 * (1 - progress / 0.15))))
            self.screen.blit(fl, (0, 0))

        t = self._t()

        # Main card
        cw, ch = 700, 320
        cx, cy = self.W // 2, self.H // 2
        card = pygame.Rect(cx - cw // 2, cy - ch // 2, cw, ch)
        self._rp(card, col=(20, 5, 12), alpha=240, rad=28, border=RED, bw=3)

        self._glow("❌  CAUGHT!", self.f_title, RED, (cx, cy - 75), gr=8)

        dot_col = _colour_name_to_rgb(colour)
        pygame.draw.circle(self.screen, dot_col, (cx - 90, cy + 5), 16)
        pygame.draw.circle(self.screen, WHITE, (cx - 90, cy + 5), 16, 2)
        self._txt(f"Player in the {colour} shirt", self.f_med, WHITE, (cx + 10, cy + 5))

        self._txt("Walk back to the start line", self.f_body, GREY_LIGHT, (cx, cy + 55))

        # Resuming countdown ring
        ring_cx = cx
        ring_cy = cy + 125
        self._progress_ring(ring_cx, ring_cy, 28, 1.0 - progress, RED, width=5)
        self._txt("Resuming…", self.f_tiny, GREY_LIGHT, (ring_cx, ring_cy + 46))

        #self.particles.tick_draw(self.screen)
        self._scanlines()
        pygame.display.flip()

    def flash_caught(self, x_cam: int, y_cam: int):
        """Call once on elimination to burst particles at that location."""
        # Map camera coords to screen coords
        r = self._cam_rect
        sx = r.x + int(x_cam / CAM_W * r.w)
        sy = r.y + int(y_cam / CAM_H * r.h)
        for _ in range(4):
            self.particles.burst(sx, sy, RED, n=20)

    # ════════════════════════════════════════════════════════════════
    #  SCREEN: WINNER
    # ════════════════════════════════════════════════════════════════

    def draw_winner_screen(self, winner_colour: str, highlight_frames: list[np.ndarray], palm_progress: float = 0.0):
        self._draw_bg("WINNER")
        t = self._t()

        # Confetti
        while len(self._confetti) < 120:
            self._confetti.append(_Confetti(self.W))
        for c in self._confetti:
            c.tick(self.H)
            c.draw(self.screen)
        self._confetti = [c for c in self._confetti if not c.dead]

        # Rainbow winner text
        hue = (t * 60) % 360
        r_v = abs(math.sin(math.radians(hue)))
        g_v = abs(math.sin(math.radians(hue + 120)))
        b_v = abs(math.sin(math.radians(hue + 240)))
        win_col = (int(r_v * 255), int(g_v * 255), int(b_v * 255))

        self._glow("🏆  WINNER!  🏆", self.f_hero, win_col, (self.W // 2, 100), gr=10)

        dot_col = _colour_name_to_rgb(winner_colour)
        pygame.draw.circle(self.screen, dot_col, (self.W // 2 - 155, 180), 20)
        pygame.draw.circle(self.screen, WHITE, (self.W // 2 - 155, 180), 20, 2)
        self._glow(f"Player in the {winner_colour} shirt!", self.f_big, WHITE, (self.W // 2 + 15, 180), gr=2)

        self._txt("CONGRATULATIONS!", self.f_med, CYAN, (self.W // 2, 240))

        # ── Highlight gallery ─────────────────────────────────────
        if highlight_frames:
            n = min(len(highlight_frames), 4)
            gap = 20
            tw = (self.W - gap * (n + 1)) // n
            th = int(tw * CAM_H / CAM_W)
            gy = 290

            for i, f in enumerate(highlight_frames[-n:]):
                fx = gap + i * (tw + gap)
                pulse = 0.6 + 0.4 * abs(math.sin(t * 1.3 + i * 0.9))
                bc = tuple(int(c * pulse) for c in win_col[:3])
                frame_r = pygame.Rect(fx - 4, gy - 4, tw + 8, th + 8)
                self._rp(frame_r, col=PANEL, alpha=230, rad=16, border=bc, bw=3)
                surf = cv2surf(f, (tw, th))
                if surf:
                    self.screen.blit(surf, (fx, gy))
                self._txt(["Action!", "Caught!", "FREEZE!", "Winner!"][i % 4], self.f_tiny, GREY_LIGHT, (fx + tw // 2, gy + th + 12))

        # ── Palm to restart ───────────────────────────────────────
        prx, pry = self.W // 2, self.H - 80
        self._progress_ring(prx, pry, 30, palm_progress, GREEN, width=6)
        if palm_progress > 0:
            self._txt(f"{int(palm_progress*100)}%", self.f_tiny, GREEN, (prx, pry))
        self._txt("✋  Raise palm to play again", self.f_small, GREY_LIGHT, (prx, pry + 48))

        self.particles.tick_draw(self.screen)
        self._scanlines(6)
        pygame.display.flip()


# ════════════════════════════════════════════════════════════════════
#  UTILITY — colour name → RGB  (for dots next to eliminated names)
# ════════════════════════════════════════════════════════════════════

_RGB_MAP = {
    "red": (220, 50, 50),
    "orange": (255, 140, 0),
    "yellow": (240, 200, 0),
    "green": (40, 200, 70),
    "cyan": (0, 200, 230),
    "blue": (50, 100, 220),
    "purple": (160, 60, 220),
    "pink": (230, 100, 180),
    "white": (230, 230, 230),
    "black": (60, 60, 60),
    "grey": (130, 130, 140),
    "unknown": (100, 100, 110),
}


def _colour_name_to_rgb(name: str) -> tuple:
    return _RGB_MAP.get(name.lower(), _RGB_MAP["unknown"])
