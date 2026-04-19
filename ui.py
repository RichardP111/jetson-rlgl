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

import pygame
import cv2
import numpy as np
import math
import time
import random
import colorsys
import os

from config import (
    FONTS_DIR,
    C_CYAN,
    C_ACCENT,
    C_BG,
    C_RED,
    C_GREEN,
    C_YELLOW,
    C_WHITE,
    C_GREY,
    C_PURPLE,
    C_PANEL,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    ASSETS_DIR,
    CAPTURES_DIR,
)


# ── Font loading helper ───────────────────────────────────────────────
def _font(name: str, size: int, bold: bool = False) -> pygame.font.Font:
    path = os.path.join(FONTS_DIR, name)
    if os.path.exists(path):
        return pygame.font.Font(path, size)
    return pygame.font.SysFont("monospace", size, bold=bold)


def cv_to_surf(frame: np.ndarray, target_size: tuple | None = None) -> pygame.Surface | None:
    """Convert an OpenCV BGR frame to a pygame Surface (optionally resized)."""
    if frame is None:
        return None
    if target_size is not None:
        frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    surf = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))
    return surf


# ─────────────────────────────────────────────────────────────────────
class UIRenderer:

    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.W, self.H = screen.get_size()
        self._t0 = time.time()

        # ── Fonts (falls back to monospace if Orbitron not present)
        # Download Orbitron from Google Fonts and drop into assets/fonts/
        self.fnt_xl = _font("Orbitron-Bold.ttf", 90, True)
        self.fnt_lg = _font("Orbitron-Bold.ttf", 58, True)
        self.fnt_md = _font("Orbitron-Regular.ttf", 34)
        self.fnt_sm = _font("Orbitron-Regular.ttf", 22)
        self.fnt_xs = _font("Orbitron-Regular.ttf", 15)
        self.fnt_mono = pygame.font.SysFont("monospace", 18)

        # ── Static background layers
        self._star_layer = self._bake_stars()
        self._grid_layer = self._bake_grid()
        self._particles = self._init_particles(70)

        # ── Elimination flash state
        self._elim_alpha = 0.0
        self._elim_pos = None  # (cx, cy) in screen coords
        self._elim_label = ""

        # ── Confetti for win screen
        self._confetti = self._init_confetti(120)

    # ═══════════════════════════════════════════════════════════════
    #  Internal helpers
    # ═══════════════════════════════════════════════════════════════

    def _t(self) -> float:
        return time.time() - self._t0

    # ── Background decorations ────────────────────────────────────

    def _bake_stars(self) -> pygame.Surface:
        surf = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        for _ in range(250):
            x = random.randint(0, self.W)
            y = random.randint(0, self.H)
            r = random.choice([1, 1, 1, 2])
            a = random.randint(30, 100)
            pygame.draw.circle(surf, (200, 210, 255, a), (x, y), r)
        return surf

    def _bake_grid(self) -> pygame.Surface:
        surf = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        sp = 64
        col = (*C_CYAN, 10)
        for x in range(0, self.W, sp):
            pygame.draw.line(surf, col, (x, 0), (x, self.H))
        for y in range(0, self.H, sp):
            pygame.draw.line(surf, col, (0, y), (self.W, y))
        return surf

    def _init_particles(self, n: int) -> list[dict]:
        return [
            {
                "x": random.uniform(0, self.W),
                "y": random.uniform(0, self.H),
                "vx": random.uniform(-0.25, 0.25),
                "vy": random.uniform(-0.6, -0.15),
                "r": random.uniform(1.2, 2.8),
                "a": random.randint(60, 180),
                "c": random.choice([C_CYAN, C_ACCENT, (110, 60, 200)]),
            }
            for _ in range(n)
        ]

    def _init_confetti(self, n: int) -> list[dict]:
        cols = [C_RED, C_GREEN, C_YELLOW, C_CYAN, C_PURPLE, (255, 100, 50)]
        return [
            {
                "x": random.uniform(0, self.W),
                "y": random.uniform(-self.H, 0),
                "vx": random.uniform(-1.5, 1.5),
                "vy": random.uniform(1.5, 4.5),
                "w": random.randint(8, 16),
                "h": random.randint(4, 8),
                "rot": random.uniform(0, 360),
                "drot": random.uniform(-4, 4),
                "c": random.choice(cols),
            }
            for _ in range(n)
        ]

    def _tick_particles(self):
        for p in self._particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            if p["y"] < -4:
                p["y"] = self.H + 4
                p["x"] = random.uniform(0, self.W)

    def _tick_confetti(self):
        for c in self._confetti:
            c["x"] += c["vx"]
            c["y"] += c["vy"]
            c["rot"] = (c["rot"] + c["drot"]) % 360
            if c["y"] > self.H + 20:
                c["y"] = random.uniform(-80, -10)
                c["x"] = random.uniform(0, self.W)

    def _draw_particles(self):
        self._tick_particles()
        for p in self._particles:
            s = pygame.Surface((int(p["r"] * 2 + 1), int(p["r"] * 2 + 1)), pygame.SRCALPHA)
            pygame.draw.circle(s, (*p["c"], p["a"]), (int(p["r"]), int(p["r"])), int(p["r"]))
            self.screen.blit(s, (int(p["x"] - p["r"]), int(p["y"] - p["r"])))

    def _draw_confetti(self):
        self._tick_confetti()
        for c in self._confetti:
            s = pygame.Surface((c["w"], c["h"]), pygame.SRCALPHA)
            s.fill((*c["c"], 220))
            rotated = pygame.transform.rotate(s, c["rot"])
            self.screen.blit(rotated, (int(c["x"]), int(c["y"])))

    # ── Text rendering ────────────────────────────────────────────

    def _txt(self, text: str, font, color, pos, center: bool = True, alpha: int = 255) -> pygame.Rect:
        surf = font.render(text, True, color)
        if alpha < 255:
            surf.set_alpha(alpha)
        rect = surf.get_rect()
        if center:
            rect.center = pos
        else:
            rect.topleft = pos
        self.screen.blit(surf, rect)
        return rect

    def _glow(
        self,
        text: str,
        font,
        color,
        pos,
        radius: int = 3,
        glow_col: tuple | None = None,
    ):
        """Text with diffuse glow."""
        gc = glow_col or tuple(min(255, c + 100) for c in color)
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                g = font.render(text, True, gc)
                g.set_alpha(max(0, 35 - abs(dx) * 5 - abs(dy) * 5))
                r = g.get_rect(center=(pos[0] + dx * 2, pos[1] + dy * 2))
                self.screen.blit(g, r)
        self._txt(text, font, color, pos)

    def _panel(
        self,
        rect,
        color=C_PANEL,
        border=C_ACCENT,
        bw: int = 1,
        alpha: int = 200,
        radius: int = 8,
    ):
        s = pygame.Surface((rect[2], rect[3]), pygame.SRCALPHA)
        s.fill((*color, alpha))
        self.screen.blit(s, (rect[0], rect[1]))
        if border:
            pygame.draw.rect(self.screen, border, pygame.Rect(rect), bw, border_radius=radius)

    def _scanlines(self, alpha: int = 12):
        """CRT scanline overlay for atmosphere."""
        sl = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        for y in range(0, self.H, 4):
            pygame.draw.line(sl, (0, 0, 0, alpha), (0, y), (self.W, y))
        self.screen.blit(sl, (0, 0))

    # ═══════════════════════════════════════════════════════════════
    #  START SCREEN
    # ═══════════════════════════════════════════════════════════════

    def draw_start_screen(self, cam_frame=None, palm_held_ratio: float = 0.0):
        """
        palm_held_ratio: 0.0 – 1.0 progress of palm-hold gesture.
        """
        self.screen.fill(C_BG)
        self.screen.blit(self._star_layer, (0, 0))
        self.screen.blit(self._grid_layer, (0, 0))
        self._draw_particles()

        t = self._t()

        # ── Animated circles (Squid Game symbols) ────────────────
        symbols = [("○", 80), ("△", 80), ("□", 80)]
        for i, (sym, sz) in enumerate(symbols):
            x = self.W // 2 - 100 + i * 100
            y = 75 + int(6 * math.sin(t * 1.2 + i * 1.1))
            col = [C_RED, C_GREEN, C_CYAN][i]
            f = _font("Orbitron-Bold.ttf", sz, True)
            self._glow(sym, f, col, (x, y), radius=2)

        # ── Title ────────────────────────────────────────────────
        title_pulse = 0.88 + 0.12 * math.sin(t * 1.8)
        r_col = tuple(int(c * title_pulse) for c in C_RED)
        self._glow("RED LIGHT", self.fnt_xl, r_col, (self.W // 2, 180), radius=5)
        self._glow("GREEN LIGHT", self.fnt_xl, C_GREEN, (self.W // 2, 285), radius=5)

        # Divider
        dw = 500
        pygame.draw.line(
            self.screen,
            C_ACCENT,
            (self.W // 2 - dw // 2, 350),
            (self.W // 2 + dw // 2, 350),
            2,
        )

        # ── Instructions panel ───────────────────────────────────
        iw, ih = 820, 300
        ix, iy = self.W // 2 - iw // 2, 370
        self._panel((ix, iy, iw, ih), alpha=180)

        lines = [
            ("HOW TO PLAY", self.fnt_sm, C_CYAN),
            ("", None, None),
            ("🟢  Green Light  →  Move toward the finish line", self.fnt_xs, C_WHITE),
            ("🔴  Red Light    →  FREEZE or you're eliminated", self.fnt_xs, C_WHITE),
            ("🏆  First player to cross the finish line WINS!", self.fnt_xs, C_YELLOW),
            ("", None, None),
            ("✋  Hold up an open palm to start the game", self.fnt_sm, C_GREEN),
        ]
        for i, (txt, fnt, col) in enumerate(lines):
            if txt and fnt:
                self._txt(txt, fnt, col, (self.W // 2, iy + 28 + i * 38))

        # ── Palm progress ring ───────────────────────────────────
        if palm_held_ratio > 0:
            cx, cy = self.W // 2, iy + ih + 55
            r = 32
            pygame.draw.circle(self.screen, C_GREY, (cx, cy), r, 3)
            # Arc
            arc_rect = pygame.Rect(cx - r, cy - r, r * 2, r * 2)
            arc_end = -math.pi / 2 + 2 * math.pi * palm_held_ratio
            pygame.draw.arc(self.screen, C_GREEN, arc_rect, -math.pi / 2, arc_end, 5)
            pct = int(palm_held_ratio * 100)
            self._txt(f"{pct}%", self.fnt_xs, C_GREEN, (cx, cy))
            self._txt("Hold steady...", self.fnt_xs, C_GREEN, (cx, cy + r + 14))

        # ── Live camera thumbnail ────────────────────────────────
        if cam_frame is not None:
            pw, ph = 300, 169
            px, py = self.W - pw - 28, self.H - ph - 28
            surf = cv_to_surf(cam_frame, (pw, ph))
            if surf:
                bc = C_GREEN if palm_held_ratio > 0 else C_ACCENT
                pygame.draw.rect(
                    self.screen,
                    bc,
                    (px - 3, py - 3, pw + 6, ph + 6),
                    2,
                    border_radius=5,
                )
                self.screen.blit(surf, (px, py))
                lbl = "✋ PALM DETECTED" if palm_held_ratio > 0 else "CAMERA LIVE"
                self._txt(
                    lbl,
                    self.fnt_xs,
                    C_GREEN if palm_held_ratio > 0 else C_ACCENT,
                    (px + pw // 2, py - 13),
                )

        # ── Footer ───────────────────────────────────────────────
        self._txt(
            "STEM DAY  •  RED LIGHT GREEN LIGHT  •  Press ESC to exit",
            self.fnt_xs,
            C_GREY,
            (self.W // 2, self.H - 18),
        )

        self._scanlines()
        pygame.display.flip()

    # ═══════════════════════════════════════════════════════════════
    #  GAME HUD  (Green Light / Red Light / Turning)
    # ═══════════════════════════════════════════════════════════════

    def draw_game_screen(
        self,
        cam_frame,
        state: str,  # "GREEN" | "RED" | "TURNING"
        players_alive: int,
        elapsed: float,
        round_num: int,
        eliminated: list[dict],  # [{"label": str, ...}]
        motion_score: float = 0.0,
        moments: list | None = None,
    ):

        self.screen.fill(C_BG)
        self.screen.blit(self._grid_layer, (0, 0))

        t = self._t()

        # ── Camera feed  (left 64% of screen) ────────────────────
        cam_w = int(self.W * 0.64)
        cam_h = int(cam_w * CAMERA_HEIGHT / CAMERA_WIDTH)
        cam_y = (self.H - cam_h) // 2

        if cam_frame is not None:
            surf = cv_to_surf(cam_frame, (cam_w, cam_h))
            if surf:
                self.screen.blit(surf, (0, cam_y))

        # State-colour accent bar at top of feed
        s_col = {"GREEN": C_GREEN, "RED": C_RED, "TURNING": C_YELLOW}.get(state, C_CYAN)
        pygame.draw.rect(self.screen, s_col, (0, cam_y, cam_w, 5))
        pygame.draw.rect(self.screen, s_col, (0, cam_y + cam_h - 5, cam_w, 5))

        # State label on feed
        labels = {
            "GREEN": "● GREEN LIGHT",
            "RED": "● RED LIGHT",
            "TURNING": "● TURNING…",
        }
        lbl_col = s_col
        lbl_txt = labels.get(state, state)
        # Blinking on RED
        alpha = 255
        if state == "RED":
            alpha = 200 + int(55 * math.sin(t * 6))
        self._glow(lbl_txt, self.fnt_lg, lbl_col, (cam_w // 2, cam_y + 44), radius=3)

        # Motion intensity bar (bottom of feed, RED state only)
        if state == "RED" and motion_score > 0:
            bw = int(cam_w * motion_score)
            bc = (int(50 + 200 * motion_score), int(200 * (1 - motion_score)), 20)
            pygame.draw.rect(self.screen, (20, 20, 20), (0, cam_y + cam_h - 14, cam_w, 10))
            pygame.draw.rect(self.screen, bc, (0, cam_y + cam_h - 14, bw, 10))
            self._txt(
                "MOTION DETECTED" if motion_score > 0.25 else "SCANNING…",
                self.fnt_xs,
                bc,
                (cam_w // 2, cam_y + cam_h - 20),
            )

        # ── Right stats panel ─────────────────────────────────────
        px = cam_w + 18
        pw = self.W - px - 18

        # State badge
        self._panel((px, 14, pw, 90), color=C_PANEL, border=s_col, alpha=180)
        self._glow(lbl_txt, self.fnt_md, s_col, (px + pw // 2, 59), radius=2)

        # Stat tiles
        tiles = [
            ("ROUND", str(round_num)),
            ("ALIVE", str(players_alive)),
            ("ELAPSED", f"{int(elapsed//60):02d}:{int(elapsed%60):02d}"),
        ]
        ty = 115
        for label, value in tiles:
            self._panel((px, ty, pw, 76), alpha=160)
            self._txt(label, self.fnt_xs, C_ACCENT, (px + 14, ty + 10), center=False)
            self._glow(value, self.fnt_md, C_WHITE, (px + pw // 2, ty + 50), radius=1)
            ty += 86

        # ── Eliminated list ───────────────────────────────────────
        ey = ty + 8
        eh = max(40, min(240, len(eliminated) * 32 + 40))
        self._panel((px, ey, pw, eh), alpha=160)
        self._txt("■  ELIMINATED", self.fnt_xs, C_RED, (px + 12, ey + 10), center=False)
        for i, ep in enumerate(eliminated[-7:]):
            row_y = ey + 38 + i * 28
            pygame.draw.circle(self.screen, C_RED, (px + 16, row_y + 9), 5)
            self._txt(
                ep.get("label", f"Player {i+1}"),
                self.fnt_xs,
                (160, 55, 55),
                (px + 30, row_y),
                center=False,
            )

        # ── Moment captures strip (below camera feed) ─────────────
        strip_y = cam_y + cam_h + 8
        strip_h = self.H - strip_y - 8
        if strip_h >= 50 and moments:
            n_show = min(4, len(moments))
            mw = (cam_w - 4 * (n_show + 1)) // n_show
            mh = min(strip_h, int(mw * 9 / 16))
            for j, m in enumerate(moments[-n_show:]):
                mx = 4 + j * (mw + 4)
                msurf = cv_to_surf(m["frame"], (mw, mh))
                if msurf:
                    self.screen.blit(msurf, (mx, strip_y))
                    pygame.draw.rect(self.screen, C_ACCENT, (mx, strip_y, mw, mh), 1, border_radius=3)
                    if m.get("label"):
                        self._txt(
                            m["label"],
                            self.fnt_xs,
                            C_WHITE,
                            (mx + mw // 2, strip_y + mh - 10),
                        )

        self._scanlines(8)
        pygame.display.flip()

    # ═══════════════════════════════════════════════════════════════
    #  ELIMINATION FLASH
    # ═══════════════════════════════════════════════════════════════

    def draw_elimination(self, cam_frame, label: str = "", screen_pos: tuple | None = None):
        """
        Full-screen red flash with camera behind it.
        screen_pos: (x, y) in screen coordinates of the eliminated player.
        """
        self.screen.fill(C_BG)

        if cam_frame is not None:
            surf = cv_to_surf(cam_frame, (self.W, self.H))
            if surf:
                self.screen.blit(surf, (0, 0))

        # Dark-red tint overlay
        tint = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        tint.fill((160, 0, 0, 130))
        self.screen.blit(tint, (0, 0))

        t = self._t()

        # Big text
        self._glow(
            "ELIMINATED",
            self.fnt_xl,
            C_RED,
            (self.W // 2, self.H // 2 - 60),
            radius=7,
            glow_col=(255, 50, 50),
        )
        self._txt(
            label or "RETURN TO START",
            self.fnt_md,
            C_WHITE,
            (self.W // 2, self.H // 2 + 40),
        )
        self._txt(
            "Walk back to the start line",
            self.fnt_sm,
            (200, 150, 150),
            (self.W // 2, self.H // 2 + 90),
        )

        # Animated downward-pointing arrow toward player
        if screen_pos:
            ax, ay = screen_pos
            bob = int(12 * abs(math.sin(t * 5)))
            tip = (ax, ay + 70 + bob)
            pts = [tip, (ax - 22, ay + 40 + bob), (ax + 22, ay + 40 + bob)]
            pygame.draw.polygon(self.screen, C_RED, pts)
            pygame.draw.rect(self.screen, C_RED, (ax - 9, ay - 40, 18, 85 + bob))
            # Pulse ring
            r = 36 + int(10 * abs(math.sin(t * 4)))
            pygame.draw.circle(self.screen, C_RED, (ax, ay), r, 3)

        self._scanlines()
        pygame.display.flip()

    # ═══════════════════════════════════════════════════════════════
    #  COUNTDOWN OVERLAY
    # ═══════════════════════════════════════════════════════════════

    def draw_countdown_overlay(self, n: int):
        """Overlay a big countdown digit (no flip — caller must flip or redraw)."""
        dim = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 140))
        self.screen.blit(dim, (0, 0))
        col = [C_GREEN, C_YELLOW, C_RED][max(0, min(2, n - 1))]
        self._glow(str(n), self.fnt_xl, col, (self.W // 2, self.H // 2), radius=10)
        self._txt("GET READY", self.fnt_sm, C_WHITE, (self.W // 2, self.H // 2 + 100))
        pygame.display.flip()

    # ═══════════════════════════════════════════════════════════════
    #  WIN SCREEN
    # ═══════════════════════════════════════════════════════════════

    def draw_win_screen(self, moments: list | None = None, winner_label: str = "WINNER"):
        self.screen.fill(C_BG)
        self.screen.blit(self._star_layer, (0, 0))
        self._draw_confetti()

        t = self._t()

        # Rainbow winner text
        hue = (t * 40) % 360
        r, g, b = colorsys.hsv_to_rgb(hue / 360, 1.0, 1.0)
        win_col = (int(r * 255), int(g * 255), int(b * 255))

        self._glow("🏆  WINNER  🏆", self.fnt_xl, win_col, (self.W // 2, 105), radius=7)
        self._glow(winner_label, self.fnt_lg, C_WHITE, (self.W // 2, 210), radius=3)
        self._txt("CONGRATULATIONS!", self.fnt_md, C_CYAN, (self.W // 2, 275))

        # ── Best moments gallery ──────────────────────────────────
        if moments:
            n = min(len(moments), 4)
            margin = 28
            tw = (self.W - margin * (n + 1)) // n
            th = int(tw * CAMERA_HEIGHT / CAMERA_WIDTH)
            gy = 330

            for i, m in enumerate(moments[:n]):
                mx = margin + i * (tw + margin)
                msurf = cv_to_surf(m["frame"], (tw, th))
                if msurf:
                    pulse = 0.5 + 0.5 * abs(math.sin(t * 1.5 + i * 1.2))
                    bc = tuple(int(c * pulse) for c in win_col)
                    pygame.draw.rect(
                        self.screen,
                        bc,
                        (mx - 4, gy - 4, tw + 8, th + 8),
                        3,
                        border_radius=8,
                    )
                    self.screen.blit(msurf, (mx, gy))
                    tag = m.get("label", "")
                    if tag:
                        self._panel((mx, gy + th - 22, tw, 22), color=(0, 0, 0), alpha=160, bw=0)
                        self._txt(tag, self.fnt_xs, C_WHITE, (mx + tw // 2, gy + th - 11))

        self._txt(
            "✋  Raise palm to play again   •   ESC to exit",
            self.fnt_sm,
            C_GREY,
            (self.W // 2, self.H - 30),
        )

        self._scanlines(6)
        pygame.display.flip()
