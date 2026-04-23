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

import pygame, cv2, numpy as np, math, time, random, os
from config import (
    FONT_PATH,
    CAM_W,
    CAM_H,
    ENABLE_FPS_COUNTER,
    ENABLE_PLAYER_COUNT,
    ENABLE_PHASE_TIMER,
    STATE_COLORS,
    MD3_BG,
    MD3_SURFACE,
    MD3_PRIMARY,
    MD3_SUCCESS,
    MD3_ERROR,
    MD3_WARNING,
    MD3_ON_BG,
    MD3_ON_BG_MED,
    MD3_ON_BG_DIM,
    MD3_SUCCESS_DIM,
    MD3_ERROR_DIM,
)


def cv2surf(frame, size=None):
    if frame is None:
        return None
    if size:
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return pygame.surfarray.make_surface(rgb.swapaxes(0, 1))


def rounded_panel(surface, rect, color, alpha=210, radius=18, border=None, border_w=2):
    s = pygame.Surface(rect.size, pygame.SRCALPHA)
    pygame.draw.rect(s, (*color[:3], alpha), (0, 0, *rect.size), border_radius=radius)
    surface.blit(s, rect.topleft)
    if border:
        pygame.draw.rect(surface, border, rect, border_w, border_radius=radius)


def text_shadow(surface, text, font, color, pos, center=True, shadow_col=(0, 0, 0), shadow_off=(2, 2)):
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


class UIRenderer:
    def __init__(self, screen):
        self.screen = screen
        self.W, self.H = screen.get_size()
        self._t0 = time.time()
        if os.path.exists(FONT_PATH):
            self.f_hero = pygame.font.Font(FONT_PATH, 96)
            self.f_title = pygame.font.Font(FONT_PATH, 64)
            self.f_big = pygame.font.Font(FONT_PATH, 48)
            self.f_med = pygame.font.Font(FONT_PATH, 32)
            self.f_body = pygame.font.Font(FONT_PATH, 24)
            self.f_small = pygame.font.Font(FONT_PATH, 18)
            self.f_tiny = pygame.font.Font(FONT_PATH, 14)
        else:
            self.f_hero = pygame.font.SysFont("sans", 96, True)
            self.f_title = pygame.font.SysFont("sans", 64, True)
            self.f_big = pygame.font.SysFont("sans", 48, True)
            self.f_med = pygame.font.SysFont("sans", 32, True)
            self.f_body = pygame.font.SysFont("sans", 24)
            self.f_small = pygame.font.SysFont("sans", 18)
            self.f_tiny = pygame.font.SysFont("sans", 14)
        cam_w = int(self.W * 0.75)
        cam_h = int(cam_w * CAM_H / CAM_W)
        self._cam_rect = pygame.Rect(20, (self.H - cam_h) // 2, cam_w, cam_h)
        self._rp_x = self._cam_rect.right + 20
        self._rp_w = self.W - self._rp_x - 24
        self._particles = []
        self._confetti = []

    def _t(self):
        return time.time() - self._t0

    def _rp(self, r, col=MD3_SURFACE, alpha=215, rad=18, border=None, bw=2):
        rounded_panel(self.screen, r, col, alpha, rad, border, bw)

    def _txt(self, text, font, color, pos, center=True):
        return text_shadow(self.screen, text, font, color, pos, center=center)

    def draw_start_screen(self, frame, palm_progress=0.0, clock=None):
        self.screen.fill(MD3_BG)
        t = self._t()
        thumb_w = int(self.W * 0.42)
        thumb_h = int(thumb_w * CAM_H / CAM_W)
        thumb_r = pygame.Rect(40, (self.H - thumb_h) // 2, thumb_w, thumb_h)
        col = MD3_SUCCESS if palm_progress > 0 else MD3_PRIMARY
        self._rp(thumb_r.inflate(10, 10), col=MD3_SURFACE, alpha=220, rad=24, border=col)
        if frame is not None:
            surf = cv2surf(frame, (thumb_w, thumb_h))
            if surf:
                self.screen.blit(surf, thumb_r.topleft)
        pygame.draw.rect(self.screen, col, thumb_r.inflate(10, 10), 2, border_radius=24)
        lbl = "✋  PALM DETECTED" if palm_progress > 0 else "CAMERA LIVE"
        self._txt(lbl, self.f_small, col, (thumb_r.centerx, thumb_r.bottom + 18))
        rx = thumb_r.right + 40
        rw = self.W - rx - 40
        cy = self.H // 2
        bob = int(7 * math.sin(t * 1.6))
        self._txt("RED LIGHT", self.f_hero, MD3_ERROR, (rx + rw // 2, cy - 175 + bob))
        self._txt("GREEN LIGHT", self.f_hero, MD3_SUCCESS, (rx + rw // 2, cy - 70 + bob))
        pygame.draw.line(self.screen, MD3_PRIMARY, (rx + int(rw * 0.1), cy - 2), (rx + int(rw * 0.9), cy - 2), 1)
        card = pygame.Rect(rx, cy + 10, rw, 230)
        self._rp(card, col=MD3_SURFACE, alpha=200, rad=20, border=MD3_ON_BG_DIM)
        lines = [
            (MD3_SUCCESS, "🟢  GREEN LIGHT  →  Move toward finish"),
            (MD3_ERROR, "🔴  RED LIGHT    →  FREEZE (any motion = OUT)"),
            (MD3_WARNING, "🏆  First to finish line WINS!"),
            (MD3_PRIMARY, "✋  Raise palm to start"),
        ]
        for i, (c, txt) in enumerate(lines):
            self._txt(txt, self.f_body, c, (card.centerx, card.y + 35 + i * 50))
        prx, pry = rx + rw // 2, card.bottom + 55
        ring_r = 38
        self._progress_ring(prx, pry, ring_r, palm_progress, MD3_SUCCESS, width=7)
        if palm_progress > 0:
            pct = int(palm_progress * 100)
            self._txt(f"{pct}%", self.f_small, MD3_SUCCESS, (prx, pry))
            self._txt("Hold steady…", self.f_small, MD3_SUCCESS, (prx, pry + ring_r + 16))
        else:
            self._txt("✋ RAISE PALM", self.f_small, MD3_ON_BG_MED, (prx, pry))
            self._txt("to start", self.f_small, MD3_ON_BG_DIM, (prx, pry + ring_r + 16))
        if ENABLE_FPS_COUNTER and clock:
            fps = int(clock.get_fps())
            fps_col = MD3_SUCCESS if fps > 20 else MD3_WARNING if fps > 15 else MD3_ERROR
            self._txt(f"{fps} FPS", self.f_small, fps_col, (self.W - 80, 30))
        pygame.display.flip()

    def draw_countdown(self, frame, n, clock):
        self.screen.fill(MD3_BG)
        self._draw_cam(frame, border_col=MD3_ON_BG_DIM, label="CAMERA")
        dim = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 120))
        self.screen.blit(dim, (0, 0))
        col = [MD3_SUCCESS, MD3_WARNING, MD3_ERROR][max(0, min(2, n - 1))]
        self._txt(str(n), self.f_hero, col, (self.W // 2, self.H // 2 - 20))
        self._txt("GET READY", self.f_med, MD3_ON_BG, (self.W // 2, self.H // 2 + 90))
        if ENABLE_FPS_COUNTER and clock:
            fps = int(clock.get_fps())
            fps_col = MD3_SUCCESS if fps > 20 else MD3_WARNING if fps > 15 else MD3_ERROR
            self._txt(f"{fps} FPS", self.f_small, fps_col, (self.W - 80, 30))
        pygame.display.flip()

    def draw_game_hud(self, frame, state, players_alive, elapsed, round_num, caught_list, motion_score, time_left, clock):
        col = STATE_COLORS.get(state, MD3_PRIMARY)
        self.screen.fill(MD3_BG)
        self._draw_cam(frame, border_col=col, label={"GREEN": "● GREEN LIGHT", "RED": "● RED LIGHT", "TURNING": "● TURNING…"}.get(state, state))
        if state == "RED":
            t = self._t()
            pulse_alpha = int(50 + 50 * abs(math.sin(t * 4)))
            pulse = pygame.Surface(self._cam_rect.inflate(8, 8).size, pygame.SRCALPHA)
            pygame.draw.rect(pulse, (*MD3_ERROR, pulse_alpha), (0, 0, *pulse.get_size()), 8, border_radius=22)
            self.screen.blit(pulse, self._cam_rect.inflate(8, 8).topleft)
        rx, rw = self._rp_x, self._rp_w
        badge = pygame.Rect(rx, 24, rw, 72)
        badge_col = {"GREEN": MD3_SUCCESS_DIM, "RED": MD3_ERROR_DIM, "TURNING": (40, 35, 10)}.get(state, MD3_SURFACE)
        self._rp(badge, col=badge_col, alpha=240, rad=18, border=col, bw=2)
        self._txt({"GREEN": "GREEN LIGHT 🟢", "RED": "RED LIGHT 🔴", "TURNING": "TURNING…"}.get(state, state), self.f_med, col, badge.center)
        th, tw = 88, (rw - 12) // 2
        tx, ty = rx, badge.bottom + 12
        elapsed_str = f"{int(elapsed//60):02d}:{int(elapsed%60):02d}"
        tiles = [
            ("ROUND", str(round_num), MD3_PRIMARY),
            ("ALIVE", str(max(0, players_alive)), MD3_SUCCESS),
            ("TIME", elapsed_str, MD3_WARNING),
            ("CAUGHT", str(len(caught_list)), MD3_ERROR),
        ]
        for i, (lbl, val, ac) in enumerate(tiles):
            col_i, row_i = i % 2, i // 2
            self._stat_tile(tx + col_i * (tw + 12), ty + row_i * (th + 8), tw, th, lbl, val, ac)
        ty2 = ty + 2 * (th + 8) + 12
        mbar_r = pygame.Rect(rx, ty2, rw, 54)
        self._rp(mbar_r, col=MD3_SURFACE, alpha=200, rad=14, border=MD3_ON_BG_DIM, bw=1)
        self._txt("MOTION DETECTOR", self.f_tiny, MD3_ON_BG_MED, (rx + rw // 2, mbar_r.y + 11))
        bx, bw, bh, by = rx + 12, rw - 24, 12, mbar_r.y + 30
        pygame.draw.rect(self.screen, (30, 30, 45), (bx, by, bw, bh), border_radius=6)
        fill = int(bw * motion_score)
        if fill > 0:
            bar_col = MD3_ERROR if motion_score > 0.4 else MD3_WARNING if motion_score > 0.15 else MD3_SUCCESS
            pygame.draw.rect(self.screen, bar_col, (bx, by, fill, bh), border_radius=6)
        if ENABLE_PLAYER_COUNT and players_alive > 0:
            badge_x, badge_y = self._cam_rect.x + 20, self._cam_rect.y + 20
            badge_w = 120
            self._rp(pygame.Rect(badge_x, badge_y, badge_w, 50), MD3_SURFACE, alpha=220, rad=12, border=MD3_PRIMARY)
            self._txt(f"👥 {players_alive}", self.f_med, MD3_PRIMARY, (badge_x + badge_w // 2, badge_y + 25))
        if ENABLE_PHASE_TIMER and state in ("GREEN", "RED") and time_left > 0:
            timer_col = STATE_COLORS[state]
            self._txt(f"{int(time_left)}s", self.f_big, timer_col, (self._cam_rect.centerx, self._cam_rect.y + 80))
        self._tick_particles()
        if ENABLE_FPS_COUNTER and clock:
            fps = int(clock.get_fps())
            fps_col = MD3_SUCCESS if fps > 20 else MD3_WARNING if fps > 15 else MD3_ERROR
            self._txt(f"{fps} FPS", self.f_small, fps_col, (self.W - 80, 30))
        pygame.display.flip()

    def draw_caught_screen(self, frame, colour, progress, clock):
        self.screen.fill(MD3_BG)
        if frame is not None:
            surf = cv2surf(frame, (self.W, self.H))
            if surf:
                surf.set_alpha(130)
                self.screen.blit(surf, (0, 0))
        tint = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        tint.fill((0, 0, 0, 155))
        self.screen.blit(tint, (0, 0))
        if progress < 0.15:
            fl = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
            fl.fill((200, 0, 0, int(140 * (1 - progress / 0.15))))
            self.screen.blit(fl, (0, 0))
        cw, ch, cx, cy = 700, 320, self.W // 2, self.H // 2
        card = pygame.Rect(cx - cw // 2, cy - ch // 2, cw, ch)
        self._rp(card, col=(20, 5, 12), alpha=240, rad=28, border=MD3_ERROR, bw=3)
        self._txt("❌  CAUGHT!", self.f_title, MD3_ERROR, (cx, cy - 75))
        dot_col = self._colour_name_to_rgb(colour)
        pygame.draw.circle(self.screen, dot_col, (cx - 90, cy + 5), 16)
        pygame.draw.circle(self.screen, MD3_ON_BG, (cx - 90, cy + 5), 16, 2)
        self._txt(f"Player in the {colour} shirt", self.f_med, MD3_ON_BG, (cx + 10, cy + 5))
        self._txt("Walk back to the start line", self.f_body, MD3_ON_BG_MED, (cx, cy + 55))
        ring_cx, ring_cy = cx, cy + 125
        self._progress_ring(ring_cx, ring_cy, 28, 1.0 - progress, MD3_ERROR, width=5)
        self._txt("Resuming…", self.f_tiny, MD3_ON_BG_DIM, (ring_cx, ring_cy + 46))
        self._tick_particles()
        if ENABLE_FPS_COUNTER and clock:
            fps = int(clock.get_fps())
            fps_col = MD3_SUCCESS if fps > 20 else MD3_WARNING if fps > 15 else MD3_ERROR
            self._txt(f"{fps} FPS", self.f_small, fps_col, (self.W - 80, 30))
        pygame.display.flip()

    def draw_winner_screen(self, winner_colour, highlight_frames, palm_progress, clock):
        self.screen.fill(MD3_BG)
        t = self._t()
        while len(self._confetti) < 120:
            self._confetti.append(self._Confetti(self.W))
        for c in self._confetti:
            c.tick(self.H)
            c.draw(self.screen)
        self._confetti = [c for c in self._confetti if not c.dead]
        hue = (t * 60) % 360
        r_v, g_v, b_v = abs(math.sin(math.radians(hue))), abs(math.sin(math.radians(hue + 120))), abs(math.sin(math.radians(hue + 240)))
        win_col = (int(r_v * 255), int(g_v * 255), int(b_v * 255))
        self._txt("🏆  WINNER!  🏆", self.f_hero, win_col, (self.W // 2, 100))
        dot_col = self._colour_name_to_rgb(winner_colour)
        pygame.draw.circle(self.screen, dot_col, (self.W // 2 - 155, 180), 20)
        pygame.draw.circle(self.screen, MD3_ON_BG, (self.W // 2 - 155, 180), 20, 2)
        self._txt(f"Player in the {winner_colour} shirt!", self.f_big, MD3_ON_BG, (self.W // 2 + 15, 180))
        self._txt("CONGRATULATIONS!", self.f_med, MD3_PRIMARY, (self.W // 2, 240))
        if highlight_frames:
            n = min(len(highlight_frames), 4)
            gap = 20
            tw = (self.W - gap * (n + 1)) // n
            th, gy = int(tw * CAM_H / CAM_W), 290
            for i, f in enumerate(highlight_frames[-n:]):
                fx = gap + i * (tw + gap)
                pulse = 0.6 + 0.4 * abs(math.sin(t * 1.3 + i * 0.9))
                bc = tuple(int(c * pulse) for c in win_col[:3])
                frame_r = pygame.Rect(fx - 4, gy - 4, tw + 8, th + 8)
                self._rp(frame_r, col=MD3_SURFACE, alpha=230, rad=16, border=bc, bw=3)
                surf = cv2surf(f, (tw, th))
                if surf:
                    self.screen.blit(surf, (fx, gy))
        prx, pry = self.W // 2, self.H - 80
        self._progress_ring(prx, pry, 30, palm_progress, MD3_SUCCESS, width=6)
        if palm_progress > 0:
            self._txt(f"{int(palm_progress*100)}%", self.f_tiny, MD3_SUCCESS, (prx, pry))
        self._txt("✋  Raise palm to play again", self.f_small, MD3_ON_BG_MED, (prx, pry + 48))
        if ENABLE_FPS_COUNTER and clock:
            fps = int(clock.get_fps())
            fps_col = MD3_SUCCESS if fps > 20 else MD3_WARNING if fps > 15 else MD3_ERROR
            self._txt(f"{fps} FPS", self.f_small, fps_col, (self.W - 80, 30))
        pygame.display.flip()

    def _draw_cam(self, frame, border_col=MD3_PRIMARY, label=""):
        r = self._cam_rect
        self._rp(r.inflate(8, 8), col=MD3_SURFACE, alpha=255, rad=22, border=border_col)
        if frame is not None:
            surf = cv2surf(frame, (r.w, r.h))
            if surf:
                clip = pygame.Surface((r.w, r.h), pygame.SRCALPHA)
                pygame.draw.rect(clip, (255, 255, 255, 255), (0, 0, r.w, r.h), border_radius=14)
                surf.blit(clip, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
                self.screen.blit(surf, r.topleft)
        else:
            self._rp(r, col=(12, 14, 24), alpha=255, rad=14)
            self._txt("NO CAMERA SIGNAL", self.f_med, MD3_ON_BG_DIM, r.center)
        pygame.draw.rect(self.screen, border_col, r.inflate(8, 8), 2, border_radius=22)
        if label:
            lw = self.f_small.size(label)[0] + 20
            lr = pygame.Rect(r.x + 14, r.y + 14, lw, 30)
            self._rp(lr, col=(10, 10, 18), alpha=220, rad=8, border=border_col)
            self._txt(label, self.f_small, border_col, lr.center)

    def _stat_tile(self, x, y, w, h, label, value, accent=MD3_PRIMARY, rad=16):
        r = pygame.Rect(x, y, w, h)
        self._rp(r, col=MD3_SURFACE, alpha=210, rad=rad, border=accent, bw=1)
        self._txt(label, self.f_tiny, MD3_ON_BG_MED, (x + w // 2, y + 16))
        self._txt(str(value), self.f_big, accent, (x + w // 2, y + h // 2 + 8))

    def _progress_ring(self, cx, cy, radius, progress, color, bg=(40, 40, 60), width=6):
        pygame.draw.circle(self.screen, bg, (cx, cy), radius, width)
        if progress <= 0:
            return
        steps = 120
        end = int(steps * progress)
        for i in range(end):
            a1, a2 = math.radians(-90 + (360 / steps) * i), math.radians(-90 + (360 / steps) * (i + 1))
            x1, y1 = int(cx + math.cos(a1) * radius), int(cy + math.sin(a1) * radius)
            x2, y2 = int(cx + math.cos(a2) * radius), int(cy + math.sin(a2) * radius)
            pygame.draw.line(self.screen, color, (x1, y1), (x2, y2), width + 1)

    def _tick_particles(self):
        alive = []
        for p in self._particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["vy"] += 0.2
            p["alpha"] = max(0, p["alpha"] - 8)
            p["r"] = max(0.0, p["r"] - 0.1)
            if p["alpha"] > 0 and p["r"] > 0:
                s = pygame.Surface((int(p["r"] * 2) + 1, int(p["r"] * 2) + 1), pygame.SRCALPHA)
                pygame.draw.circle(s, (*p["color"][:3], int(p["alpha"])), (int(p["r"]), int(p["r"])), int(p["r"]))
                self.screen.blit(s, (int(p["x"] - p["r"]), int(p["y"] - p["r"])))
                alive.append(p)
        self._particles = alive

    def flash_caught(self, x_cam, y_cam):
        r = self._cam_rect
        sx, sy = r.x + int(x_cam / CAM_W * r.w), r.y + int(y_cam / CAM_H * r.h)
        for _ in range(4):
            for __ in range(20):
                angle, speed = random.uniform(0, 2 * math.pi), random.uniform(1.5, 5.0)
                self._particles.append(
                    {
                        "x": sx,
                        "y": sy,
                        "vx": math.cos(angle) * speed,
                        "vy": math.sin(angle) * speed,
                        "r": random.uniform(3, 7),
                        "alpha": 255,
                        "color": MD3_ERROR,
                        "dead": False,
                    }
                )

    def _colour_name_to_rgb(self, name):
        m = {
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
        return m.get(name.lower(), m["unknown"])

    class _Confetti:
        def __init__(self, W):
            self.x, self.y = random.uniform(0, W), random.uniform(-200, 0)
            self.vx, self.vy = random.uniform(-1.5, 1.5), random.uniform(2.5, 5.5)
            self.w, self.h = random.randint(8, 20), random.randint(4, 10)
            self.rot, self.drot = random.uniform(0, 360), random.uniform(-5, 5)
            colors = [MD3_ERROR, MD3_SUCCESS, MD3_PRIMARY, MD3_WARNING, (255, 100, 50), (255, 220, 50)]
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
