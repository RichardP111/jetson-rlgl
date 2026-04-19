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
from config import (
    CAMERA_WIDTH, CAMERA_HEIGHT, FONT_MAIN_BOLD, FONT_MAIN_REG,
    C_BG, C_PANEL, C_RED, C_GREEN, C_WHITE, C_GREY, C_YELLOW, C_CYAN
)

class UIRenderer:
    def __init__(self, surface: pygame.Surface):
        self.screen = surface
        pygame.font.init()
        
        try:
            self.font_xl = pygame.font.Font(FONT_MAIN_BOLD, 80)
            self.font_lg = pygame.font.Font(FONT_MAIN_BOLD, 48)
            self.font_md = pygame.font.Font(FONT_MAIN_BOLD, 32)
            self.font_sm = pygame.font.Font(FONT_MAIN_REG, 24)
        except FileNotFoundError:
            print("[UI] Custom fonts missing. Falling back to SysFont.")
            self.font_xl = pygame.font.SysFont(None, 80)
            self.font_lg = pygame.font.SysFont(None, 48)
            self.font_md = pygame.font.SysFont(None, 32)
            self.font_sm = pygame.font.SysFont(None, 24)

    def _cv2_to_surface(self, frame: np.ndarray) -> pygame.Surface:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = np.rot90(frame_rgb)
        frame_rgb = pygame.surfarray.make_surface(frame_rgb)
        return pygame.transform.flip(frame_rgb, True, False)

    def draw_start_screen(self, frame: np.ndarray | None, palm_held_ratio: float = 0.0):
        self.screen.fill(C_BG)
        if frame is not None:
            self.screen.blit(self._cv2_to_surface(frame), (0, 0))
            
        overlay = pygame.Surface((CAMERA_WIDTH, CAMERA_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))

        title = self.font_xl.render("PROJECT: RED LIGHT", True, C_RED)
        self.screen.blit(title, (CAMERA_WIDTH//2 - title.get_width()//2, 200))

        prompt = self.font_md.render("Raise your hand to the camera to begin", True, C_WHITE)
        self.screen.blit(prompt, (CAMERA_WIDTH//2 - prompt.get_width()//2, 350))

        if palm_held_ratio > 0:
            bar_w, bar_h = 400, 20
            px, py = CAMERA_WIDTH//2 - bar_w//2, 420
            pygame.draw.rect(self.screen, C_GREY, (px, py, bar_w, bar_h), border_radius=10)
            pygame.draw.rect(self.screen, C_CYAN, (px, py, int(bar_w * palm_held_ratio), bar_h), border_radius=10)

        pygame.display.flip()

    def draw_countdown_overlay(self, number: int):
        txt = self.font_xl.render(str(number), True, C_YELLOW)
        self.screen.blit(txt, (CAMERA_WIDTH//2 - txt.get_width()//2, CAMERA_HEIGHT//2 - txt.get_height()//2))
        pygame.display.flip()

    def draw_game_screen(self, frame: np.ndarray | None, state: str, players_tracked: int, elapsed: float, round_num: int):
        self.screen.fill(C_BG)
        if frame is not None:
            self.screen.blit(self._cv2_to_surface(frame), (0, 0))

        # Top HUD
        pygame.draw.rect(self.screen, C_PANEL, (0, 0, CAMERA_WIDTH, 60))
        
        color = C_GREEN if state == "GREEN" else (C_RED if state == "RED" else C_YELLOW)
        txt_state = self.font_lg.render(f"STATUS: {state}", True, color)
        self.screen.blit(txt_state, (20, 10))

        txt_info = self.font_md.render(f"TRACKING: {players_tracked} | TIME: {elapsed:.1f}s", True, C_WHITE)
        self.screen.blit(txt_info, (CAMERA_WIDTH - txt_info.get_width() - 20, 15))

        pygame.display.flip()

    def draw_caught_screen(self, frame: np.ndarray | None, label: str):
        if frame is not None:
            self.screen.blit(self._cv2_to_surface(frame), (0, 0))
            
        overlay = pygame.Surface((CAMERA_WIDTH, CAMERA_HEIGHT), pygame.SRCALPHA)
        overlay.fill((255, 0, 0, 100))
        self.screen.blit(overlay, (0, 0))

        txt = self.font_xl.render("MOTION DETECTED", True, C_WHITE)
        self.screen.blit(txt, (CAMERA_WIDTH//2 - txt.get_width()//2, CAMERA_HEIGHT//2 - 100))
        
        sub = self.font_lg.render(f"{label} CAUGHT! RETURN TO START", True, C_YELLOW)
        self.screen.blit(sub, (CAMERA_WIDTH//2 - sub.get_width()//2, CAMERA_HEIGHT//2))

        pygame.display.flip()

    def draw_win_screen(self, winner_label: str):
        self.screen.fill(C_BG)
        txt = self.font_xl.render(f"{winner_label}", True, C_GREEN)
        self.screen.blit(txt, (CAMERA_WIDTH//2 - txt.get_width()//2, 200))
        
        sub = self.font_md.render("Raise hand to restart", True, C_GREY)
        self.screen.blit(sub, (CAMERA_WIDTH//2 - sub.get_width()//2, 300))
        
        pygame.display.flip()