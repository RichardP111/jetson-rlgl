#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Project:      Red Light Green Light (Jetson Orin Nano)
File:         game.py
Description:  Main Finite State Machine (FSM) governing game phases, integrating
              YOLOv8 multi-person tracking, Color-Caller AI, and hardware
              laser break-beam for the finish line.

Author:       Richard Pu
Last Updated: April 2026
===============================================================================
"""

# --- Standard Library Imports ---
import time
import random
import math
from enum import Enum, auto
from dataclasses import dataclass, field

# --- Third-Party Imports ---
import pygame

# --- Local Application Imports ---
from config import (
    FPS_CAP,
    GREEN_LIGHT_MIN,
    GREEN_LIGHT_MAX,
    RED_LIGHT_MIN,
    RED_LIGHT_MAX,
    TURN_SETTLE_TIME,
    GRACE_PERIOD,
    MOTION_THRESHOLD,
    ELIMINATION_HOLD,
)
from hardware import Camera, ServoController, LaserBreakBeam
from vision import ProPoseTracker, ColorAnalyzer
from audio import AudioManager
from ui import UIRenderer


class State(Enum):
    START = auto()
    COUNTDOWN = auto()
    GREEN = auto()
    TURNING_RED = auto()
    RED = auto()
    ELIMINATION = auto()
    TURNING_GREEN = auto()
    WINNER = auto()
    RESET = auto()


@dataclass
class EliminatedEntry:
    player_id: int
    label: str
    last_pos: tuple
    ts: float = field(default_factory=time.time)


class GameEngine:
    def __init__(self, camera: Camera, servo: ServoController, laser: LaserBreakBeam, vision: ProPoseTracker, audio: AudioManager, ui: UIRenderer):

        self.camera = camera
        self.servo = servo
        self.laser = laser
        self.vision = vision
        self.audio = audio
        self.ui = ui

        # --- Game State ---
        self._state = State.START
        self._state_time = time.time()
        self._game_start = 0.0
        self._round = 0
        self._light_dur = 0.0
        self._eliminated = []
        self._winner_label = ""
        self._last_elim = None

        # --- YOLO Tracking Data ---
        self._baseline_positions = {}  # Stores ID -> (x, y) when Red Light hits
        self._palm_since = None
        self._PALM_HOLD = 2.0

        self._last_frame = None

    # =========================================================================
    # --- State Helpers ---
    # =========================================================================

    def _go(self, new_state: State):
        print(f"[FSM] Transition: {self._state.name} -> {new_state.name}")
        self._state = new_state
        self._state_time = time.time()

    def _in_state(self) -> float:
        return time.time() - self._state_time

    # =========================================================================
    # --- Main Loop ---
    # =========================================================================

    def run(self, clock: pygame.time.Clock):
        print("[ENGINE] Game FSM Live. Press ESC to quit.")
        self.servo.face_away()
        self.audio.play_music("bgm")

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    self._handle_key(event.key)

            frame = self.camera.read()
            self._last_frame = frame
            if frame is None:
                continue

            # Run YOLO AI
            pose_data, overlay = self.vision.process_frame(frame)

            # Dispatch to state handler
            self._dispatch(frame, overlay, pose_data)
            clock.tick(FPS_CAP)

    def _dispatch(self, frame, overlay, pose):
        s = self._state
        if s == State.START:
            self._do_start(frame, overlay, pose)
        elif s == State.COUNTDOWN:
            self._do_countdown(frame, overlay)
        elif s == State.GREEN:
            self._do_green(frame, overlay, pose)
        elif s == State.TURNING_RED:
            self._do_turning_red(frame, overlay, pose)
        elif s == State.RED:
            self._do_red(frame, overlay, pose)
        elif s == State.ELIMINATION:
            self._do_elimination(frame, overlay)
        elif s == State.TURNING_GREEN:
            self._do_turning_green(frame, overlay, pose)
        elif s == State.WINNER:
            self._do_winner(frame, overlay, pose)
        elif s == State.RESET:
            self._do_reset()

    # =========================================================================
    # --- State Behaviors ---
    # =========================================================================

    def _do_start(self, frame, overlay, pose):
        palm_up = self._check_palm_yolo(pose)
        prog = self._palm_progress(palm_up)
        self.ui.draw_start_screen(overlay, palm_held_ratio=prog)

        if prog >= 1.0:
            self._palm_since = None
            self.camera.capture_moment("Game Start")
            self.audio.say("Get ready!")
            self._go(State.COUNTDOWN)

    def _do_countdown(self, frame, overlay):
        t = self._in_state()
        n = 3 - int(t)
        self.ui.draw_start_screen(overlay)
        if n > 0:
            self.ui.draw_countdown_overlay(n)
        else:
            self._round = 1
            self._eliminated = []
            self._game_start = time.time()
            self._light_dur = random.uniform(GREEN_LIGHT_MIN, GREEN_LIGHT_MAX)
            self.servo.face_away()
            self.audio.announce_green()
            self._go(State.GREEN)

    def _do_green(self, frame, overlay, pose):
        elapsed = time.time() - self._game_start
        self.ui.draw_game_screen(
            overlay,
            state="GREEN",
            players_alive=pose.get("players_alive", 0) if pose else 0,
            elapsed=elapsed,
            round_num=self._round,
            eliminated=[{"label": e.label} for e in self._eliminated],
            moments=self.camera.moment_captures,
        )

        # --- HARDWARE FINISH LINE ---
        if self.laser.broken:
            self._trigger_winner("FINISH LINE REACHED!")
            return

        if self._in_state() >= self._light_dur:
            self._round += 1
            self.servo.face_players()
            self.audio.announce_red()
            self._light_dur = random.uniform(RED_LIGHT_MIN, RED_LIGHT_MAX)
            self._go(State.TURNING_RED)

    def _do_turning_red(self, frame, overlay, pose):
        self.ui.draw_game_screen(
            overlay,
            state="TURNING",
            players_alive=pose.get("players_alive", 0) if pose else 0,
            elapsed=time.time() - self._game_start,
            round_num=self._round,
            eliminated=[{"label": e.label} for e in self._eliminated],
            moments=self.camera.moment_captures,
        )

        if self.servo.is_facing_players or self._in_state() >= TURN_SETTLE_TIME:
            time.sleep(GRACE_PERIOD)
            self._baseline_positions.clear()  # Clear old baselines
            self._go(State.RED)

    def _do_red(self, frame, overlay, pose):
        # 1. Update UI
        self.ui.draw_game_screen(
            overlay,
            state="RED",
            players_alive=pose.get("players_alive", 0) if pose else 0,
            elapsed=time.time() - self._game_start,
            round_num=self._round,
            eliminated=[{"label": e.label} for e in self._eliminated],
            moments=self.camera.moment_captures,
        )

        # 2. YOLO Velocity Tracking Logic
        boxes = pose.get("raw_data").boxes if pose and pose.get("raw_data") else None

        if boxes and boxes.id is not None:
            for box, trk_id in zip(boxes.xyxy, boxes.id):
                tid = int(trk_id.item())
                cx = float((box[0] + box[2]) / 2)
                cy = float((box[1] + box[3]) / 2)

                if tid not in self._baseline_positions:
                    # Register new player position
                    self._baseline_positions[tid] = (cx, cy)
                else:
                    # Calculate Euclidean distance from baseline
                    bx, by = self._baseline_positions[tid]
                    dist = math.hypot(cx - bx, cy - by)

                    if dist > MOTION_THRESHOLD and self._in_state() >= TURN_SETTLE_TIME:
                        # --- COLOR AI UPGRADE ---
                        raw_box = box.cpu().numpy()

                        # Strict type-checking to satisfy Pylance
                        shirt_color = "unknown"
                        if self._last_frame is not None:
                            shirt_color = ColorAnalyzer.get_shirt_color(self._last_frame, raw_box)

                        player_label = f"Player in the {shirt_color} shirt"

                        print(f"[Elimination] {player_label} (ID {tid}) moved {int(dist)} pixels!")
                        self.camera.capture_moment("CAUGHT!")

                        self._do_eliminate(frame, tid, (cx, cy), player_label)
                        return

        if self._in_state() >= self._light_dur:
            self.servo.face_away()
            self._light_dur = random.uniform(GREEN_LIGHT_MIN, GREEN_LIGHT_MAX)
            self.audio.announce_green()
            self._go(State.TURNING_GREEN)

    def _do_turning_green(self, frame, overlay, pose):
        self.ui.draw_game_screen(
            overlay,
            state="TURNING",
            players_alive=pose.get("players_alive", 0) if pose else 0,
            elapsed=time.time() - self._game_start,
            round_num=self._round,
            eliminated=[{"label": e.label} for e in self._eliminated],
            moments=self.camera.moment_captures,
        )

        if not self.servo.is_facing_players and self._in_state() > 0.8:
            self._go(State.GREEN)

    # =========================================================================
    # --- Elimination / Win Handlers ---
    # =========================================================================

    def _do_eliminate(self, frame, player_id, pos, label):
        entry = EliminatedEntry(player_id=player_id, label=label, last_pos=pos)
        self._eliminated.append(entry)
        self.audio.announce_elimination(label)
        self._last_elim = entry
        self._go(State.ELIMINATION)

    def _do_elimination(self, frame, overlay):
        self.ui.draw_elimination(overlay, label=getattr(self._last_elim, "label", "UNKNOWN") if self._last_elim else "UNKNOWN")
        if self._in_state() >= ELIMINATION_HOLD:
            self._baseline_positions.clear()  # Reset baselines so survivors aren't instantly caught
            self._go(State.RED)

    def _trigger_winner(self, label: str):
        self._winner_label = label
        self.camera.capture_moment("WINNER!")
        self.servo.face_away()
        self.audio.announce_winner()
        self._go(State.WINNER)

    def _do_winner(self, frame, overlay, pose):
        self.ui.draw_win_screen(moments=self.camera.moment_captures, winner_label=self._winner_label)
        if self._palm_progress(self._check_palm_yolo(pose)) >= 1.0:
            self._palm_since = None
            self._do_reset()

    def _do_reset(self):
        self.servo.face_away()
        self._eliminated = []
        self._baseline_positions.clear()
        self._round = 0
        self.camera.moment_captures.clear()
        self.audio.play_music("bgm")
        self._go(State.START)

    # =========================================================================
    # --- YOLO Utility Logic ---
    # =========================================================================

    def _check_palm_yolo(self, pose_data) -> bool:
        """Heuristic: Are wrists significantly higher than shoulders?"""
        if not pose_data or not pose_data.get("raw_data"):
            return False
        kpts = pose_data["raw_data"].keypoints
        if kpts is None or not hasattr(kpts, "xy") or len(kpts.xy) == 0:
            return False

        for person in kpts.xy:
            if len(person) > 10:
                shoulder_y = min(person[5][1], person[6][1])  # Y-axis origin is top-left
                wrist_y = min(person[9][1], person[10][1])
                if shoulder_y > 0 and wrist_y > 0 and wrist_y < (shoulder_y - 80):
                    return True
        return False

    def _palm_progress(self, palm_up: bool) -> float:
        if palm_up:
            if self._palm_since is None:
                self._palm_since = time.time()
            return min(1.0, (time.time() - self._palm_since) / self._PALM_HOLD)
        self._palm_since = None
        return 0.0

    def _handle_key(self, key):
        if key == pygame.K_ESCAPE:
            raise SystemExit
        if key == pygame.K_g:
            self._go(State.GREEN)
        if key == pygame.K_r:
            self._baseline_positions.clear()
            self._go(State.RED)
        if key == pygame.K_w:
            self._trigger_winner("DEBUG WIN")
        if key == pygame.K_SPACE:
            self._palm_since = time.time() - self._PALM_HOLD
