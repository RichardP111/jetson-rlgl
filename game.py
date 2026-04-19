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

States:
    START          Idle / attract screen, waiting for palm gesture
    COUNTDOWN      3 - 2 - 1 before the game begins
    GREEN          Players may move; servo faces away
    TURNING_RED    Servo sweeping to face players; grace window
    RED            Freeze! Motion detection active
    CAUGHT_PAUSE   Someone was caught; game pauses for them to walk back
    TURNING_GREEN  Servo sweeping back; transition to green
    WINNER         Finish crossed; winner screen
    RESET          Cleanup before returning to START
===============================================================================
"""

import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
import pygame

from audio import AudioManager
from config import (CAM_H, CAM_W, CAUGHT_PAUSE_S, COUNTDOWN_N, FPS_CAP,
                    GRACE_S, GREEN_MAX, GREEN_MIN, MOTION_PX, PALM_HOLD_S,
                    RED_MAX, RED_MIN, SETTLE_S)
from hardware import Camera, LaserBreakBeam, ServoController
from ui import UIRenderer
from vision import PoseTracker, check_tape_finish, detect_palm_raise

# ════════════════════════════════════════════════════════════════════


class State(Enum):
    START = auto()
    COUNTDOWN = auto()
    GREEN = auto()
    TURNING_RED = auto()
    RED = auto()
    CAUGHT_PAUSE = auto()
    TURNING_GREEN = auto()
    WINNER = auto()
    RESET = auto()


@dataclass
class CaughtEntry:
    track_id: int
    shirt_colour: str
    cx: float  # camera-space centroid X
    cy: float
    ts: float = field(default_factory=time.time)


# ════════════════════════════════════════════════════════════════════
class GameEngine:

    def __init__(self, camera: Camera, servo: ServoController, laser: LaserBreakBeam, tracker: PoseTracker, audio: AudioManager, ui: UIRenderer):

        self.camera = camera
        self.servo = servo
        self.laser = laser
        self.tracker = tracker
        self.audio = audio
        self.ui = ui

        # ── State machine ─────────────────────────────────────────
        self._state = State.START
        self._state_ts = time.time()

        # ── Game vars ─────────────────────────────────────────────
        self._round = 0
        self._game_start_ts = 0.0
        self._light_dur = 0.0
        self._caught: list[CaughtEntry] = []
        self._last_caught: CaughtEntry | None = None
        self._winner_colour = "unknown"
        self._highlights: list[np.ndarray] = []  # frames for win gallery

        # ── Per-player baseline positions for motion detection ────
        # {track_id: (cx, cy)}  — set when red light activates
        self._baselines: dict[int, tuple[float, float]] = {}

        # ── Palm-hold gesture ─────────────────────────────────────
        self._palm_since: float | None = None

        # ── Latest vision data ────────────────────────────────────
        self._last_frame = None
        self._last_pose_data = None

    # ════════════════════════════════════════════════════════════════
    #  RUN LOOP
    # ════════════════════════════════════════════════════════════════

    def run(self, clock: pygame.time.Clock):
        print("[ENGINE] Game live.  Keys: G=Green R=Red W=Win E=Elim SPACE=palm ESC=quit")
        self.servo.face_away()
        self.audio.play_music("bgm")

        while True:
            # ── Events ───────────────────────────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    self._handle_key(event.key)

            # ── Grab frame + run YOLO ─────────────────────────────
            frame = self.camera.read()
            self._last_frame = frame

            pose_data, overlay = self.tracker.process_frame(frame)
            self._last_pose_data = pose_data

            # ── Dispatch ─────────────────────────────────────────
            self._dispatch(frame, overlay, pose_data)

            clock.tick(FPS_CAP)

    # ════════════════════════════════════════════════════════════════
    #  DISPATCH
    # ════════════════════════════════════════════════════════════════

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
        elif s == State.CAUGHT_PAUSE:
            self._do_caught_pause(frame, overlay)
        elif s == State.TURNING_GREEN:
            self._do_turning_green(frame, overlay, pose)
        elif s == State.WINNER:
            self._do_winner(frame, overlay, pose)
        elif s == State.RESET:
            self._do_reset()

    # ════════════════════════════════════════════════════════════════
    #  STATE HANDLERS
    # ════════════════════════════════════════════════════════════════

    # ── START ─────────────────────────────────────────────────────

    def _do_start(self, frame, overlay, pose):
        palm = detect_palm_raise(pose)
        prog = self._palm_progress(palm)
        self.ui.draw_start_screen(overlay, palm_progress=prog)
        if prog >= 1.0:
            self._palm_since = None
            self.audio.on_game_start()
            self._go(State.COUNTDOWN)

    # ── COUNTDOWN ─────────────────────────────────────────────────

    def _do_countdown(self, frame, overlay):
        t = self._in_state()
        n = COUNTDOWN_N - int(t)
        self.ui.draw_countdown(overlay, max(1, n))

        # Beep each integer boundary
        elapsed_floor = int(t)
        if not hasattr(self, "_last_beep") or self._last_beep != elapsed_floor:
            self._last_beep = elapsed_floor
            if n > 0:
                self.audio.on_countdown(n)

        if t >= COUNTDOWN_N:
            self._round = 1
            self._caught = []
            self._highlights = []
            self._game_start_ts = time.time()
            self._light_dur = random.uniform(GREEN_MIN, GREEN_MAX)
            self.servo.face_away()
            self.audio.on_green()
            self._go(State.GREEN)

    # ── GREEN ─────────────────────────────────────────────────────

    def _do_green(self, frame, overlay, pose):
        elapsed = time.time() - self._game_start_ts
        alive = pose["players_alive"] if pose else 0
        motion_s = 0.0

        self.ui.draw_game_hud(overlay, "GREEN", alive, elapsed, self._round, [e.shirt_colour for e in self._caught], motion_s)

        # Snapshot a highlight every ~8 seconds
        if frame is not None and random.random() < 0.003:
            self._highlights.append(frame.copy())

        # Finish-line check
        if self._check_finish(frame, pose):
            return

        # Time's up → go red
        if self._in_state() >= self._light_dur:
            self._initiate_red()

    def _initiate_red(self):
        self._round += 1
        self.servo.face_players()
        self.audio.on_red()
        self._light_dur = random.uniform(RED_MIN, RED_MAX)
        self._go(State.TURNING_RED)

    # ── TURNING → RED ─────────────────────────────────────────────

    def _do_turning_red(self, frame, overlay, pose):
        elapsed = time.time() - self._game_start_ts
        alive = pose["players_alive"] if pose else 0

        self.ui.draw_game_hud(overlay, "TURNING", alive, elapsed, self._round, [e.shirt_colour for e in self._caught])

        # Wait for servo to reach position (or fall back to time)
        settled = self.servo.is_facing_players or self._in_state() >= SETTLE_S + 0.8
        if settled and self._in_state() >= SETTLE_S:
            # Grace period — snapshot for baseline
            time.sleep(GRACE_S)
            self._snapshot_baselines(pose)
            self._go(State.RED)

    # ── RED ───────────────────────────────────────────────────────

    def _do_red(self, frame, overlay, pose):
        elapsed = time.time() - self._game_start_ts
        alive = pose["players_alive"] if pose else 0
        motion_score = 0.0

        caught_entry, motion_score = self._check_motion(pose, frame)

        self.ui.draw_game_hud(overlay, "RED", alive, elapsed, self._round, [e.shirt_colour for e in self._caught], motion_score)

        if caught_entry is not None:
            self._last_caught = caught_entry
            self._caught.append(caught_entry)
            # Particle burst at caught position
            self.ui.flash_caught(int(caught_entry.cx), int(caught_entry.cy))
            # Snapshot
            if frame is not None:
                self._highlights.append(frame.copy())
            self.audio.on_caught(caught_entry.shirt_colour)
            self._go(State.CAUGHT_PAUSE)
            return

        # Finish-line check (can still win during red light)
        if self._check_finish(frame, pose):
            return

        # Phase over → go green
        if self._in_state() >= self._light_dur:
            self._initiate_green()

    def _initiate_green(self):
        self.servo.face_away()
        self._baselines.clear()
        self._light_dur = random.uniform(GREEN_MIN, GREEN_MAX)
        self.audio.on_green()
        self._go(State.TURNING_GREEN)

    # ── CAUGHT PAUSE ──────────────────────────────────────────────

    def _do_caught_pause(self, frame, overlay):
        progress = min(1.0, self._in_state() / CAUGHT_PAUSE_S)
        colour = self._last_caught.shirt_colour if self._last_caught else "unknown"
        self.ui.draw_caught_screen(frame, colour, progress)

        if self._in_state() >= CAUGHT_PAUSE_S:
            # Refresh baselines so returning player doesn't re-trigger
            self._snapshot_baselines(self._last_pose_data)
            self._go(State.RED)

    # ── TURNING → GREEN ───────────────────────────────────────────

    def _do_turning_green(self, frame, overlay, pose):
        elapsed = time.time() - self._game_start_ts
        alive = pose["players_alive"] if pose else 0

        self.ui.draw_game_hud(overlay, "TURNING", alive, elapsed, self._round, [e.shirt_colour for e in self._caught])

        # Proceed once servo has turned away (or timeout)
        if not self.servo.is_facing_players and self._in_state() > 0.8:
            self._go(State.GREEN)

    # ── WINNER ────────────────────────────────────────────────────

    def _do_winner(self, frame, overlay, pose):
        palm = detect_palm_raise(pose)
        prog = self._palm_progress(palm)

        self.ui.draw_winner_screen(
            self._winner_colour,
            self._highlights,
            palm_progress=prog,
        )

        if prog >= 1.0:
            self._palm_since = None
            self._go(State.RESET)

    # ── RESET ─────────────────────────────────────────────────────

    def _do_reset(self):
        self.servo.face_away()
        self._baselines.clear()
        self._caught = []
        self._highlights = []
        self._round = 0
        self._winner_colour = "unknown"
        self._last_caught = None
        self._palm_since = None
        self.audio.play_music("bgm")
        self._go(State.START)

    # ════════════════════════════════════════════════════════════════
    #  MOTION DETECTION (position-based, per tracked ID)
    # ════════════════════════════════════════════════════════════════

    def _snapshot_baselines(self, pose):
        """Lock current bounding-box centroids as 'frozen' positions."""
        self._baselines.clear()
        if pose is None:
            return
        boxes = pose.get("boxes", [])
        track_ids = pose.get("track_ids", [])
        for box, tid in zip(boxes, track_ids):
            if tid is None:
                continue
            cx = float((box[0] + box[2]) / 2)
            cy = float((box[1] + box[3]) / 2)
            self._baselines[tid] = (cx, cy)

    def _check_motion(self, pose, frame) -> tuple[CaughtEntry | None, float]:
        """
        Compare current positions against baselines.
        Returns (CaughtEntry, motion_score) — CaughtEntry is None if no one moved.
        motion_score is 0.0 – 1.0 for the motion bar.
        """
        if pose is None or not self._baselines:
            return None, 0.0
        if self._in_state() < SETTLE_S:
            return None, 0.0

        boxes = pose.get("boxes", [])
        track_ids = pose.get("track_ids", [])
        shirt_cols = pose.get("shirt_colours", [])
        max_dist = 0.0

        for i, (box, tid, colour) in enumerate(zip(boxes, track_ids, shirt_cols)):
            if tid is None or tid not in self._baselines:
                continue
            cx = float((box[0] + box[2]) / 2)
            cy = float((box[1] + box[3]) / 2)
            bx, by = self._baselines[tid]
            dist = math.hypot(cx - bx, cy - by)
            max_dist = max(max_dist, dist)

            if dist > MOTION_PX:
                entry = CaughtEntry(
                    track_id=tid,
                    shirt_colour=colour,
                    cx=cx,
                    cy=cy,
                )
                return entry, min(1.0, dist / (MOTION_PX * 4))

        # No catch — still compute score for motion bar
        score = min(1.0, max_dist / (MOTION_PX * 3)) if self._baselines else 0.0
        return None, score

    # ════════════════════════════════════════════════════════════════
    #  FINISH LINE
    # ════════════════════════════════════════════════════════════════

    def _check_finish(self, frame, pose) -> bool:
        if self.laser.broken:
            self._trigger_winner(pose, frame, "LASER CROSSED!")
            return True
        if check_tape_finish(frame, pose):
            self._trigger_winner(pose, frame, "FINISH LINE!")
            return True
        return False

    def _trigger_winner(self, pose, frame, reason: str = ""):
        print(f"[WIN] {reason}")
        # Identify winner by shirt colour
        colour = "unknown"
        if pose and pose.get("shirt_colours"):
            colour = pose["shirt_colours"][0]
        self._winner_colour = colour
        if frame is not None:
            self._highlights.append(frame.copy())
        self.servo.face_away()
        self.audio.on_winner()
        self._go(State.WINNER)

    # ════════════════════════════════════════════════════════════════
    #  UTILITIES
    # ════════════════════════════════════════════════════════════════

    def _go(self, new_state: State):
        print(f"  [{self._state.name}] → [{new_state.name}]")
        self._state = new_state
        self._state_ts = time.time()

    def _in_state(self) -> float:
        return time.time() - self._state_ts

    def _palm_progress(self, palm_up: bool) -> float:
        if palm_up:
            if self._palm_since is None:
                self._palm_since = time.time()
            return min(1.0, (time.time() - self._palm_since) / PALM_HOLD_S)
        self._palm_since = None
        return 0.0

    def _handle_key(self, key: int):
        """Debug shortcuts — remove before STEM day if desired."""
        if key == pygame.K_ESCAPE:
            raise SystemExit
        if key == pygame.K_g:
            self.servo.face_away()
            self._light_dur = random.uniform(GREEN_MIN, GREEN_MAX)
            self.audio.on_green()
            self._go(State.GREEN)
        if key == pygame.K_r:
            self._snapshot_baselines(self._last_pose_data)
            self._light_dur = random.uniform(RED_MIN, RED_MAX)
            self.servo.face_players()
            self.audio.on_red()
            self._go(State.RED)
        if key == pygame.K_w:
            self._winner_colour = "blue"
            self.audio.on_winner()
            self._go(State.WINNER)
        if key == pygame.K_e:
            # Fake an elimination
            dummy = CaughtEntry(track_id=99, shirt_colour="red", cx=640, cy=360)
            self._last_caught = dummy
            self._caught.append(dummy)
            self.ui.flash_caught(640, 360)
            self.audio.on_caught("red")
            self._go(State.CAUGHT_PAUSE)
        if key == pygame.K_SPACE:
            # Shortcut palm hold
            self._palm_since = time.time() - PALM_HOLD_S
