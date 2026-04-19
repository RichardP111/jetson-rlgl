#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Project:      Red Light Green Light (Jetson Orin Nano)
File:         game.py
Description:  Runs the core finite state machine governing the
              Green Light, Red Light, and Elimination phases.

Author:       Richard Pu
Last Updated: April 2026
===============================================================================
"""

import time
import random
import pygame
from enum import Enum, auto
from dataclasses import dataclass, field

from config import (
    FPS_CAP,
    GREEN_LIGHT_MIN,
    GREEN_LIGHT_MAX,
    RED_LIGHT_MIN,
    RED_LIGHT_MAX,
    TURN_SETTLE_TIME,
    GRACE_PERIOD,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    ELIMINATION_HOLD,
)
from hardware import Camera, ServoController, LaserBreakBeam
from vision import PoseTracker, MotionDetector, check_finish_line
from audio import AudioManager
from ui import UIRenderer


# ─────────────────────────────────────────────────────────────────────
class State(Enum):
    START = auto()  # Idle / attract screen
    COUNTDOWN = auto()  # 3-2-1 before game
    GREEN = auto()  # Players may move
    TURNING_RED = auto()  # Head rotating to face players
    RED = auto()  # Players must freeze
    ELIMINATION = auto()  # Someone was caught
    TURNING_GREEN = auto()  # Head rotating away
    WINNER = auto()  # Someone crossed the finish
    RESET = auto()  # Brief cleanup before looping


# ─────────────────────────────────────────────────────────────────────
@dataclass
class EliminatedEntry:
    player_id: int
    label: str
    last_pos: tuple  # (cx, cy) in camera-space
    colour_sig: dict | None = None
    ts: float = field(default_factory=time.time)


# ─────────────────────────────────────────────────────────────────────
class GameEngine:

    def __init__(
        self,
        camera: Camera,
        servo: ServoController,
        laser: LaserBreakBeam,
        tracker: PoseTracker,
        motion: MotionDetector,
        audio: AudioManager,
        ui: UIRenderer,
    ):

        self.camera = camera
        self.servo = servo
        self.laser = laser
        self.tracker = tracker
        self.motion = motion
        self.audio = audio
        self.ui = ui

        # ── Game state ────────────────────────────────────────────
        self._state = State.START
        self._state_time = time.time()
        self._game_start = 0.0
        self._round = 0
        self._light_dur = 0.0  # current phase duration
        self._eliminated: list[EliminatedEntry] = []
        self._winner_label = ""

        # ── Palm-hold gesture tracking ────────────────────────────
        self._palm_since: float | None = None
        PALM_HOLD = 2.0  # seconds to hold palm up

        self._PALM_HOLD = PALM_HOLD

        # ── Last seen pose data ───────────────────────────────────
        self._last_pose = None
        self._last_frame = None

    # ═══════════════════════════════════════════════════════════════
    #  Helpers
    # ═══════════════════════════════════════════════════════════════

    def _go(self, new_state: State):
        print(f"  [{self._state.name}] → [{new_state.name}]")
        self._state = new_state
        self._state_time = time.time()

    def _in_state(self) -> float:
        return time.time() - self._state_time

    def _rand_green(self) -> float:
        return random.uniform(GREEN_LIGHT_MIN, GREEN_LIGHT_MAX)

    def _rand_red(self) -> float:
        return random.uniform(RED_LIGHT_MIN, RED_LIGHT_MAX)

    def _palm_progress(self, palm_up: bool) -> float:
        """Returns 0.0-1.0 completion of palm-hold gesture."""
        if palm_up:
            if self._palm_since is None:
                self._palm_since = time.time()
            held = time.time() - self._palm_since
            return min(1.0, held / self._PALM_HOLD)
        else:
            self._palm_since = None
            return 0.0

    # ═══════════════════════════════════════════════════════════════
    #  Main loop
    # ═══════════════════════════════════════════════════════════════

    def run(self, clock: pygame.time.Clock):
        print("[GAME] Engine started. Press G/R for debug shortcuts, ESC to quit.")
        self.servo.face_away()
        self.audio.play_music("bgm")

        while True:
            # ── Event handling ────────────────────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    self._handle_key(event.key)

            # ── Frame grab ────────────────────────────────────────
            frame = self.camera.read()
            self._last_frame = frame

            # ── Pose tracking (every frame) ───────────────────────
            pose_data, overlay = self.tracker.process_frame(frame)
            self._last_pose = pose_data

            # ── Dispatch ─────────────────────────────────────────
            self._dispatch(frame, overlay, pose_data)

            clock.tick(FPS_CAP)

    def _handle_key(self, key: int):
        """Debug keyboard shortcuts (remove for production)."""
        if key == pygame.K_ESCAPE:
            raise SystemExit
        if key == pygame.K_g:
            self._go(State.GREEN)
            self._light_dur = self._rand_green()
        if key == pygame.K_r:
            self._start_red_sequence()
        if key == pygame.K_w:
            self._trigger_winner()
        if key == pygame.K_e:
            self._eliminate_debug()
        if key == pygame.K_SPACE:
            # Force palm-start
            self._palm_since = time.time() - self._PALM_HOLD

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

    # ═══════════════════════════════════════════════════════════════
    #  State handlers
    # ═══════════════════════════════════════════════════════════════

    # ── START / idle screen ───────────────────────────────────────
    def _do_start(self, frame, overlay, pose):
        palm_up = self.tracker.detect_palm_raise(frame)
        prog = self._palm_progress(palm_up)

        self.ui.draw_start_screen(frame, palm_held_ratio=prog)

        if prog >= 1.0:
            self._palm_since = None
            print("[GAME] Palm gesture confirmed — starting countdown")
            self.camera.capture_moment("Game Start")
            self.audio.say("Get ready! The game is starting!")
            self._go(State.COUNTDOWN)

    # ── COUNTDOWN ────────────────────────────────────────────────
    def _do_countdown(self, frame, overlay):
        t = self._in_state()
        n = 3 - int(t)

        # Draw start screen behind overlay
        self.ui.draw_start_screen(frame)
        if n > 0:
            self.ui.draw_countdown_overlay(n)
            if abs(t - (3 - n)) < 0.05:  # first tick of each second
                self.audio.countdown_beep()
        else:
            # Launch green light
            self._round = 1
            self._eliminated = []
            self._game_start = time.time()
            self._light_dur = self._rand_green()
            self.servo.face_away()
            self.audio.announce_green()
            self._go(State.GREEN)

    # ── GREEN LIGHT ───────────────────────────────────────────────
    def _do_green(self, frame, overlay, pose):
        elapsed = time.time() - self._game_start
        elim_dicts = [{"label": e.label} for e in self._eliminated]

        self.ui.draw_game_screen(
            overlay,
            state="GREEN",
            players_alive=self._alive_count(pose),
            elapsed=elapsed,
            round_num=self._round,
            eliminated=elim_dicts,
            moments=self.camera.moment_captures,
        )

        # Occasional action captures
        if random.random() < 0.004:
            self.camera.capture_moment("Action")

        # Check finish
        if self._check_finish(frame, pose):
            return

        # Time up → go red
        if self._in_state() >= self._light_dur:
            self._start_red_sequence()

    def _start_red_sequence(self):
        self._round += 1
        self.servo.face_players()
        self.audio.announce_red()
        self._light_dur = self._rand_red()
        self._go(State.TURNING_RED)

    # ── TURNING → RED ────────────────────────────────────────────
    def _do_turning_red(self, frame, overlay, pose):
        elapsed = time.time() - self._game_start
        elim_dicts = [{"label": e.label} for e in self._eliminated]

        self.ui.draw_game_screen(
            overlay,
            state="TURNING",
            players_alive=self._alive_count(pose),
            elapsed=elapsed,
            round_num=self._round,
            eliminated=elim_dicts,
            moments=self.camera.moment_captures,
        )

        # Wait for head to settle, then activate red
        settled = self.servo.is_facing_players or self._in_state() >= TURN_SETTLE_TIME + 1.0
        if settled:
            time.sleep(GRACE_PERIOD)
            self.motion.set_reference(self._last_frame or frame)
            self._go(State.RED)

    # ── RED LIGHT ─────────────────────────────────────────────────
    def _do_red(self, frame, overlay, pose):
        motion, _, score = self.motion.detect(frame)
        elapsed = time.time() - self._game_start
        elim_dicts = [{"label": e.label} for e in self._eliminated]

        self.ui.draw_game_screen(
            overlay,
            state="RED",
            players_alive=self._alive_count(pose),
            elapsed=elapsed,
            round_num=self._round,
            eliminated=elim_dicts,
            motion_score=score,
            moments=self.camera.moment_captures,
        )

        # Motion detected → eliminate
        if motion and self._in_state() >= TURN_SETTLE_TIME:
            # Capture the moment (hopefully a funny face)
            self.camera.capture_moment("CAUGHT!")
            self._do_eliminate(frame, pose)
            return

        # Red light phase ended → go green
        if self._in_state() >= self._light_dur:
            self._start_green_sequence()

    def _start_green_sequence(self):
        self.servo.face_away()
        self.motion.reset()
        self._light_dur = self._rand_green()
        self.audio.announce_green()
        self._go(State.TURNING_GREEN)

    # ── TURNING → GREEN ───────────────────────────────────────────
    def _do_turning_green(self, frame, overlay, pose):
        elapsed = time.time() - self._game_start
        elim_dicts = [{"label": e.label} for e in self._eliminated]

        self.ui.draw_game_screen(
            overlay,
            state="TURNING",
            players_alive=self._alive_count(pose),
            elapsed=elapsed,
            round_num=self._round,
            eliminated=elim_dicts,
            moments=self.camera.moment_captures,
        )

        # Once servo finishes turning away (or timeout), proceed to green
        turned_away = not self.servo.is_facing_players or self._in_state() >= TURN_SETTLE_TIME + 0.5
        if turned_away and self._in_state() > 0.8:
            self._go(State.GREEN)

    # ── ELIMINATION ──────────────────────────────────────────────
    def _do_eliminate(self, frame, pose):
        pid = len(self._eliminated) + 1
        colour_sig = None
        pos = (CAMERA_WIDTH // 2, CAMERA_HEIGHT // 2)  # default centre

        if pose is not None:
            pos = pose["centroid"]
            colour_sig = self.tracker.get_colour_signature(frame, pose)

        label = f"Player {pid}"
        entry = EliminatedEntry(player_id=pid, label=label, last_pos=pos, colour_sig=colour_sig)
        self._eliminated.append(entry)
        self.audio.announce_elimination(label)
        self._last_elim = entry
        self._go(State.ELIMINATION)

    def _do_elimination(self, frame, overlay):
        entry = getattr(self, "_last_elim", None)

        # Convert camera-space pos to screen-space
        screen_pos = None
        if entry:
            cx, cy = entry.last_pos
            fh, fw = (CAMERA_HEIGHT, CAMERA_WIDTH)
            # camera feed occupies left 64% of screen
            cam_sw = int(self.ui.W * 0.64)
            cam_sh = int(cam_sw * fh / fw)
            cam_sy = (self.ui.H - cam_sh) // 2
            sx = int(cx / fw * cam_sw)
            sy = int(cy / fh * cam_sh) + cam_sy
            screen_pos = (sx, sy)

        self.ui.draw_elimination(frame, label=entry.label if entry else "", screen_pos=screen_pos)

        if self._in_state() >= ELIMINATION_HOLD:
            # Resume red light (set fresh reference so returner doesn't trigger)
            self.motion.set_reference(self._last_frame or frame)
            self._go(State.RED)

    # ── WINNER ───────────────────────────────────────────────────
    def _trigger_winner(self, label: str = "FINISH LINE CROSSED!"):
        self._winner_label = label
        self.camera.capture_moment("WINNER!")
        self.servo.face_away()
        self.audio.announce_winner()
        self._go(State.WINNER)

    def _do_winner(self, frame, overlay, pose):
        self.ui.draw_win_screen(moments=self.camera.moment_captures, winner_label=self._winner_label)

        # Palm raise to restart
        palm_up = self.tracker.detect_palm_raise(frame)
        prog = self._palm_progress(palm_up)
        if prog >= 1.0:
            self._palm_since = None
            self._go(State.RESET)

    # ── RESET ─────────────────────────────────────────────────────
    def _do_reset(self):
        self.servo.face_away()
        self.motion.reset()
        self._eliminated = []
        self._round = 0
        self._winner_label = ""
        self.camera.moment_captures.clear()
        self.audio.play_music("bgm")
        self._go(State.START)

    # ═══════════════════════════════════════════════════════════════
    #  Utility
    # ═══════════════════════════════════════════════════════════════

    def _check_finish(self, frame, pose) -> bool:
        """Returns True and triggers winner state if finish detected."""
        crossed = check_finish_line(frame, pose, laser_broken=self.laser.broken)
        if crossed:
            self._trigger_winner("FINISH LINE REACHED!")
            return True
        return False

    def _alive_count(self, pose) -> int:
        """
        Best-effort alive count. MediaPipe tracks one body at a time,
        so we report 'detected' (1 or 0) + known-total heuristic.
        In a real multi-person setup you'd track each player.
        """
        detected = 1 if pose is not None else 0
        return max(detected, 0)

    def _eliminate_debug(self):
        """Debug: instant fake elimination."""
        entry = EliminatedEntry(
            player_id=len(self._eliminated) + 1,
            label=f"Player {len(self._eliminated)+1}",
            last_pos=(640, 360),
        )
        self._eliminated.append(entry)
        self._last_elim = entry
        self._go(State.ELIMINATION)
