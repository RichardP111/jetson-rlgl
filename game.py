#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Project:      Red Light Green Light (Jetson Orin Nano)
File:         game.py
Description:  Finite state machine governing game phases. Owns no rendering;
              it delegates all visuals to UIRenderer and all hardware to the
              Camera / ServoController / LaserBreakBeam objects.


Author:       Richard Pu
Last Updated: April 2026

States:
    START           Attract screen, waiting for palm gesture
    COUNTDOWN       3 - 2 - 1 before the game begins
    GREEN           Players may move; servo faces away
    TURNING_RED     Servo sweeping to face players; grace window
    RED             Freeze! Motion detection active
    CAUGHT_PAUSE    Someone was caught; game pauses for them to reset
    TURNING_GREEN   Servo sweeping back
    WINNER          Finish crossed
    RESET           Cleanup before returning to START
===============================================================================
"""

from __future__ import annotations

import math
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import cv2  # type: ignore
import numpy as np
import pygame

from audio import AudioManager
from config import (
    CAUGHT_RETURN_DWELL_S,
    CAUGHT_RETURN_GRACE_S,
    CAUGHT_RETURN_MAX_S,
    CAUGHT_RETURN_RECHECK_HZ,
    COUNTDOWN_N,
    DEBUG_SKIP_FINISH,
    DEFAULT_DIFFICULTY,
    DIFFICULTY_PRESETS,
    EASE_CHECK_EVERY_S,
    EASE_ENABLED,
    EASE_MAX_GREEN_S,
    EASE_MAX_MOTION_PX,
    EASE_MIN_RED_S,
    EASE_STEP_GREEN_S,
    EASE_STEP_MOTION_PX,
    EASE_STEP_RED_S,
    EASE_TRIGGER_AFTER_S,
    FPS_CAP,
    GRACE_PERIOD,
    HIGHLIGHT_MAX,
    HIGHLIGHT_SCALE,
    LEADERBOARD_1ST_CELEBRATE_S,
    LEADERBOARD_1ST_DELAY_S,
    LEADERBOARD_1ST_REVEAL_S,
    LEADERBOARD_BLAST_OUT_S,
    LEADERBOARD_PALM_ARMED_S,
    PALM_HOLD,
    PALM_HOLD_LEADERBOARD,
    PHASE_EXTREME_BIAS,
    PHASE_FAKEOUT_MAX,
    PHASE_FAKEOUT_MIN,
    PHASE_FAKEOUT_PROB,
    PROFILE_CAPTURE_ENABLED,
    PROFILE_CAPTURE_MIN_BBOX_PX,
    SETTLE_TIME,
    START_LINE_DWELL_S,
    START_LINE_MAX_WAIT_S,
    START_LINE_REQUIRED,
    USE_LASER,
    USE_TAPE_FINISH,
    CAM_W,
    CAM_H,
    DISPLAY_W, 
    DISPLAY_H
)
from hardware import Camera, LaserBreakBeam, ServoController
from ui import UIRenderer
from vision import (
    LineDetector,
    PhotoCapture,
    PlayerDescriber,
    PoseWorker,
    ProPoseTracker,
    detect_palm_raise,
)


# ===========================================================================
# State
# ===========================================================================
class State(Enum):
    START = auto()
    CALIBRATE_LINES = auto()
    COUNTDOWN = auto()
    WAIT_START_LINE = auto()
    TURNING_GREEN = auto()
    GREEN = auto()
    TURNING_RED = auto()
    RED = auto()
    CAUGHT_RETURN = auto()
    ALL_ELIMINATED_HOLD = auto()
    LEADERBOARD = auto()
    RESET = auto()


_BANNER_LABEL = {
    State.START: "START",
    State.CALIBRATE_LINES: "CALIBRATION",
    State.COUNTDOWN: "COUNTDOWN",
    State.WAIT_START_LINE: "START_LINE",
    State.TURNING_GREEN: "TURNING",
    State.GREEN: "GREEN",
    State.TURNING_RED: "TURNING",
    State.RED: "RED",
    State.CAUGHT_RETURN: "ELIMINATED",
    State.ALL_ELIMINATED_HOLD: "ELIMINATED",
    State.LEADERBOARD: "LEADERBOARD",
    State.RESET: "START",
}


# ===========================================================================
# Player record — persists across the whole round
# ===========================================================================
@dataclass
class Player:
    track_id: int
    descriptor: str = "Player"
    photo_crop: Optional[np.ndarray] = None  # best BGR crop
    photo_quality: float = 0.0
    photo_surface: Optional[pygame.Surface] = None  # built lazily for leaderboard
    finished: bool = False
    rank: int = 0
    finish_time_s: float = 0.0
    times_caught: int = 0
    baseline_pos: tuple[float, float] = (0.0, 0.0)
    last_box: Optional[np.ndarray] = None
    last_seen_ts: float = 0.0
    is_caught_this_phase: bool = False
    needs_to_return: bool = False
    is_back_at_start: bool = False
    descriptor_pending: bool = False  # async describer in flight


# ===========================================================================
# Game engine
# ===========================================================================
class GameEngine:
    def __init__(
        self,
        camera: Camera,
        servo: ServoController,
        laser: LaserBreakBeam,
        tracker: ProPoseTracker,
        audio: AudioManager,
        ui: UIRenderer,
        pose_worker: PoseWorker,
        describer: PlayerDescriber,
    ) -> None:
        self.camera = camera
        self.servo = servo
        self.laser = laser
        self.tracker = tracker
        self.pose_worker = pose_worker
        self.audio = audio
        self.ui = ui
        self.describer = describer
        self.line_detector = LineDetector()
        self._calib_points: list[tuple[int, int]] = []

        self._state = State.START
        self._state_ts = time.time()
        self._game_start_ts = 0.0

        self._players: dict[int, Player] = {}
        self._players_locked = False  # True after WAIT_START_LINE → GREEN
        self._next_rank = 1
        self._last_finish_ts = 0.0

        # Light-phase state
        self._light_dur = 0.0
        self._motion_score_smooth = 0.0
        self._motion_score = 0.0

        # Caught-return state
        self._return_grace_until = 0.0

        # Photo retry budget
        self._photo_retry_phases_used = 0

        # Highlights
        self._highlights: deque[np.ndarray] = deque(maxlen=HIGHLIGHT_MAX)

        # Palm gate
        self._palm_since: float | None = None
        self._last_beep = -1

        # Last frame/pose pair we saw
        self._last_frame: np.ndarray | None = None
        self._last_frame_id: int = 0
        # Apr 2026 — runtime debug toggle. Seeded from config; flipped
        # by the F key. When True, _check_finishes() is a no-op so the
        # round never auto-advances to LEADERBOARD; players can still
        # be caught and walk back, which is exactly what's needed to
        # test red/green light timing without the finish tape set up.
        self._skip_finish_active: bool = bool(DEBUG_SKIP_FINISH)
        self._last_pose: dict | None = None
        self._last_pose_id: int = -1

        # Apr 2026 — fake leaderboard demo (K key). When non-None, the
        # leaderboard renders these results instead of building from
        # self._players, so the choreography can be tested without
        # actually playing a round. Cleared on RESET.
        self._demo_leaderboard_results: list[dict] | None = None

        # Apr 2026 — track-id-independent motion baselines. List of
        # (cx, cy, Player) snapshotted at start of each RED phase.
        # Greedy nearest-neighbour matched against each frame's
        # detections so motion catches still work when bytetrack
        # re-assigns track IDs mid-round (which it does often when a
        # player is briefly occluded). Cleared on RED → !RED.
        self._red_baselines: list[tuple[float, float, "Player"]] = []

        # Difficulty
        self._difficulty = DEFAULT_DIFFICULTY
        d = DIFFICULTY_PRESETS[self._difficulty]
        self._green_min, self._green_max = d["green_min"], d["green_max"]
        self._red_min, self._red_max = d["red_min"], d["red_max"]
        self._motion_px = d["motion_px"]
        self._ease_steps = 0
        self._last_ease_check_ts = 0.0

        # Cached line positions for the UI overlay
        self._cached_start_y = 0
        self._cached_finish_y = 0

    # ==================================================================
    # Main loop
    # ==================================================================
    def run(self, clock: pygame.time.Clock) -> None:
        print("[ENGINE] Running. Hotkeys: H dev | G/R force | W debug-finish | " "L leaderboard | F skip-finish | SPACE bypass | ESC quit")
        self.servo.face_away()
        self.audio.play_music("bgm")

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    self._handle_key(event)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self._handle_mouse(event)

            frame, frame_id = self.camera.read_with_id()
            pose, pose_id = self.pose_worker.latest()

            self._last_frame = frame
            self._last_frame_id = frame_id
            if pose_id != self._last_pose_id:
                self._last_pose = pose
                self._last_pose_id = pose_id

            # Update line-detector cache (used by UI overlay AND by the FSM)
            if frame is not None:
                start_line, sc = self.line_detector.detect_start(frame)
                finish_line, fc = self.line_detector.detect_finish(frame)
                self._cached_start_line = start_line
                self._cached_finish_line = finish_line
                
                # Pass the camera dimensions to the UI so it can scale the tilt properly
                self.ui.set_line_calibration(frame.shape[1], frame.shape[0], start_line, sc, finish_line, fc)

            # Hand the UI the current frame_id so it can dedupe identical
            # frames in its render path (cuts ~12ms/frame on the home screen).
            self.ui.set_frame_id(frame_id)
            self.ui.set_debug_state(skip_finish=self._skip_finish_active)

            self._dispatch(frame, self._last_pose, clock)
            clock.tick(FPS_CAP)

    # ==================================================================
    # State dispatch
    # ==================================================================
    def _dispatch(self, frame, pose, clock: pygame.time.Clock) -> None:
        d = {
            State.START: self._do_start,
            State.CALIBRATE_LINES: self._do_calibrate_lines,
            State.COUNTDOWN: self._do_countdown,
            State.WAIT_START_LINE: self._do_wait_start_line,
            State.TURNING_GREEN: self._do_turning_green,
            State.GREEN: self._do_green,
            State.TURNING_RED: self._do_turning_red,
            State.RED: self._do_red,
            State.CAUGHT_RETURN: self._do_caught_return,
            State.ALL_ELIMINATED_HOLD: self._do_all_eliminated_hold,
            State.LEADERBOARD: self._do_leaderboard,
            State.RESET: self._do_reset,
        }
        fn = d.get(self._state)
        if fn is None:
            return
        if self._state == State.RESET:
            fn()
        else:
            fn(frame, pose, clock)

        if self.ui.is_dev_mode():
            self.ui.draw_dev_panel(self._dev_metrics(clock))
        self.ui.present()

    # ==================================================================
    # State implementations
    # ==================================================================
    def _do_start(self, frame, pose, clock) -> None:
        palm = detect_palm_raise(pose)
        prog = self._palm_progress(palm)
        self.ui.draw_start_screen(frame, prog, clock)
        if prog >= 1.0:
            self._palm_since = None
            self.audio.announce_game_start()
            self._go(State.COUNTDOWN)

    def _handle_mouse(self, event: pygame.event.Event) -> None:
        if self._state == State.CALIBRATE_LINES:
            if event.button == 1:  # Left click
                self._calib_points.append(event.pos)
                if len(self._calib_points) == 4:
                    self._apply_calibration()

    def _apply_calibration(self) -> None:
        from config import DISPLAY_W, DISPLAY_H
        
        # 1. Grab the actual frame dimensions to ensure a 1:1 match
        fw, fh = DISPLAY_W, DISPLAY_H
        if self._last_frame is not None:
            fh, fw = self._last_frame.shape[:2]

        def screen_to_native(p):
            # 2. Scale the click using the true camera frame resolution
            native_x = p[0] * (fw / DISPLAY_W)
            native_y = p[1] * (fh / DISPLAY_H)
            return native_x, native_y

        # Translate all 4 raw mouse clicks into true camera space
        pts = [screen_to_native(p) for p in self._calib_points]

        def calc_line(p1, p2):
            x1, y1 = p1
            x2, y2 = p2
            if abs(x2 - x1) < 0.1:  # Prevent divide-by-zero on perfectly vertical lines
                x2 += 0.1
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            return float(m), float(b)

        # Calculate the mathematical slopes
        start_line = calc_line(pts[0], pts[1])
        finish_line = calc_line(pts[2], pts[3])

        # Overwrite the vision system!
        self.line_detector.locked_start = start_line
        self.line_detector.locked_finish = finish_line
        self.line_detector.locked = True
        print(f"[ENGINE] Manual calibration applied! Start={start_line}, Finish={finish_line}")

        pygame.mouse.set_visible(False)
        self._go(State.START)

    def _do_all_eliminated_hold(self, frame, pose, clock) -> None:
        self._draw_game_hud(frame, pose, "ELIMINATED", clock)
        
        if self._in_state() >= 4.0 and not self.audio.is_tts_busy():
            self.audio.announce_all_finished()
            self._go(State.LEADERBOARD)

    def _do_calibrate_lines(self, frame, pose, clock) -> None:
        self.ui.draw_calibration(frame, self._calib_points)

    def _do_countdown(self, frame, pose, clock) -> None:
        t = self._in_state()
        n = COUNTDOWN_N - int(t)
        self.ui.draw_countdown(frame, max(1, n), clock)
        if int(t) != self._last_beep and n > 0:
            self._last_beep = int(t)
            self.audio.announce_countdown(n)
        if t >= COUNTDOWN_N:
            self._begin_round()

    def _begin_round(self) -> None:
        self._players.clear()
        self._players_locked = False
        self._next_rank = 1
        self._photo_retry_phases_used = 0
        self._highlights.clear()
        self._game_start_ts = time.time()
        self._last_finish_ts = self._game_start_ts
        self._last_ease_check_ts = self._game_start_ts
        self._ease_steps = 0
        self._reset_difficulty()
        self.audio.play_music("game_bgm")
        if START_LINE_REQUIRED:
            self.audio.announce_wait_for_start()
            self._go(State.WAIT_START_LINE)
        else:
            self.servo.face_away()
            self.audio.announce_green()
            self._light_dur = self._pick_phase_duration("green")
            self._register_players_from_pose(self._last_pose)
            self._players_locked = True
            self._go(State.GREEN)

    # ---- Wait at start line -----------------------------------------
    def _do_wait_start_line(self, frame, pose, clock) -> None:
        elapsed = self._in_state()
        time_left = max(0.0, START_LINE_MAX_WAIT_S - elapsed)
        ratio = time_left / max(0.001, START_LINE_MAX_WAIT_S)

        # Count how many visible players are currently behind the line
        behind, total = self._count_behind_start(pose)

        # Draw screen
        self.ui.draw_wait_start_line(frame, behind, total, time_left, clock)
        # Bar drawn explicitly so we can pass an exact ratio
        self.ui.draw_progress_bar_centered(
            DISPLAY_BAR_Y,
            ratio,
            color=(255, 220, 160),
            width=560,
        )

        # Periodic verbal nudge
        if int(elapsed) != self._last_beep and total > 0 and behind < total and elapsed < START_LINE_MAX_WAIT_S - 1:
            if int(elapsed) % 4 == 3 and elapsed > 2.0:
                self.audio.announce_wait_for_start()
                self._last_beep = int(elapsed)

        # All ready? Need to hold position briefly so a player crossing
        # back-and-forth doesn't trigger an instant start.
        ready = total > 0 and behind == total
        if ready and self._dwell_satisfied("start_line", elapsed, START_LINE_DWELL_S):
            self._enter_first_green(pose)
            return
        # Force-start on timeout
        if elapsed >= START_LINE_MAX_WAIT_S:
            print("[ENGINE] Start-line wait timed out — proceeding anyway")
            self._enter_first_green(pose)
            return

    def _enter_first_green(self, pose) -> None:
        self._dwell_reset()
        self._register_players_from_pose(pose)
        self._players_locked = True
        self.servo.face_away()
        self.audio.announce_green()
        self._light_dur = self._pick_phase_duration("green")
        self._go(State.GREEN)

    def _count_behind_start(self, pose) -> tuple[int, int]:
        if pose is None or pose.get("boxes") is None:
            return 0, 0
        boxes = pose["boxes"]
        if len(boxes) == 0:
            return 0, 0

        start_line = getattr(self, "_cached_start_line", (0.0, 0.0))
        behind = 0
        for box in boxes:
            fx, fy = LineDetector.get_foot_pos(box)
            if LineDetector.is_behind_start(fx, fy, start_line):
                behind += 1
        return behind, len(boxes)

    def _register_players_from_pose(self, pose) -> None:
        """Lock the roster — every track_id visible right now becomes a
        Player. Players that join later (e.g. someone who arrived after
        the first GREEN) will not appear in the leaderboard, by design."""
        if pose is None or pose.get("boxes") is None:
            return
        boxes = pose["boxes"]
        ids = pose.get("track_ids", [])
        for box, tid in zip(boxes, ids):
            if tid is None:
                continue
            if tid not in self._players:
                p = Player(track_id=tid)
                p.descriptor = self._initial_label(tid)
                p.last_box = box
                p.last_seen_ts = time.time()
                self._players[tid] = p
        print(f"[ENGINE] Registered {len(self._players)} players")

    @staticmethod
    def _initial_label(tid: int) -> str:
        return f"Player #{tid}"

    # ---- Turning green / green --------------------------------------
    def _do_turning_green(self, frame, pose, clock) -> None:
        self._draw_game_hud(frame, pose, "TURNING", clock)
        if not self.servo.is_facing_players and self._in_state() > 0.6:
            self.audio.announce_green()
            self._light_dur = self._pick_phase_duration("green")
            self._go(State.GREEN)

    def _do_green(self, frame, pose, clock) -> None:
        self._draw_game_hud(frame, pose, "GREEN", clock)
        self._update_player_boxes(pose)
        # Finish detection
        self._check_finishes(frame, pose)
        # Difficulty easing
        self._maybe_ease()
        if self._all_finished():
            if all(p.rank == -1 for p in self._players.values()):
                self._go(State.ALL_ELIMINATED_HOLD)
            else:
                self.audio.announce_all_finished()
                self._go(State.LEADERBOARD)
            return
        if self._in_state() >= self._light_dur:
            self.servo.face_players()
            self.audio.announce_red()
            self._light_dur = self._pick_phase_duration("red")
            self._go(State.TURNING_RED)

    # ---- Turning red / red ------------------------------------------
    def _do_turning_red(self, frame, pose, clock) -> None:
        self._draw_game_hud(frame, pose, "TURNING", clock)
        settled = self.servo.is_facing_players or self._in_state() >= SETTLE_TIME + 0.6
        if settled and self._in_state() >= SETTLE_TIME + GRACE_PERIOD:
            self._snapshot_baselines(pose)
            # Reset per-phase caught flags
            for p in self._players.values():
                p.is_caught_this_phase = False
            self._go(State.RED)

    def _do_red(self, frame, pose, clock) -> None:
        self._draw_game_hud(frame, pose, "RED", clock)
        self._update_player_boxes(pose)
        self._detect_motion_caught(frame, pose)
        # Capture / improve photos
        if PROFILE_CAPTURE_ENABLED and frame is not None:
            self._capture_photos(frame, pose)
        # Finishes can still occur during RED if a player is shoved past
        # the line by their own motion — counts as a normal finish, no
        # penalty (this matches user expectation).
        self._check_finishes(frame, pose)
        if self._all_finished():
            if all(p.rank == -1 for p in self._players.values()):
                self._go(State.ALL_ELIMINATED_HOLD)
            else:
                self.audio.announce_all_finished()
                self._go(State.LEADERBOARD)
            return
        if self._in_state() >= self._light_dur:
            if not self.audio.is_tts_busy():
                self._end_red_phase()

    def _end_red_phase(self) -> None:
        # Apr 2026 — clear baselines now so the next RED phase rebuilds
        # them from a fresh pose snapshot. Otherwise stale baselines
        # from this phase would silently apply during the next one.
        from config import PERMANENT_ELIMINATION
        self._red_baselines = []
        # Are there caught players who need to walk back?
        any_caught = any(p.needs_to_return for p in self._players.values())
        if any_caught and not PERMANENT_ELIMINATION:
            self._return_grace_until = time.time() + CAUGHT_RETURN_GRACE_S
            self._go(State.CAUGHT_RETURN)
            return
        # Otherwise resume the green phase
        self.servo.face_away()
        self._photo_retry_phases_used += 1
        self._go(State.TURNING_GREEN)

    # ---- Caught return ----------------------------------------------
    def _do_caught_return(self, frame, pose, clock) -> None:
        elapsed = self._in_state()
        time_left = max(0.0, CAUGHT_RETURN_MAX_S - elapsed)
        ratio = time_left / max(0.001, CAUGHT_RETURN_MAX_S)

        # Refresh "is back" status for each captive
        if time.time() >= self._return_grace_until:
            self._update_returning_status(pose)

        captives = [(p.descriptor, p.is_back_at_start) for p in self._players.values() if p.needs_to_return]
        # Apr 2026: ELIMINATED rebrand — pass caught_ids and labels so the
        # walk-back screen can paint a red bbox + pill on every captive
        # in the live camera feed (not just the list-card on the right).
        caught_ids = {p.track_id for p in self._players.values() if p.needs_to_return}
        labels_by_id = {p.track_id: p.descriptor for p in self._players.values()}
        self.ui.draw_caught_return(
            frame,
            captives,
            ratio,
            CAUGHT_RETURN_MAX_S,
            clock,
            pose=pose,
            caught_ids=caught_ids,
            labels_by_id=labels_by_id,
        )

        # Resume conditions
        all_back = all(is_back for _, is_back in captives) and len(captives) > 0
        
        # Wait until everyone is back AND the TTS is done talking
        if all_back and not self.audio.is_tts_busy():
            self._resume_after_return()
        # If it times out, still make sure TTS is done before forcing resume
        elif elapsed >= CAUGHT_RETURN_MAX_S and not self.audio.is_tts_busy():
            print("[ENGINE] Caught-return timed out   resuming anyway")
            self._resume_after_return()

    def _resume_after_return(self) -> None:
        for p in self._players.values():
            if p.needs_to_return:
                p.needs_to_return = False
                p.is_back_at_start = False
        self.audio.announce_return_complete()
        self.servo.face_away()
        self._photo_retry_phases_used += 1
        self._go(State.TURNING_GREEN)

    def _update_returning_status(self, pose) -> None:
        if pose is None or pose.get("boxes") is None:
            return
        boxes = pose["boxes"]
        ids = pose.get("track_ids", [])
        
        # FIX: Get the cached tuple instead of the old flat Y value
        start_line = getattr(self, "_cached_start_line", None)
        if start_line is None:
            return
            
        # Build a fresh map tid -> box (only for currently visible)
        seen: dict[int, np.ndarray] = {}
        for box, tid in zip(boxes, ids):
            if tid is not None:
                seen[tid] = box
                
        # Each captive: if visible AND behind the line for the dwell window, mark back
        now = time.time()
        for p in self._players.values():
            if not p.needs_to_return:
                continue
                
            box = seen.get(p.track_id)
            if box is None:
                # Not visible — keep their last status, don't reset
                continue
                
            # FIX: Extract X and Y foot positions and pass to the new tilted line math
            fx, fy = LineDetector.get_foot_pos(box)
            if LineDetector.is_behind_start(fx, fy, start_line):
                # Begin / continue dwell
                if not getattr(p, "_dwell_start", None):
                    p._dwell_start = now  # type: ignore[attr-defined]
                if now - p._dwell_start >= CAUGHT_RETURN_DWELL_S:  # type: ignore[attr-defined]
                    p.is_back_at_start = True
            else:
                p._dwell_start = None  # type: ignore[attr-defined]
                p.is_back_at_start = False

    # ---- Leaderboard -------------------------------------------------
    def _do_leaderboard(self, frame, pose, clock) -> None:
        # Build photo surfaces lazily on first entry
        for p in self._players.values():
            if p.photo_crop is not None and p.photo_surface is None:
                try:
                    thumb = PhotoCapture.make_thumbnail(p.photo_crop)
                    rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)  # type: ignore
                    h, w = rgb.shape[:2]
                    p.photo_surface = pygame.image.frombuffer(rgb.tobytes(), (w, h), "RGB")
                except Exception as exc:
                    print(f"[ENGINE] Photo surface error: {exc}")

        # Apr 2026 — K-key demo overrides real results when set.
        if self._demo_leaderboard_results is not None:
            results = self._demo_leaderboard_results
        else:
            results = self._build_leaderboard_results()

        # Apr 2026: Kahoot-style sequenced reveal. The UI tracks per-card
        # entry timing internally (seeded by the screen's elapsed time),
        # so we just hand it the time-in-state and let it pace the cards.
        elapsed = self._in_state()

        # Palm-to-restart — only armed once 1st place has fully revealed,
        # the spotlight has blasted open, and players have had a moment
        # to celebrate. The new dramatic sequence runs ~12s before the
        # winner is even on screen, so this guard is critical.
        palm_armed_at = (
            LEADERBOARD_1ST_DELAY_S
            + LEADERBOARD_1ST_REVEAL_S
            + LEADERBOARD_BLAST_OUT_S
            + LEADERBOARD_1ST_CELEBRATE_S
            + LEADERBOARD_PALM_ARMED_S
        )
        palm_armed = elapsed >= palm_armed_at
        palm_up = detect_palm_raise(pose) if palm_armed else False
        palm_prog = self._leaderboard_palm_progress(palm_up)

        self.ui.draw_leaderboard(results, elapsed, clock, palm_progress=palm_prog)

        if palm_prog >= 1.0:
            self._palm_since = None
            print("[ENGINE] Palm-restart from leaderboard")
            self._go(State.RESET)

    def _leaderboard_palm_progress(self, palm_up: bool) -> float:
        if palm_up:
            if self._palm_since is None:
                self._palm_since = time.time()
            return min(1.0, (time.time() - self._palm_since) / max(0.1, PALM_HOLD_LEADERBOARD))
        self._palm_since = None
        return 0.0

    def _build_demo_leaderboard(self) -> list[dict]:
        """Generate 5 fake players for the K-key demo.

        Each gets a synthetic avatar (vertical-gradient pygame Surface
        with the player's colour + a big initial), a descriptor, and a
        plausible finish time. Returned in the standard
        ``_build_leaderboard_results`` shape so the UI can't tell the
        difference."""
        # A small palette of colourful "shirt" tints + matching names
        # so each fake player has a distinct identity.
        palette = [
            ((255, 90, 95), "Red Hoodie"),
            ((80, 175, 255), "Blue Jacket"),
            ((140, 220, 110), "Green Shirt"),
            ((255, 195, 60), "Yellow Tee"),
            ((200, 130, 255), "Purple Vest"),
            ((255, 140, 70), "Orange Cap"),
            ((80, 220, 220), "Teal Hoodie"),
        ]
        random.shuffle(palette)
        n = 5
        chosen = palette[:n]

        # Increasing finish times — winner is fastest.
        base_time = random.uniform(18.0, 28.0)
        out: list[dict] = []
        for i, (colour, name) in enumerate(chosen):
            avatar = self._make_demo_avatar(colour, name[:1])
            out.append(
                {
                    "rank": i + 1,
                    "descriptor": name,
                    "time_s": base_time + i * random.uniform(2.5, 5.5),
                    "photo_surface": avatar,
                    "times_caught": random.randint(0, 2),
                }
            )
        return out

    @staticmethod
    def _make_demo_avatar(tint: tuple[int, int, int], initial: str) -> pygame.Surface:
        """Build a 200×200 pygame Surface for a fake demo player.

        Vertical gradient from a lighter top to a darker bottom of the
        given tint, with a big white initial centred on it. Looks like
        the avatar UI used by real players just enough that the
        leaderboard's photo masking and corner-rounding still kick in."""
        size = 200
        surf = pygame.Surface((size, size)).convert()
        # Build gradient via numpy → pygame surface for speed.
        try:
            top = np.array(
                (min(255, tint[0] + 40), min(255, tint[1] + 40), min(255, tint[2] + 40)),
                dtype=np.float32,
            )
            bot = np.array(
                (max(0, tint[0] - 40), max(0, tint[1] - 40), max(0, tint[2] - 40)),
                dtype=np.float32,
            )
            ts = np.linspace(0.0, 1.0, size, dtype=np.float32).reshape(size, 1, 1)
            grad = (top * (1 - ts) + bot * ts).astype(np.uint8)
            grad = np.broadcast_to(grad, (size, size, 3)).copy()
            grad_surf = pygame.surfarray.make_surface(grad.swapaxes(0, 1)).convert()
            surf.blit(grad_surf, (0, 0))
        except Exception:
            surf.fill(tint)

        # Big white initial in the centre.
        try:
            font = pygame.font.SysFont("arial", 130, bold=True)
            text = font.render(initial.upper(), True, (255, 255, 255))
            surf.blit(text, text.get_rect(center=(size // 2, size // 2 - 6)))
        except Exception:
            pass
        return surf

    def _build_leaderboard_results(self) -> list[dict]:
        # ONLY grab players who actually crossed the finish line (rank > 0)
        finishers = sorted(
            [p for p in self._players.values() if p.finished and p.rank > 0],
            key=lambda p: p.rank,
        )
        # Non-finishers AND permanently eliminated players (rank == -1)
        non_finishers = [p for p in self._players.values() if not p.finished or p.rank == -1]

        out: list[dict] = []
        for p in finishers:
            out.append(
                {
                    "rank": p.rank,
                    "descriptor": p.descriptor,
                    "time_s": p.finish_time_s,
                    "photo_surface": p.photo_surface,
                    "times_caught": p.times_caught,
                }
            )
            
        for p in non_finishers:
            # If permanently eliminated, use their elimination time. Otherwise, time since start.
            time_val = p.finish_time_s if p.rank == -1 else (time.time() - self._game_start_ts)
            out.append(
                {
                    "rank": "-",  # Use a dash instead of a broken negative number
                    "descriptor": p.descriptor + " (ELIMINATED)",
                    "time_s": time_val,
                    "photo_surface": p.photo_surface,
                    "times_caught": p.times_caught,
                }
            )
            
        return out

    # ---- Reset -------------------------------------------------------
    def _do_reset(self) -> None:
        self.servo.face_away()
        self._players.clear()
        self._players_locked = False
        self._next_rank = 1
        self._highlights.clear()
        self._palm_since = None
        self._reset_difficulty()
        self._ease_steps = 0
        # Apr 2026 — clear demo state on reset.
        self._demo_leaderboard_results = None
        self._red_baselines = []
        self.audio.fade_music(300) 
        self.audio.play_music("bgm")
        self._go(State.START)

    # ==================================================================
    # Helpers — finishes, motion, photos
    # ==================================================================
    def _check_finishes(self, frame, pose) -> None:
        # Apr 2026 — debug skip switch. When ON, no player ever gets
        # marked as finished, so the round runs forever in green/red
        # cycles. Use F to toggle, L to escape to the leaderboard.
        if self._skip_finish_active:
            return
            
        # FIX: Grab the cached tuple for the finish line
        finish_line = getattr(self, "_cached_finish_line", (0.0, 0.0))
        
        # 1. LASER takes priority — fire a finish for the closest player
        if self.laser.in_use and USE_LASER:
            if self.laser.broken:
                # FIX: Pass the finish_line tuple into the laser attribution method
                closest = self._closest_unfinished_to_finish(pose, finish_line)
                if closest is not None:
                    self._mark_finished(closest, frame)
                return
                
        # 2. Camera tape — only used if USE_TAPE_FINISH
        if not USE_TAPE_FINISH:
            return
            
        if pose is None or pose.get("boxes") is None:
            return
            
        boxes = pose["boxes"]
        ids = pose.get("track_ids", [])
        
        for box, tid in zip(boxes, ids):
            if tid is None or tid not in self._players:
                continue
                
            p = self._players[tid]
            
            # FIX: Ignore players who have been caught and need to return
            if p.finished or p.needs_to_return:
                continue

            # FIX: Use the new 2D foot coordinate extractor and tilted line math
            fx, fy = LineDetector.get_foot_pos(box)
            if LineDetector.has_crossed_finish(fx, fy, finish_line):
                self._mark_finished(p, frame)

    def _closest_unfinished_to_finish(self, pose, finish_line: tuple[float, float]) -> Player | None:
        if pose is None or pose.get("boxes") is None:
            return None
        boxes = pose["boxes"]
        ids = pose.get("track_ids", [])
        best: tuple[float, Player] | None = None
        
        m, b = finish_line
        
        for box, tid in zip(boxes, ids):
            if tid is None or tid not in self._players:
                continue
            p = self._players[tid]
            if p.finished or p.needs_to_return:
                continue

            fx, fy = LineDetector.get_foot_pos(box)
            line_y_at_x = m * fx + b
            dist = abs(fy - line_y_at_x) # Vertical distance to the tilted line
            
            if best is None or dist < best[0]:
                best = (dist, p)
        return best[1] if best else None

    def _mark_finished(self, p: Player, frame) -> None:
        if p.finished:
            return
        p.finished = True
        p.rank = self._next_rank
        self._next_rank += 1
        p.finish_time_s = time.time() - self._game_start_ts
        self._last_finish_ts = time.time()
        # If they didn't have a profile photo yet, grab one now from
        # whatever we have on hand — better than nothing for the podium.
        if p.photo_crop is None and p.last_box is not None and frame is not None:
            crop = PhotoCapture.crop_player(frame, p.last_box)
            if crop is not None:
                p.photo_crop = crop
                p.photo_quality = PhotoCapture.quality_score(crop)
                self._dispatch_describer(p, crop)
        # Also push a highlight thumbnail for the recap reel.
        if frame is not None:
            try:
                small = cv2.resize(  # type: ignore
                    frame, (int(frame.shape[1] * HIGHLIGHT_SCALE), int(frame.shape[0] * HIGHLIGHT_SCALE)), interpolation=cv2.INTER_AREA  # type: ignore
                )
                self._highlights.append(small)
            except Exception:
                pass
        print(f"[ENGINE] Finish #{p.rank}: {p.descriptor} (track {p.track_id}) " f"@ {p.finish_time_s:.1f}s")
        self.audio.announce_finished(p.rank, p.descriptor)

    def _detect_motion_caught(self, frame, pose) -> None:
        if pose is None or pose.get("boxes") is None:
            return
        if self._in_state() < GRACE_PERIOD:
            return

        boxes = pose["boxes"]
        
        # 1. Gather all current bounding box centers (ignoring tracking IDs)
        current_centroids = []
        for box in boxes:
            cx = float((box[0] + box[2]) / 2.0)
            cy = float((box[1] + box[3]) / 2.0)
            current_centroids.append((cx, cy))

        max_dist = 0.0
        caught_this_frame = []

        # 2. Iterate over the stable baselines we locked in at the start of RED
        for bx, by, p in self._red_baselines:
            if p.finished or getattr(p, 'needs_to_return', False):
                continue
            
            best_dist = float("inf")
            best_cx, best_cy = bx, by
            
            # 3. Find the closest physical player to this baseline
            for cx, cy in current_centroids:
                d = math.hypot(cx - bx, cy - by)
                if d < best_dist:
                    best_dist = d
                    best_cx = cx
                    best_cy = cy

            # 4. Anti-Occlusion Guard: If the closest person is absurdly far away (>150px), 
            # the player is likely hidden behind someone else. Ignore them this frame.
            if best_dist > 150.0:  
                continue

            # 5. Check if they exceeded the actual movement threshold
            if best_dist > max_dist:
                max_dist = best_dist

            if best_dist > self._motion_px and not p.is_caught_this_phase:
                self._on_player_caught(p, frame, best_cx, best_cy)
                caught_this_frame.append(p)

        if caught_this_frame:
            if len(caught_this_frame) == 1:
                self.audio.announce_caught(caught_this_frame[0].descriptor)
            else:
                # If 2 or more people get caught in the exact same frame
                self.audio.announce_caught(f"{len(caught_this_frame)} players")

        # Update the UI motion meter accurately
        target = min(1.0, max_dist / max(1.0, self._motion_px))
        self._motion_score = 0.7 * self._motion_score + 0.3 * target

    def _on_player_caught(self, p: Player, frame, cx: float, cy: float) -> None:
        bx, by = p.baseline_pos
        dist = math.hypot(cx - bx, cy - by)
        print(
            f"[ENGINE] CAUGHT {p.descriptor}  "
            f"(moved {dist:.1f}px from baseline ({bx:.0f},{by:.0f}) "
            f"to ({cx:.0f},{cy:.0f}); threshold={self._motion_px}px)"
        )
        p.is_caught_this_phase = True
        p.times_caught += 1
        p._dwell_start = None  # type: ignore[attr-defined]
        # Try to upgrade their photo while we're here, since they're
        # presumably stationary (caught means they barely moved past
        # threshold, so the next few frames are a good capture window).
        from config import PERMANENT_ELIMINATION
        if PERMANENT_ELIMINATION:
            # Mark as finished so the tracker stops checking them
            p.finished = True
            p.needs_to_return = False
            p.rank = -1  # Special rank for losers
            print(f"[ENGINE] Permanent Elimination: {p.descriptor}")
        else:
            p.needs_to_return = True
            p.is_back_at_start = False

        if frame is not None and p.last_box is not None:
            crop = PhotoCapture.crop_player(frame, p.last_box)
            if crop is not None:
                q = PhotoCapture.quality_score(crop)
                if q > p.photo_quality:
                    p.photo_crop = crop
                    p.photo_quality = q
                    self._dispatch_describer(p, crop)
        # UI feedback
        self.ui.flash_caught(
            int(cx * (1920 / max(1.0, frame.shape[1] if frame is not None else 1920))),
            int(cy * (1080 / max(1.0, frame.shape[0] if frame is not None else 1080))),
        )
        self.ui.log_sent_back(p.descriptor)
        #self.audio.announce_caught(p.descriptor)

    def _capture_photos(self, frame, pose) -> None:
        """During RED, take the best-looking crop we can of each player."""
        if pose is None or pose.get("boxes") is None or frame is None:
            return
        boxes = pose["boxes"]
        ids = pose.get("track_ids", [])
        for box, tid in zip(boxes, ids):
            if tid is None or tid not in self._players:
                continue
            p = self._players[tid]
            if p.finished:
                continue
            # Skip if the bbox is too small (player far away)
            bw = float(box[2] - box[0])
            bh = float(box[3] - box[1])
            if bh < PROFILE_CAPTURE_MIN_BBOX_PX:
                continue
            crop = PhotoCapture.crop_player(frame, box)
            if crop is None:
                continue
            q = PhotoCapture.quality_score(crop)
            # Replace if higher quality OR if they have nothing yet
            if p.photo_crop is None or q > p.photo_quality * 1.05:
                p.photo_crop = crop
                p.photo_quality = q
                # Kick off async describer
                self._dispatch_describer(p, crop)

    def _dispatch_describer(self, p: Player, crop: np.ndarray) -> None:
        """Fire-and-forget background description. Same player can be
        re-described later if a higher-quality crop arrives — the latest
        result wins."""
        if p.descriptor_pending:
            return
        p.descriptor_pending = True

        def _worker(player=p, snap=crop.copy()) -> None:
            try:
                desc = self.describer.describe(snap)
            except Exception as exc:
                print(f"[ENGINE] describer worker error: {exc}")
                desc = None
            if desc:
                player.descriptor = desc
            player.descriptor_pending = False

        threading.Thread(target=_worker, daemon=True, name="describe").start()

    def _update_player_boxes(self, pose) -> None:
        if pose is None or pose.get("boxes") is None:
            return
            
        boxes = pose["boxes"]
        ids = pose.get("track_ids", [])
        now = time.time()
        
        matched_players = set()
        unmatched_boxes = []

        # 1. First pass: Match by exact track ID
        for box, tid in zip(boxes, ids):
            if tid is not None and tid in self._players:
                p = self._players[tid]
                p.last_box = box
                p.last_seen_ts = now
                matched_players.add(tid)
            else:
                unmatched_boxes.append((box, tid))

        # 2. Second pass: Heal broken tracks by geometry (centroid distance)
        # Find all active players who suddenly lost their bounding box this frame
        missing_players = [
            p for p in self._players.values() 
            if p.track_id not in matched_players and not p.finished
        ]

        for box, tid in unmatched_boxes:
            if not missing_players:
                break  # No more missing players to heal
                
            bx = float((box[0] + box[2]) / 2.0)
            by = float((box[1] + box[3]) / 2.0)
            
            best_dist = float("inf")
            best_p = None
            
            for p in missing_players:
                if p.last_box is None:
                    continue
                px = float((p.last_box[0] + p.last_box[2]) / 2.0)
                py = float((p.last_box[1] + p.last_box[3]) / 2.0)
                d = math.hypot(bx - px, by - py)
                
                if d < best_dist:
                    best_dist = d
                    best_p = p
                    
            # If the unmatched box is physically close to where we last saw a missing player, 
            # assume ByteTrack swapped their ID and re-assign them.
            if best_p is not None and best_dist < 250.0:
                best_p.last_box = box
                best_p.last_seen_ts = now
                
                # Update dictionary keys so the finish line and UI can find them under their new ID
                old_tid = best_p.track_id
                if tid is not None and tid not in self._players:
                    print(f"[ENGINE] Healing track ID: {old_tid} -> {tid} for {best_p.descriptor}")
                    best_p.track_id = tid
                    self._players[tid] = best_p
                    del self._players[old_tid]
                    
                missing_players.remove(best_p)

    def _snapshot_baselines(self, pose) -> None:
        if pose is None or pose.get("boxes") is None:
            return
        boxes = pose["boxes"]
        ids = pose.get("track_ids", [])
        for box, tid in zip(boxes, ids):
            if tid is None or tid not in self._players:
                continue
            p = self._players[tid]
            cx = float((box[0] + box[2]) / 2.0)
            cy = float((box[1] + box[3]) / 2.0)
            p.baseline_pos = (cx, cy)
        # Apr 2026 — also seed track-id-independent baselines used by
        # the new _detect_motion_caught path. Match each visible bbox
        # to the nearest registered Player by their last-known box.
        # If a player isn't visible right now we skip them; the
        # motion-detect fallback will seed any latecomers on the next
        # frame.
        self._snapshot_red_baselines(pose)

    def _snapshot_red_baselines(self, pose) -> None:
        """Build the list of (cx, cy, Player) baselines that
        ``_detect_motion_caught`` matches detections against.

        This is independent of track IDs — we use the position of
        each registered Player's last_box to match to a current
        detection by nearest centroid. That way a registered Player
        whose YOLO track ID has drifted still gets a fresh baseline
        anchored to wherever they actually are right now.
        """
        if pose is None or pose.get("boxes") is None:
            return
        boxes = pose["boxes"]
        if len(boxes) == 0:
            return

        # Build current centroids
        current = []
        for box in boxes:
            cx = float((box[0] + box[2]) / 2.0)
            cy = float((box[1] + box[3]) / 2.0)
            current.append((cx, cy))

        # For each unfinished registered player, find the closest
        # current detection and use it as their baseline. Greedy NN —
        # a current detection can only be claimed once.
        used: set[int] = set()
        baselines: list[tuple[float, float, "Player"]] = []
        candidates = [p for p in self._players.values() if not p.finished and not p.needs_to_return]

        # Sort candidates by how confident we are about their last
        # known position (most recently seen first), so the player
        # who definitely was somewhere gets matched before someone
        # who hasn't been seen in a while.
        candidates.sort(key=lambda p: -p.last_seen_ts)

        for p in candidates:
            if p.last_box is None:
                continue
            px = float((p.last_box[0] + p.last_box[2]) / 2.0)
            py = float((p.last_box[1] + p.last_box[3]) / 2.0)
            best_idx = -1
            best_dist = float("inf")
            for i, (cx, cy) in enumerate(current):
                if i in used:
                    continue
                d = math.hypot(cx - px, cy - py)
                if d < best_dist:
                    best_dist = d
                    best_idx = i
            if best_idx == -1:
                continue
            # Reasonable proximity — don't claim a detection that's
            # way off from where the player was last seen (probably a
            # different person).
            if best_dist > 250:
                continue
            used.add(best_idx)
            cx, cy = current[best_idx]
            baselines.append((cx, cy, p))
            p.baseline_pos = (cx, cy)
            p.last_seen_ts = time.time()

        self._red_baselines = baselines
        print(
            f"[ENGINE] RED baselines seeded: matched {len(baselines)}/{len(self._players)} "
            f"players  (motion threshold = {self._motion_px}px)"
        )

    def _all_finished(self) -> bool:
        return self._players_locked and len(self._players) > 0 and all(p.finished for p in self._players.values())

    # ==================================================================
    # Difficulty / phase duration / easing
    # ==================================================================
    def _reset_difficulty(self) -> None:
        d = DIFFICULTY_PRESETS[self._difficulty]
        self._green_min, self._green_max = d["green_min"], d["green_max"]
        self._red_min, self._red_max = d["red_min"], d["red_max"]
        self._motion_px = d["motion_px"]

    def _pick_phase_duration(self, kind: str) -> float:
        if random.random() < PHASE_FAKEOUT_PROB:
            return random.uniform(PHASE_FAKEOUT_MIN, PHASE_FAKEOUT_MAX)
        if kind == "green":
            lo, hi = self._green_min, self._green_max
        else:
            lo, hi = self._red_min, self._red_max
        # Mix uniform with triangular biased to the extremes.
        u = random.uniform(lo, hi)
        # Triangular: peaks at lo and hi, valley at midpoint
        if random.random() < 0.5:
            extreme = random.triangular(lo, lo + (hi - lo) * 0.35, lo)
        else:
            extreme = random.triangular(hi - (hi - lo) * 0.35, hi, hi)
        bias = PHASE_EXTREME_BIAS
        return (1.0 - bias) * u + bias * extreme

    def _maybe_ease(self) -> None:
        if not EASE_ENABLED:
            return
        now = time.time()
        if now - self._last_ease_check_ts < EASE_CHECK_EVERY_S:
            return
        self._last_ease_check_ts = now
        # Trigger condition: been too long without a finisher
        if (now - self._last_finish_ts) < EASE_TRIGGER_AFTER_S:
            return
        # Apply one step
        applied = False
        if self._green_max < EASE_MAX_GREEN_S:
            self._green_min = min(EASE_MAX_GREEN_S - 1.0, self._green_min + EASE_STEP_GREEN_S)
            self._green_max = min(EASE_MAX_GREEN_S, self._green_max + EASE_STEP_GREEN_S)
            applied = True
        if self._red_min > EASE_MIN_RED_S:
            self._red_min = max(EASE_MIN_RED_S, self._red_min - EASE_STEP_RED_S)
            self._red_max = max(EASE_MIN_RED_S + 1.0, self._red_max - EASE_STEP_RED_S)
            applied = True
        if self._motion_px < EASE_MAX_MOTION_PX:
            self._motion_px = min(EASE_MAX_MOTION_PX, self._motion_px + EASE_STEP_MOTION_PX)
            applied = True
        if applied:
            self._ease_steps += 1
            print(
                f"[ENGINE] Eased difficulty (step {self._ease_steps}): "
                f"green=[{self._green_min:.1f}-{self._green_max:.1f}] "
                f"red=[{self._red_min:.1f}-{self._red_max:.1f}] "
                f"motion={self._motion_px}px"
            )
            self.audio.announce_easing()

    # ==================================================================
    # Misc helpers
    # ==================================================================
    def _draw_game_hud(self, frame, pose, label: str, clock: pygame.time.Clock) -> None:
        elapsed = time.time() - self._game_start_ts if self._game_start_ts else 0.0
        finished = sum(1 for p in self._players.values() if p.finished)
        total = len(self._players)
        in_play = total - finished
        finished_ids = {p.track_id for p in self._players.values() if p.finished}
        caught_ids = {
            p.track_id for p in self._players.values() 
            if p.needs_to_return or (p.finished and getattr(p, 'rank', 0) == -1)
        }
        labels_by_id = {p.track_id: p.descriptor for p in self._players.values()}
        self.ui.draw_game_hud(
            frame,
            pose,
            label,
            in_play=in_play,
            finished=finished,
            total=total,
            elapsed=elapsed,
            motion_score=self._motion_score,
            finished_ids=finished_ids,
            caught_ids=caught_ids,
            labels_by_id=labels_by_id,
            ease_steps=self._ease_steps,
            clock=clock,
        )

    def _go(self, new_state: State) -> None:
        print(f"  [{self._state.name}] → [{new_state.name}]")
        self._state = new_state
        self._state_ts = time.time()
        self._last_beep = -1
        self._dwell_reset()

    def _in_state(self) -> float:
        return time.time() - self._state_ts

    def _palm_progress(self, palm_up: bool) -> float:
        if palm_up:
            if self._palm_since is None:
                self._palm_since = time.time()
            return min(1.0, (time.time() - self._palm_since) / PALM_HOLD)
        self._palm_since = None
        return 0.0

    # Generic "must hold a condition for N seconds" dwell tracker
    _dwell_key: str = ""
    _dwell_start: float = 0.0

    def _dwell_satisfied(self, key: str, elapsed: float, dwell_s: float) -> bool:
        now = time.time()
        if self._dwell_key != key:
            self._dwell_key = key
            self._dwell_start = now
            return False
        return (now - self._dwell_start) >= dwell_s

    def _dwell_reset(self) -> None:
        self._dwell_key = ""
        self._dwell_start = 0.0

    # ==================================================================
    # Input
    # ==================================================================
    def _handle_key(self, event: pygame.event.Event) -> None:
        key = event.key
        mods = pygame.key.get_mods()

        # Dev-mode toggle: H or CTRL+D
        if key == pygame.K_h or (key == pygame.K_d and (mods & pygame.KMOD_CTRL)):
            self.ui.toggle_dev_mode()
            return

        if key == pygame.K_ESCAPE:
            raise SystemExit

        # Dev-only hardware tests
        if self.ui.is_dev_mode():
            if key == pygame.K_1:
                self.servo.face_players()
                return
            if key == pygame.K_2:
                self.servo.face_away()
                return
            if key == pygame.K_3:
                print(f"[DEV] Laser broken = {self.laser.broken} (in_use={self.laser.in_use})")
                return
            if key == pygame.K_4:
                self.audio.test_chime()
                return

        # Force GREEN
        if key == pygame.K_g:
            if self._state in (State.START, State.COUNTDOWN):
                self._begin_round()
            self.servo.face_away()
            self._light_dur = self._pick_phase_duration("green")
            self.audio.announce_green()
            self._go(State.GREEN)
            return

        # Force RED
        if key == pygame.K_r:
            if self._state in (State.START, State.COUNTDOWN):
                return
            self._snapshot_baselines(self._last_pose)
            self._light_dur = self._pick_phase_duration("red")
            self.servo.face_players()
            self.audio.announce_red()
            self._go(State.RED)
            return

        # Debug-finish closest player
        if key == pygame.K_w:
            if self._players:
                # Pick the first non-finished player
                for p in self._players.values():
                    if not p.finished:
                        self._mark_finished(p, self._last_frame)
                        break
            return

        # Jump to leaderboard
        if key == pygame.K_l:
            # Mark all unfinished as finished so the board has data
            for p in self._players.values():
                if not p.finished:
                    self._mark_finished(p, self._last_frame)
            self.audio.announce_all_finished()
            self._go(State.LEADERBOARD)
            return

        # K — fake leaderboard demo. Generates 5 fake players with
        # synthetic avatars + names + finish times, jumps straight to
        # the LEADERBOARD state, and plays the entire reveal
        # choreography end-to-end. Lets you test podium animations,
        # spotlight sweep, confetti burst, and audio cues without
        # needing to run a real round. Press SPACE on the leaderboard
        # to escape, or wait for palm-restart to arm.
        if key == pygame.K_k:
            self._demo_leaderboard_results = self._build_demo_leaderboard()
            print("[ENGINE] Fake leaderboard demo (K) — playing reveal choreography")
            self.audio.announce_all_finished()            
            self._go(State.LEADERBOARD)
            return

        # F or F9 — toggle DEBUG_SKIP_FINISH. With this on the round
        # runs forever in green/red cycles so you can iterate on the
        # red/green light timing, the catch loop, and the servo
        # behaviour without needing to set up the finish tape or laser.
        # The HUD shows a ribbon banner + corner pill while it's active.
        # Available outside dev mode so it's reachable straight from the
        # home screen.
        if key == pygame.K_f or key == pygame.K_F9:
            self._skip_finish_active = not self._skip_finish_active
            self.ui.set_debug_state(skip_finish=self._skip_finish_active)
            print(f"[ENGINE] DEBUG_SKIP_FINISH = {self._skip_finish_active}")
            return

        # SPACE — bypass palm OR play again from leaderboard
        if key == pygame.K_SPACE:
            if self._state == State.LEADERBOARD:
                self._go(State.RESET)
            elif self._state == State.START:
                # Bypass the palm gate and jump straight to the countdown
                self._palm_since = None
                self.audio.announce_game_start()
                self._go(State.COUNTDOWN)
            return
        
        if key == pygame.K_t:
            self.line_detector.locked = not self.line_detector.locked
            print(f"[ENGINE] Tape Lock = {self.line_detector.locked}")
            return
        
        if key == pygame.K_c:
            if self._state in (State.START, State.CALIBRATE_LINES):
                if self._state == State.CALIBRATE_LINES:
                    pygame.mouse.set_visible(False)
                    self._go(State.START)
                else:
                    self._calib_points = []
                    pygame.mouse.set_visible(True) # Turn on the mouse pointer!
                    self._go(State.CALIBRATE_LINES)
            return

    # ==================================================================
    # Dev metrics
    # ==================================================================
    def _dev_metrics(self, clock: pygame.time.Clock) -> dict:
        alive = self._last_pose.get("players_alive", 0) if self._last_pose else 0
        return {
            "fps": clock.get_fps(),
            "cam_fps": self.camera.fps,
            "inference_ms": self.tracker.last_inference_ms,
            "state": self._state.name,
            "state_time": self._in_state(),
            "servo_angle": self.servo.angle,
            "servo_target": self.servo.target,
            "laser_in_use": self.laser.in_use,
            "players": alive,
            "finishers": sum(1 for p in self._players.values() if p.finished),
            "ease_steps": self._ease_steps,
            "id_mode": self.describer.mode,
            "tape_locked": self.line_detector.locked,
            "dev_hints": [
                f"Backend: {self.camera.backend}",
                f"Servo HW: {self.servo.is_hardware}  Laser: {self.laser.in_use}",
                f"Pose worker id={self._last_pose_id}",
                f"Start Y={self._cached_start_y}  Finish Y={self._cached_finish_y}",
                f"Difficulty: G[{self._green_min:.1f}-{self._green_max:.1f}] " f"R[{self._red_min:.1f}-{self._red_max:.1f}] mv={self._motion_px}",
            ],
        }


# ---------------------------------------------------------------------------
# Constant used by the wait-start-line drawing path. The bar Y is in display
# coordinates; we keep it module-local to avoid bloating config with a UI
# constant.
# ---------------------------------------------------------------------------
DISPLAY_BAR_Y = 740