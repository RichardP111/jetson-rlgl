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

import cv2
import numpy as np
import pygame

from audio import AudioManager
from config import (
    CAUGHT_RETURN_DWELL_S,
    CAUGHT_RETURN_GRACE_S,
    CAUGHT_RETURN_MAX_S,
    CAUGHT_RETURN_RECHECK_HZ,
    COUNTDOWN_N,
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
    PALM_HOLD,
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
    COUNTDOWN = auto()
    WAIT_START_LINE = auto()
    TURNING_GREEN = auto()
    GREEN = auto()
    TURNING_RED = auto()
    RED = auto()
    CAUGHT_RETURN = auto()
    LEADERBOARD = auto()
    RESET = auto()


_BANNER_LABEL = {
    State.START: "START",
    State.COUNTDOWN: "COUNTDOWN",
    State.WAIT_START_LINE: "START_LINE",
    State.TURNING_GREEN: "TURNING",
    State.GREEN: "GREEN",
    State.TURNING_RED: "TURNING",
    State.RED: "RED",
    State.CAUGHT_RETURN: "RETURN",
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
        self._last_pose: dict | None = None
        self._last_pose_id: int = -1

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
        print("[ENGINE] Running. Hotkeys: H dev | G/R force | W debug-finish | " "L leaderboard | SPACE bypass | ESC quit")
        self.servo.face_away()
        self.audio.play_music("bgm")

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    self._handle_key(event)

            frame = self.camera.read()
            pose, pose_id = self.pose_worker.latest()

            self._last_frame = frame
            if pose_id != self._last_pose_id:
                self._last_pose = pose
                self._last_pose_id = pose_id

            # Update line-detector cache (used by UI overlay AND by the FSM)
            if frame is not None:
                start_y, sc = self.line_detector.detect_start(frame)
                finish_y, fc = self.line_detector.detect_finish(frame)
                self._cached_start_y = start_y
                self._cached_finish_y = finish_y
                self.ui.set_line_calibration(frame.shape[0], start_y, sc, finish_y, fc)

            self._dispatch(frame, self._last_pose, clock)
            clock.tick(FPS_CAP)

    # ==================================================================
    # State dispatch
    # ==================================================================
    def _dispatch(self, frame, pose, clock: pygame.time.Clock) -> None:
        d = {
            State.START: self._do_start,
            State.COUNTDOWN: self._do_countdown,
            State.WAIT_START_LINE: self._do_wait_start_line,
            State.TURNING_GREEN: self._do_turning_green,
            State.GREEN: self._do_green,
            State.TURNING_RED: self._do_turning_red,
            State.RED: self._do_red,
            State.CAUGHT_RETURN: self._do_caught_return,
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
        start_y = self._cached_start_y or 0
        if start_y <= 0:
            return 0, len(boxes)
        behind = 0
        for box in boxes:
            foot = LineDetector.player_foot_y(box)
            if LineDetector.is_behind_start(foot, start_y):
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
        # Check if everyone has finished
        if self._all_finished():
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
            self.audio.announce_all_finished()
            self._go(State.LEADERBOARD)
            return
        if self._in_state() >= self._light_dur:
            self._end_red_phase()

    def _end_red_phase(self) -> None:
        # Are there caught players who need to walk back?
        any_caught = any(p.needs_to_return for p in self._players.values())
        if any_caught:
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
        self.ui.draw_caught_return(frame, captives, ratio, CAUGHT_RETURN_MAX_S, clock)

        # Resume conditions
        all_back = all(is_back for _, is_back in captives) and len(captives) > 0
        if all_back:
            self._resume_after_return()
        elif elapsed >= CAUGHT_RETURN_MAX_S:
            print("[ENGINE] Caught-return timed out — resuming anyway")
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
        start_y = self._cached_start_y or 0
        if start_y <= 0:
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
            foot = LineDetector.player_foot_y(box)
            if LineDetector.is_behind_start(foot, start_y):
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

        results = self._build_leaderboard_results()
        replay_progress = 0.0
        self.ui.draw_leaderboard(results, replay_progress, clock)

    def _build_leaderboard_results(self) -> list[dict]:
        finishers = sorted(
            [p for p in self._players.values() if p.finished],
            key=lambda p: p.rank,
        )
        non_finishers = [p for p in self._players.values() if not p.finished]
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
            out.append(
                {
                    "rank": len(out) + 1,
                    "descriptor": p.descriptor + " (DNF)",
                    "time_s": time.time() - self._game_start_ts,
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
        self.audio.play_music("bgm")
        self._go(State.START)

    # ==================================================================
    # Helpers — finishes, motion, photos
    # ==================================================================
    def _check_finishes(self, frame, pose) -> None:
        finish_y = self._cached_finish_y or 0
        # 1. LASER takes priority — fire a finish for the closest player
        if self.laser.in_use and USE_LASER:
            if self.laser.broken:
                closest = self._closest_unfinished_to_finish(pose, finish_y)
                if closest is not None:
                    self._mark_finished(closest, frame)
                return
        # 2. Camera tape — only used if USE_TAPE_FINISH
        if not USE_TAPE_FINISH or finish_y <= 0:
            return
        if pose is None or pose.get("boxes") is None:
            return
        boxes = pose["boxes"]
        ids = pose.get("track_ids", [])
        for box, tid in zip(boxes, ids):
            if tid is None or tid not in self._players:
                continue
            p = self._players[tid]
            if p.finished:
                continue
            foot = LineDetector.player_foot_y(box)
            if LineDetector.has_crossed_finish(foot, finish_y):
                self._mark_finished(p, frame)

    def _closest_unfinished_to_finish(self, pose, finish_y: int) -> Player | None:
        """Used for laser-trigger attribution: pick the player whose feet
        are closest to the finish line, that's still in play."""
        if pose is None or pose.get("boxes") is None:
            return None
        boxes = pose["boxes"]
        ids = pose.get("track_ids", [])
        best: tuple[float, Player] | None = None
        for box, tid in zip(boxes, ids):
            if tid is None or tid not in self._players:
                continue
            p = self._players[tid]
            if p.finished:
                continue
            foot = LineDetector.player_foot_y(box)
            dist = abs(foot - finish_y)
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
        ids = pose.get("track_ids", [])
        max_dist = 0.0
        for box, tid in zip(boxes, ids):
            if tid is None or tid not in self._players:
                continue
            p = self._players[tid]
            if p.finished or p.is_caught_this_phase or p.needs_to_return:
                continue
            cx = float((box[0] + box[2]) / 2.0)
            cy = float((box[1] + box[3]) / 2.0)
            bx, by = p.baseline_pos
            dist = math.hypot(cx - bx, cy - by)
            if dist > max_dist:
                max_dist = dist
            if dist > self._motion_px:
                self._on_player_caught(p, frame, cx, cy)
        # Smooth the meter
        target = min(1.0, max_dist / max(1.0, self._motion_px * 3.0))
        self._motion_score = 0.7 * self._motion_score + 0.3 * target

    def _on_player_caught(self, p: Player, frame, cx: float, cy: float) -> None:
        p.is_caught_this_phase = True
        p.needs_to_return = True
        p.is_back_at_start = False
        p.times_caught += 1
        p._dwell_start = None  # type: ignore[attr-defined]
        # Try to upgrade their photo while we're here, since they're
        # presumably stationary (caught means they barely moved past
        # threshold, so the next few frames are a good capture window).
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
        self.audio.announce_caught(p.descriptor)

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
        for box, tid in zip(boxes, ids):
            if tid is None or tid not in self._players:
                continue
            p = self._players[tid]
            p.last_box = box
            p.last_seen_ts = now

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
            if key == pygame.K_5:
                self.audio.test_tts()
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
            self._go(State.LEADERBOARD)
            return

        # SPACE — bypass palm OR play again from leaderboard
        if key == pygame.K_SPACE:
            if self._state == State.LEADERBOARD:
                self._go(State.RESET)
            elif self._state == State.START:
                self._palm_since = time.time() - PALM_HOLD
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
