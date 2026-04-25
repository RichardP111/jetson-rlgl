#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Project:      Red Light Green Light (Jetson Orin Nano)
File:         game.py
Description:  Finite state machine governing game phases. Owns no rendering;
              it delegates all visuals to UIRenderer and all hardware to the
              Camera / ServoController / LaserBreakBeam objects.

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
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto

import cv2
import numpy as np
import pygame

from audio import AudioManager
from config import (
    CAUGHT_HOLD,
    COUNTDOWN_N,
    DEFAULT_DIFFICULTY,
    DIFFICULTY_PRESETS,
    FPS_CAP,
    GRACE_PERIOD,
    HIGHLIGHT_MAX,
    HIGHLIGHT_SCALE,
    PALM_HOLD,
    SETTLE_TIME,
)
from hardware import Camera, LaserBreakBeam, ServoController
from ui import UIRenderer
from vision import PoseWorker, ProPoseTracker, check_tape_finish, detect_palm_raise


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


# Maps internal State -> banner label used by the UI
_BANNER_LABEL = {
    State.START: "START",
    State.COUNTDOWN: "COUNTDOWN",
    State.GREEN: "GREEN",
    State.TURNING_RED: "TURNING",
    State.RED: "RED",
    State.CAUGHT_PAUSE: "CAUGHT",
    State.TURNING_GREEN: "TURNING",
    State.WINNER: "WINNER",
    State.RESET: "START",
}


@dataclass
class CaughtEntry:
    track_id: int
    shirt_colour: str
    cx: float
    cy: float
    ts: float = field(default_factory=time.time)


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
    ) -> None:
        self.camera = camera
        self.servo = servo
        self.laser = laser
        self.tracker = tracker  # kept for dev panel (last_inference_ms)
        self.pose_worker = pose_worker  # NEW: source of (frame, pose) each tick
        self.audio = audio
        self.ui = ui

        self._state = State.START
        self._state_ts = time.time()
        self._round = 0
        self._game_start_ts = 0.0
        self._light_dur = 0.0

        self._caught: list[CaughtEntry] = []
        self._last_caught: CaughtEntry | None = None
        self._winner_colour = "unknown"

        # Highlights: bounded ring of downsampled frames
        self._highlights: deque[np.ndarray] = deque(maxlen=HIGHLIGHT_MAX)

        self._baselines: dict[int, tuple[float, float]] = {}
        self._palm_since: float | None = None

        self._last_frame: np.ndarray | None = None
        self._last_pose: dict | None = None
        self._last_pose_id: int = -1
        self._last_beep = -1

        self._difficulty = DEFAULT_DIFFICULTY
        d = DIFFICULTY_PRESETS[self._difficulty]
        self._green_min, self._green_max = d["green_min"], d["green_max"]
        self._red_min, self._red_max = d["red_min"], d["red_max"]
        self._motion_px = d["motion_px"]

        self._laser_last_state = False

    # ==================================================================
    # Main loop
    # ==================================================================
    def run(self, clock: pygame.time.Clock) -> None:
        print("[ENGINE] Running. Dev: CTRL+D  Cheats: G R W E SPACE  ESC to quit")
        self.servo.face_away()
        self.audio.play_music("bgm")

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    self._handle_key(event)

            # Non-blocking pull from the pose worker. The worker keeps
            # YOLO running on its own thread; we just consume the most
            # recent (frame, pose) pair. If the worker hasn't produced a
            # new pair yet, we re-render with the previous one — which is
            # exactly the behaviour we want at 60 FPS UI on top of a
            # ~10–30 FPS detector.
            frame, pose, pose_id = self.pose_worker.latest()
            self._last_frame = frame
            if pose_id != self._last_pose_id:
                self._last_pose = pose
                self._last_pose_id = pose_id

            self._dispatch(frame, self._last_pose, clock)

            clock.tick(FPS_CAP)

    # ==================================================================
    # State dispatch
    # ==================================================================
    def _dispatch(
        self,
        frame: np.ndarray | None,
        pose: dict | None,
        clock: pygame.time.Clock,
    ) -> None:
        dispatchers = {
            State.START: self._do_start,
            State.COUNTDOWN: self._do_countdown,
            State.GREEN: self._do_green,
            State.TURNING_RED: self._do_turning_red,
            State.RED: self._do_red,
            State.CAUGHT_PAUSE: self._do_caught_pause,
            State.TURNING_GREEN: self._do_turning_green,
            State.WINNER: self._do_winner,
            State.RESET: self._do_reset,
        }
        fn = dispatchers.get(self._state)
        if fn is None:
            return
        if self._state == State.RESET:
            fn()
        else:
            fn(frame, pose, clock)

        # Dev panel draws on top of whatever screen was just rendered.
        if self.ui.is_dev_mode():
            self.ui.draw_dev_panel(self._dev_metrics(clock))
        self.ui.present()

    # ==================================================================
    # State implementations
    # ==================================================================
    def _do_start(
        self,
        frame: np.ndarray | None,
        pose: dict | None,
        clock: pygame.time.Clock,
    ) -> None:
        palm = detect_palm_raise(pose)
        prog = self._palm_progress(palm)
        self.ui.draw_start_screen(frame, prog, clock)
        if prog >= 1.0:
            self._palm_since = None
            self.audio.on_game_start()
            self._go(State.COUNTDOWN)

    def _do_countdown(
        self,
        frame: np.ndarray | None,
        pose: dict | None,
        clock: pygame.time.Clock,
    ) -> None:
        t = self._in_state()
        n = COUNTDOWN_N - int(t)
        self.ui.draw_countdown(frame, max(1, n), clock)
        if int(t) != self._last_beep and n > 0:
            self._last_beep = int(t)
            self.audio.on_countdown(n)
        if t >= COUNTDOWN_N:
            self._begin_game()

    def _begin_game(self) -> None:
        self._round = 1
        self._caught = []
        self._highlights.clear()
        self._game_start_ts = time.time()
        self._light_dur = random.uniform(self._green_min, self._green_max)
        self.servo.face_away()
        self.audio.on_green()
        self._go(State.GREEN)

    def _do_green(
        self,
        frame: np.ndarray | None,
        pose: dict | None,
        clock: pygame.time.Clock,
    ) -> None:
        elapsed = time.time() - self._game_start_ts
        alive = pose["players_alive"] if pose else 0
        time_left = max(0.0, self._light_dur - self._in_state())
        self.ui.draw_game_hud(
            frame,
            pose,
            "GREEN",
            alive,
            elapsed,
            self._round,
            [e.shirt_colour for e in self._caught],
            motion_score=0.0,
            time_left=time_left,
            time_total=self._light_dur,  # ⬅ progress bar now scales correctly
            clock=clock,
        )

        if frame is not None and random.random() < 0.003:
            self._push_highlight(frame)
        if self._check_finish(frame, pose):
            return
        if self._in_state() >= self._light_dur:
            self._initiate_red()

    def _initiate_red(self) -> None:
        self._round += 1
        self.servo.face_players()
        self.audio.on_red()
        self._light_dur = random.uniform(self._red_min, self._red_max)
        self._go(State.TURNING_RED)

    def _do_turning_red(
        self,
        frame: np.ndarray | None,
        pose: dict | None,
        clock: pygame.time.Clock,
    ) -> None:
        elapsed = time.time() - self._game_start_ts
        alive = pose["players_alive"] if pose else 0
        # Time_left reflects the upcoming RED duration, held until RED starts.
        self.ui.draw_game_hud(
            frame,
            pose,
            "TURNING",
            alive,
            elapsed,
            self._round,
            [e.shirt_colour for e in self._caught],
            motion_score=0.0,
            time_left=self._light_dur,
            time_total=self._light_dur,
            clock=clock,
        )
        settled = self.servo.is_facing_players or self._in_state() >= SETTLE_TIME + 0.8
        if settled and self._in_state() >= SETTLE_TIME + GRACE_PERIOD:
            self._snapshot_baselines(pose)
            self._state_ts = time.time()  # RED duration starts now
            self._go(State.RED)

    def _do_red(
        self,
        frame: np.ndarray | None,
        pose: dict | None,
        clock: pygame.time.Clock,
    ) -> None:
        elapsed = time.time() - self._game_start_ts
        alive = pose["players_alive"] if pose else 0
        caught_entry, motion_score = self._check_motion(pose)
        time_left = max(0.0, self._light_dur - self._in_state())
        self.ui.draw_game_hud(
            frame,
            pose,
            "RED",
            alive,
            elapsed,
            self._round,
            [e.shirt_colour for e in self._caught],
            motion_score=motion_score,
            time_left=time_left,
            time_total=self._light_dur,
            clock=clock,
        )

        if caught_entry is not None:
            self._process_caught(caught_entry, frame)
            return
        if self._check_finish(frame, pose):
            return
        if self._in_state() >= self._light_dur:
            self._initiate_green()

    def _process_caught(self, entry: CaughtEntry, frame: np.ndarray | None) -> None:
        self._last_caught = entry
        self._caught.append(entry)
        sx = 1.0
        sy = 1.0
        if self._last_pose is not None:
            fw = self._last_pose.get("frame_w", 0) or 0
            fh = self._last_pose.get("frame_h", 0) or 0
            from config import DISPLAY_H, DISPLAY_W

            if fw:
                sx = DISPLAY_W / fw
            if fh:
                sy = DISPLAY_H / fh
        self.ui.flash_caught(int(entry.cx * sx), int(entry.cy * sy))
        self.ui.log_elimination(entry.shirt_colour)
        if frame is not None:
            self._push_highlight(frame)
        self.audio.on_caught(entry.shirt_colour)
        self._go(State.CAUGHT_PAUSE)

    def _initiate_green(self) -> None:
        self.servo.face_away()
        self._baselines.clear()
        self._light_dur = random.uniform(self._green_min, self._green_max)
        self.audio.on_green()
        self._go(State.TURNING_GREEN)

    def _do_caught_pause(
        self,
        frame: np.ndarray | None,
        pose: dict | None,
        clock: pygame.time.Clock,
    ) -> None:
        progress = min(1.0, self._in_state() / CAUGHT_HOLD)
        colour = self._last_caught.shirt_colour if self._last_caught else "unknown"
        self.ui.draw_caught_screen(frame, colour, progress, clock)
        if self._in_state() >= CAUGHT_HOLD:
            self._snapshot_baselines(self._last_pose)
            self._state_ts = time.time()
            self._go(State.RED)

    def _do_turning_green(
        self,
        frame: np.ndarray | None,
        pose: dict | None,
        clock: pygame.time.Clock,
    ) -> None:
        elapsed = time.time() - self._game_start_ts
        alive = pose["players_alive"] if pose else 0
        self.ui.draw_game_hud(
            frame,
            pose,
            "TURNING",
            alive,
            elapsed,
            self._round,
            [e.shirt_colour for e in self._caught],
            motion_score=0.0,
            time_left=self._light_dur,
            time_total=self._light_dur,
            clock=clock,
        )
        if not self.servo.is_facing_players and self._in_state() > 0.8:
            self._state_ts = time.time()
            self._go(State.GREEN)

    def _do_winner(
        self,
        frame: np.ndarray | None,
        pose: dict | None,
        clock: pygame.time.Clock,
    ) -> None:
        palm = detect_palm_raise(pose)
        prog = self._palm_progress(palm)
        self.ui.draw_winner_screen(frame, self._winner_colour, prog, clock)
        if prog >= 1.0:
            self._palm_since = None
            self._go(State.RESET)

    def _do_reset(self) -> None:
        self.servo.face_away()
        self._baselines.clear()
        self._caught = []
        self._highlights.clear()
        self._round = 0
        self._winner_colour = "unknown"
        self._last_caught = None
        self._palm_since = None
        self.audio.play_music("bgm")
        self._go(State.START)

    # ==================================================================
    # Game logic helpers
    # ==================================================================
    def _snapshot_baselines(self, pose: dict | None) -> None:
        self._baselines.clear()
        if not pose:
            return
        boxes = pose.get("boxes")
        ids = pose.get("track_ids", [])
        if boxes is None:
            return
        for box, tid in zip(boxes, ids):
            if tid is None:
                continue
            cx = float((box[0] + box[2]) / 2.0)
            cy = float((box[1] + box[3]) / 2.0)
            self._baselines[tid] = (cx, cy)

    def _check_motion(self, pose: dict | None) -> tuple[CaughtEntry | None, float]:
        if pose is None or not self._baselines or self._in_state() < GRACE_PERIOD:
            return None, 0.0
        boxes = pose.get("boxes")
        ids = pose.get("track_ids", [])
        shirts = pose.get("shirt_colours", [])
        if boxes is None:
            return None, 0.0
        max_dist = 0.0
        for box, tid, colour in zip(boxes, ids, shirts):
            if tid is None or tid not in self._baselines:
                continue
            cx = float((box[0] + box[2]) / 2.0)
            cy = float((box[1] + box[3]) / 2.0)
            bx, by = self._baselines[tid]
            dist = math.hypot(cx - bx, cy - by)
            if dist > max_dist:
                max_dist = dist
            if dist > self._motion_px:
                return CaughtEntry(tid, colour, cx, cy), min(1.0, dist / (self._motion_px * 4.0))
        return None, min(1.0, max_dist / (self._motion_px * 3.0))

    def _check_finish(self, frame: np.ndarray | None, pose: dict | None) -> bool:
        if self.laser.broken:
            self._trigger_winner(pose, frame, "LASER")
            return True
        if check_tape_finish(frame, pose):
            self._trigger_winner(pose, frame, "TAPE")
            return True
        return False

    def _trigger_winner(
        self,
        pose: dict | None,
        frame: np.ndarray | None,
        reason: str,
    ) -> None:
        print(f"[WIN] {reason}")
        colour = "unknown"
        if pose and pose.get("shirt_colours"):
            colour = pose["shirt_colours"][0]
        self._winner_colour = colour
        if frame is not None:
            self._push_highlight(frame)
        self.servo.face_away()
        self.audio.on_winner()
        self._go(State.WINNER)

    def _push_highlight(self, frame: np.ndarray) -> None:
        """Downsample and bound-cache a frame to avoid memory growth."""
        try:
            h, w = frame.shape[:2]
            small = cv2.resize(
                frame,
                (max(1, int(w * HIGHLIGHT_SCALE)), max(1, int(h * HIGHLIGHT_SCALE))),
                interpolation=cv2.INTER_AREA,
            )
            self._highlights.append(small)
        except Exception:
            pass

    def _go(self, new_state: State) -> None:
        print(f"  [{self._state.name}] -> [{new_state.name}]")
        self._state = new_state
        self._state_ts = time.time()

    def _in_state(self) -> float:
        return time.time() - self._state_ts

    def _palm_progress(self, palm_up: bool) -> float:
        if palm_up:
            if self._palm_since is None:
                self._palm_since = time.time()
            return min(1.0, (time.time() - self._palm_since) / PALM_HOLD)
        self._palm_since = None
        return 0.0

    # ==================================================================
    # Input
    # ==================================================================
    def _handle_key(self, event: pygame.event.Event) -> None:
        key = event.key
        mods = pygame.key.get_mods()

        # Dev mode toggle
        if key == pygame.K_d and (mods & pygame.KMOD_CTRL):
            self.ui.toggle_dev_mode()
            return

        if key == pygame.K_ESCAPE:
            raise SystemExit

        # Dev-only hardware tests (active when dev mode is on)
        if self.ui.is_dev_mode():
            if key == pygame.K_1:
                self.servo.face_players()
                return
            if key == pygame.K_2:
                self.servo.face_away()
                return
            if key == pygame.K_3:
                print(f"[DEV] Laser broken = {self.laser.broken}")
                return
            if key == pygame.K_4:
                self.audio.test_chime()
                return

        # Gameplay cheat keys
        if key == pygame.K_g:
            self.servo.face_away()
            self._light_dur = random.uniform(self._green_min, self._green_max)
            self.audio.on_green()
            self._go(State.GREEN)
        elif key == pygame.K_r:
            self._snapshot_baselines(self._last_pose)
            self._light_dur = random.uniform(self._red_min, self._red_max)
            self.servo.face_players()
            self.audio.on_red()
            self._go(State.RED)
        elif key == pygame.K_w:
            self._winner_colour = "blue"
            self.audio.on_winner()
            self._go(State.WINNER)
        elif key == pygame.K_e:
            dummy = CaughtEntry(0, "blue", 640.0, 360.0)
            self._last_caught = dummy
            self._caught.append(dummy)
            self.ui.log_elimination("blue")
            self.audio.on_caught("blue")
            self._go(State.CAUGHT_PAUSE)
        elif key == pygame.K_SPACE:
            self._palm_since = time.time() - PALM_HOLD

    # ==================================================================
    # Dev metrics
    # ==================================================================
    def _dev_metrics(self, clock: pygame.time.Clock) -> dict:
        alive = 0
        if self._last_pose is not None:
            alive = self._last_pose.get("players_alive", 0)
        return {
            "fps": clock.get_fps(),
            "cam_fps": self.camera.fps,
            "inference_ms": self.tracker.last_inference_ms,
            "state": self._state.name,
            "state_time": self._in_state(),
            "servo_angle": self.servo.angle,
            "servo_target": self.servo.target,
            "laser_broken": self.laser.broken,
            "players": alive,
            "dev_hints": [
                f"Backend: {self.camera.backend}",
                f"Servo HW: {self.servo.is_hardware}  Laser HW: {self.laser.enabled}",
                f"Pose worker: id={self._last_pose_id}",
            ],
        }
