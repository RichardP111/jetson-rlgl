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

import time, random, math, pygame
from enum import Enum, auto
from dataclasses import dataclass, field
import config
from config import (
    YOLO_SKIP_FRAMES,
    FPS_CAP,
    COUNTDOWN_N,
    SETTLE_TIME,
    GRACE_PERIOD,
    CAUGHT_HOLD,
    PALM_HOLD,
)
from hardware import Camera, ServoController, LaserBreakBeam
from vision import ProPoseTracker, detect_palm_raise, check_tape_finish
from audio import AudioManager
from ui import UIRenderer


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
    cx: float
    cy: float
    ts: float = field(default_factory=time.time)


class GameEngine:
    def __init__(self, camera, servo, laser, tracker, audio, ui):
        self.camera = camera
        self.servo = servo
        self.laser = laser
        self.tracker = tracker
        self.audio = audio
        self.ui = ui
        self._state = State.START
        self._state_ts = time.time()
        self._round = 0
        self._game_start_ts = 0.0
        self._light_dur = 0.0
        self._caught = []
        self._last_caught = None
        self._winner_colour = "unknown"
        self._highlights = []
        self._baselines = {}
        self._palm_since = None
        self._last_frame = None
        self._last_pose_data = None
        self._yolo_skip = 0
        self._last_beep = -1
        self._difficulty = "Normal"
        d = config.DIFFICULTY_PRESETS[self._difficulty]
        self._green_min, self._green_max = d["green_min"], d["green_max"]
        self._red_min, self._red_max = d["red_min"], d["red_max"]
        self._motion_px = d["motion_px"]

    def run(self, clock):
        print("[ENGINE] Game live. Keys: G=Green R=Red W=Win E=Elim SPACE=palm ESC=quit")
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
            self._yolo_skip += 1
            if self._yolo_skip % (YOLO_SKIP_FRAMES + 1) == 0:
                pose_data, overlay = self.tracker.process_frame(frame)
                self._last_pose_data = pose_data
            else:
                pose_data = self._last_pose_data
                overlay = frame
            self._dispatch(frame, overlay, pose_data, clock)
            clock.tick(FPS_CAP)

    def _dispatch(self, frame, overlay, pose, clock):
        s = self._state
        if s == State.START:
            self._do_start(frame, overlay, pose, clock)
        elif s == State.COUNTDOWN:
            self._do_countdown(frame, overlay, clock)
        elif s == State.GREEN:
            self._do_green(frame, overlay, pose, clock)
        elif s == State.TURNING_RED:
            self._do_turning_red(frame, overlay, pose, clock)
        elif s == State.RED:
            self._do_red(frame, overlay, pose, clock)
        elif s == State.CAUGHT_PAUSE:
            self._do_caught_pause(frame, overlay, clock)
        elif s == State.TURNING_GREEN:
            self._do_turning_green(frame, overlay, pose, clock)
        elif s == State.WINNER:
            self._do_winner(frame, overlay, pose, clock)
        elif s == State.RESET:
            self._do_reset()

    def _do_start(self, frame, overlay, pose, clock):
        palm = detect_palm_raise(pose)
        prog = self._palm_progress(palm)
        self.ui.draw_start_screen(overlay, palm_progress=prog, clock=clock)
        if prog >= 1.0:
            self._palm_since = None
            self.audio.on_game_start()
            self._go(State.COUNTDOWN)

    def _do_countdown(self, frame, overlay, clock):
        t = self._in_state()
        n = COUNTDOWN_N - int(t)
        self.ui.draw_countdown(overlay, max(1, n), clock)
        if int(t) != self._last_beep and n > 0:
            self._last_beep = int(t)
            self.audio.on_countdown(n)
        if t >= COUNTDOWN_N:
            self._round = 1
            self._caught = []
            self._highlights = []
            self._game_start_ts = time.time()
            self._light_dur = random.uniform(self._green_min, self._green_max)
            self.servo.face_away()
            self.audio.on_green()
            self._go(State.GREEN)

    def _do_green(self, frame, overlay, pose, clock):
        elapsed = time.time() - self._game_start_ts
        alive = pose["players_alive"] if pose else 0
        time_left = max(0, self._light_dur - self._in_state())
        self.ui.draw_game_hud(overlay, "GREEN", alive, elapsed, self._round, [e.shirt_colour for e in self._caught], 0.0, time_left, clock)
        if frame is not None and random.random() < 0.003:
            self._highlights.append(frame.copy())
        if self._check_finish(frame, pose):
            return
        if self._in_state() >= self._light_dur:
            self._initiate_red()

    def _initiate_red(self):
        self._round += 1
        self.servo.face_players()
        self.audio.on_red()
        self._light_dur = random.uniform(self._red_min, self._red_max)
        self._go(State.TURNING_RED)

    def _do_turning_red(self, frame, overlay, pose, clock):
        elapsed = time.time() - self._game_start_ts
        alive = pose["players_alive"] if pose else 0
        time_left = max(0, self._light_dur)
        self.ui.draw_game_hud(overlay, "TURNING", alive, elapsed, self._round, [e.shirt_colour for e in self._caught], 0.0, time_left, clock)
        settled = self.servo.is_facing_players or self._in_state() >= SETTLE_TIME + 0.8
        if settled and self._in_state() >= SETTLE_TIME + GRACE_PERIOD:
            self._snapshot_baselines(pose)
            self._go(State.RED)

    def _do_red(self, frame, overlay, pose, clock):
        elapsed = time.time() - self._game_start_ts
        alive = pose["players_alive"] if pose else 0
        caught_entry, motion_score = self._check_motion(pose, frame)
        time_left = max(0, self._light_dur - self._in_state())
        self.ui.draw_game_hud(overlay, "RED", alive, elapsed, self._round, [e.shirt_colour for e in self._caught], motion_score, time_left, clock)
        if caught_entry:
            self._last_caught = caught_entry
            self._caught.append(caught_entry)
            self.ui.flash_caught(int(caught_entry.cx), int(caught_entry.cy))
            if frame is not None:
                self._highlights.append(frame.copy())
            self.audio.on_caught(caught_entry.shirt_colour)
            self._go(State.CAUGHT_PAUSE)
            return
        if self._check_finish(frame, pose):
            return
        if self._in_state() >= self._light_dur:
            self._initiate_green()

    def _initiate_green(self):
        self.servo.face_away()
        self._baselines.clear()
        self._light_dur = random.uniform(self._green_min, self._green_max)
        self.audio.on_green()
        self._go(State.TURNING_GREEN)

    def _do_caught_pause(self, frame, overlay, clock):
        progress = min(1.0, self._in_state() / CAUGHT_HOLD)
        colour = self._last_caught.shirt_colour if self._last_caught else "unknown"
        self.ui.draw_caught_screen(frame, colour, progress, clock)
        if self._in_state() >= CAUGHT_HOLD:
            self._snapshot_baselines(self._last_pose_data)
            self._go(State.RED)

    def _do_turning_green(self, frame, overlay, pose, clock):
        elapsed = time.time() - self._game_start_ts
        alive = pose["players_alive"] if pose else 0
        time_left = max(0, self._light_dur)
        self.ui.draw_game_hud(overlay, "TURNING", alive, elapsed, self._round, [e.shirt_colour for e in self._caught], 0.0, time_left, clock)
        if not self.servo.is_facing_players and self._in_state() > 0.8:
            self._go(State.GREEN)

    def _do_winner(self, frame, overlay, pose, clock):
        palm = detect_palm_raise(pose)
        prog = self._palm_progress(palm)
        self.ui.draw_winner_screen(self._winner_colour, self._highlights, prog, clock)
        if prog >= 1.0:
            self._palm_since = None
            self._go(State.RESET)

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

    def _snapshot_baselines(self, pose):
        self._baselines.clear()
        if not pose:
            return
        for box, tid in zip(pose.get("boxes", []), pose.get("track_ids", [])):
            if tid is None:
                continue
            cx, cy = float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)
            self._baselines[tid] = (cx, cy)

    def _check_motion(self, pose, frame):
        if not pose or not self._baselines or self._in_state() < SETTLE_TIME:
            return None, 0.0
        max_dist = 0.0
        for i, (box, tid, colour) in enumerate(zip(pose.get("boxes", []), pose.get("track_ids", []), pose.get("shirt_colours", []))):
            if tid is None or tid not in self._baselines:
                continue
            cx, cy = float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)
            bx, by = self._baselines[tid]
            dist = math.hypot(cx - bx, cy - by)
            max_dist = max(max_dist, dist)
            if dist > self._motion_px:
                return CaughtEntry(tid, colour, cx, cy), min(1.0, dist / (self._motion_px * 4))
        return None, min(1.0, max_dist / (self._motion_px * 3))

    def _check_finish(self, frame, pose):
        if self.laser.broken:
            self._trigger_winner(pose, frame, "LASER")
            return True
        if check_tape_finish(frame, pose):
            self._trigger_winner(pose, frame, "TAPE")
            return True
        return False

    def _trigger_winner(self, pose, frame, reason):
        print(f"[WIN] {reason}")
        colour = "unknown"
        if pose and pose.get("shirt_colours"):
            colour = pose["shirt_colours"][0]
        self._winner_colour = colour
        if frame is not None:
            self._highlights.append(frame.copy())
        self.servo.face_away()
        self.audio.on_winner()
        self._go(State.WINNER)

    def _go(self, new_state):
        print(f"  [{self._state.name}] → [{new_state.name}]")
        self._state = new_state
        self._state_ts = time.time()

    def _in_state(self):
        return time.time() - self._state_ts

    def _palm_progress(self, palm_up):
        if palm_up:
            if self._palm_since is None:
                self._palm_since = time.time()
            return min(1.0, (time.time() - self._palm_since) / PALM_HOLD)
        self._palm_since = None
        return 0.0

    def _handle_key(self, key):
        if key == pygame.K_ESCAPE:
            raise SystemExit
        if key == pygame.K_g:
            self.servo.face_away()
            self._light_dur = random.uniform(self._green_min, self._green_max)
            self.audio.on_green()
            self._go(State.GREEN)
        if key == pygame.K_r:
            self._snapshot_baselines(self._last_pose_data)
            self._light_dur = random.uniform(self._red_min, self._red_max)
            self.servo.face_players()
            self.audio.on_red()
            self._go(State.RED)
        if key == pygame.K_w:
            self._winner_colour = "blue"
            self.audio.on_winner()
            self._go(State.WINNER)
        if key == pygame.K_e:
            dummy = CaughtEntry(0, "blue", 640.0, 360.0)
            self._last_caught = dummy
            self._caught.append(dummy)
            self.audio.on_caught("blue")
            self._go(State.CAUGHT_PAUSE)
        if key == pygame.K_SPACE:
            self._palm_since = time.time() - PALM_HOLD
