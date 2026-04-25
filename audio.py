#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Project:      Red Light Green Light (Jetson Orin Nano)
File:         audio.py
Description:  Background music, low-latency SFX, and espeak TTS announcements.

Author:       Richard Pu
Last Updated: April 2026
===============================================================================
"""

from __future__ import annotations

import os
import subprocess
import threading

import pygame

from config import (
    AUDIO_BUFFER,
    AUDIO_FREQUENCY,
    IS_WINDOWS,
    SOUNDS,
    SOUNDS_DIR,
    TTS_LINES,
    TTS_WPM,
    VOL_MUSIC,
    VOL_SFX,
)


def pre_init() -> None:
    """Configure the mixer before pygame.init(). USB-C friendly settings."""
    try:
        pygame.mixer.pre_init(
            frequency=AUDIO_FREQUENCY,
            size=-16,
            channels=2,
            buffer=AUDIO_BUFFER,
        )
    except Exception as exc:
        print(f"[AUD] pre_init failed: {exc}")


class AudioManager:
    def __init__(self) -> None:
        self._silent = False
        self._sfx: dict[str, pygame.mixer.Sound] = {}
        self._music: dict[str, str] = {}
        self._tts_lock = threading.Lock()
        self._espeak_ok = self._check_espeak()

        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(
                    frequency=AUDIO_FREQUENCY,
                    size=-16,
                    channels=2,
                    buffer=AUDIO_BUFFER,
                )
        except Exception as exc:
            print(f"[AUD] No audio device ({exc}) - silent mode")
            self._silent = True
            return

        os.makedirs(SOUNDS_DIR, exist_ok=True)
        for key, fname in SOUNDS.items():
            path = os.path.join(SOUNDS_DIR, fname)
            if not os.path.exists(path):
                continue
            try:
                if fname.endswith(".mp3"):
                    self._music[key] = path
                else:
                    snd = pygame.mixer.Sound(path)
                    snd.set_volume(VOL_SFX)
                    self._sfx[key] = snd
            except Exception as exc:
                print(f"[AUD] Load error ({fname}): {exc}")
        print(f"[AUD] Loaded {len(self._sfx)} sfx, {len(self._music)} music, " f"espeak={self._espeak_ok}")

    @staticmethod
    def _check_espeak() -> bool:
        if IS_WINDOWS:
            return False
        try:
            subprocess.run(
                ["espeak", "--version"],
                capture_output=True,
                timeout=2,
            )
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return False

    # ------------------------------------------------------------------
    # Primitives
    # ------------------------------------------------------------------
    def play(self, key: str) -> None:
        if self._silent:
            return
        if key in self._sfx:
            try:
                self._sfx[key].play()
                return
            except Exception:
                pass
        line = TTS_LINES.get(key, "")
        if line:
            self.say(line)

    def stop_sfx(self, key: str) -> None:
        if self._silent:
            return
        if key in self._sfx:
            try:
                self._sfx[key].stop()
            except Exception:
                pass

    def play_music(self, key: str = "bgm", loop: bool = True) -> None:
        if self._silent or key not in self._music:
            return
        try:
            pygame.mixer.music.load(self._music[key])
            pygame.mixer.music.set_volume(VOL_MUSIC)
            pygame.mixer.music.play(-1 if loop else 0)
        except Exception as exc:
            print(f"[AUD] Music error ({key}): {exc}")

    def stop_music(self) -> None:
        if self._silent:
            return
        try:
            pygame.mixer.music.stop()
        except Exception:
            pass

    def fade_music(self, ms: int = 1200) -> None:
        if self._silent:
            return
        try:
            pygame.mixer.music.fadeout(ms)
        except Exception:
            pass

    def say(self, text: str, block: bool = False) -> None:
        if self._silent or not self._espeak_ok:
            print(f"[TTS] {text}")
            return

        def _speak() -> None:
            with self._tts_lock:
                try:
                    subprocess.run(
                        ["espeak", "-v", "en+f3", f"-s{TTS_WPM}", "--", text],
                        timeout=15,
                        capture_output=True,
                    )
                except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                    pass

        if block:
            _speak()
        else:
            threading.Thread(target=_speak, daemon=True, name="tts").start()

    # ------------------------------------------------------------------
    # Game-level events
    # ------------------------------------------------------------------
    def on_green(self) -> None:
        if self._silent:
            print("[AUD] GREEN")
            return
        self.play_music("bgm")
        self.play("green")

    def on_red(self) -> None:
        if self._silent:
            print("[AUD] RED")
            return
        if "mugunghwa" in self._sfx:
            self.play("mugunghwa")
        else:
            self.play("red")

    def on_caught(self, colour: str = "") -> None:
        if self._silent:
            print(f"[AUD] CAUGHT: {colour}")
            return
        self.play("caught")
        if colour and colour != "unknown":
            threading.Timer(1.1, self.say, args=[f"Player in {colour} shirt! Return to start!"]).start()
        else:
            threading.Timer(1.1, self.say, args=["You moved! Return to start!"]).start()

    def on_winner(self) -> None:
        if self._silent:
            print("[AUD] WINNER")
            return
        self.fade_music(600)
        self.play("winner")
        threading.Timer(1.8, self.say, args=["We have a winner! Amazing!"]).start()

    def on_countdown(self, n: int) -> None:
        if self._silent:
            print(f"[AUD] {n}")
            return
        self.say(str(n), block=False)

    def on_game_start(self) -> None:
        if self._silent:
            print("[AUD] START")
            return
        self.say(TTS_LINES["start"])

    # ------------------------------------------------------------------
    # Dev-mode test hooks
    # ------------------------------------------------------------------
    def test_chime(self) -> None:
        if self._silent:
            print("[AUD] TEST CHIME")
            return
        if "chime" in self._sfx:
            self.play("chime")
        elif "tick" in self._sfx:
            self.play("tick")
        else:
            self.say("Chime")
