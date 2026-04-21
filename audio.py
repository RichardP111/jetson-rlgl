#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Project:      Red Light Green Light (Jetson Orin Nano)
File:         audio.py
Description:  Handles background music, low-latency sound effects,
              and espeak text-to-speech announcements.

Author:       Richard Pu
Last Updated: April 2026
===============================================================================
"""

import os
import subprocess
import threading

import pygame

from config import SOUNDS_DIR, TTS_WPM, VOL_MUSIC, VOL_SFX

# ── Sound file map ───────────────────────────────────────────────────
# Key → filename in assets/sounds/
# If a file is missing, TTS fallback is used where defined.
SOUNDS = {
    "bgm": "bgm.mp3",  # background music loop
    "green": "green_light.wav",  # "Green Light!" announcement
    "red": "red_light.wav",  # buzzer or voice
    "mugunghwa": "mugunghwa.wav",  # Korean freeze phrase
    "caught": "eliminated.wav",  # elimination sting
    "winner": "winner.wav",  # victory fanfare
    "tick": "tick.wav",  # countdown beep
    "tension": "tension.mp3",  # optional tension loop for red phase
}

TTS_LINES = {
    "green": "Green light! Go go go!",
    "red": "Red light! Freeze!",
    "caught": "You moved! Return to the start!",
    "winner": "We have a winner! Congratulations!",
    "start": "Get ready! The game is about to begin!",
}

class AudioManager:
    
    def __init__(self):
        self._docker_mode = False
        self._sfx = {}
        self._music = {}
        self._tts_lock = threading.Lock()

        # Detect if we're in Docker (no audio device)
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
            pygame.mixer.quit()  # Just testing, quit immediately
        except:
            print("[AUD] No audio device — running in silent mode")
            self._docker_mode = True
            return

        os.makedirs(SOUNDS_DIR, exist_ok=True)

        for key, fname in SOUNDS.items():
            path = os.path.join(SOUNDS_DIR, fname)
            if not os.path.exists(path):
                continue
            try:
                if fname.endswith(".mp3"):
                    self._music[key] = path
                    print(f"[AUD] Music   → {fname}")
                else:
                    s = pygame.mixer.Sound(path)
                    s.set_volume(VOL_SFX)
                    self._sfx[key] = s
                    print(f"[AUD] SFX     → {fname}")
            except Exception as exc:
                print(f"[AUD] Load error ({fname}): {exc}")

        print(f"[AUD] Loaded {len(self._sfx)} sfx, {len(self._music)} music tracks")

    # Stub methods that do nothing in Docker mode
    def play(self, key: str):
        if self._docker_mode:
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

    def stop_sfx(self, key: str):
        if self._docker_mode:
            return
        if key in self._sfx:
            try:
                self._sfx[key].stop()
            except Exception:
                pass

    def play_music(self, key: str = "bgm", loop: bool = True):
        if self._docker_mode:
            return
        if key not in self._music:
            return
        try:
            pygame.mixer.music.load(self._music[key])
            pygame.mixer.music.set_volume(VOL_MUSIC)
            pygame.mixer.music.play(-1 if loop else 0)
        except Exception as exc:
            print(f"[AUD] Music play error ({key}): {exc}")

    def stop_music(self):
        if self._docker_mode:
            return
        try:
            pygame.mixer.music.stop()
        except Exception:
            pass

    def fade_music(self, ms: int = 1200):
        if self._docker_mode:
            return
        try:
            pygame.mixer.music.fadeout(ms)
        except Exception:
            pass

    def say(self, text: str, block: bool = False):
        if self._docker_mode:
            print(f"[TTS] {text}")
            return
        # ... rest of original say() method

    def on_green(self):
        if self._docker_mode:
            print("[AUD] GREEN LIGHT")
            return
        self.play_music("bgm")
        self.play("green")

    def on_red(self):
        if self._docker_mode:
            print("[AUD] RED LIGHT")
            return
        if "mugunghwa" in self._sfx:
            self.play("mugunghwa")
        else:
            self.play("red")

    def on_caught(self, colour: str = ""):
        if self._docker_mode:
            print(f"[AUD] CAUGHT: {colour}")
            return
        self.play("caught")
        if colour and colour != "unknown":
            threading.Timer(1.1, self.say,
                            args=[f"The player in the {colour} shirt! Return to start!"]).start()
        else:
            threading.Timer(1.1, self.say, args=["You moved! Return to the start!"]).start()

    def on_winner(self):
        if self._docker_mode:
            print("[AUD] WINNER!")
            return
        self.fade_music(600)
        self.play("winner")
        threading.Timer(1.8, self.say, args=["We have a winner! Amazing!"]).start()

    def on_countdown(self, n: int):
        if self._docker_mode:
            print(f"[AUD] {n}")
            return
        self.say(str(n), block=False)

    def on_game_start(self):
        if self._docker_mode:
            print("[AUD] GAME START")
            return
        self.say(TTS_LINES["start"])