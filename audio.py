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

import os, threading, subprocess, pygame
from config import SOUNDS_DIR, SOUNDS, TTS_LINES, VOL_SFX, VOL_MUSIC, TTS_WPM


class AudioManager:
    def __init__(self):
        self._docker_mode, self._sfx, self._music, self._tts_lock = False, {}, {}, threading.Lock()
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
        except Exception:
            print("[AUD] No audio device — silent mode")
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
                    print(f"[AUD] Music → {fname}")
                else:
                    s = pygame.mixer.Sound(path)
                    s.set_volume(VOL_SFX)
                    self._sfx[key] = s
                    print(f"[AUD] SFX → {fname}")
            except Exception as exc:
                print(f"[AUD] Load error ({fname}): {exc}")
        print(f"[AUD] Loaded {len(self._sfx)} sfx, {len(self._music)} music")

    def play(self, key):
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

    def stop_sfx(self, key):
        if self._docker_mode:
            return
        if key in self._sfx:
            try:
                self._sfx[key].stop()
            except Exception:
                pass

    def play_music(self, key="bgm", loop=True):
        if self._docker_mode:
            return
        if key not in self._music:
            return
        try:
            pygame.mixer.music.load(self._music[key])
            pygame.mixer.music.set_volume(VOL_MUSIC)
            pygame.mixer.music.play(-1 if loop else 0)
        except Exception as exc:
            print(f"[AUD] Music error ({key}): {exc}")

    def stop_music(self):
        if self._docker_mode:
            return
        try:
            pygame.mixer.music.stop()
        except Exception:
            pass

    def fade_music(self, ms=1200):
        if self._docker_mode:
            return
        try:
            pygame.mixer.music.fadeout(ms)
        except Exception:
            pass

    def say(self, text, block=False):
        if self._docker_mode:
            print(f"[TTS] {text}")
            return

        def _speak():
            with self._tts_lock:
                try:
                    subprocess.run(["espeak", "-v", "en+f3", f"-s{TTS_WPM}", "--", text], timeout=15, capture_output=True)
                except FileNotFoundError:
                    print(f"[TTS] {text}")
                except subprocess.TimeoutExpired:
                    pass

        if block:
            _speak()
        else:
            threading.Thread(target=_speak, daemon=True, name="tts").start()

    def on_green(self):
        if self._docker_mode:
            print("[AUD] GREEN")
            return
        self.play_music("bgm")
        self.play("green")

    def on_red(self):
        if self._docker_mode:
            print("[AUD] RED")
            return
        if "mugunghwa" in self._sfx:
            self.play("mugunghwa")
        else:
            self.play("red")

    def on_caught(self, colour=""):
        if self._docker_mode:
            print(f"[AUD] CAUGHT: {colour}")
            return
        self.play("caught")
        if colour and colour != "unknown":
            threading.Timer(1.1, self.say, args=[f"Player in {colour} shirt! Return to start!"]).start()
        else:
            threading.Timer(1.1, self.say, args=["You moved! Return to start!"]).start()

    def on_winner(self):
        if self._docker_mode:
            print("[AUD] WINNER")
            return
        self.fade_music(600)
        self.play("winner")
        threading.Timer(1.8, self.say, args=["We have a winner! Amazing!"]).start()

    def on_countdown(self, n):
        if self._docker_mode:
            print(f"[AUD] {n}")
            return
        self.say(str(n), block=False)

    def on_game_start(self):
        if self._docker_mode:
            print("[AUD] START")
            return
        self.say(TTS_LINES["start"])

    def announce_green(self):
        self.on_green()

    def announce_red(self):
        self.on_red()

    def announce_caught(self, label):
        self.on_caught(label.split()[-2] if "shirt" in label else "")

    def announce_winner(self):
        self.on_winner()
