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

from config import SOUNDS_DIR, TTS_SPEED, VOLUME_MUSIC, VOLUME_SFX


# Mapping of sound keys to their filenames in the assets/sounds/ directory
SOUNDS: dict[str, str] = {
    "green": "green_light.wav",
    "red": "red_light.wav",
    "mugunghwa": "mugunghwa.wav",
    "elim": "eliminated.wav",
    "winner": "winner.wav",
    "tick": "countdown.wav",
    "bgm": "bgm.mp3",
    "tension": "tension.mp3",
}

# Fallback text-to-speech phrases if a required sound file is missing
TTS_FALLBACK: dict[str, str] = {
    "green": "Green Light!",
    "red": "Red Light!",
    "mugunghwa": "Mugunghwa!",
    "elim": "Eliminated!",
    "winner": "We have a winner!",
}


class AudioManager:
    """
    Unified audio interface for the game.
    Manages low-latency sound effects (.wav), background music (.mp3),
    and text-to-speech announcements via espeak.
    """

    def __init__(self):
        self._sfx: dict[str, pygame.mixer.Sound] = {}
        self._music: dict[str, str] = {}
        self._tts_lock = threading.Lock()

        os.makedirs(SOUNDS_DIR, exist_ok=True)

        for key, filename in SOUNDS.items():
            path = os.path.join(SOUNDS_DIR, filename)
            if not os.path.exists(path):
                continue

            try:
                # Load .mp3 files as music streams, .wav files as pre-loaded sound effects
                if filename.endswith(".mp3"):
                    self._music[key] = path
                    print(f"[AUD] Music loaded: {filename}")
                else:
                    snd = pygame.mixer.Sound(path)
                    snd.set_volume(VOLUME_SFX)
                    self._sfx[key] = snd
                    print(f"[AUD] SFX loaded: {filename}")
            except Exception as exc:
                print(f"[AUD] Error loading {filename}: {exc}")

    def play(self, key: str):
        """Plays a sound effect by key, falling back to TTS if the file is missing."""
        if key in self._sfx:
            try:
                self._sfx[key].play()
                return
            except Exception as exc:
                print(f"[AUD] Error playing SFX {key}: {exc}")

        # Fallback to text-to-speech if the sound effect is unavailable
        self.say(TTS_FALLBACK.get(key, key))

    def stop(self, key: str):
        """Stops a currently playing sound effect."""
        if key in self._sfx:
            self._sfx[key].stop()

    def play_music(self, key: str = "bgm", loop: bool = True, volume: float | None = None):
        """Starts streaming background music."""
        if key not in self._music:
            return

        try:
            pygame.mixer.music.load(self._music[key])
            vol = volume if volume is not None else VOLUME_MUSIC
            pygame.mixer.music.set_volume(vol)
            pygame.mixer.music.play(-1 if loop else 0)
        except Exception as exc:
            print(f"[AUD] Error playing music {key}: {exc}")

    def stop_music(self):
        """Stops the currently playing background music."""
        try:
            pygame.mixer.music.stop()
        except Exception:
            pass

    def fade_music(self, ms: int = 1500):
        """Fades out the background music over the specified duration in milliseconds."""
        try:
            pygame.mixer.music.fadeout(ms)
        except Exception:
            pass

    def say(self, text: str, block: bool = False):
        """Speaks the provided text using the espeak subprocess."""

        def _speak():
            with self._tts_lock:
                try:
                    subprocess.run(["espeak", "-v", "en-us", f"-s{TTS_SPEED}", "--", text], timeout=12, capture_output=True, check=False)
                except FileNotFoundError:
                    # espeak is not installed, output to console instead
                    print(f"[TTS] {text}")
                except subprocess.TimeoutExpired:
                    pass

        if block:
            _speak()
        else:
            threading.Thread(target=_speak, daemon=True, name="tts").start()

    def announce_green(self):
        """Announces the start of the green light phase."""
        self.stop_music()
        self.play_music("bgm")
        self.play("green")

    def announce_red(self):
        """Announces the start of the red light phase."""
        self.play("mugunghwa")
        if "mugunghwa" not in self._sfx:
            self.play("red")

    def announce_elimination(self, player_label: str):
        """Announces a player's elimination."""
        self.play("elim")
        msg = f"{player_label} — eliminated! Return to start!" if player_label else "You moved! Return to start!"
        threading.Timer(1.0, self.say, args=[msg]).start()

    def announce_winner(self, winner_name: str = ""):
        """Announces the winner of the game."""
        self.fade_music(500)
        self.play("winner")
        msg = f"Congratulations {winner_name}! We have a winner!" if winner_name else "We have a winner!"
        threading.Timer(1.5, self.say, args=[msg]).start()

    def countdown_beep(self):
        """Plays a beep sound for countdowns."""
        self.play("tick")
