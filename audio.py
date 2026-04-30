#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ===============================================================================
# Project:      Red Light Green Light (Jetson Orin Nano)
# File:         audio.py
# Description:  Background music, SFX, and strictly-scoped ElevenLabs TTS.
# Author:       Richard Pu
# Last Updated: April 2026
# ===============================================================================

import os
import threading
import pygame
from dotenv import load_dotenv

from config import (
    AUDIO_BUFFER,
    AUDIO_FREQUENCY,
    SOUNDS,
    SOUNDS_DIR,
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


# ---------------------------------------------------------------------------
# TTS Backend (ElevenLabs with Local Caching)
# ---------------------------------------------------------------------------
class _ElevenLabsBackend:
    name = "elevenlabs"

    def __init__(self) -> None:
        load_dotenv()
        self.api_key = os.getenv("ELEVENLABS_API_KEY")

        if not self.api_key:
            print("[TTS] Warning: No ELEVENLABS_API_KEY found in .env file. TTS is disabled.")
            self.client = None
            return

        # Replace with your chosen Voice ID from ElevenLabs
        self.voice_id = "EXAVITQu4vr4xnSDxMaL"
        self.cache_dir = os.path.join(SOUNDS_DIR, "tts_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        try:
            from elevenlabs import ElevenLabs

            self.client = ElevenLabs(api_key=self.api_key)
            print("[TTS] ElevenLabs initialized.")
        except ImportError as exc:
            print("[TTS] Error: 'elevenlabs' library not installed.")
            self.client = None

    def _sanitize_filename(self, text: str) -> str:
        """Convert text into a safe filename (e.g., 'Player in red' -> 'player_in_red.mp3')"""
        safe_name = "".join(c for c in text.lower() if c.isalnum() or c == " ")
        return safe_name.strip().replace(" ", "_") + ".mp3"

    def speak(self, text: str) -> None:
        if not self.client:
            print(f"[TTS-SIMULATED] {text}")
            return

        filename = self._sanitize_filename(text)
        filepath = os.path.join(self.cache_dir, filename)

        # 1. Check if we already generated this exact phrase
        if not os.path.exists(filepath):
            print(f"[TTS] ElevenLabs generating new clip: '{text}'")
            try:
                # 2. Call the API
                audio_generator = self.client.text_to_speech.convert(
                    voice_id=self.voice_id, text=text, model_id="eleven_turbo_v2", output_format="mp3_44100_128"
                )

                # 3. Save to disk
                with open(filepath, "wb") as f:
                    for chunk in audio_generator:
                        if chunk:
                            f.write(chunk)

            except Exception as exc:
                print(f"[TTS] ElevenLabs API error: {exc}")
                # Cleanup corrupted file if quota exceeded/network failed
                if os.path.exists(filepath):
                    os.remove(filepath)
                return

        # 4. Play the file instantly using pygame mixer
        try:
            pygame.mixer.Sound(filepath).play()
        except Exception as exc:
            print(f"[TTS] Playback error: {exc}")


# ---------------------------------------------------------------------------
# AudioManager
# ---------------------------------------------------------------------------
class AudioManager:
    def __init__(self) -> None:
        self._silent = False
        self._sfx: dict[str, pygame.mixer.Sound] = {}
        self._music: dict[str, str] = {}
        self._tts_lock = threading.Lock()

        # Initialize our single TTS backend
        self._tts = _ElevenLabsBackend()

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
                # Anything with "bgm" in the key is treated as music
                if "bgm" in key or fname.endswith(".mp3"):
                    self._music[key] = path
                else:
                    snd = pygame.mixer.Sound(path)
                    snd.set_volume(VOL_SFX)
                    self._sfx[key] = snd
            except Exception as exc:
                print(f"[AUD] Load error ({fname}): {exc}")

        print(f"[AUD] Loaded {len(self._sfx)} sfx, {len(self._music)} music.")

    # ------------------------------------------------------------------
    # Primitives
    # ------------------------------------------------------------------
    def play(self, key: str) -> None:
        if self._silent:
            return
        if key in self._sfx:
            try:
                self._sfx[key].play()
            except Exception:
                pass

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
            pygame.mixer.music.play(loops=-1 if loop else 0)
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
        if self._silent:
            print(f"[TTS] {text}")
            return

        def _speak() -> None:
            with self._tts_lock:
                self._tts.speak(text)

        if block:
            _speak()
        else:
            threading.Thread(target=_speak, daemon=True, name="tts").start()

    # ------------------------------------------------------------------
    # Game-level events
    # ------------------------------------------------------------------
    def announce_green(self) -> None:
        if self._silent:
            return
        self.play("green")

    def announce_red(self) -> None:
        if self._silent:
            return
        self.play("red")

    def announce_caught(self, descriptor: str = "") -> None:
        if self._silent:
            print(f"[AUD] CAUGHT: {descriptor}")
            return

        # The ONLY ElevenLabs call in the game. No buzzer SFX played beforehand.
        if descriptor:
            threading.Thread(target=self.say, args=[f"{descriptor}, eliminated."]).start()
        else:
            print("[AUD] No descriptor provided, skipping TTS.")

    def announce_finished(self, rank: int, descriptor: str = "") -> None:
        if self._silent:
            return
        self.play("winner")

    def announce_all_finished(self) -> None:
        if self._silent:
            return

        self.fade_music(800)
        self.play("applause")
        threading.Timer(1.0, self.play_music, args=["leaderboard_bgm"]).start()

    def announce_wait_for_start(self) -> None:
        if self._silent:
            return
        self.play("wait_start")

    def announce_return_complete(self) -> None:
        pass  # Silent

    def announce_easing(self) -> None:
        pass  # Silent

    def announce_countdown(self, n: int) -> None:
        if self._silent:
            return

        num_key = f"num_{n}"
        if num_key in self._sfx:
            self.play(num_key)
        else:
            self.play("tick")

    def announce_game_start(self) -> None:
        pass  # Silent

    # ------------------------------------------------------------------
    # Legacy aliases
    # ------------------------------------------------------------------
    def on_green(self) -> None:
        self.announce_green()

    def on_red(self) -> None:
        self.announce_red()

    def on_caught(self, descriptor: str = "") -> None:
        self.announce_caught(descriptor)

    def on_winner(self) -> None:
        self.announce_all_finished()

    def on_countdown(self, n: int) -> None:
        self.announce_countdown(n)

    def on_game_start(self) -> None:
        self.announce_game_start()

    def test_chime(self) -> None:
        if self._silent:
            return
        if "chime" in self._sfx:
            self.play("chime")
        elif "tick" in self._sfx:
            self.play("tick")

    def cleanup(self) -> None:
        try:
            pygame.mixer.music.stop()
        except Exception:
            pass
        try:
            pygame.mixer.quit()
        except Exception:
            pass
