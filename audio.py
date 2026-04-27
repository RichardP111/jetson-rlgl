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
import shutil
import subprocess
import threading

import pygame

from config import (
    AUDIO_BUFFER,
    AUDIO_FREQUENCY,
    IS_WINDOWS,
    SOUNDS,
    SOUNDS_DIR,
    TTS_ENGINES,
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


# ---------------------------------------------------------------------------
# TTS backends
# ---------------------------------------------------------------------------
class _TtsBackend:
    name: str = "noop"

    def speak(self, text: str) -> None:  # pragma: no cover
        print(f"[TTS] {text}")


class _EspeakNgBackend(_TtsBackend):
    name = "espeak-ng"

    def __init__(self) -> None:
        self._bin: str = shutil.which("espeak-ng") or ""
        if not self._bin:
            raise FileNotFoundError("espeak-ng not on PATH")

    def speak(self, text: str) -> None:
        try:
            subprocess.run(
                [self._bin, "-v", "en+f3", f"-s{TTS_WPM}", "-a", "180", "--", text],
                timeout=15,
                capture_output=True,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
            print(f"[TTS] espeak-ng error: {exc}")


class _EspeakBackend(_TtsBackend):
    name = "espeak"

    def __init__(self) -> None:
        self._bin: str = shutil.which("espeak") or ""
        if not self._bin:
            raise FileNotFoundError("espeak not on PATH")

    def speak(self, text: str) -> None:
        try:
            subprocess.run(
                [self._bin, "-v", "en+f3", f"-s{TTS_WPM}", "--", text],
                timeout=15,
                capture_output=True,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
            print(f"[TTS] espeak error: {exc}")


class _Pyttsx3Backend(_TtsBackend):
    """pyttsx3 wraps SAPI/NSSpeechSynth/espeak; useful on Windows / dev boxes."""

    name = "pyttsx3"

    def __init__(self) -> None:
        try:
            import pyttsx3  # type: ignore  # noqa: WPS433
        except ImportError as exc:
            raise FileNotFoundError("pyttsx3 not installed") from exc
        self._mod = pyttsx3
        # Quick init test - raises if the platform driver isn't usable.
        eng = self._mod.init()
        eng.setProperty("rate", TTS_WPM + 30)
        del eng

    def speak(self, text: str) -> None:
        try:
            eng = self._mod.init()
            eng.setProperty("rate", TTS_WPM + 30)
            eng.say(text)
            eng.runAndWait()
            try:
                eng.stop()
            except Exception:
                pass
        except Exception as exc:
            print(f"[TTS] pyttsx3 error: {exc}")


_BACKEND_FACTORIES: dict[str, type[_TtsBackend]] = {
    "espeak-ng": _EspeakNgBackend,
    "espeak": _EspeakBackend,
    "pyttsx3": _Pyttsx3Backend,
}


def _select_tts_backend() -> _TtsBackend:
    """Return the first usable TTS backend from TTS_ENGINES, else a noop."""
    for name in TTS_ENGINES:
        cls = _BACKEND_FACTORIES.get(name)
        if cls is None:
            continue
        try:
            backend = cls()
            print(f"[AUD] TTS backend: {backend.name}")
            return backend
        except Exception as exc:
            print(f"[AUD] TTS backend '{name}' unavailable: {exc}")
    print("[AUD] No TTS backend available — running in print-only mode.")
    print("      Install espeak-ng:   sudo apt install espeak-ng")
    return _TtsBackend()


# ---------------------------------------------------------------------------
# AudioManager
# ---------------------------------------------------------------------------
class AudioManager:
    def __init__(self) -> None:
        self._silent = False
        self._sfx: dict[str, pygame.mixer.Sound] = {}
        self._music: dict[str, str] = {}
        self._tts_lock = threading.Lock()
        self._tts: _TtsBackend = _select_tts_backend()

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
        print(f"[AUD] Loaded {len(self._sfx)} sfx, {len(self._music)} music, " f"tts={self._tts.name}")

    @property
    def has_tts(self) -> bool:
        return not isinstance(self._tts, _TtsBackend) or self._tts.__class__ is not _TtsBackend

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
        # No SFX file present → speak the equivalent line.
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
        """Speak ``text`` via the active TTS backend.

        Threading note: backends are inherently serialised by ``_tts_lock``
        — this prevents two threads from invoking ``espeak`` concurrently,
        which on Jetson manifests as audio crackle and/or dropped phrases.
        """
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
            print("[AUD] GREEN")
            return
        self.play_music("bgm")
        self.play("green")

    def announce_red(self) -> None:
        if self._silent:
            print("[AUD] RED")
            return
        if "mugunghwa" in self._sfx:
            self.play("mugunghwa")
        else:
            self.play("red")

    def announce_caught(self, descriptor: str = "") -> None:
        if self._silent:
            print(f"[AUD] CAUGHT: {descriptor}")
            return
        self.play("caught")
        if descriptor:
            threading.Timer(1.1, self.say, args=[f"{descriptor}, walk back to the start."]).start()
        else:
            threading.Timer(1.1, self.say, args=[TTS_LINES["caught"]]).start()

    def announce_finished(self, rank: int, descriptor: str = "") -> None:
        """Called the moment a single player crosses the line."""
        if self._silent:
            print(f"[AUD] FINISHED #{rank}: {descriptor}")
            return
        self.play("winner")
        # Match rank to a friendly suffix.
        suffix = {1: "first", 2: "second", 3: "third"}.get(rank, f"number {rank}")
        line = f"{descriptor or 'A player'} finished {suffix}!" if rank <= 3 else f"{descriptor or 'A player'} crossed the finish line."
        threading.Timer(0.6, self.say, args=[line]).start()

    def announce_all_finished(self) -> None:
        if self._silent:
            print("[AUD] ALL FINISHED")
            return
        self.fade_music(800)
        self.play("applause")
        threading.Timer(1.4, self.say, args=[TTS_LINES["all_finished"]]).start()

    def announce_wait_for_start(self) -> None:
        if self._silent:
            print("[AUD] WAIT FOR START")
            return
        self.say(TTS_LINES["wait_for_start"])

    def announce_return_complete(self) -> None:
        if self._silent:
            print("[AUD] RETURN COMPLETE")
            return
        self.say(TTS_LINES["return_complete"])

    def announce_easing(self) -> None:
        if self._silent:
            print("[AUD] EASING")
            return
        self.say(TTS_LINES["easing"])

    def announce_countdown(self, n: int) -> None:
        if self._silent:
            print(f"[AUD] {n}")
            return
        self.say(str(n), block=False)

    def announce_game_start(self) -> None:
        if self._silent:
            print("[AUD] START")
            return
        self.say(TTS_LINES["start"])

    # ------------------------------------------------------------------
    # Legacy aliases — keep old call-sites working until everything is
    # migrated. New code should call the announce_* methods above.
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
            self.say("Chime test")

    def test_tts(self) -> None:
        self.say("Audio test. One, two, three.")

    def cleanup(self) -> None:
        try:
            pygame.mixer.music.stop()
        except Exception:
            pass
        try:
            pygame.mixer.quit()
        except Exception:
            pass
