#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Project:      Red Light Green Light (Jetson Orin Nano)
File:         main.py
Description:  Entry point — initialises all subsystems and hands off to GameEngine.

Author:       Richard Pu
Last Updated: April 2026

To Run:
Run laucher.sh OR VS CODE: Ctrl+Shift+B
===============================================================================
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

# Headless support (must be set before pygame imports a display module)
if "--headless" in sys.argv or os.environ.get("RLGL_HEADLESS") == "1":
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
else:
    # Apr 2026 — force GPU compositing on Jetson Orin Nano. Pygame's
    # default SDL2 backend on Linux silently falls back to a software
    # blit pipeline, which pegs a single CPU core and leaves the GPU
    # idle. Telling SDL to use the OpenGL ES 2 renderer offloads the
    # final framebuffer composite to the Jetson's integrated GPU and
    # frees the CPU up for camera + pose work.
    #
    # Setting these as DEFAULTS via setdefault() lets the user override
    # them from the shell if a particular system needs something else
    # (e.g. SDL_RENDER_DRIVER=opengl on a desktop dev box).
    os.environ.setdefault("SDL_RENDER_DRIVER", "opengles2")
    os.environ.setdefault("SDL_RENDER_VSYNC", "1")
    # On Jetson, KMSDRM gives us a clean fullscreen path without going
    # through X11; keep it as a fallback if the user is running headless
    # without a desktop.
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        os.environ.setdefault("SDL_VIDEODRIVER", "kmsdrm")

import pygame

# Audio pre_init must happen BEFORE pygame.init() to set the mixer buffer
import audio as audio_mod

audio_mod.pre_init()

from audio import AudioManager
from config import (
    DISPLAY_H,
    DISPLAY_W,
    FPS_CAP,
    IDENTIFICATION_MODE,
    USE_LASER,
    USE_TAPE_FINISH,
    WINDOW_TITLE,
)
from game import GameEngine
from hardware import Camera, LaserBreakBeam, ServoController
from ui import UIRenderer
from vision import PlayerDescriber, PoseWorker, ProPoseTracker


# ---------------------------------------------------------------------------
# Pretty banner
# ---------------------------------------------------------------------------
_BANNER = r"""
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║      RED LIGHT, GREEN LIGHT  ·  Jetson Orin Nano  ·  Apr 2026            ║
║                                                                          ║
║      • Camera + (optional) laser break-beam finish detection             ║
║      • YOLOv8-pose tracking, palm-raise to start                         ║
║      • {id_mode_pad}player identification (VLM / CLIP / colour)        ║
║      • Auto-easing difficulty if a round drags on                        ║
║      • Kahoot-style leaderboard with podium photos                       ║
║                                                                          ║
║      HOTKEYS:  H or Ctrl+D  Dev panel                                    ║
║                G              Force GREEN  (begins round if needed)      ║
║                R              Force RED                                  ║
║                W              Debug-finish first remaining player        ║
║                L              Jump to leaderboard                        ║
║                F              Toggle DEBUG_SKIP_FINISH (no finish line)  ║
║                SPACE          Bypass palm gate / Play again              ║
║                ESC            Quit                                       ║
║      DEV-ONLY: 1 servo→face   2 servo→away   3 laser status              ║
║                4 chime         5 TTS test                                ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
"""


def _print_banner() -> None:
    print(_BANNER.format(id_mode_pad=f"{IDENTIFICATION_MODE:>5} "))
    print(f"  USE_LASER       = {USE_LASER}")
    print(f"  USE_TAPE_FINISH = {USE_TAPE_FINISH}")
    print(f"  ID mode         = {IDENTIFICATION_MODE}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="Run with dummy SDL drivers (no display, no audio device)")
    parser.add_argument("--fullscreen", action="store_true", default=True, help="Run fullscreen (default)")
    parser.add_argument("--windowed", dest="fullscreen", action="store_false", help="Run in a window (useful for dev)")
    args = parser.parse_args()

    _print_banner()

    pygame.init()
    pygame.display.set_caption(WINDOW_TITLE)
    # Apr 2026 — pygame.SCALED routes the framebuffer through SDL2's
    # GPU-accelerated renderer (matched with SDL_RENDER_DRIVER=opengles2
    # set above) and DOUBLEBUF gives us a proper backbuffer flip rather
    # than a software blit. Together with the SDL hints these flags are
    # what actually gets the Jetson GPU off idle.
    base_flags = pygame.SCALED | pygame.DOUBLEBUF
    if args.fullscreen and not args.headless:
        base_flags |= pygame.FULLSCREEN
    try:
        screen = pygame.display.set_mode((DISPLAY_W, DISPLAY_H), base_flags, vsync=1)
    except Exception as exc:
        # SDL_RENDER_DRIVER=opengles2 + SCALED can fail on systems without
        # a working GLES2 stack; fall back to plain DOUBLEBUF so we at
        # least keep DOUBLEBUF's flip-buffer benefit.
        print(f"[main] GPU display path unavailable ({exc}); falling back to plain DOUBLEBUF")
        fallback_flags = pygame.DOUBLEBUF
        if args.fullscreen and not args.headless:
            fallback_flags |= pygame.FULLSCREEN
        screen = pygame.display.set_mode((DISPLAY_W, DISPLAY_H), fallback_flags)
    pygame.mouse.set_visible(False)
    clock = pygame.time.Clock()

    # ---- Construct subsystems ----------------------------------------
    camera: Camera | None = None
    servo: ServoController | None = None
    laser: LaserBreakBeam | None = None
    tracker: ProPoseTracker | None = None
    pose_worker: PoseWorker | None = None
    audio: AudioManager | None = None
    ui: UIRenderer | None = None
    describer: PlayerDescriber | None = None

    rc = 0
    try:
        camera = Camera()
        servo = ServoController()
        laser = LaserBreakBeam()
        tracker = ProPoseTracker()
        pose_worker = PoseWorker(camera, tracker)
        audio = AudioManager()
        ui = UIRenderer(screen)
        ui.set_audio_hook(audio)  # let the leaderboard fire podium SFX directly
        describer = PlayerDescriber()

        engine = GameEngine(
            camera=camera,
            servo=servo,
            laser=laser,
            tracker=tracker,
            audio=audio,
            ui=ui,
            pose_worker=pose_worker,
            describer=describer,
        )
        engine.run(clock)
    except SystemExit:
        pass
    except KeyboardInterrupt:
        print("\n[main] Interrupted.")
    except Exception as exc:
        print(f"[main] Fatal: {exc}")
        traceback.print_exc()
        rc = 1
    finally:
        # Reverse-order teardown
        try:
            if pose_worker is not None:
                pose_worker.stop()
        except Exception:
            pass
        try:
            if camera is not None:
                camera.release()
        except Exception:
            pass
        try:
            if servo is not None:
                servo.cleanup()
        except Exception:
            pass
        try:
            if laser is not None:
                laser.cleanup()
        except Exception:
            pass
        try:
            if audio is not None:
                audio.cleanup()
        except Exception:
            pass
        try:
            pygame.quit()
        except Exception:
            pass

    return rc


if __name__ == "__main__":
    raise SystemExit(main())