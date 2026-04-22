#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Project:      Red Light Green Light (Jetson Orin Nano)
File:         game.py
Description:  Main Finite State Machine (FSM) governing game phases, integrating
              YOLOv8 multi-person tracking for movement elimination.

Author:       Richard Pu
Last Updated: April 2026
Run via Docker: 
sudo docker ps -a 
sudo docker start squid-game-live
sudo docker exec -it squid-game-live bash
export DISPLAY=:0
export PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH
export LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:$LD_LIBRARY_PATH
python3 main.py
===============================================================================
"""

import os
import sys
import traceback

# Must be set before pygame touches display
os.environ.setdefault("DISPLAY", ":1")

import pygame

from audio import AudioManager
from config import (DISPLAY_H, DISPLAY_W, FONTS_DIR, FPS_CAP, FULLSCREEN,
                    SOUNDS_DIR)
from game import GameEngine
from hardware import Camera, LaserBreakBeam, ServoController
from ui import UIRenderer
from vision import PoseTracker

BANNER = """
  ╔══════════════════════════════════════════════╗
  ║   🔴  RED LIGHT  /  GREEN LIGHT  🟢         ║
  ║    STEM Day Edition  ·  Jetson Orin Nano     ║
  ╚══════════════════════════════════════════════╝
"""


def main():
    print(BANNER)

    # ── Asset dirs ────────────────────────────────────────────────
    for d in [SOUNDS_DIR, FONTS_DIR]:
        os.makedirs(d, exist_ok=True)

    # ── PyGame ────────────────────────────────────────────────────
    pygame.init()
    #pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)

    if FULLSCREEN:
        flags = pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF
        screen = pygame.display.set_mode((DISPLAY_W, DISPLAY_H), flags)
    else:
        screen = pygame.display.set_mode((DISPLAY_W, DISPLAY_H))

    pygame.display.set_caption("Red Light Green Light — STEM Day")
    pygame.mouse.set_visible(False)
    clock = pygame.time.Clock()

    # ── Hardware ─────────────────────────────────────────────────
    print("[INIT] Camera…")
    camera = Camera()

    print("[INIT] Servo controller…")
    servo = ServoController()

    print("[INIT] Laser break-beam…")
    laser = LaserBreakBeam()

    # ── Vision ───────────────────────────────────────────────────
    print("[INIT] YOLOv8 pose tracker…")
    tracker = PoseTracker()

    # ── Audio ────────────────────────────────────────────────────
    print("[INIT] Audio manager…")
    audio = AudioManager()

    # ── UI ───────────────────────────────────────────────────────
    print("[INIT] UI renderer…")
    ui = UIRenderer(screen)

    # ── Game ─────────────────────────────────────────────────────
    print("[INIT] Game engine…")
    engine = GameEngine(camera=camera, servo=servo, laser=laser, tracker=tracker, audio=audio, ui=ui)

    print()
    print("  ✓  All systems ready!")
    print()
    print("  Debug keys (active during game):")
    print("    G      → Force Green Light")
    print("    R      → Force Red Light")
    print("    W      → Force Winner screen")
    print("    E      → Fake elimination")
    print("    SPACE  → Skip palm gesture")
    print("    ESC    → Quit")
    print()

    # ── Run ──────────────────────────────────────────────────────
    try:
        engine.run(clock)
    except SystemExit:
        print("\n[EXIT] Quit by user")
    except KeyboardInterrupt:
        print("\n[EXIT] Keyboard interrupt")
    except Exception:
        print("\n[ERROR] Unhandled exception:")
        traceback.print_exc()
    finally:
        print("[CLEANUP] Releasing resources…")
        try:
            camera.release()
        except Exception:
            pass
        try:
            servo.cleanup()
        except Exception:
            pass
        try:
            laser.cleanup()
        except Exception:
            pass
        pygame.quit()
        print("[CLEANUP] Done. Goodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
