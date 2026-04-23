#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Project:      Red Light Green Light (Jetson Orin Nano)
File:         main.py
Description:  Entry point — initialises all subsystems and hands off to GameEngine.

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

import pygame
from config import DISPLAY_H, DISPLAY_W, FONTS_DIR, FULLSCREEN, SOUNDS_DIR
from hardware import Camera, ServoController, LaserBreakBeam
from vision import ProPoseTracker
from audio import AudioManager
from ui import UIRenderer
from game import GameEngine


def main():
    os.environ.setdefault("DISPLAY", ":1")
    print("\n  ╔══════════════════════════════════════════╗")
    print("  ║   🔴  RED LIGHT  /  GREEN LIGHT  🟢     ║")
    print("  ║     STEM Day · Material Design 3         ║")
    print("  ╚══════════════════════════════════════════╝\n")
    for d in [SOUNDS_DIR, FONTS_DIR]:
        os.makedirs(d, exist_ok=True)
    pygame.init()
    flags = pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF if FULLSCREEN else 0
    screen = pygame.display.set_mode((DISPLAY_W, DISPLAY_H), flags)
    pygame.display.set_caption("Red Light Green Light")
    pygame.mouse.set_visible(False)
    clock = pygame.time.Clock()
    print("[INIT] Camera…")
    camera = Camera()
    print("[INIT] Servo…")
    servo = ServoController()
    print("[INIT] Laser…")
    laser = LaserBreakBeam()
    print("[INIT] YOLOv8…")
    tracker = ProPoseTracker()
    print("[INIT] Audio…")
    audio = AudioManager()
    print("[INIT] UI…")
    ui = UIRenderer(screen)
    print("[INIT] Game engine…")
    engine = GameEngine(camera, servo, laser, tracker, audio, ui)
    print("\n  ✓  All systems ready!\n")
    print("  Debug keys:")
    print("    G=Green  R=Red  W=Win  SPACE=palm  ESC=quit\n")
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
