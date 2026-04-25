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

import os
import sys
import traceback

# ── Docker Display Routing ───────────────────────────────────────────────────
# Force X11 to use the primary physical monitor by default if not set
if "DISPLAY" not in os.environ:
    os.environ["DISPLAY"] = ":0"
# Hide Pygame community prompt for a cleaner console boot
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

# ── Local Imports ────────────────────────────────────────────────────────────
import pygame  # noqa: E402
from config import DISPLAY_W, DISPLAY_H, FULLSCREEN, SOUNDS_DIR, FONTS_DIR  # noqa: E402
from hardware import Camera, LaserBreakBeam, ServoController  # noqa: E402
from vision import PoseWorker, ProPoseTracker  # noqa: E402
from audio import AudioManager  # noqa: E402
from ui import UIRenderer  # noqa: E402
from game import GameEngine  # noqa: E402


# ── Terminal UI Formatting ───────────────────────────────────────────────────
class C:
    MAGENTA = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


BANNER = f"""
{C.MAGENTA}{C.BOLD}╔════════════════════════════════════════════════════════════╗
║   {C.RED}🔴 RED LIGHT{C.RESET}{C.MAGENTA}{C.BOLD}  /  {C.GREEN}GREEN LIGHT 🟢{C.RESET}{C.MAGENTA}{C.BOLD}                         ║
║   {C.CYAN}Material 3 Edition  ·  Jetson Orin Nano Accelerated{C.MAGENTA}{C.BOLD}      ║
╚════════════════════════════════════════════════════════════╝{C.RESET}
"""


def main():
    print(BANNER)
    print(f"{C.CYAN}[SYSTEM]{C.RESET} Boot sequence initiated...")

    # 1. Asset Directory Checks
    for d in [SOUNDS_DIR, FONTS_DIR]:
        os.makedirs(d, exist_ok=True)

    # 2. Audio Engine Hard-Fix (USB-C DAC Compatibility)
    print(f"{C.CYAN}[INIT]{C.RESET} Pre-configuring audio hardware (48kHz USB-C)...")
    try:
        pygame.mixer.pre_init(frequency=48000, size=-16, channels=2, buffer=1024)
    except Exception as e:
        print(f"{C.RED}[WARN]{C.RESET} Audio pre-init failed: {e}")

    # 3. Pygame Display Engine
    pygame.init()

    if FULLSCREEN:
        flags = pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF
        screen = pygame.display.set_mode((DISPLAY_W, DISPLAY_H), flags)
        print(f"{C.CYAN}[INIT]{C.RESET} Display: {DISPLAY_W}x{DISPLAY_H} (Fullscreen Native)")
    else:
        screen = pygame.display.set_mode((DISPLAY_W, DISPLAY_H))
        print(f"{C.CYAN}[INIT]{C.RESET} Display: {DISPLAY_W}x{DISPLAY_H} (Windowed)")

    pygame.display.set_caption("Red Light Green Light — M3 Edition")
    pygame.mouse.set_visible(False)  # Hide cursor for immersive experience
    clock = pygame.time.Clock()

    # 4. Hardware & AI Orchestration
    print(f"\n{C.YELLOW}--- Activating Hardware Subsystems ---{C.RESET}")
    camera = Camera()
    servo = ServoController()
    laser = LaserBreakBeam()

    print(f"\n{C.YELLOW}--- Loading AI Vision Engine ---{C.RESET}")
    tracker = ProPoseTracker()

    # PoseWorker spawns its own daemon thread that runs YOLO continuously
    # against whatever the camera has produced most recently. The main
    # Pygame loop then never blocks on inference, which is the change
    # that takes the engine to a steady 60 FPS.
    print(f"{C.CYAN}[INIT]{C.RESET} Starting pose worker thread...")
    pose_worker = PoseWorker(camera, tracker)

    print(f"\n{C.YELLOW}--- Booting Multimedia ---{C.RESET}")
    audio = AudioManager()
    ui = UIRenderer(screen)

    # 5. Core Engine Linkage
    print(f"\n{C.GREEN}✓ All subsystems active. Linking Game Engine...{C.RESET}")
    engine = GameEngine(
        camera=camera,
        servo=servo,
        laser=laser,
        tracker=tracker,
        audio=audio,
        ui=ui,
        pose_worker=pose_worker,
    )

    # Terminal Dashboard
    print(
        f"""
{C.BOLD}══════════════ [ DEBUG DASHBOARD / HOTKEYS ] ══════════════{C.RESET}
  {C.BOLD}H{C.RESET}      → Toggle On-Screen Debug Stats (FPS, AI Time)
  {C.BOLD}G{C.RESET}      → Force State: {C.GREEN}GREEN LIGHT{C.RESET}
  {C.BOLD}R{C.RESET}      → Force State: {C.RED}RED LIGHT{C.RESET}
  {C.BOLD}W{C.RESET}      → Force State: WINNER (Blue Shirt)
  {C.BOLD}E{C.RESET}      → Force State: FAKE ELIMINATION
  {C.BOLD}SPACE{C.RESET}  → Bypass Palm-Raise Startup Check
  {C.BOLD}ESC{C.RESET}    → Graceful Shutdown
═══════════════════════════════════════════════════════════
"""
    )

    # 6. Main Run Loop
    try:
        engine.run(clock)
    except SystemExit:
        print(f"\n{C.CYAN}[EXIT]{C.RESET} Esc key pressed. Shutting down cleanly.")
    except KeyboardInterrupt:
        print(f"\n{C.CYAN}[EXIT]{C.RESET} Ctrl+C detected. Shutting down cleanly.")
    except Exception:
        print(f"\n{C.RED}[FATAL ERROR]{C.RESET} Engine crashed unexpectedly:")
        traceback.print_exc()
    finally:
        # 7. Guaranteed Resource Cleanup — order matters: stop the pose
        # worker BEFORE releasing the camera so it doesn't try to read
        # from a closed capture.
        print(f"\n{C.YELLOW}[CLEANUP] Releasing hardware locks...{C.RESET}")
        try:
            pose_worker.stop()
            print("  ✓ Pose worker stopped")
        except Exception:
            pass

        try:
            if hasattr(camera, "release"):
                camera.release()
            print("  ✓ Camera thread closed")
        except Exception:
            pass

        try:
            if hasattr(servo, "cleanup"):
                servo.cleanup()
            print("  ✓ Servos disengaged")
        except Exception:
            pass

        try:
            if hasattr(laser, "cleanup"):
                laser.cleanup()
            print("  ✓ Laser GPIO unmapped")
        except Exception:
            pass

        pygame.quit()
        print(f"{C.GREEN}[CLEANUP] Complete. Goodbye!{C.RESET}\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
