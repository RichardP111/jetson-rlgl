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
===============================================================================
"""

import sys
import pygame
from config import CAMERA_WIDTH, CAMERA_HEIGHT
from hardware import Camera, ServoController, LaserBreakBeam
from audio import AudioManager
from ui import UIRenderer
from vision import ProPoseTracker
from game import GameEngine

def main():
    print("=========================================")
    print("  INITIALIZING STEM NIGHT RLGL ENGINE")
    print("=========================================")
    
    pygame.init()
    screen = pygame.display.set_mode((CAMERA_WIDTH, CAMERA_HEIGHT))
    pygame.display.set_caption("Red Light Green Light - Operator Console")
    clock = pygame.time.Clock()

    camera = Camera()
    servo  = ServoController()
    laser  = LaserBreakBeam()
    audio  = AudioManager()
    ui     = UIRenderer(screen)
    
    try:
        vision = ProPoseTracker(model_size='n')
    except Exception as e:
        print(f"[ERROR] AI Engine failed to load: {e}")
        camera.release()
        pygame.quit()
        sys.exit(1)

    engine = GameEngine(camera, servo, laser, vision, audio, ui)
    
    try:
        engine.run(clock)
    except KeyboardInterrupt:
        print("\n[SYSTEM] Operator triggered shutdown.")
    finally:
        print("[SYSTEM] Releasing hardware...")
        camera.release()
        pygame.quit()

if __name__ == "__main__":
    sys.exit(main())
