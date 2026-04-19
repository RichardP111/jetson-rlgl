#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Project:      Red Light Green Light (Jetson Orin Nano)
File:         test_laser.py
Description:  Isolated hardware test for the GPIO Break-Beam Sensor.

Author:       Richard Pu
Last Updated: April 2026
===============================================================================
"""

# --- Standard Library Imports ---
import sys
import time

# --- Third-Party Imports ---
try:
    import Jetson.GPIO as GPIO
except ImportError:
    print("Error: Jetson.GPIO not found. Are you running this on the Jetson?")
    sys.exit(1)

# --- Constants ---
LASER_PIN: int = 12


def main() -> int:
    print("--- LASER BREAK-BEAM TEST ---")
    print(f"Listening on BCM Pin {LASER_PIN}...")
    print("Press Ctrl+C to exit.\n")

    try:
        GPIO.setmode(GPIO.BCM)
        # Pull-up resistor enabled. The sensor pulls the pin LOW when the beam breaks.
        GPIO.setup(LASER_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

        while True:
            if GPIO.input(LASER_PIN) == GPIO.LOW:
                print("🚨 BEAM BROKEN! 🚨")
            else:
                print("Beam Intact ✓", end="\r")

            time.sleep(0.1)  # 10Hz refresh rate

    except KeyboardInterrupt:
        print("\nTest terminated by user.")
    finally:
        GPIO.cleanup()
        print("GPIO pins cleaned up.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
