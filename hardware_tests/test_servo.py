#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Project:      Red Light Green Light (Jetson Orin Nano)
File:         test_servo.py
Description:  Isolated hardware test for the PCA9685 I2C Servo Controller.
              Sweeps the head motor from 0 to 180 degrees.

Author:       Richard Pu
Last Updated: April 2026
===============================================================================
"""

# --- Standard Library Imports ---
import sys
import time

# --- Third-Party Imports ---
try:
    import board
    import busio
    from adafruit_pca9685 import PCA9685
    from adafruit_motor import servo
except ImportError:
    print("Error: Adafruit libraries not found. Are you running this on the Jetson?")
    sys.exit(1)


def main() -> int:
    print("--- SERVO SWEEP TEST ---")
    print("Initializing I2C bus...")

    try:
        # Initialize I2C and PCA9685
        i2c = busio.I2C(board.SCL, board.SDA)
        pca = PCA9685(i2c, address=0x40)
        pca.frequency = 50

        # Initialize Servo on Channel 0
        head_servo = servo.Servo(pca.channels[0], min_pulse=500, max_pulse=2500)

        print("Moving to 0 degrees (Green Light / Away)...")
        head_servo.angle = 0
        time.sleep(2)

        print("Moving to 180 degrees (Red Light / Facing Players)...")
        head_servo.angle = 180
        time.sleep(2)

        print("Resetting to 0 degrees...")
        head_servo.angle = 0
        time.sleep(1)

        print("Test Complete. Shutting down PCA9685.")
        pca.deinit()
        return 0

    except Exception as e:
        print(f"Hardware Error: {e}")
        print("Did you run 'sudo i2cdetect -y 7' to confirm the address is 0x40?")
        return 1


if __name__ == "__main__":
    sys.exit(main())
