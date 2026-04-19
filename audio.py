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
import pygame
import platform
import subprocess
from config import SOUNDS_DIR, VOLUME_MUSIC, VOLUME_SFX

class AudioManager:
    def __init__(self):
        pygame.mixer.init()
        self.os_type = platform.system()
        
        self.sounds = {}
        sfx_path = os.path.join(SOUNDS_DIR, "elimination.wav")
        if os.path.exists(sfx_path):
            self.sounds['caught'] = pygame.mixer.Sound(sfx_path)
            self.sounds['caught'].set_volume(VOLUME_SFX)

    def play_music(self, track_name: str):
        path = os.path.join(SOUNDS_DIR, f"{track_name}.mp3")
        if os.path.exists(path):
            pygame.mixer.music.load(path)
            pygame.mixer.music.set_volume(VOLUME_MUSIC)
            pygame.mixer.music.play(-1)
        else:
            print(f"[AUDIO] Missing BGM: {path}")

    def say(self, text: str):
        print(f"[TTS] {text}")
        if self.os_type == "Windows":
            cmd = f'PowerShell -Command "Add-Type –AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{text}\');"'
            subprocess.Popen(cmd, shell=True)
        else:
            subprocess.Popen(['espeak', text])

    def announce_green(self):
        self.say("Green Light")

    def announce_red(self):
        self.say("Red Light")

    def announce_caught(self, player_label: str):
        if 'caught' in self.sounds:
            self.sounds['caught'].play()
        self.say(f"Motion detected. {player_label}, return to the starting line.")

    def announce_winner(self):
        self.say("We have a winner!")