<div align="center">

<br />

# Jetson RLGL

### AI-Powered Red Light Green Light — Built for the Real World

**An interactive, computer-vision game prop running on the NVIDIA Jetson Orin Nano.**
Inspired by *Squid Game* · Built for a Grade 7 STEM Day · Powered by YOLOv8

<br />

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![NVIDIA Jetson](https://img.shields.io/badge/NVIDIA-Jetson%20Orin%20Nano-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Pose-FF6B35?style=flat-square)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)
[![Ubuntu](https://img.shields.io/badge/Ubuntu-22.04-E95420?style=flat-square&logo=ubuntu&logoColor=white)](https://ubuntu.com/)

<br />

</div>

---

## Overview

Jetson RLGL is a fully autonomous game that runs a live *Red Light, Green Light* game using AI computer vision. A physical animatronic head rotates on a servo motor, controlled by an NVIDIA Jetson Orin Nano. The system uses YOLOv8 pose estimation to track every player simultaneously, detect motion during the red-light phase, and identify which player moved by their shirt colour.

The game is projected onto a screen in real time, complete with a live skeleton overlay, motion detection bar, player statistics, and dramatic screens for eliminations and winners — no human referee required.

<br />

## Features

- **Multi-person pose tracking** via YOLOv8-pose with persistent player IDs across frames
- **Palm-raise gesture** starts and resets the game — no keyboard needed during play
- **Shirt colour identification** announces exactly who was caught ("Player in the blue shirt!")
- **Live skeleton overlay** renders neon-green stick figures on the camera feed
- **Dual finish-line detection** — laser break-beam for precision + tape colour vision as backup
- **Animatronic head** sweeps smoothly back and forth via a PCA9685 servo driver
- **PA speaker integration** with `espeak` TTS and optional `.wav`/`.mp3` sound effects
- **Full game UI** projected live: start screen, game HUD, caught screen, winner gallery
- **Graceful hardware fallback** — runs in simulation mode on any laptop for development

<br />

## Architecture

```
main.py          Entry point — initialise hardware, pygame, run engine
│
├── config.py    Single source of truth for all constants and tuning values
│
├── hardware.py
│   ├── Camera           Background-threaded V4L2 camera reader
│   ├── ServoController  Smooth PCA9685 servo sweep (threaded)
│   └── LaserBreakBeam   GPIO finish-line sensor (Physical Pin 13)
│
├── vision.py
│   ├── PoseTracker      YOLOv8-pose multi-person tracking + skeleton overlay
│   ├── get_shirt_colour HSV-based shirt colour identification
│   ├── detect_palm_raise Wrist/shoulder keypoint gesture detection
│   └── check_tape_finish Coloured-tape finish-line detection
│
├── audio.py
│   └── AudioManager     pygame SFX · espeak TTS · background music
│
├── ui.py
│   └── UIRenderer       All PyGame screens — start, HUD, caught, winner
│
└── game.py
    └── GameEngine       Finite state machine governing all game logic
```

<br />

## Game Flow

```
┌─────────────────────────────────────────────────────────┐
│                    START SCREEN                          │
│         Raise palm for 2 seconds to begin               │
└────────────────────┬────────────────────────────────────┘
                     │
              3 … 2 … 1 …
                     │
┌────────────────────▼────────────────────────────────────┐
│               🟢  GREEN LIGHT                            │
│    Head faces away · Players move toward finish line     │
└────────────────────┬────────────────────────────────────┘
                     │  Timer expires
┌────────────────────▼────────────────────────────────────┐
│                   TURNING                                │
│        Head rotates · Grace period active                │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│               🔴  RED LIGHT                              │
│    Head faces players · Motion detection active          │
│                                                          │
│   Motion detected ──► CAUGHT SCREEN (player walks back)  │
│   Timer expires   ──► Back to GREEN LIGHT                │
│   Finish crossed  ──► WINNER SCREEN                      │
└──────────────────────────────────────────────────────────┘
```

<br />

## Hardware

### Components

| Component | Model | Interface |
|---|---|---|
| Edge Computer | NVIDIA Jetson Orin Nano | — |
| Camera | Waveshare IMX219-120 (CSI) | V4L2 `/dev/video0` |
| Servo Driver | PCA9685 16-Channel PWM | I2C Bus 7 · `0x40` |
| Servo Motor | Standard 180° servo | PCA9685 Channel 0 |
| Finish Line | Laser Break-Beam Sensor | GPIO Physical Pin 13 |
| Speaker | PA Speaker | 3.5mm audio out |
| Display | Monitor / Projector | HDMI |

### Wiring

**PCA9685 → Jetson Orin Nano (40-pin header)**

| PCA9685 | Jetson Pin | Function |
|---|---|---|
| VCC | Pin 1 | 3.3V |
| GND | Pin 6 | Ground |
| SDA | Pin 3 | I2C-7 SDA |
| SCL | Pin 5 | I2C-7 SCL |

> ⚠️ **Power the servo from an external 5–6V supply**, not from the Jetson header. Servos draw too much current.

**Laser Break-Beam → GPIO**

| Sensor | Jetson Pin | Note |
|---|---|---|
| Signal OUT | Physical Pin 13 | Pull-up, active LOW |
| VCC | 3.3V | — |
| GND | GND | — |

**Verify your connections before first boot:**
```bash
# Check camera
v4l2-ctl --list-devices

# Check PCA9685
sudo i2cdetect -y 7
# Should show 40 in the grid
```

<br />

## Software Setup

### Prerequisites

- Ubuntu 22.04 with JetPack 5.x / 6.x installed
- CUDA, cuDNN, and OpenCV already present (standard JetPack)
- NoMachine for remote desktop access (headless setup)

### Install

```bash
# Clone the repo
git clone https://github.com/RichardP111/jetson-rlgl.git
cd jetson-rlgl

# Run the setup script (installs all dependencies)
bash setup.sh
```

The setup script installs: `ultralytics` · `pygame` · `adafruit-circuitpython-pca9685` · `adafruit-circuitpython-motor` · `espeak` · `v4l-utils` · `i2c-tools`

### Assets

Drop your media files into the correct directories:

**`assets/fonts/`**
```
GoogleSans-Bold.ttf
GoogleSans-Regular.ttf
```

**`assets/sounds/`**

| File | Purpose |
|---|---|
| `bgm.mp3` | Background music loop |
| `mugunghwa.wav` | Korean "freeze!" phrase |
| `green_light.wav` | Green light announcement |
| `red_light.wav` | Red light buzzer |
| `eliminated.wav` | Elimination sting |
| `winner.wav` | Victory fanfare |
| `tick.wav` | Countdown beep |

> All sound files are optional — the system falls back to `espeak` TTS if a file is missing.

**Getting `mugunghwa.wav`:**
```bash
# Download from YouTube and convert
yt-dlp -x --audio-format wav "https://youtube.com/..." -o assets/sounds/mugunghwa.wav
```

### Run

```bash
export DISPLAY=:1 && python3 main.py
```

<br />

## Configuration

All tunable values live in `config.py`. The most important ones:

**Gameplay timing**

```python
GREEN_MIN  = 3.5   # shortest green-light phase (seconds)
GREEN_MAX  = 8.0   # longest  green-light phase (seconds)
RED_MIN    = 3.0   # shortest red-light phase (seconds)
RED_MAX    = 6.5   # longest  red-light phase (seconds)
GRACE_S    = 0.35  # delay after RED before motion triggers elimination
MOTION_PX  = 14    # pixel movement delta to count as "moved"
```

**Finish-line tape colour** (adjust for whatever tape you use)

```python
# Default: yellow tape
TAPE_HSV_LOW  = (18, 100, 100)
TAPE_HSV_HIGH = (35, 255, 255)

# Orange tape
TAPE_HSV_LOW  = (5,  120, 120)
TAPE_HSV_HIGH = (22, 255, 255)

# Pink tape
TAPE_HSV_LOW  = (140, 80, 100)
TAPE_HSV_HIGH = (170, 255, 255)
```

**Servo positions**

```python
SERVO_AWAY_DEG = 0    # GREEN LIGHT — head faces away from players
SERVO_FACE_DEG = 180  # RED LIGHT   — head faces players
```

<br />

## Debug Controls

These keyboard shortcuts work at any point during the game for testing without playing through:

| Key | Action |
|---|---|
| `G` | Force Green Light |
| `R` | Force Red Light |
| `W` | Force Winner screen |
| `E` | Simulate an elimination |
| `SPACE` | Skip palm-raise gesture |
| `ESC` | Quit |

<br />

## Troubleshooting

**Camera not opening**
```bash
ls /dev/video*
# If empty, run Jetson-IO to enable the IMX219:
sudo /opt/nvidia/jetson-io/jetson-io.py
```

**PCA9685 not detected**
```bash
sudo i2cdetect -y 7
# 0x40 should appear. If not, check SDA/SCL wiring and 3.3V supply.
```

**GPIO error: "A different mode has already been set"**
This is handled automatically — the code checks `GPIO.getmode()` before setting BOARD mode. If it still appears, another process is holding GPIO. Reboot and try again.

**NoMachine display issues**
```bash
# Ensure DISPLAY points to the NoMachine virtual session
export DISPLAY=:1
# Not :0 — that's the physical display which may not exist on a headless setup
```

**Motion too sensitive / not sensitive enough**
Adjust `MOTION_PX` in `config.py`. Lower = more sensitive. Start at `14` and tune up if the gym floor vibrations are triggering false positives.

**YOLOv8 running slowly**
The nano model (`yolov8n-pose.pt`) is already the fastest. If FPS is still low, reduce camera resolution in `config.py`:
```python
CAM_W = 640
CAM_H = 360
```

<br />

## Project Structure

```
jetson-rlgl/
├── main.py              Entry point
├── config.py            All constants and tuning values
├── hardware.py          Camera, servo, laser beam
├── vision.py            YOLOv8 tracking, gesture, finish line
├── audio.py             Sound effects and TTS
├── ui.py                PyGame UI renderer (all screens)
├── game.py              Game state machine
├── setup.sh             Dependency installer
├── yolov8n-pose.pt      YOLOv8 pose model weights
├── assets/
│   ├── fonts/           GoogleSans font files
│   └── sounds/          .wav and .mp3 audio files
└── hardware_tests/      Individual hardware test scripts
```

<br />

## Built With

[NVIDIA Jetson](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/) · [YOLOv8 by Ultralytics](https://github.com/ultralytics/ultralytics) · [OpenCV](https://opencv.org/) · [PyGame](https://www.pygame.org/) · [Adafruit CircuitPython](https://github.com/adafruit/Adafruit_CircuitPython_PCA9685)

<br />

---

<div align="center">

Built by **Richard Pu** · STEM Day 2026

*Squid Game inspiration, Jetson execution.*

</div>