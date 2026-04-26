<div align="center">

# Jetson RLGL

### A real-time, AI *Red Light, Green Light* Game

A self-officiating *Squid Game*-style game built on the NVIDIA Jetson Orin Nano. The system tracks every player simultaneously with YOLOv8-pose accelerated through TensorRT, detects motion during the red phase to the pixel, identifies eliminated players by shirt colour, and projects the entire game — animated overlays, dramatic eliminations, winner sequences — in real time at 60 FPS.

<br />

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jetson Orin Nano](https://img.shields.io/badge/Jetson_Orin_Nano-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)
[![TensorRT](https://img.shields.io/badge/TensorRT-FP16-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/tensorrt)
[![YOLOv8](https://img.shields.io/badge/YOLOv8--pose-FF6B35?style=for-the-badge)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

<br />

**~30 ms inference · 60 FPS display · multi-player tracking · zero human referee required**

</div>

---

## Table of Contents

- [Performance at a glance](#performance-at-a-glance)
- [How it plays](#how-it-plays)
- [Architecture](#architecture)
- [Hardware](#hardware)
- [Installation](#installation)
- [Running the game](#running-the-game)
- [Calibration tools](#calibration-tools)
- [Configuration](#configuration)
- [Optimisation notes](#optimisation-notes)
- [Troubleshooting](#troubleshooting)
- [Project structure](#project-structure)
- [Credits](#credits)

---

## Performance at a glance

| Metric | Value | How |
|---|---|---|
| Pose inference | **~15–30 ms** | YOLOv8n-pose exported to TensorRT FP16, pre-resized to 640×384 |
| Display loop | **60 FPS** | Decoupled from inference via dedicated `PoseWorker` thread |
| Camera capture | **60 FPS** | CSI IMX219 → NVMM zero-copy → hardware scaler |
| Colour analysis | **~0 ms typical** | Lazy: only runs at catch / win events, not every frame |
| End-to-end latency | **~50 ms** | From a player moving to elimination on screen |

Tested on Jetson Orin Nano (8 GB) with `MAXN` power mode, `jetson_clocks` locked, JetPack 6.0.

---

## How it plays

```
┌────────────────────────────────────────────────────────────┐
│                    START SCREEN                             │
│            Raise a palm for 2 seconds to begin              │
└──────────────────────────────┬─────────────────────────────┘
                               │
                       3 … 2 … 1 …
                               │
┌──────────────────────────────▼─────────────────────────────┐
│                  🟢  GREEN LIGHT                            │
│        Head faces away · Players advance toward finish      │
└──────────────────────────────┬─────────────────────────────┘
                               │  random 3–6 s
┌──────────────────────────────▼─────────────────────────────┐
│                       TURNING                               │
│             Servo rotates · 0.5 s grace window              │
└──────────────────────────────┬─────────────────────────────┘
                               │
┌──────────────────────────────▼─────────────────────────────┐
│                  🔴  RED LIGHT                              │
│      Head faces players · Per-player motion check active    │
│                                                             │
│   Movement detected ──► CAUGHT — return to start            │
│   Timer expires     ──► Back to GREEN                       │
│   Beam crossed      ──► WINNER sequence                     │
└─────────────────────────────────────────────────────────────┘
```

Every player has an independent baseline position the moment red starts. They're checked individually each frame — motion in one person doesn't affect the others. Caught players are identified by shirt colour and announced over a PA speaker by `espeak` TTS: *"Player in the blue shirt — return to the start!"*

---

## Architecture

The system is built around a strict producer-consumer split. Inference runs on its own thread, the camera runs on its own thread, and the main game loop never blocks on either.

```
┌─────────────────────────┐    ┌─────────────────────────┐
│      Camera (thread)    │    │   PoseWorker (thread)   │
│                         │    │                         │
│   nvarguscamerasrc      │───▶│   YOLOv8-pose (TRT FP16)│
│   1280×720 @ 60 FPS     │    │   ~15–30 ms / frame     │
│   ISP-tuned colour      │    │   ByteTrack persistent  │
└──────────┬──────────────┘    └────────────┬────────────┘
           │                                │
           │  latest frame                  │  latest pose
           │  (lock-protected)              │  (lock-protected)
           ▼                                ▼
┌─────────────────────────────────────────────────────────┐
│                    GameEngine (main thread)              │
│                                                          │
│   ┌────────────┐  ┌───────────┐  ┌────────────────────┐ │
│   │ State FSM  │  │   Audio   │  │  ServoController   │ │
│   │ GREEN/RED  │  │  Manager  │  │  (PCA9685, thread) │ │
│   └────────────┘  └───────────┘  └────────────────────┘ │
│                                                          │
│   ┌──────────────────────────────────────────────────┐  │
│   │  UIRenderer — Pygame, 60 FPS Material 3 UI       │  │
│   └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

Critically, the main loop reads the *most recent* camera frame and the *most recent* pose result — never waits for either. If inference takes 30 ms and the display does 16 ms frames, the game stays at 60 FPS and simply re-uses the most recent pose data for ~2 frames at a time. The eye doesn't notice; the inference budget no longer caps the display.

---

## Hardware

### Bill of materials

| Component | Model | Interface | Notes |
|---|---|---|---|
| Compute | NVIDIA Jetson Orin Nano (8 GB) | — | JetPack 6.0+ |
| Camera | Waveshare IMX219-120 (CSI) | CSI / nvarguscamerasrc | 60 FPS, 120° FOV |
| Servo driver | PCA9685 16-channel PWM | I²C bus 7 @ 0x40 | External 5–6 V supply |
| Servo | Standard 180° hobby servo | PCA9685 ch. 0 | DS3218 or similar |
| Finish line | Laser break-beam module | GPIO physical pin 7 | Active LOW, PUD_UP |
| Audio | PA speaker | 3.5 mm or USB DAC | `espeak` TTS supported |
| Display | Monitor / projector | DP | Native 1920×1080 |

### Wiring

**PCA9685 → Jetson 40-pin header**

| PCA9685 | Jetson | Function |
|:---:|:---:|:---|
| VCC | Pin 1 (3.3 V) | Logic |
| GND | Pin 6 | Ground |
| SDA | Pin 3 | I²C-7 SDA |
| SCL | Pin 5 | I²C-7 SCL |
| **V+** | **External 5–6 V** | **Servo power — DO NOT use Jetson 5V** |

> ⚠️ **Power servos from a separate 5–6 V supply.** Servos can briefly draw >1 A under load; the Jetson rail will sag and reset the board.

**Laser break-beam → Jetson**

| Sensor | Jetson | Note |
|:---:|:---:|:---|
| Signal | Pin 7 (BOARD) | Active LOW, internal pull-up |
| VCC | Pin 1 / 17 (3.3 V) | Most modules also accept 5 V |
| GND | Any GND | Common ground with Jetson |

**Pre-flight verification**

```bash
v4l2-ctl --list-devices         # Camera should appear at /dev/video0
sudo i2cdetect -y 7             # PCA9685 should appear as 40 in the grid
gpio readall                    # Optional — sanity-check pin states
```

---

## Installation

### Prerequisites

- Jetson Orin Nano with JetPack 6.0 or later
- CUDA, cuDNN, TensorRT (all bundled with JetPack)
- A working DISPLAY — local HDMI, or NoMachine for headless
- ~3 GB free disk space

### Setup

```bash
git clone https://github.com/RichardP111/jetson-rlgl.git
cd jetson-rlgl
bash setup.sh
```

`setup.sh` installs `ultralytics`, `pygame`, `adafruit-circuitpython-pca9685`, `Jetson.GPIO`, `espeak`, `v4l-utils`, and `i2c-tools`. It also verifies the camera node and I²C bus.

### One-time TensorRT export

The repo ships with `yolov8n-pose.pt`. On first run the engine is auto-built — but you can trigger it manually for predictable startup time:

```bash
yolo export model=yolov8n-pose.pt format=engine half=True imgsz=480 device=0
```

This takes **~5 minutes** the first time. The resulting `yolov8n-pose.engine` is what `vision.py` actually loads.

### Lock the Jetson at full clocks (recommended)

```bash
sudo nvpmodel -m 0          # MAXN power mode
sudo jetson_clocks          # Pin clocks to maximum
```

Without this, dynamic frequency scaling causes inference times to fluctuate by ~30%.

### Assets

Drop your media into the assets folders:

**`assets/sounds/`** — all optional; `espeak` TTS is the fallback.

| File | Used for |
|---|---|
| `bgm.mp3` | Background music loop |
| `green_light.wav` | Green-light cue |
| `red_light.wav` | Red-light cue |
| `mugunghwa.wav` | Korean "freeze" phrase |
| `eliminated.wav` | Caught sting |
| `winner.wav` | Victory fanfare |
| `tick.wav` / `chime.wav` | Countdown / generic |

**`assets/fonts/`** — `GoogleSans-Bold.ttf` recommended (Material 3 default).

---

## Running the game

```bash
bash launcher.sh
```

`launcher.sh` is a menu-driven controller that handles the hairy stuff: display permissions, `nvargus-daemon` restart, audio device routing, and Docker container coordination. Pick option **3** to start the game.

To bypass the launcher (developer mode):

```bash
export DISPLAY=:0
python3 main.py
```

### Debug hotkeys

These are live during gameplay and useful for testing without playing through the full FSM.

| Key | Action |
|:---:|---|
| `H` | Toggle on-screen debug panel (FPS, inference ms, player count) |
| `G` | Force GREEN LIGHT |
| `R` | Force RED LIGHT |
| `W` | Force WINNER screen |
| `E` | Simulate elimination |
| `SPACE` | Skip palm-raise gesture |
| `ESC` | Graceful shutdown |

---

## Calibration tools

Three standalone GUIs are bundled to handle hardware bring-up, tuning, and debugging without modifying the game code. Each writes a snippet you paste back into `config.py` (or `hardware.py` for the camera).

### `test_camera.py` — ISP & colour

```bash
python3 test_camera.py
```

Live preview with sliders for every `nvarguscamerasrc` knob — white balance, saturation, exposure, gain caps, ISP digital gain, temporal noise reduction, edge enhancement. Tabbed by category. Shows the GStreamer pipeline string updating in real time, and emits a drop-in `_gstreamer_pipeline()` snippet for `hardware.py`. Includes presets for fluorescent gym lighting, daylight, and warm tungsten.

**Most useful for:** colours looking dull or grey, sensor noise in low light, AE/AWB hunting causing blur cycles.

### `test_servo.py` — PCA9685

```bash
python3 test_servo.py
```

Animated semicircular gauge, manual angle and pulse-width sliders, four automated tests (sweep / step response / hold / random walk), I²C bus scanner, PCA9685 ping, and a guided MIN/MAX calibration wizard. Hard-stop protection — won't drive past mechanical limits once calibrated. Generates the full servo block for `config.py`.

**Most useful for:** first-time install, finding your specific servo's real `MIN_US`/`MAX_US`, debugging jitter or buzzing.

### `tets_lazer.py` — Break-beam

```bash
python3 tets_lazer.py
```

Live beam-state visualisation, scrolling 10-second history strip (great for spotting flaky sensors), debounce calibration with samples × window, and three automated tests:

- **Alignment helper** — confirms a stable beam for 10 seconds straight
- **Drift watch** — 60-second false-positive logger
- **Latency probe** — measures debounce delay across 10 real crossings

GPIO diagnostics tab runs all the standard checks (gpio group membership, gpiochip device list, current pin state).

**Most useful for:** the day-of-event debugging when a previously-working sensor has decided to act up, alignment after physical setup, dialing in debounce so a single crossing isn't counted as five wins.

All three tools fall back to **simulation mode** when hardware isn't available, so they're useful for UI testing on a laptop too.

---

## Configuration

Everything tunable lives in `config.py`. The values most worth knowing:

### Gameplay timing

```python
GREEN_LIGHT_MIN  = 3.0   # shortest green-light phase
GREEN_LIGHT_MAX  = 6.0   # longest green-light phase
RED_LIGHT_MIN    = 2.0   # shortest red-light phase
RED_LIGHT_MAX    = 5.0   # longest red-light phase
GRACE_PERIOD     = 0.5   # delay after RED before motion can catch
MOTION_THRESHOLD = 15.0  # pixels of movement to trigger elimination
CAUGHT_HOLD      = 3.5   # screen pause after a catch
```

`MOTION_THRESHOLD` is in pixel space at 1280×720 capture resolution. If you change camera resolution, scale this proportionally — at 960×540 you'd want `~11`.

### Difficulty presets

For age-appropriate scaling, three pre-tuned profiles ship in `config.py`:

| Preset | Green min/max | Red min/max | Motion px |
|---|:---:|:---:|:---:|
| Easy | 5.0 – 9.0 s | 4.0 – 7.0 s | 20 |
| **Normal** | 3.5 – 7.0 s | 3.0 – 5.5 s | 15 |
| Hard | 2.5 – 5.0 s | 2.0 – 4.0 s | 10 |

### Hardware pinout

```python
I2C_BUS         = 7        # Bus 7 on Orin Nano 40-pin header
PCA9685_ADDRESS = 0x40
SERVO_CHANNEL   = 0
SERVO_FREQ      = 50       # Hz; some cheap servos prefer 60
SERVO_MIN_PULSE = 500      # µs — calibrate per-servo with servo_tuner.py
SERVO_MAX_PULSE = 2500     # µs
LASER_PIN       = 7        # BOARD numbering
```

---

## Optimisation notes

The system has been optimised hard. The key wins, in order of impact:

1. **TensorRT FP16 export.** ~100 ms PyTorch inference → ~15–30 ms TRT inference. Single biggest win.
2. **Pre-resize the YOLO input.** Feeding 1920×1080 frames into a 480-input model wastes ~5 ms per frame on a CPU-side letterbox. Pre-sizing to 640×384 in OpenCV cuts that to ~1 ms.
3. **Async `PoseWorker` thread.** Decouples display loop from inference. Display fps is now bounded by `FPS_CAP`, not by YOLO.
4. **Lazy shirt-colour detection.** Eleven `cv2.inRange` operations per detected player, every frame, was costing ~3–6 ms with multiple players. Now runs only at catch / winner events — saves several ms continuously.
5. **ISP-tuned camera pipeline.** Saturation boost, white balance lock, digital gain capped at 1× to kill amplification noise, edge enhancement disabled. Better-looking image *and* faster — no more per-frame CPU colour correction.
6. **Power mode.** `nvpmodel -m 0` + `jetson_clocks` is free 30% perf.

If inference is still slow after all of this, drop to `imgsz=384` in the export command. INT8 export shaves another few milliseconds but requires calibration data and slightly hurts keypoint accuracy on small bodies.

---

## Troubleshooting

| Symptom | First thing to check |
|---|---|
| **Black camera feed** | `v4l2-ctl --list-devices` — if empty, run `sudo /opt/nvidia/jetson-io/jetson-io.py` and enable IMX219 |
| **PCA9685 not detected** | `sudo i2cdetect -y 7` — should show `40`. If not, check SDA/SCL wiring |
| **GPIO permission denied** | `sudo usermod -aG gpio $USER && sudo reboot` |
| **Servo jitters / buzzes** | Power supply too weak. Use a separate 5–6 V supply, ≥1 A. Common ground required. |
| **Beam triggers spuriously** | Run `laser_tuner.py` → Drift Watch test. Increase debounce samples to 5+ |
| **Inference > 50 ms** | Confirm `.engine` is loading, not `.pt`. Check `sudo nvpmodel -q` shows MAXN |
| **Display fps < 60** | Check `FPS_CAP=60` in `config.py`. Confirm `PoseWorker` is publishing pose only, not frames |
| **Colours look grey / dull** | Run `camera_tuner.py` — bump saturation to 1.4, lock AWB after calibration |
| **NoMachine no display** | `export DISPLAY=:1` (NoMachine virtual session, not `:0`) |

For each hardware subsystem, the matching tuner script has a dedicated **Troubleshooting** tab with extended checklists.

---

## Project structure

```
jetson-rlgl/
├── main.py                 Entry point — initialises subsystems, runs engine
├── game.py                 GameEngine — finite-state machine, motion logic
├── vision.py               ProPoseTracker (TRT) + PoseWorker (async thread)
├── ui.py                   UIRenderer — Material 3 Pygame screens
├── audio.py                AudioManager — SFX + music + espeak TTS
├── hardware.py             Camera (CSI), ServoController (PCA9685), LaserBreakBeam
├── config.py               All tunable constants + difficulty presets
│
├── camera_tuner.py         ISP / GStreamer live tuner
├── servo_tuner.py          PCA9685 calibration & test rig
├── laser_tuner.py          GPIO break-beam debugger
│
├── setup.sh                One-shot dependency installer
├── launcher.sh              Menu-driven game launcher
│
├── yolov8n-pose.pt         Source model (4 MB)
├── yolov8n-pose.engine     TensorRT engine, auto-generated on first run (~10 MB)
│
├── assets/
│   ├── fonts/              GoogleSans .ttf files
│   └── sounds/             .wav and .mp3 audio
│
└── hardware_tests/         Standalone subsystem test scripts
```

---

## Built with

[NVIDIA Jetson](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/) ·
[TensorRT](https://developer.nvidia.com/tensorrt) ·
[YOLOv8 by Ultralytics](https://github.com/ultralytics/ultralytics) ·
[OpenCV](https://opencv.org/) ·
[GStreamer](https://gstreamer.freedesktop.org/) ·
[Pygame](https://www.pygame.org/) ·
[Adafruit CircuitPython](https://github.com/adafruit/Adafruit_CircuitPython_PCA9685) ·
[espeak](http://espeak.sourceforge.net/)

---
## Acknowledgements

### Models
- **YOLOv8-pose** by [Ultralytics](https://github.com/ultralytics/ultralytics)
  — released under [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).
  The bundled `yolov8n-pose.pt` and the derived `yolov8n-pose.engine`
  remain under AGPL. Code in this repository that loads or wraps these
  weights is intended for personal and educational use; commercial
  deployment requires either an Ultralytics enterprise license or
  AGPL-compatible release of all dependent code.

### Fonts
- **Google Sans** by Google — licensed under
  [SIL Open Font License v1.1](assets/fonts/OFL.txt). Redistributed
  unmodified.

### Inspiration
- The *Red Light, Green Light* game itself is a traditional
  playground game in the public domain.
- Aesthetic and dramatic framing inspired by *Squid Game* (Netflix, 2021).
  This project is not affiliated with, endorsed by, or sponsored by
  Netflix or any related entity. All series-specific trademarks,
  character likenesses, and copyrighted material remain the property
  of their respective owners.

### Author
Built by **Richard P** and **Cindy X** for STEM Day 2026.

Released under the [MIT License](LICENSE) — note that this license
applies to the project's original code only. Bundled third-party
assets (fonts, model weights) retain their original licenses.

<br />

<div align="center">

*"red light…   green light…"*

</div>