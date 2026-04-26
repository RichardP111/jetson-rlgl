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

from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any


# ===========================================================================
# Defaults — match your current config.py
# ===========================================================================
DEFAULTS: dict[str, Any] = {
    "i2c_bus": 7,
    "address": 0x40,
    "channel": 0,
    "frequency": 50,
    "min_us": 500,
    "max_us": 2500,
    "away_deg": 0,
    "face_deg": 180,
    "step_deg": 4,
    "tick_s": 0.012,
}


# ===========================================================================
# Hardware abstraction — Adafruit stack with simulation fallback
# ===========================================================================
class ServoBackend:
    """Wraps PCA9685 with the same try-import pattern as your hardware.py.
    Exposes: set_pulse_us(us), set_freq(hz), is_hw, deinit(), ping()."""

    def __init__(self, address: int, channel: int, frequency: int) -> None:
        self.address = address
        self.channel = channel
        self.frequency = frequency
        self.is_hw = False
        self._pca: Any | None = None
        self._i2c: Any | None = None
        self._last_pulse_us: float = 1500.0
        self._init_error: str = ""

        try:
            import board  # type: ignore
            import busio  # type: ignore
            from adafruit_pca9685 import PCA9685 as _PCA9685  # type: ignore

            self._i2c = busio.I2C(board.SCL, board.SDA)
            self._pca = _PCA9685(self._i2c, address=address)
            self._pca.frequency = frequency
            self.is_hw = True
        except ImportError as exc:
            self._init_error = f"Adafruit libs missing: {exc}"
        except Exception as exc:
            self._init_error = f"PCA9685 init failed: {exc}"

    @property
    def init_error(self) -> str:
        return self._init_error

    @property
    def last_pulse_us(self) -> float:
        return self._last_pulse_us

    def set_pulse_us(self, pulse_us: float) -> None:
        self._last_pulse_us = pulse_us
        if not self.is_hw or self._pca is None:
            return
        period_us = 1_000_000.0 / self.frequency
        duty = int(round((pulse_us / period_us) * 65535))
        duty = max(0, min(65535, duty))
        try:
            self._pca.channels[self.channel].duty_cycle = duty
        except Exception as exc:
            print(f"[BACKEND] write failed: {exc}")

    def set_freq(self, hz: int) -> None:
        self.frequency = hz
        if self.is_hw and self._pca is not None:
            try:
                self._pca.frequency = hz
            except Exception as exc:
                print(f"[BACKEND] freq set failed: {exc}")

    def ping(self) -> tuple[bool, str]:
        """Try a write+readback to confirm the chip is alive."""
        if not self.is_hw or self._pca is None:
            return False, f"Not in hardware mode ({self._init_error or 'sim'})"
        try:
            current = self._pca.frequency
            return True, f"PCA9685 alive @ 0x{self.address:02X}, freq={current}Hz"
        except Exception as exc:
            return False, f"Ping failed: {exc}"

    def deinit(self) -> None:
        if self._pca is not None:
            try:
                self._pca.deinit()
            except Exception:
                pass


# ===========================================================================
# I2C bus scanner — uses i2cdetect, gracefully handles missing tool
# ===========================================================================
def i2c_scan(bus: int) -> tuple[bool, list[int], str]:
    """Returns (success, list_of_addresses_found, raw_output)."""
    try:
        result = subprocess.run(
            ["i2cdetect", "-y", str(bus)],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError:
        return False, [], "i2cdetect not installed (apt install i2c-tools)"
    except subprocess.TimeoutExpired:
        return False, [], "i2cdetect timed out"
    except PermissionError:
        return False, [], "Permission denied — try `sudo i2cdetect -y {bus}` from terminal"

    if result.returncode != 0:
        err = result.stderr.strip() or "unknown error"
        return False, [], f"i2cdetect returned {result.returncode}: {err}"

    addresses: list[int] = []
    for line in result.stdout.splitlines()[1:]:  # skip header
        if ":" not in line:
            continue
        try:
            row_hex, cells = line.split(":", 1)
            row_base = int(row_hex.strip(), 16) << 4
        except ValueError:
            continue
        tokens = cells.split()
        for col_idx, tok in enumerate(tokens):
            if tok in ("--", "UU"):
                continue
            try:
                addresses.append(int(tok, 16))
            except ValueError:
                pass
    return True, addresses, result.stdout


# ===========================================================================
# Config snippet generator
# ===========================================================================
def build_config_snippet(s: dict[str, Any]) -> str:
    return f"""# ── PCA9685 servo config — calibrated via servo_tuner.py on {time.strftime("%Y-%m-%d")}
I2C_BUS = {int(s['i2c_bus'])}
PCA9685_ADDR = 0x{int(s['address']):02X}
SERVO_CHANNEL = {int(s['channel'])}
SERVO_FREQ = {int(s['frequency'])}
SERVO_MIN_US = {int(s['min_us'])}
SERVO_MAX_US = {int(s['max_us'])}
SERVO_AWAY_DEG = {int(s['away_deg'])}
SERVO_FACE_DEG = {int(s['face_deg'])}
SERVO_STEP_DEG = {int(s['step_deg'])}
SERVO_TICK_S = {float(s['tick_s']):.4f}
"""


# ===========================================================================
# Main app
# ===========================================================================
class ServoTuner:
    GAUGE_W = 520
    GAUGE_H = 320

    def __init__(self) -> None:
        self.settings = DEFAULTS.copy()

        # Hardware
        self.backend = ServoBackend(
            self.settings["address"],
            self.settings["channel"],
            self.settings["frequency"],
        )

        # Sweep worker state
        self._target_deg = 90.0
        self._current_deg = 90.0
        self._smooth = True
        self._sweep_lock = threading.Lock()
        self._sweep_running = True
        self._sweep_thread = threading.Thread(target=self._sweep_loop, daemon=True, name="sweep")

        # Auto-test state (sweep / step / jitter)
        self._test_running = False
        self._test_thread: threading.Thread | None = None

        # Tk
        self.root = tk.Tk()
        self.root.title("Servo Tuner — jetson-rlgl")
        self.root.geometry("1280x880")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()
        self._sweep_thread.start()
        self._tick_gauge()

        # Push initial position so something visibly happens at boot
        self._command_angle(90.0)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        pane = ttk.PanedWindow(self.root, orient="horizontal")
        pane.pack(fill="both", expand=True)

        # ── LEFT: gauge + presets + status ───────────────────────────
        left = ttk.Frame(pane, padding=12)
        pane.add(left, weight=1)

        ttk.Label(left, text="Servo Position", font=("TkDefaultFont", 14, "bold")).pack(anchor="w")

        self.gauge = tk.Canvas(left, width=self.GAUGE_W, height=self.GAUGE_H, bg="#1e1e2e", highlightthickness=0)
        self.gauge.pack(pady=(6, 8))

        # Status badge (HW vs SIM)
        status_row = ttk.Frame(left)
        status_row.pack(fill="x", pady=(0, 10))
        if self.backend.is_hw:
            badge = ttk.Label(status_row, text=" ✓ HARDWARE ", foreground="white", background="#16a34a", font=("TkDefaultFont", 11, "bold"))
        else:
            badge = ttk.Label(status_row, text=" ⚠ SIMULATION ", foreground="white", background="#dc2626", font=("TkDefaultFont", 11, "bold"))
        badge.pack(side="left", ipadx=6, ipady=2)

        self.status_label = ttk.Label(
            status_row,
            text=(
                self.backend.init_error
                or f"PCA9685 @ 0x{self.settings['address']:02X}, " f"ch{self.settings['channel']}, " f"{self.settings['frequency']}Hz"
            ),
            foreground="#666",
        )
        self.status_label.pack(side="left", padx=10)

        # Live readouts
        readout = ttk.LabelFrame(left, text="Live", padding=10)
        readout.pack(fill="x", pady=(0, 12))

        self.readout_angle = self._make_readout(readout, "Angle (°)", "90.0", row=0)
        self.readout_target = self._make_readout(readout, "Target (°)", "90.0", row=1)
        self.readout_pulse = self._make_readout(readout, "Pulse (µs)", "1500", row=2)
        self.readout_duty = self._make_readout(readout, "Duty cycle", "7.5%", row=3)

        # Presets
        presets = ttk.LabelFrame(left, text="Presets", padding=10)
        presets.pack(fill="x", pady=(0, 12))

        for label, deg, color in (
            ("◄ FACE AWAY  (0°)", 0, "#0ea5e9"),
            ("◊ CENTER  (90°)", 90, "#a855f7"),
            ("FACE PLAYERS  (180°) ►", 180, "#dc2626"),
        ):
            btn = tk.Button(
                presets,
                text=label,
                command=lambda d=deg: self._command_angle(float(d)),
                bg=color,
                fg="white",
                font=("TkDefaultFont", 11, "bold"),
                activebackground=color,
                relief="flat",
                padx=12,
                pady=8,
            )
            btn.pack(fill="x", pady=2)

        # ── RIGHT: tabbed controls ───────────────────────────────────
        right = ttk.Frame(pane, padding=12)
        pane.add(right, weight=1)

        nb = ttk.Notebook(right)
        nb.pack(fill="both", expand=True)

        self._build_tab_manual(nb)
        self._build_tab_calibration(nb)
        self._build_tab_tests(nb)
        self._build_tab_diagnostics(nb)
        self._build_tab_config(nb)
        self._build_tab_troubleshooting(nb)

    @staticmethod
    def _make_readout(parent: ttk.Frame | ttk.LabelFrame, label: str, initial: str, row: int) -> ttk.Label:
        ttk.Label(parent, text=label, font=("TkDefaultFont", 10), foreground="#666").grid(row=row, column=0, sticky="w", padx=(0, 16), pady=2)
        val = ttk.Label(parent, text=initial, font=("Courier", 13, "bold"))
        val.grid(row=row, column=1, sticky="w")
        return val

    # --------------------------------------------------------------
    # Tab: Manual
    # --------------------------------------------------------------
    def _build_tab_manual(self, nb: ttk.Notebook) -> None:
        tab = ttk.Frame(nb, padding=12)
        nb.add(tab, text="Manual")

        ttk.Label(tab, text="Angle control", font=("TkDefaultFont", 12, "bold")).pack(anchor="w")
        ttk.Label(tab, text="Drag to command an angle. Smooth mode mimics " "the in-game sweep behaviour.", foreground="#888").pack(
            anchor="w", pady=(0, 8)
        )

        # Angle slider
        self._angle_var = tk.DoubleVar(value=90.0)
        angle_top = ttk.Frame(tab)
        angle_top.pack(fill="x")
        ttk.Label(angle_top, text="Angle (°)", font=("TkDefaultFont", 11)).pack(side="left")
        self._angle_lbl = ttk.Label(angle_top, text="90.0", font=("Courier", 12, "bold"))
        self._angle_lbl.pack(side="right")

        angle_scale = ttk.Scale(
            tab,
            from_=0,
            to=180,
            variable=self._angle_var,
            command=self._on_angle_drag,
            orient="horizontal",
        )
        angle_scale.pack(fill="x", pady=(2, 8))

        # Smooth toggle + speed
        opts = ttk.Frame(tab)
        opts.pack(fill="x", pady=(0, 14))
        self._smooth_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="Smooth motion (background sweep)", variable=self._smooth_var, command=self._on_smooth_toggle).pack(side="left")

        ttk.Separator(tab, orient="horizontal").pack(fill="x", pady=8)

        # Direct pulse-width control
        ttk.Label(tab, text="Direct pulse width (advanced)", font=("TkDefaultFont", 12, "bold")).pack(anchor="w", pady=(4, 0))
        ttk.Label(
            tab,
            text="Bypasses angle-to-pulse mapping. Use during MIN/MAX " "calibration, otherwise the angle slider above.",
            foreground="#888",
        ).pack(anchor="w", pady=(0, 6))

        self._pulse_var = tk.DoubleVar(value=1500.0)
        pulse_top = ttk.Frame(tab)
        pulse_top.pack(fill="x")
        ttk.Label(pulse_top, text="Pulse (µs)", font=("TkDefaultFont", 11)).pack(side="left")
        self._pulse_lbl = ttk.Label(pulse_top, text="1500", font=("Courier", 12, "bold"))
        self._pulse_lbl.pack(side="right")

        pulse_scale = ttk.Scale(
            tab,
            from_=400,
            to=2700,
            variable=self._pulse_var,
            command=self._on_pulse_drag,
            orient="horizontal",
        )
        pulse_scale.pack(fill="x", pady=(2, 8))

        # Step / tick controls (sweep speed tuning)
        ttk.Separator(tab, orient="horizontal").pack(fill="x", pady=8)
        ttk.Label(tab, text="Sweep speed", font=("TkDefaultFont", 12, "bold")).pack(anchor="w", pady=(4, 0))
        ttk.Label(
            tab,
            text="step° per tick × tick interval = effective speed.",
            foreground="#888",
        ).pack(anchor="w", pady=(0, 6))

        self._make_setting_slider(tab, "Step degrees", "step_deg", 1, 12, integer=True)
        self._make_setting_slider(tab, "Tick interval (s)", "tick_s", 0.004, 0.05, resolution=0.001)

    # --------------------------------------------------------------
    # Tab: Calibration
    # --------------------------------------------------------------
    def _build_tab_calibration(self, nb: ttk.Notebook) -> None:
        tab = ttk.Frame(nb, padding=12)
        nb.add(tab, text="Calibration")

        ttk.Label(tab, text="MIN / MAX pulse calibration", font=("TkDefaultFont", 13, "bold")).pack(anchor="w")
        ttk.Label(
            tab,
            text=(
                "Cheap servos rarely match the textbook 500–2500 µs range. "
                "The wrong values mean 0° and 180° in code don't actually\n"
                "hit the mechanical limits — or worse, drive the servo "
                "into its endstop and burn it out."
            ),
            foreground="#888",
            justify="left",
        ).pack(anchor="w", pady=(0, 12))

        # MIN
        min_frame = ttk.LabelFrame(tab, text="Step 1 — find MIN_US (0° endpoint)", padding=10)
        min_frame.pack(fill="x", pady=4)
        ttk.Label(
            min_frame,
            text=(
                "1. Click 'Set 500 µs' below.\n"
                "2. Watch / listen to the servo. If it's straining or "
                "buzzing, your MIN is too low — raise it.\n"
                "3. Decrease the slider in the Manual tab's pulse-width "
                "control until the servo just stops moving further.\n"
                "4. Click 'Capture as MIN' to record that value."
            ),
            justify="left",
        ).pack(anchor="w")
        row = ttk.Frame(min_frame)
        row.pack(fill="x", pady=(8, 0))
        ttk.Button(row, text="Set 500 µs", command=lambda: self._command_pulse(500)).pack(side="left", padx=2)
        ttk.Button(row, text="Set 600 µs", command=lambda: self._command_pulse(600)).pack(side="left", padx=2)
        ttk.Button(row, text="Set 700 µs", command=lambda: self._command_pulse(700)).pack(side="left", padx=2)
        ttk.Button(
            row,
            text="📍 Capture current as MIN_US",
            command=self._capture_min,
        ).pack(side="right", padx=2)
        self._min_label = ttk.Label(min_frame, text=f"Current MIN_US = {self.settings['min_us']}", font=("Courier", 11, "bold"))
        self._min_label.pack(anchor="w", pady=(8, 0))

        # MAX
        max_frame = ttk.LabelFrame(tab, text="Step 2 — find MAX_US (180° endpoint)", padding=10)
        max_frame.pack(fill="x", pady=10)
        ttk.Label(
            max_frame,
            text=("Same idea, but increase the pulse width until the servo " "just stops, then capture."),
            justify="left",
        ).pack(anchor="w")
        row = ttk.Frame(max_frame)
        row.pack(fill="x", pady=(8, 0))
        ttk.Button(row, text="Set 2300 µs", command=lambda: self._command_pulse(2300)).pack(side="left", padx=2)
        ttk.Button(row, text="Set 2400 µs", command=lambda: self._command_pulse(2400)).pack(side="left", padx=2)
        ttk.Button(row, text="Set 2500 µs", command=lambda: self._command_pulse(2500)).pack(side="left", padx=2)
        ttk.Button(
            row,
            text="📍 Capture current as MAX_US",
            command=self._capture_max,
        ).pack(side="right", padx=2)
        self._max_label = ttk.Label(max_frame, text=f"Current MAX_US = {self.settings['max_us']}", font=("Courier", 11, "bold"))
        self._max_label.pack(anchor="w", pady=(8, 0))

        # Verify
        verify = ttk.LabelFrame(tab, text="Step 3 — verify", padding=10)
        verify.pack(fill="x", pady=4)
        ttk.Label(
            verify,
            text=("Click each preset. The servo should reach its mechanical " "limit cleanly without straining."),
        ).pack(anchor="w")
        row = ttk.Frame(verify)
        row.pack(fill="x", pady=(8, 0))
        ttk.Button(row, text="Drive to 0°", command=lambda: self._command_angle(0)).pack(side="left", padx=2)
        ttk.Button(row, text="Drive to 90°", command=lambda: self._command_angle(90)).pack(side="left", padx=2)
        ttk.Button(row, text="Drive to 180°", command=lambda: self._command_angle(180)).pack(side="left", padx=2)

        # Frequency
        ttk.Separator(tab, orient="horizontal").pack(fill="x", pady=12)
        ttk.Label(tab, text="PWM frequency", font=("TkDefaultFont", 12, "bold")).pack(anchor="w")
        ttk.Label(
            tab,
            text=("Almost all hobby servos want 50 Hz. Some cheap ones " "behave better at 60 Hz. Don't go above 100 Hz."),
            foreground="#888",
        ).pack(anchor="w", pady=(0, 6))
        self._make_setting_slider(tab, "Frequency (Hz)", "frequency", 40, 100, integer=True, on_change=self._on_freq_change)

    # --------------------------------------------------------------
    # Tab: Tests
    # --------------------------------------------------------------
    def _build_tab_tests(self, nb: ttk.Notebook) -> None:
        tab = ttk.Frame(nb, padding=12)
        nb.add(tab, text="Tests")

        ttk.Label(tab, text="Automated tests", font=("TkDefaultFont", 13, "bold")).pack(anchor="w")
        ttk.Label(
            tab,
            text="Each test runs in the background. Press STOP to abort.",
            foreground="#888",
        ).pack(anchor="w", pady=(0, 12))

        # Sweep
        sweep = ttk.LabelFrame(tab, text="Continuous sweep", padding=10)
        sweep.pack(fill="x", pady=4)
        ttk.Label(
            sweep,
            text="Sweeps slowly between AWAY_DEG and FACE_DEG. Visual sanity " "check — the head should travel smoothly with no buzzing or stalls.",
        ).pack(anchor="w")
        ttk.Button(
            sweep,
            text="▶ Start sweep",
            command=lambda: self._run_test(self._test_sweep),
        ).pack(side="left", pady=(8, 0))

        # Step response
        step = ttk.LabelFrame(tab, text="Step response", padding=10)
        step.pack(fill="x", pady=4)
        ttk.Label(
            step,
            text="Jumps instantly between 0° and 180°. Measures how fast " "your particular servo can travel a half-turn.",
        ).pack(anchor="w")
        ttk.Button(
            step,
            text="▶ Run step test",
            command=lambda: self._run_test(self._test_step),
        ).pack(side="left", pady=(8, 0))

        # Jitter / hold
        jitter = ttk.LabelFrame(tab, text="Hold test", padding=10)
        jitter.pack(fill="x", pady=4)
        ttk.Label(
            jitter,
            text="Holds the current position for 30 seconds. If the servo " "hums or twitches, you have power supply / ground issues.",
        ).pack(anchor="w")
        ttk.Button(
            jitter,
            text="▶ Hold for 30s",
            command=lambda: self._run_test(self._test_hold),
        ).pack(side="left", pady=(8, 0))

        # Random
        rnd = ttk.LabelFrame(tab, text="Random walk", padding=10)
        rnd.pack(fill="x", pady=4)
        ttk.Label(
            rnd,
            text="Picks random target angles every 1.5 s. Stress test for " "the smooth-sweep code path.",
        ).pack(anchor="w")
        ttk.Button(
            rnd,
            text="▶ Random walk",
            command=lambda: self._run_test(self._test_random),
        ).pack(side="left", pady=(8, 0))

        # STOP
        ttk.Separator(tab, orient="horizontal").pack(fill="x", pady=14)
        stop_btn = tk.Button(
            tab,
            text="■ STOP all tests",
            command=self._stop_test,
            bg="#dc2626",
            fg="white",
            font=("TkDefaultFont", 11, "bold"),
            relief="flat",
            padx=12,
            pady=8,
        )
        stop_btn.pack(fill="x")

        self._test_status = ttk.Label(tab, text="Idle.", foreground="#888")
        self._test_status.pack(anchor="w", pady=8)

    # --------------------------------------------------------------
    # Tab: Diagnostics
    # --------------------------------------------------------------
    def _build_tab_diagnostics(self, nb: ttk.Notebook) -> None:
        tab = ttk.Frame(nb, padding=12)
        nb.add(tab, text="Diagnostics")

        ttk.Label(tab, text="I²C bus scan", font=("TkDefaultFont", 13, "bold")).pack(anchor="w")
        ttk.Label(
            tab,
            text="Runs `i2cdetect -y <bus>` and looks for the PCA9685 " "(default address 0x40).",
            foreground="#888",
        ).pack(anchor="w", pady=(0, 6))

        scan_row = ttk.Frame(tab)
        scan_row.pack(fill="x")
        ttk.Label(scan_row, text="Bus number:").pack(side="left")
        self._scan_bus = tk.IntVar(value=self.settings["i2c_bus"])
        ttk.Spinbox(scan_row, from_=0, to=11, textvariable=self._scan_bus, width=4).pack(side="left", padx=8)
        ttk.Button(scan_row, text="🔍 Scan", command=self._do_scan).pack(side="left", padx=8)

        self._scan_text = tk.Text(tab, height=14, wrap="none", font=("Courier", 10), bg="#1e1e1e", fg="#d4d4d4")
        self._scan_text.pack(fill="x", pady=8)
        self._scan_text.insert("1.0", "Click Scan to probe the I²C bus.\n")

        # Ping
        ttk.Separator(tab, orient="horizontal").pack(fill="x", pady=8)
        ttk.Label(tab, text="PCA9685 ping", font=("TkDefaultFont", 13, "bold")).pack(anchor="w")
        ttk.Label(
            tab,
            text="Reads back the chip's frequency register to confirm it " "actually responds, not just that the address is on the bus.",
            foreground="#888",
        ).pack(anchor="w", pady=(0, 6))
        ttk.Button(tab, text="📡 Ping PCA9685", command=self._do_ping).pack(anchor="w")

        self._ping_label = ttk.Label(tab, text="—", font=("Courier", 11))
        self._ping_label.pack(anchor="w", pady=8)

    # --------------------------------------------------------------
    # Tab: Config snippet
    # --------------------------------------------------------------
    def _build_tab_config(self, nb: ttk.Notebook) -> None:
        tab = ttk.Frame(nb, padding=12)
        nb.add(tab, text="Config")

        ttk.Label(tab, text="Drop into config.py", font=("TkDefaultFont", 13, "bold")).pack(anchor="w")
        ttk.Label(
            tab,
            text=("Replace your existing servo settings in config.py with " "the block below."),
            foreground="#888",
        ).pack(anchor="w", pady=(0, 8))

        self._config_text = tk.Text(tab, height=18, wrap="none", font=("Courier", 11), bg="#1e1e1e", fg="#d4d4d4", insertbackground="white")
        self._config_text.pack(fill="both", expand=True)

        btn_row = ttk.Frame(tab)
        btn_row.pack(fill="x", pady=8)
        ttk.Button(btn_row, text="🔄 Refresh", command=self._refresh_config_snippet).pack(side="left", padx=2)
        ttk.Button(btn_row, text="📋 Copy", command=self._copy_config).pack(side="left", padx=2)
        ttk.Button(btn_row, text="💾 Save preset", command=self._save_preset).pack(side="left", padx=2)
        ttk.Button(btn_row, text="📂 Load preset", command=self._load_preset).pack(side="left", padx=2)
        ttk.Button(btn_row, text="↺ Reset", command=self._reset_defaults).pack(side="left", padx=2)

        self._refresh_config_snippet()

    # --------------------------------------------------------------
    # Tab: Troubleshooting
    # --------------------------------------------------------------
    def _build_tab_troubleshooting(self, nb: ttk.Notebook) -> None:
        tab = ttk.Frame(nb, padding=12)
        nb.add(tab, text="Troubleshooting")

        canvas = tk.Canvas(tab, highlightthickness=0)
        scroll = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        issues = [
            (
                "🔇 Servo doesn't move at all",
                [
                    "Check the SIMULATION badge in the upper-left — if you're " "in sim mode the script can't talk to the chip.",
                    "Run an I²C scan in the Diagnostics tab. 0x40 should appear. "
                    "If it doesn't, check SDA → pin 3, SCL → pin 5 on the "
                    "Jetson 40-pin header.",
                    "Verify the PCA9685 has 3.3V on its VCC pin (powers the "
                    "logic) AND 5–6V on V+ (powers the servos). Both are "
                    "required. The Jetson 3.3V cannot drive a servo.",
                    "Check that the servo's PWM wire (yellow/orange) is on " "channel 0's signal pin, not GND or VCC.",
                ],
            ),
            (
                "📳 Servo moves but jitters / buzzes",
                [
                    "Power supply too weak — most servos need 1A+ inrush. " "USB power banks usually aren't enough; use a wall adapter.",
                    "Ground continuity: the PCA9685 GND, the servo PSU GND, " "and the Jetson GND must all be tied together.",
                    "PWM frequency wrong — try 60 Hz instead of 50 Hz in the " "Calibration tab.",
                    "Cable too long or too thin — voltage drops at the servo. " "Use 22 AWG or thicker, keep under 30 cm.",
                ],
            ),
            (
                "🎯 Servo hits the wrong angle (off by 5–30°)",
                [
                    "Run MIN/MAX calibration in the Calibration tab. Most " "servos aren't perfectly 500/2500 µs.",
                    "After calibration, check that the gauge angle matches "
                    "what you see physically. If they disagree, AWAY_DEG / "
                    "FACE_DEG might need to swap (depends on servo orientation).",
                ],
            ),
            (
                "🔥 Servo gets hot / strains at endpoints",
                [
                    "MIN or MAX is past the mechanical stop. Re-run calibration " "and pull the values 50 µs back from the limit you found.",
                    "Stop using the servo immediately if it smells warm — " "running into a hard stop will burn out the motor coil " "in minutes.",
                ],
            ),
            (
                "🔌 I²C address not detected",
                [
                    "Run `sudo i2cdetect -y 7` from the terminal. If that "
                    "needs sudo but the script doesn't, the i2c-dev udev "
                    "rule isn't applied — add your user to the `i2c` group "
                    "and reboot.",
                    "Check for solder bridges between A0–A5 on the PCA9685 — " "they change the I²C address. All open = 0x40.",
                    "Try the other I²C bus (1 vs 7 on Jetson Orin Nano). " "Bus 7 is the default for the 40-pin header.",
                ],
            ),
            (
                "🐢 Sweep is jerky / stuttery",
                [
                    "TICK_S too high — drop to 0.008 in Manual tab.",
                    "STEP_DEG too high — drop to 2 for smoother motion at the " "cost of slower travel.",
                    "Background system load — check `htop`; if YOLO is using "
                    "100% of one core, the sweep thread might not be getting "
                    "scheduled often enough.",
                ],
            ),
            (
                "💀 Worked yesterday, doesn't work today",
                [
                    "First: I²C scan. Did the chip drop off the bus?",
                    "Power: was the PSU unplugged? Is the 5V rail still 5V " "(measure with a meter — voltage drops as power supplies age).",
                    "Wiring: tug-test every connector. Loose Dupont jumpers " "are the #1 cause of intermittent failures.",
                    "Heat damage: if the chip got too hot at any point, " "the I²C interface can fail before the PWM does. Replace.",
                    "Software: did you `pip upgrade` recently? Adafruit's "
                    "circuitpython libs occasionally break compatibility. "
                    "Pin to a known-good version in setup.sh.",
                ],
            ),
        ]

        for title, items in issues:
            grp = ttk.LabelFrame(inner, text=title, padding=10)
            grp.pack(fill="x", padx=8, pady=6)
            for item in items:
                lbl = ttk.Label(grp, text="• " + item, justify="left", wraplength=620, foreground="#222")
                lbl.pack(anchor="w", pady=2)

    # ------------------------------------------------------------------
    # Generic slider helpers
    # ------------------------------------------------------------------
    def _make_setting_slider(
        self,
        parent: ttk.Frame,
        label: str,
        key: str,
        min_val: float,
        max_val: float,
        integer: bool = False,
        resolution: float = 0.001,
        on_change: Any = None,
    ) -> None:
        outer = ttk.Frame(parent)
        outer.pack(fill="x", pady=4)
        top = ttk.Frame(outer)
        top.pack(fill="x")
        ttk.Label(top, text=label, width=20, anchor="w").pack(side="left")

        var = tk.DoubleVar(value=float(self.settings[key]))
        val_text = tk.StringVar(value=self._fmt(self.settings[key], integer))
        ttk.Label(top, textvariable=val_text, width=12, anchor="e", font=("Courier", 10, "bold")).pack(side="right")

        def cb(v: str) -> None:
            val: Any = float(v)
            if integer:
                val = int(round(val))
            self.settings[key] = val
            val_text.set(self._fmt(val, integer))
            if on_change is not None:
                on_change(val)
            self._refresh_config_snippet()

        scale = ttk.Scale(outer, from_=min_val, to=max_val, variable=var, command=cb, orient="horizontal")
        scale.pack(fill="x", pady=(2, 0))

    @staticmethod
    def _fmt(val: Any, integer: bool) -> str:
        if integer:
            return str(int(val))
        return f"{float(val):.3f}"

    # ------------------------------------------------------------------
    # Manual-tab slider callbacks
    # ------------------------------------------------------------------
    def _on_angle_drag(self, v: str) -> None:
        deg = float(v)
        self._angle_lbl.config(text=f"{deg:.1f}")
        self._command_angle(deg)

    def _on_pulse_drag(self, v: str) -> None:
        us = float(v)
        self._pulse_lbl.config(text=f"{us:.0f}")
        self._command_pulse(us)

    def _on_smooth_toggle(self) -> None:
        self._smooth = bool(self._smooth_var.get())

    def _on_freq_change(self, hz: int) -> None:
        self.backend.set_freq(int(hz))
        self.status_label.config(
            text=f"PCA9685 @ 0x{self.settings['address']:02X}, " f"ch{self.settings['channel']}, {hz}Hz",
        )

    # ------------------------------------------------------------------
    # Command helpers
    # ------------------------------------------------------------------
    def _angle_to_pulse(self, deg: float) -> float:
        deg = max(0.0, min(180.0, deg))
        return self.settings["min_us"] + (deg / 180.0) * (self.settings["max_us"] - self.settings["min_us"])

    def _command_angle(self, deg: float) -> None:
        deg = max(0.0, min(180.0, deg))
        with self._sweep_lock:
            self._target_deg = deg
            if not self._smooth:
                self._current_deg = deg
                self.backend.set_pulse_us(self._angle_to_pulse(deg))
        # Update slider widget if changed externally
        try:
            self._angle_var.set(deg)
            self._angle_lbl.config(text=f"{deg:.1f}")
        except (AttributeError, tk.TclError):
            pass

    def _command_pulse(self, us: float) -> None:
        self.backend.set_pulse_us(us)
        try:
            self._pulse_var.set(us)
            self._pulse_lbl.config(text=f"{us:.0f}")
        except (AttributeError, tk.TclError):
            pass

    # ------------------------------------------------------------------
    # Sweep worker thread
    # ------------------------------------------------------------------
    def _sweep_loop(self) -> None:
        while self._sweep_running:
            with self._sweep_lock:
                target = self._target_deg
                current = self._current_deg
                step = self.settings["step_deg"]
                tick = self.settings["tick_s"]
                smooth = self._smooth

            if smooth:
                diff = target - current
                if abs(diff) > 0.5:
                    move = step if diff > 0 else -step
                    if abs(move) > abs(diff):
                        move = diff
                    new_deg = max(0.0, min(180.0, current + move))
                    with self._sweep_lock:
                        self._current_deg = new_deg
                    self.backend.set_pulse_us(self._angle_to_pulse(new_deg))

            time.sleep(max(0.001, tick))

    # ------------------------------------------------------------------
    # Gauge tick (Tk main thread, ~30 Hz)
    # ------------------------------------------------------------------
    def _tick_gauge(self) -> None:
        with self._sweep_lock:
            current = self._current_deg
            target = self._target_deg

        # Live readouts
        pulse = self.backend.last_pulse_us
        period_us = 1_000_000.0 / self.settings["frequency"]
        duty_pct = (pulse / period_us) * 100.0

        self.readout_angle.config(text=f"{current:6.1f}")
        self.readout_target.config(text=f"{target:6.1f}")
        self.readout_pulse.config(text=f"{pulse:6.0f}")
        self.readout_duty.config(text=f"{duty_pct:5.2f}%")

        self._draw_gauge(current, target)
        self.root.after(33, self._tick_gauge)

    def _draw_gauge(self, current: float, target: float) -> None:
        c = self.gauge
        c.delete("all")

        cx = self.GAUGE_W // 2
        cy = self.GAUGE_H - 30
        radius = 220

        # Arc background (180° at top)
        c.create_arc(
            cx - radius,
            cy - radius,
            cx + radius,
            cy + radius,
            start=0,
            extent=180,
            style="arc",
            outline="#3f3f55",
            width=18,
        )

        # Tick marks every 30°
        for tick in range(0, 181, 30):
            ang = math.radians(180 - tick)
            x1 = cx + (radius - 18) * math.cos(ang)
            y1 = cy - (radius - 18) * math.sin(ang)
            x2 = cx + (radius + 6) * math.cos(ang)
            y2 = cy - (radius + 6) * math.sin(ang)
            c.create_line(x1, y1, x2, y2, fill="#9ca3af", width=2)
            tx = cx + (radius + 24) * math.cos(ang)
            ty = cy - (radius + 24) * math.sin(ang)
            c.create_text(tx, ty, text=f"{tick}°", fill="#9ca3af", font=("TkDefaultFont", 9))

        # Coloured progress arc
        if current > 0:
            c.create_arc(
                cx - radius,
                cy - radius,
                cx + radius,
                cy + radius,
                start=180 - current,
                extent=current,
                style="arc",
                outline="#a855f7",
                width=18,
            )

        # Target indicator (thin)
        target_ang = math.radians(180 - target)
        tx1 = cx + (radius - 30) * math.cos(target_ang)
        ty1 = cy - (radius - 30) * math.sin(target_ang)
        tx2 = cx + (radius + 12) * math.cos(target_ang)
        ty2 = cy - (radius + 12) * math.sin(target_ang)
        c.create_line(tx1, ty1, tx2, ty2, fill="#fbbf24", width=2, dash=(4, 2))

        # Needle for current position
        ang = math.radians(180 - current)
        nx = cx + (radius - 8) * math.cos(ang)
        ny = cy - (radius - 8) * math.sin(ang)
        c.create_line(cx, cy, nx, ny, fill="#fff", width=4, capstyle="round")

        # Hub
        c.create_oval(cx - 12, cy - 12, cx + 12, cy + 12, fill="#1e1e2e", outline="#a855f7", width=3)

        # Centre numeric readout
        c.create_text(cx, cy + 50, text=f"{current:.1f}°", fill="#fff", font=("TkDefaultFont", 24, "bold"))

    # ------------------------------------------------------------------
    # Calibration captures
    # ------------------------------------------------------------------
    def _capture_min(self) -> None:
        us = int(round(self.backend.last_pulse_us))
        self.settings["min_us"] = us
        self._min_label.config(text=f"Current MIN_US = {us}")
        self._refresh_config_snippet()

    def _capture_max(self) -> None:
        us = int(round(self.backend.last_pulse_us))
        self.settings["max_us"] = us
        self._max_label.config(text=f"Current MAX_US = {us}")
        self._refresh_config_snippet()

    # ------------------------------------------------------------------
    # Tests (run on background thread)
    # ------------------------------------------------------------------
    def _run_test(self, target: Any) -> None:
        if self._test_running:
            self._test_status.config(text="A test is already running. Stop it first.", foreground="#dc2626")
            return
        self._test_running = True
        self._test_thread = threading.Thread(target=target, daemon=True, name="test")
        self._test_thread.start()

    def _stop_test(self) -> None:
        self._test_running = False
        self._test_status.config(text="Stopping…", foreground="#666")

    def _test_sweep(self) -> None:
        self._test_status.config(text="Sweep running.", foreground="#16a34a")
        a = self.settings["away_deg"]
        b = self.settings["face_deg"]
        try:
            while self._test_running:
                self._command_angle(a)
                if not self._wait_until_at(a, timeout=4.0):
                    break
                self._command_angle(b)
                if not self._wait_until_at(b, timeout=4.0):
                    break
        finally:
            self._test_running = False
            self._test_status.config(text="Sweep stopped.", foreground="#666")

    def _test_step(self) -> None:
        self._test_status.config(text="Step test running.", foreground="#16a34a")
        try:
            for _ in range(5):
                if not self._test_running:
                    break
                self._command_angle(0)
                t0 = time.time()
                self._wait_until_at(0, timeout=3.0)
                dt0 = time.time() - t0
                self._command_angle(180)
                t1 = time.time()
                self._wait_until_at(180, timeout=3.0)
                dt1 = time.time() - t1
                self._test_status.config(
                    text=f"Step: 0°→180° took {dt1*1000:.0f}ms, 180°→0° took {dt0*1000:.0f}ms",
                    foreground="#16a34a",
                )
        finally:
            self._test_running = False

    def _test_hold(self) -> None:
        self._test_status.config(text="Holding for 30s — listen for jitter.", foreground="#16a34a")
        end = time.time() + 30
        try:
            while self._test_running and time.time() < end:
                time.sleep(0.1)
        finally:
            self._test_running = False
            self._test_status.config(text="Hold complete.", foreground="#666")

    def _test_random(self) -> None:
        import random

        self._test_status.config(text="Random walk running.", foreground="#16a34a")
        try:
            while self._test_running:
                target = random.uniform(0, 180)
                self._command_angle(target)
                t = 0.0
                while self._test_running and t < 1.5:
                    time.sleep(0.05)
                    t += 0.05
        finally:
            self._test_running = False
            self._test_status.config(text="Random walk stopped.", foreground="#666")

    def _wait_until_at(self, target: float, timeout: float = 3.0, tolerance: float = 1.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self._test_running:
                return False
            with self._sweep_lock:
                cur = self._current_deg
            if abs(cur - target) <= tolerance:
                return True
            time.sleep(0.02)
        return False

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def _do_scan(self) -> None:
        bus = int(self._scan_bus.get())
        self._scan_text.delete("1.0", "end")
        self._scan_text.insert("end", f"Scanning I²C bus {bus}…\n\n")
        self.root.update_idletasks()

        ok, addrs, raw = i2c_scan(bus)
        if not ok:
            self._scan_text.insert("end", f"❌ {raw}\n")
            return

        self._scan_text.insert("end", raw + "\n")
        if not addrs:
            self._scan_text.insert("end", "\n⚠ No devices found on this bus.\n")
        else:
            self._scan_text.insert("end", "\nFound:\n")
            for a in addrs:
                tag = "  ← PCA9685 (your servo driver) ✓" if a == self.settings["address"] else ""
                self._scan_text.insert("end", f"  0x{a:02X}{tag}\n")

    def _do_ping(self) -> None:
        ok, msg = self.backend.ping()
        colour = "#16a34a" if ok else "#dc2626"
        prefix = "✓" if ok else "✗"
        self._ping_label.config(text=f"{prefix} {msg}", foreground=colour)

    # ------------------------------------------------------------------
    # Config snippet
    # ------------------------------------------------------------------
    def _refresh_config_snippet(self) -> None:
        if not hasattr(self, "_config_text"):
            return
        snippet = build_config_snippet(self.settings)
        self._config_text.delete("1.0", "end")
        self._config_text.insert("1.0", snippet)

    def _copy_config(self) -> None:
        snippet = build_config_snippet(self.settings)
        self.root.clipboard_clear()
        self.root.clipboard_append(snippet)
        self.root.update()
        messagebox.showinfo("Copied", "config snippet copied to clipboard.")

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------
    def _save_preset(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON preset", "*.json")],
            initialfile="servo_preset.json",
        )
        if not path:
            return
        with open(path, "w") as f:
            json.dump(self.settings, f, indent=2)
        messagebox.showinfo("Saved", f"Preset saved to:\n{path}")

    def _load_preset(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("JSON preset", "*.json")])
        if not path:
            return
        try:
            with open(path) as f:
                loaded = json.load(f)
            merged = DEFAULTS.copy()
            merged.update({k: v for k, v in loaded.items() if k in DEFAULTS})
            self.settings = merged
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))
            return
        # Quick & dirty: rebuild whole UI to reflect new values
        for widget in self.root.winfo_children():
            widget.destroy()
        self._build_ui()
        self._refresh_config_snippet()

    def _reset_defaults(self) -> None:
        self.settings = DEFAULTS.copy()
        for widget in self.root.winfo_children():
            widget.destroy()
        self._build_ui()
        self._refresh_config_snippet()

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    def _on_close(self) -> None:
        self._sweep_running = False
        self._test_running = False
        try:
            self._sweep_thread.join(timeout=1.0)
        except Exception:
            pass
        # Park at AWAY before quitting
        try:
            self.backend.set_pulse_us(self._angle_to_pulse(self.settings["away_deg"]))
            time.sleep(0.3)
        except Exception:
            pass
        self.backend.deinit()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


# ===========================================================================
# Entry point
# ===========================================================================
def main() -> int:
    if platform.system() != "Linux":
        print("[INFO] Not on Linux — PCA9685 won't be available, " "running in SIMULATION mode for UI testing.")
    tuner = ServoTuner()
    tuner.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
