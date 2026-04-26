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

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import threading
import time
import tkinter as tk
from collections import deque
from tkinter import filedialog, messagebox, ttk
from typing import Any


# ===========================================================================
# Defaults — match config.py
# ===========================================================================
DEFAULTS: dict[str, Any] = {
    "pin": 7,  # BOARD numbering — pin 7 on 40-pin header
    "numbering": "BOARD",  # BOARD or BCM
    "pull": "PUD_UP",  # PUD_UP / PUD_DOWN / PUD_OFF
    "active_low": True,  # True: beam broken = LOW. False: broken = HIGH.
    "debounce_samples": 3,  # consecutive matching reads to count as stable
    "debounce_window_ms": 30,  # ms between samples
    "poll_rate_hz": 200,  # background poll rate
}

PULL_OPTIONS = {
    "PUD_UP": "Pull-UP (most common)",
    "PUD_DOWN": "Pull-DOWN",
    "PUD_OFF": "Pull-OFF (sensor has its own resistor)",
}

NUMBERING_OPTIONS = {
    "BOARD": "BOARD (physical pin number — 7, 11, 13…)",
    "BCM": "BCM (chip pin number — 4, 17, 27…)",
}


# ===========================================================================
# GPIO backend — Jetson.GPIO with simulation fallback
# ===========================================================================
class LaserBackend:
    """Wraps Jetson.GPIO. Exposes .read() returning raw pin state (0 or 1)."""

    def __init__(self) -> None:
        self.is_hw = False
        self._GPIO: Any = None
        self._init_error: str = ""
        self._pin: int | None = None
        self._numbering: str = "BOARD"
        self._pull: str = "PUD_UP"

        # Simulation state — flipped by the "simulate break" button
        self._sim_value: int = 1  # 1 = HIGH = beam OK with PUD_UP

        try:
            import Jetson.GPIO as GPIO  # type: ignore

            self._GPIO = GPIO
            self.is_hw = True
        except ImportError as exc:
            self._init_error = f"Jetson.GPIO not installed ({exc})"
        except Exception as exc:
            self._init_error = f"Jetson.GPIO import error ({exc})"

    @property
    def init_error(self) -> str:
        return self._init_error

    def setup(self, pin: int, numbering: str, pull: str) -> tuple[bool, str]:
        """Configure the pin. Returns (success, message)."""
        self._pin = pin
        self._numbering = numbering
        self._pull = pull

        if not self.is_hw:
            return False, f"Simulation mode ({self._init_error or 'no GPIO'})"

        try:
            GPIO = self._GPIO
            # If a different mode was already set we must respect it.
            current = GPIO.getmode()
            desired = GPIO.BOARD if numbering == "BOARD" else GPIO.BCM
            if current is None:
                GPIO.setmode(desired)
            elif current != desired:
                # Can't change mid-process. Honour what's set.
                actual = "BOARD" if current == GPIO.BOARD else "BCM"
                return False, (f"GPIO mode already set to {actual} elsewhere — " f"can't switch to {numbering} without restart")

            pull_const = {
                "PUD_UP": GPIO.PUD_UP,
                "PUD_DOWN": GPIO.PUD_DOWN,
                "PUD_OFF": GPIO.PUD_OFF,
            }[pull]

            GPIO.setup(pin, GPIO.IN, pull_up_down=pull_const)
            return True, f"Pin {pin} ({numbering}) configured with {pull}"
        except PermissionError:
            return False, "Permission denied — add user to gpio group, or run with sudo"
        except RuntimeError as exc:
            return False, f"Runtime error: {exc}"
        except Exception as exc:
            return False, f"Unexpected error: {exc}"

    def read(self) -> int:
        """Read raw pin state. Returns 0 or 1."""
        if not self.is_hw:
            return self._sim_value
        if self._pin is None:
            return 1
        try:
            return int(self._GPIO.input(self._pin))
        except Exception:
            return 1

    def sim_set(self, value: int) -> None:
        """Set the simulated pin value (only matters in sim mode)."""
        self._sim_value = 0 if value == 0 else 1

    def sim_pulse_break(self, duration_s: float = 0.2) -> None:
        """Simulate a brief beam break, for testing the debouncer."""
        if self.is_hw:
            return
        self._sim_value = 0
        threading.Timer(duration_s, lambda: setattr(self, "_sim_value", 1)).start()

    def cleanup(self) -> None:
        if not self.is_hw or self._GPIO is None:
            return
        try:
            self._GPIO.cleanup()
        except Exception:
            pass


# ===========================================================================
# GPIO diagnostics helpers
# ===========================================================================
def gpio_permission_check() -> tuple[bool, str]:
    """Check whether the current user is in the gpio group."""
    try:
        result = subprocess.run(["groups"], capture_output=True, text=True, timeout=2)
        groups = result.stdout.split()
        if "gpio" in groups:
            return True, "User is in 'gpio' group ✓"
        return False, ("User is NOT in 'gpio' group — " "run: sudo usermod -aG gpio $USER && reboot")
    except Exception as exc:
        return False, f"Couldn't check groups: {exc}"


def gpio_module_check() -> tuple[bool, str]:
    """Check Jetson.GPIO is installed & importable."""
    try:
        import Jetson.GPIO as GPIO  # type: ignore  # noqa: F401

        return True, "Jetson.GPIO importable ✓"
    except ImportError as exc:
        return False, f"Jetson.GPIO missing: {exc}"
    except Exception as exc:
        return False, f"Jetson.GPIO import error: {exc}"


def list_gpiochip_devices() -> tuple[bool, list[str], str]:
    """List /dev/gpiochip* devices."""
    try:
        result = subprocess.run(["ls", "-la", "/dev/"], capture_output=True, text=True, timeout=2)
        chips = [line for line in result.stdout.splitlines() if "gpiochip" in line]
        if not chips:
            return False, [], "No /dev/gpiochip* devices found"
        return True, chips, "\n".join(chips)
    except Exception as exc:
        return False, [], str(exc)


# ===========================================================================
# Config snippet generator
# ===========================================================================
def build_config_snippet(s: dict[str, Any]) -> str:
    return f"""# ── Laser break-beam config — calibrated via laser_tuner.py on {time.strftime("%Y-%m-%d")}
LASER_PIN = {int(s['pin'])}
LASER_NUMBERING = "{s['numbering']}"
LASER_PULL = "{s['pull']}"
LASER_ACTIVE_LOW = {bool(s['active_low'])}
LASER_DEBOUNCE_SAMPLES = {int(s['debounce_samples'])}
LASER_DEBOUNCE_WINDOW_MS = {int(s['debounce_window_ms'])}
LASER_POLL_RATE_HZ = {int(s['poll_rate_hz'])}
"""


# ===========================================================================
# Main app
# ===========================================================================
class LaserTuner:
    INDICATOR_W = 520
    INDICATOR_H = 220
    HISTORY_W = 520
    HISTORY_H = 100
    HISTORY_SECONDS = 10

    def __init__(self) -> None:
        self.settings = DEFAULTS.copy()
        self.backend = LaserBackend()

        # Apply initial setup so reads work immediately
        self._setup_status: tuple[bool, str] = self.backend.setup(
            self.settings["pin"],
            self.settings["numbering"],
            self.settings["pull"],
        )

        # Beam state tracking
        self._raw_value: int = 1
        self._stable_value: int = 1
        self._is_broken: bool = False
        self._sample_buffer: deque[int] = deque(maxlen=max(1, int(self.settings["debounce_samples"])))

        # History — fixed-rate samples for the strip chart
        self._history: deque[tuple[float, bool]] = deque(maxlen=2000)

        # Statistics
        self._break_count: int = 0
        self._last_break_ts: float | None = None
        self._false_positive_window: deque[float] = deque(maxlen=50)
        self._latency_samples: list[float] = []

        # Test state
        self._test_running = False
        self._test_thread: threading.Thread | None = None

        # Polling thread
        self._poll_running = True
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True, name="poll")

        # Tk
        self.root = tk.Tk()
        self.root.title("Laser Break-Beam Tuner — jetson-rlgl")
        self.root.geometry("1300x900")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()
        self._poll_thread.start()
        self._tick_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        pane = ttk.PanedWindow(self.root, orient="horizontal")
        pane.pack(fill="both", expand=True)

        # ── LEFT: visual indicator + history + stats ────────────────
        left = ttk.Frame(pane, padding=12)
        pane.add(left, weight=1)

        # Status badge
        badge_row = ttk.Frame(left)
        badge_row.pack(fill="x", pady=(0, 8))
        if self.backend.is_hw:
            badge = ttk.Label(badge_row, text=" ✓ HARDWARE ", foreground="white", background="#16a34a", font=("TkDefaultFont", 11, "bold"))
        else:
            badge = ttk.Label(badge_row, text=" ⚠ SIMULATION ", foreground="white", background="#dc2626", font=("TkDefaultFont", 11, "bold"))
        badge.pack(side="left", ipadx=6, ipady=2)

        ok, msg = self._setup_status
        self._setup_label = ttk.Label(
            badge_row,
            text=msg,
            foreground="#16a34a" if ok else "#dc2626",
        )
        self._setup_label.pack(side="left", padx=10)

        # Big indicator
        ttk.Label(left, text="Beam state", font=("TkDefaultFont", 14, "bold")).pack(anchor="w")
        self.indicator = tk.Canvas(left, width=self.INDICATOR_W, height=self.INDICATOR_H, bg="#1e1e2e", highlightthickness=0)
        self.indicator.pack(pady=(4, 12))

        # Live readouts
        readout = ttk.LabelFrame(left, text="Live", padding=10)
        readout.pack(fill="x")
        self.r_raw = self._make_readout(readout, "Raw pin", "—", row=0)
        self.r_stable = self._make_readout(readout, "Debounced state", "—", row=1)
        self.r_breaks = self._make_readout(readout, "Break count", "0", row=2)
        self.r_last = self._make_readout(readout, "Last break (s ago)", "—", row=3)

        # History strip
        ttk.Label(left, text=f"History — last {self.HISTORY_SECONDS}s", font=("TkDefaultFont", 12, "bold")).pack(anchor="w", pady=(12, 4))
        self.history_canvas = tk.Canvas(left, width=self.HISTORY_W, height=self.HISTORY_H, bg="#0f0f1a", highlightthickness=0)
        self.history_canvas.pack()

        # Reset stats button
        btn_row = ttk.Frame(left)
        btn_row.pack(fill="x", pady=(12, 0))
        ttk.Button(btn_row, text="↺ Reset counters", command=self._reset_counters).pack(side="left")
        ttk.Button(btn_row, text="🔄 Re-apply pin setup", command=self._reapply_setup).pack(side="left", padx=8)

        # Sim trigger (only useful when not hardware, but visible always)
        if not self.backend.is_hw:
            ttk.Separator(left, orient="horizontal").pack(fill="x", pady=10)
            ttk.Label(left, text="Simulation controls", font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
            sim_row = ttk.Frame(left)
            sim_row.pack(fill="x", pady=4)
            tk.Button(
                sim_row,
                text="HOLD — beam broken",
                bg="#dc2626",
                fg="white",
                command=lambda: self.backend.sim_set(0),
            ).pack(side="left", padx=2)
            tk.Button(
                sim_row,
                text="RELEASE — beam OK",
                bg="#16a34a",
                fg="white",
                command=lambda: self.backend.sim_set(1),
            ).pack(side="left", padx=2)
            tk.Button(
                sim_row,
                text="⚡ Pulse (200ms break)",
                command=lambda: self.backend.sim_pulse_break(0.2),
            ).pack(side="left", padx=2)

        # ── RIGHT: tabs ─────────────────────────────────────────────
        right = ttk.Frame(pane, padding=12)
        pane.add(right, weight=1)

        nb = ttk.Notebook(right)
        nb.pack(fill="both", expand=True)

        self._build_tab_setup(nb)
        self._build_tab_debounce(nb)
        self._build_tab_tests(nb)
        self._build_tab_diagnostics(nb)
        self._build_tab_config(nb)
        self._build_tab_troubleshooting(nb)

    @staticmethod
    def _make_readout(parent: tk.Misc, label: str, initial: str, row: int) -> ttk.Label:
        ttk.Label(parent, text=label, font=("TkDefaultFont", 10), foreground="#666").grid(row=row, column=0, sticky="w", padx=(0, 16), pady=2)
        val = ttk.Label(parent, text=initial, font=("Courier", 13, "bold"))
        val.grid(row=row, column=1, sticky="w")
        return val

    # --------------------------------------------------------------
    # Tab: Pin setup
    # --------------------------------------------------------------
    def _build_tab_setup(self, nb: ttk.Notebook) -> None:
        tab = ttk.Frame(nb, padding=12)
        nb.add(tab, text="Pin setup")

        ttk.Label(tab, text="GPIO configuration", font=("TkDefaultFont", 13, "bold")).pack(anchor="w")
        ttk.Label(
            tab,
            text="Changes here apply when you click 'Re-apply pin setup' " "on the left.",
            foreground="#888",
        ).pack(anchor="w", pady=(0, 12))

        # Numbering
        self._make_dropdown(tab, "Pin numbering", "numbering", NUMBERING_OPTIONS, help_text="BOARD=physical pin (header), BCM=Tegra chip pin number")

        # Pin
        self._make_setting_slider(tab, "Pin number", "pin", 1, 40, integer=True, help_text="Default 7 = BOARD pin 7 (active 3.3V tolerant input)")

        # Pull
        self._make_dropdown(tab, "Pull resistor", "pull", PULL_OPTIONS, help_text="Most break-beam sensors are open-collector → use PUD_UP")

        # Active state
        active_frame = ttk.Frame(tab)
        active_frame.pack(fill="x", pady=8)
        ttk.Label(active_frame, text="Beam-broken polarity", font=("TkDefaultFont", 11)).pack(anchor="w")
        ttk.Label(
            active_frame,
            text="Most laser break-beam modules pull the line LOW when " "the beam is broken. If yours is opposite, flip this.",
            foreground="#888",
        ).pack(anchor="w", pady=(0, 6))

        self._active_var = tk.BooleanVar(value=bool(self.settings["active_low"]))
        ttk.Radiobutton(
            active_frame, text="Active LOW (broken = 0V) — DEFAULT", variable=self._active_var, value=True, command=self._on_active_change
        ).pack(anchor="w")
        ttk.Radiobutton(
            active_frame, text="Active HIGH (broken = 3.3V)", variable=self._active_var, value=False, command=self._on_active_change
        ).pack(anchor="w")

    # --------------------------------------------------------------
    # Tab: Debounce
    # --------------------------------------------------------------
    def _build_tab_debounce(self, nb: ttk.Notebook) -> None:
        tab = ttk.Frame(nb, padding=12)
        nb.add(tab, text="Debounce")

        ttk.Label(tab, text="Debounce tuning", font=("TkDefaultFont", 13, "bold")).pack(anchor="w")
        ttk.Label(
            tab,
            text=(
                "Mechanical / optical break-beams produce dirty edges — the\n"
                "signal can flip-flop for 1–10ms when an object enters the beam.\n"
                "Without debouncing, one player crossing can register as 5 wins.\n\n"
                "The check here: signal must hold its new value for "
                "(samples × window_ms) ms before we trust it."
            ),
            foreground="#888",
            justify="left",
        ).pack(anchor="w", pady=(0, 12))

        self._make_setting_slider(
            tab,
            "Debounce samples",
            "debounce_samples",
            1,
            20,
            integer=True,
            help_text="3 = sweet spot. Higher = more reliable but slower response.",
            on_change=self._on_debounce_change,
        )
        self._make_setting_slider(
            tab,
            "Sample window (ms)",
            "debounce_window_ms",
            5,
            100,
            integer=True,
            help_text="30ms × 3 samples = 90ms latency. Game logic doesn't notice.",
        )
        self._make_setting_slider(
            tab,
            "Poll rate (Hz)",
            "poll_rate_hz",
            50,
            1000,
            integer=True,
            help_text="200 Hz is plenty. Higher rates burn CPU for no real gain.",
        )

        # Live demo
        ttk.Separator(tab, orient="horizontal").pack(fill="x", pady=12)
        ttk.Label(tab, text="Live debounce demo", font=("TkDefaultFont", 12, "bold")).pack(anchor="w")
        ttk.Label(
            tab,
            text="Wave your hand through the beam. Compare the raw pin " "(noisy) vs the stable state (clean).",
            foreground="#888",
        ).pack(anchor="w", pady=(0, 8))

        demo = ttk.Frame(tab)
        demo.pack(fill="x")
        self._demo_raw = ttk.Label(demo, text="RAW: —", width=20, font=("Courier", 12, "bold"))
        self._demo_raw.pack(side="left", padx=4)
        self._demo_stable = ttk.Label(demo, text="STABLE: —", width=20, font=("Courier", 12, "bold"))
        self._demo_stable.pack(side="left", padx=4)

    # --------------------------------------------------------------
    # Tab: Tests
    # --------------------------------------------------------------
    def _build_tab_tests(self, nb: ttk.Notebook) -> None:
        tab = ttk.Frame(nb, padding=12)
        nb.add(tab, text="Tests")

        ttk.Label(tab, text="Automated tests", font=("TkDefaultFont", 13, "bold")).pack(anchor="w")
        ttk.Label(
            tab,
            text="Press STOP to abort any running test.",
            foreground="#888",
        ).pack(anchor="w", pady=(0, 12))

        # Alignment
        align = ttk.LabelFrame(tab, text="Alignment helper", padding=10)
        align.pack(fill="x", pady=4)
        ttk.Label(
            align,
            text=(
                "Use during physical install. Beeps and turns the indicator "
                "GREEN whenever the beam is intact.\n"
                "Adjust the laser/receiver until the indicator stays solid "
                "green for 10s straight."
            ),
        ).pack(anchor="w")
        ttk.Button(
            align,
            text="▶ Start alignment helper",
            command=lambda: self._run_test(self._test_align),
        ).pack(side="left", pady=(8, 0))

        # Drift watch
        drift = ttk.LabelFrame(tab, text="Drift watch (60s)", padding=10)
        drift.pack(fill="x", pady=4)
        ttk.Label(
            drift,
            text=(
                "Logs every spurious break for 60 seconds with nobody "
                "touching the beam. Tells you if the sensor will trigger\n"
                "false wins during gameplay (gym vibrations, sunlight, "
                "fluorescent flicker, etc.)."
            ),
        ).pack(anchor="w")
        ttk.Button(
            drift,
            text="▶ Run drift watch (60s)",
            command=lambda: self._run_test(self._test_drift),
        ).pack(side="left", pady=(8, 0))

        # Latency
        lat = ttk.LabelFrame(tab, text="Latency probe", padding=10)
        lat.pack(fill="x", pady=4)
        ttk.Label(
            lat,
            text=(
                "Wait for a real beam break. Measures how long debounce "
                "takes from raw flip → stable flip. Repeats 10×.\n"
                "Use to verify your debounce settings produce <100ms latency."
            ),
        ).pack(anchor="w")
        ttk.Button(
            lat,
            text="▶ Latency probe (10 trips)",
            command=lambda: self._run_test(self._test_latency),
        ).pack(side="left", pady=(8, 0))

        # Stop
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

        self._test_log = tk.Text(tab, height=10, wrap="word", font=("Courier", 10), bg="#1e1e1e", fg="#d4d4d4")
        self._test_log.pack(fill="both", expand=True, pady=(8, 0))

    # --------------------------------------------------------------
    # Tab: Diagnostics
    # --------------------------------------------------------------
    def _build_tab_diagnostics(self, nb: ttk.Notebook) -> None:
        tab = ttk.Frame(nb, padding=12)
        nb.add(tab, text="Diagnostics")

        ttk.Label(tab, text="GPIO diagnostics", font=("TkDefaultFont", 13, "bold")).pack(anchor="w")
        ttk.Label(
            tab,
            text="Run all checks to confirm GPIO is set up correctly.",
            foreground="#888",
        ).pack(anchor="w", pady=(0, 8))

        ttk.Button(tab, text="🩺 Run all diagnostics", command=self._do_diagnostics).pack(anchor="w")

        self._diag_text = tk.Text(tab, wrap="word", height=20, font=("Courier", 10), bg="#1e1e1e", fg="#d4d4d4")
        self._diag_text.pack(fill="both", expand=True, pady=8)
        self._diag_text.insert("1.0", "Click 'Run all diagnostics' to start.\n")

    # --------------------------------------------------------------
    # Tab: Config
    # --------------------------------------------------------------
    def _build_tab_config(self, nb: ttk.Notebook) -> None:
        tab = ttk.Frame(nb, padding=12)
        nb.add(tab, text="Config")

        ttk.Label(tab, text="Drop into config.py", font=("TkDefaultFont", 13, "bold")).pack(anchor="w")
        ttk.Label(
            tab,
            text="Replace your existing laser settings in config.py with " "the block below.",
            foreground="#888",
        ).pack(anchor="w", pady=(0, 8))

        self._config_text = tk.Text(tab, height=14, wrap="none", font=("Courier", 11), bg="#1e1e1e", fg="#d4d4d4", insertbackground="white")
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
                "🔴 Indicator says BROKEN even with no obstruction",
                [
                    "Laser not aligned with the receiver — the beam needs "
                    "to land squarely on the photodetector. Use the "
                    "Alignment helper test.",
                    "Active polarity backwards — try toggling Active LOW vs " "Active HIGH in the Pin Setup tab.",
                    "Pull resistor wrong — most modules need PUD_UP. If " "yours has its own pull-up resistor on board, use PUD_OFF.",
                    "Sensor not powered — check 3.3V or 5V on the laser " "module and the receiver module.",
                    "Wrong pin — confirm physical wiring matches the " "configured pin number AND numbering mode (BOARD vs BCM).",
                ],
            ),
            (
                "🟢 Indicator says OK but should be BROKEN (no detection)",
                [
                    "Beam not actually hitting the receiver — visible light "
                    "lasers help here. If yours is IR-only, hold a phone "
                    "camera up to the receiver to see the dot.",
                    "Active polarity backwards — try the other Active " "LOW/HIGH option.",
                    "Receiver saturated by ambient light (sun, stage lights) " "— shield the receiver or move it.",
                    "Damaged sensor — try a different break-beam module.",
                ],
            ),
            (
                "💥 Random spurious breaks (false positives)",
                [
                    "Run Drift Watch test for 60 seconds with nobody near " "the beam. If it logs breaks, the sensor is unreliable.",
                    "Increase debounce samples to 5 or 6 — most short-pulse " "noise will get filtered.",
                    "Power supply noise — laser modules are noisy loads. " "Run them off a separate 5V supply, not the Jetson rail.",
                    "Long unshielded wiring — keep the receiver-to-Jetson " "wire under 1m and away from servo / motor wires.",
                    "Fluorescent / LED room lights flickering — visible "
                    "light leaks into the IR receiver. Use an IR-pass "
                    "filter on the receiver, or a focused beam.",
                ],
            ),
            (
                "🚪 Permission denied opening GPIO",
                [
                    "Run: `sudo usermod -aG gpio $USER` then reboot. The " "user needs to be in the gpio group.",
                    "Verify with `groups` — 'gpio' should be listed.",
                    "On JetPack 5+ you may also need to install: " "`sudo apt install python3-jetson-gpio`.",
                    "Last resort: run with sudo. Not recommended long-term " "but helpful for quick verification.",
                ],
            ),
            (
                "🔁 'A different mode has already been set'",
                [
                    "Another process is holding the GPIO library in BCM " "mode while you want BOARD (or vice versa).",
                    "Kill all Python processes: `pkill -9 python3`. Then " "re-run.",
                    "If running inside Docker — the GPIO state persists " "between container restarts on the host. Reboot the " "Jetson to clear it.",
                ],
            ),
            (
                "⏰ Game logic ignores beam crosses",
                [
                    "Confirm the indicator on this tool actually flips when "
                    "you cross the beam. If it doesn't, the problem is "
                    "hardware, not game code.",
                    "Game polls beam state at finite rate — if your "
                    "GameEngine.run loop is blocked by YOLO, beam "
                    "transitions <30ms can be missed. Increase debounce "
                    "window so transitions persist longer.",
                    "Check check_tape_finish in vision.py isn't masking " "the laser path with its own logic.",
                ],
            ),
            (
                "💀 Worked yesterday, doesn't work today",
                [
                    "First: Diagnostics tab → run all checks.",
                    "Receiver dust — give the photodiode a quick blow with " "compressed air.",
                    "Laser dimming — cheap laser modules degrade. If the " "visible dot looks weak, replace the module ($2 each).",
                    "Loose Dupont jumpers — the #1 cause of intermittent " "GPIO failures. Tug-test every connection.",
                    "User got removed from gpio group by a system update " "— re-add with usermod.",
                    "Pin damaged — Jetson GPIO pins are 3.3V tolerant, "
                    "not 5V. If you accidentally fed 5V into the input, "
                    "the pin can be permanently damaged. Try a different pin.",
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
    # Generic widget helpers
    # ------------------------------------------------------------------
    def _make_setting_slider(
        self,
        parent: ttk.Frame,
        label: str,
        key: str,
        min_val: float,
        max_val: float,
        integer: bool = False,
        help_text: str = "",
        on_change: Any = None,
    ) -> None:
        outer = ttk.Frame(parent)
        outer.pack(fill="x", pady=6)
        top = ttk.Frame(outer)
        top.pack(fill="x")
        ttk.Label(top, text=label, width=22, anchor="w").pack(side="left")

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

        if help_text:
            ttk.Label(outer, text=help_text, foreground="#888", font=("TkDefaultFont", 9)).pack(anchor="w")

    def _make_dropdown(
        self,
        parent: ttk.Frame,
        label: str,
        key: str,
        options: dict[str, str],
        help_text: str = "",
    ) -> None:
        outer = ttk.Frame(parent)
        outer.pack(fill="x", pady=6)
        top = ttk.Frame(outer)
        top.pack(fill="x")
        ttk.Label(top, text=label, width=22, anchor="w").pack(side="left")

        items = [f"{k}  -  {v}" for k, v in options.items()]
        current = self.settings[key]
        var = tk.StringVar(value=f"{current}  -  {options.get(current, '?')}")

        def on_change(_event: Any = None) -> None:
            sel = combo.get()
            new_key = sel.split("  -  ")[0]
            self.settings[key] = new_key
            self._refresh_config_snippet()

        combo = ttk.Combobox(top, textvariable=var, values=items, state="readonly", width=42)
        combo.pack(side="right")
        combo.bind("<<ComboboxSelected>>", on_change)

        if help_text:
            ttk.Label(outer, text=help_text, foreground="#888", font=("TkDefaultFont", 9)).pack(anchor="w")

    @staticmethod
    def _fmt(val: Any, integer: bool) -> str:
        if integer:
            return str(int(val))
        return f"{float(val):.3f}"

    # ------------------------------------------------------------------
    # Settings change callbacks
    # ------------------------------------------------------------------
    def _on_active_change(self) -> None:
        self.settings["active_low"] = bool(self._active_var.get())
        self._refresh_config_snippet()

    def _on_debounce_change(self, val: int) -> None:
        self._sample_buffer = deque(self._sample_buffer, maxlen=max(1, int(val)))

    def _reapply_setup(self) -> None:
        ok, msg = self.backend.setup(
            int(self.settings["pin"]),
            self.settings["numbering"],
            self.settings["pull"],
        )
        self._setup_status = (ok, msg)
        self._setup_label.config(text=msg, foreground="#16a34a" if ok else "#dc2626")
        self._reset_counters()

    def _reset_counters(self) -> None:
        self._break_count = 0
        self._last_break_ts = None
        self._false_positive_window.clear()
        self._latency_samples.clear()
        self._history.clear()

    # ------------------------------------------------------------------
    # Polling thread
    # ------------------------------------------------------------------
    def _poll_loop(self) -> None:
        last_stable = self._is_broken
        last_change_t: float | None = None

        while self._poll_running:
            interval = 1.0 / max(1, int(self.settings["poll_rate_hz"]))
            t = time.time()

            raw = self.backend.read()
            self._raw_value = raw

            broken_now = (raw == 0) if self.settings["active_low"] else (raw == 1)

            # Debounce: append, check whether buffer is uniform
            self._sample_buffer.append(1 if broken_now else 0)
            if len(self._sample_buffer) == self._sample_buffer.maxlen:
                if all(x == 1 for x in self._sample_buffer):
                    new_stable = True
                elif all(x == 0 for x in self._sample_buffer):
                    new_stable = False
                else:
                    new_stable = self._is_broken  # ambiguous, keep last
            else:
                new_stable = self._is_broken

            if new_stable != self._is_broken:
                self._is_broken = new_stable
                if new_stable:
                    self._break_count += 1
                    self._last_break_ts = t
                    self._false_positive_window.append(t)
                    if last_change_t is not None:
                        self._latency_samples.append(t - last_change_t)
                last_change_t = t
                last_stable = new_stable

            # History: append every poll
            self._history.append((t, self._is_broken))

            # Wait until next sample window for debouncer accuracy
            time.sleep(max(interval, self.settings["debounce_window_ms"] / 1000.0 / max(1, int(self.settings["debounce_samples"]))))

    # ------------------------------------------------------------------
    # UI tick (Tk main thread)
    # ------------------------------------------------------------------
    def _tick_ui(self) -> None:
        # Readouts
        raw_str = f"{self._raw_value} ({'HIGH' if self._raw_value else 'LOW'})"
        self.r_raw.config(text=raw_str)
        self.r_stable.config(
            text="🔴 BROKEN" if self._is_broken else "🟢 OK",
            foreground="#dc2626" if self._is_broken else "#16a34a",
        )
        self.r_breaks.config(text=str(self._break_count))
        if self._last_break_ts is None:
            self.r_last.config(text="—")
        else:
            self.r_last.config(text=f"{time.time() - self._last_break_ts:.1f}")

        # Demo readouts (same data, different layout)
        if hasattr(self, "_demo_raw"):
            self._demo_raw.config(
                text=f"RAW: {'BROKEN' if (self._raw_value == 0) == self.settings['active_low'] else 'OK'}",
                foreground="#dc2626" if (self._raw_value == 0) == self.settings["active_low"] else "#16a34a",
            )
            self._demo_stable.config(
                text=f"STABLE: {'BROKEN' if self._is_broken else 'OK'}",
                foreground="#dc2626" if self._is_broken else "#16a34a",
            )

        self._draw_indicator()
        self._draw_history()
        self.root.after(50, self._tick_ui)

    def _draw_indicator(self) -> None:
        c = self.indicator
        c.delete("all")
        cx = self.INDICATOR_W // 2
        cy = self.INDICATOR_H // 2

        if self._is_broken:
            color = "#dc2626"
            label = "BEAM BROKEN"
            sublabel = "(player crossing)"
            outer = "#7f1d1d"
        else:
            color = "#16a34a"
            label = "BEAM OK"
            sublabel = "(no obstruction)"
            outer = "#14532d"

        # Outer pulsing ring
        pulse = 0.5 + 0.5 * abs((time.time() % 1.5) / 1.5 - 0.5) * 2
        ring_r = int(70 + 8 * pulse)
        c.create_oval(cx - ring_r, cy - ring_r, cx + ring_r, cy + ring_r, outline=outer, width=3)

        # Solid centre dot
        c.create_oval(cx - 60, cy - 60, cx + 60, cy + 60, fill=color, outline="")

        # Beam representation: two rectangles + dashed line
        c.create_rectangle(40, cy - 8, 90, cy + 8, fill="#9ca3af", outline="")  # laser
        c.create_rectangle(self.INDICATOR_W - 90, cy - 8, self.INDICATOR_W - 40, cy + 8, fill="#9ca3af", outline="")  # receiver
        if self._is_broken:
            c.create_line(95, cy, cx - 60, cy, fill="#dc2626", width=3, dash=(8, 4))
        else:
            c.create_line(95, cy, self.INDICATOR_W - 95, cy, fill="#fbbf24", width=3)

        # Label
        c.create_text(cx, cy, text=label, fill="white", font=("TkDefaultFont", 13, "bold"))
        c.create_text(cx, cy + 22, text=sublabel, fill="white", font=("TkDefaultFont", 9))

    def _draw_history(self) -> None:
        c = self.history_canvas
        c.delete("all")

        now = time.time()
        cutoff = now - self.HISTORY_SECONDS
        # Background grid
        for i in range(self.HISTORY_SECONDS + 1):
            x = int(i * self.HISTORY_W / self.HISTORY_SECONDS)
            c.create_line(x, 0, x, self.HISTORY_H, fill="#1e1e2e", width=1)

        # Horizontal centre line
        mid = self.HISTORY_H // 2
        c.create_line(0, mid, self.HISTORY_W, mid, fill="#3f3f55", width=1)

        # Plot — broken=top half (red), ok=bottom half (green)
        prev_x = None
        prev_state = None
        for ts, broken in self._history:
            if ts < cutoff:
                continue
            x = int((ts - cutoff) / self.HISTORY_SECONDS * self.HISTORY_W)
            y_top = 8 if broken else mid + 4
            y_bot = mid - 4 if broken else self.HISTORY_H - 8
            color = "#dc2626" if broken else "#16a34a"
            c.create_rectangle(x - 1, y_top, x + 2, y_bot, fill=color, outline="")
            prev_x, prev_state = x, broken

        # Labels
        c.create_text(8, 12, text="BROKEN", fill="#dc2626", anchor="nw", font=("TkDefaultFont", 8))
        c.create_text(8, self.HISTORY_H - 14, text="OK", fill="#16a34a", anchor="nw", font=("TkDefaultFont", 8))
        c.create_text(self.HISTORY_W - 4, self.HISTORY_H - 4, text="now", fill="#666", anchor="se", font=("TkDefaultFont", 8))

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------
    def _run_test(self, target: Any) -> None:
        if self._test_running:
            self._test_status.config(text="A test is already running. Stop it first.", foreground="#dc2626")
            return
        self._test_running = True
        self._test_log.delete("1.0", "end")
        self._test_thread = threading.Thread(target=target, daemon=True, name="test")
        self._test_thread.start()

    def _stop_test(self) -> None:
        self._test_running = False
        self._log("Test stopped by user.")

    def _log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self._test_log.insert("end", f"[{ts}] {msg}\n")
        self._test_log.see("end")

    def _test_align(self) -> None:
        self._test_status.config(text="Alignment helper running.", foreground="#16a34a")
        self._log("Adjust laser/receiver. Goal: solid green for 10s straight.")
        last_state = self._is_broken
        ok_streak_start = time.time() if not last_state else None
        try:
            while self._test_running:
                if self._is_broken != last_state:
                    if self._is_broken:
                        self._log("✗ Beam lost.")
                        ok_streak_start = None
                    else:
                        self._log("✓ Beam restored.")
                        ok_streak_start = time.time()
                    last_state = self._is_broken

                if ok_streak_start is not None:
                    held = time.time() - ok_streak_start
                    if held >= 10.0:
                        self._log("✓✓✓ 10s solid — alignment is good. Done.")
                        break
                time.sleep(0.05)
        finally:
            self._test_running = False
            self._test_status.config(text="Alignment helper stopped.", foreground="#666")

    def _test_drift(self) -> None:
        self._test_status.config(text="Drift watch running (60s).", foreground="#16a34a")
        self._log("Stay clear of the beam for 60s. Logging spurious breaks.")
        end = time.time() + 60.0
        spurious = 0
        last_break_count = self._break_count
        try:
            while self._test_running and time.time() < end:
                if self._break_count > last_break_count:
                    spurious += self._break_count - last_break_count
                    self._log(f"⚠ Spurious break #{spurious}")
                    last_break_count = self._break_count
                time.sleep(0.05)
            remaining = max(0, end - time.time())
            self._log(f"=== DONE === Spurious breaks: {spurious} in 60s.")
            if spurious == 0:
                self._log("✓ Sensor is rock solid.")
            elif spurious <= 2:
                self._log("⚠ A few false positives — bump debounce_samples.")
            else:
                self._log("✗ Too noisy. See troubleshooting tab.")
        finally:
            self._test_running = False
            self._test_status.config(text="Drift watch complete.", foreground="#666")

    def _test_latency(self) -> None:
        self._test_status.config(text="Latency probe running.", foreground="#16a34a")
        self._log("Walk through the beam 10 times. Logging debounce latency.")
        last_count = self._break_count
        last_samples_count = len(self._latency_samples)
        trips = 0
        try:
            while self._test_running and trips < 10:
                if self._break_count > last_count:
                    trips = self._break_count - last_count
                    if len(self._latency_samples) > last_samples_count:
                        new_samples = self._latency_samples[last_samples_count:]
                        for s in new_samples:
                            self._log(f"Trip #{trips}: latency = {s*1000:.0f}ms")
                        last_samples_count = len(self._latency_samples)
                time.sleep(0.05)
            if self._latency_samples:
                avg = sum(self._latency_samples[-10:]) / min(10, len(self._latency_samples))
                self._log(f"=== DONE === Average latency: {avg*1000:.0f}ms")
                if avg < 0.1:
                    self._log("✓ Latency is fine for game use.")
                else:
                    self._log("⚠ Reduce debounce_samples or window for snappier response.")
        finally:
            self._test_running = False
            self._test_status.config(text="Latency probe complete.", foreground="#666")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def _do_diagnostics(self) -> None:
        out = self._diag_text
        out.delete("1.0", "end")

        def write(line: str) -> None:
            out.insert("end", line + "\n")
            out.see("end")
            out.update_idletasks()

        write("Running diagnostics…\n")

        # 1. Module check
        ok, msg = gpio_module_check()
        write(f"[1/4] Jetson.GPIO module: {'✓' if ok else '✗'} {msg}")

        # 2. Permission check
        ok, msg = gpio_permission_check()
        write(f"[2/4] User permissions: {'✓' if ok else '✗'} {msg}")

        # 3. /dev/gpiochip*
        ok, chips, raw = list_gpiochip_devices()
        if ok:
            write(f"[3/4] gpiochip devices: ✓ Found {len(chips)} device(s):")
            for c in chips:
                write(f"        {c.strip()}")
        else:
            write(f"[3/4] gpiochip devices: ✗ {raw}")

        # 4. Read current state
        write(f"[4/4] Current pin state:")
        write(f"        Pin: {self.settings['pin']} ({self.settings['numbering']})")
        write(f"        Pull: {self.settings['pull']}")
        write(f"        Active: {'LOW' if self.settings['active_low'] else 'HIGH'}")
        write(f"        Raw read: {self._raw_value} " f"({'HIGH' if self._raw_value else 'LOW'})")
        write(f"        Beam: {'BROKEN' if self._is_broken else 'OK'}")
        write("")
        write("If everything is ✓ but the beam still doesn't behave right,")
        write("check the Troubleshooting tab.")

    # ------------------------------------------------------------------
    # Config
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
            initialfile="laser_preset.json",
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
        for widget in self.root.winfo_children():
            widget.destroy()
        self._build_ui()
        self._refresh_config_snippet()
        self._reapply_setup()

    def _reset_defaults(self) -> None:
        self.settings = DEFAULTS.copy()
        for widget in self.root.winfo_children():
            widget.destroy()
        self._build_ui()
        self._refresh_config_snippet()
        self._reapply_setup()

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    def _on_close(self) -> None:
        self._poll_running = False
        self._test_running = False
        try:
            self._poll_thread.join(timeout=1.0)
        except Exception:
            pass
        self.backend.cleanup()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


# ===========================================================================
# Entry point
# ===========================================================================
def main() -> int:
    if platform.system() != "Linux":
        print("[INFO] Not on Linux — Jetson.GPIO won't be available, " "running in SIMULATION mode for UI testing.")
    tuner = LaserTuner()
    tuner.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
