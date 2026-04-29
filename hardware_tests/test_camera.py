#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Project:      Red Light Green Light (Jetson Orin Nano)
File:         test_camera.py
Description:  Isolated hardware test for the IMX219 CSI Camera using GStreamer.

Author:       Richard Pu
Last Updated: April 2026
===============================================================================
"""

from __future__ import annotations

import json
import os
import platform
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

import cv2  # type: ignore
import numpy as np

try:
    from PIL import Image, ImageTk

    _PIL_OK = True
except ImportError:
    _PIL_OK = False
    print("[ERR] Pillow not installed. Run: pip install pillow")
    sys.exit(1)


# ===========================================================================
# Defaults — match your current hardware.py
# ===========================================================================
DEFAULTS: dict[str, Any] = {
    # Capture geometry
    "capture_w": 1280,
    "capture_h": 720,
    "fps": 60,
    "output_w": 1920,
    "output_h": 1080,
    "flip_method": 2,
    # White balance & colour
    "wbmode": 1,
    "saturation": 1.40,
    "exposurecompensation": 0.0,
    # Noise reduction & sharpness
    "tnr_mode": 2,
    "tnr_strength": 0.7,
    "ee_mode": 0,
    "ee_strength": 0.0,
    # Gain & exposure
    "gain_min": 1.0,
    "gain_max": 8.0,
    "isp_digital_gain_min": 1.0,
    "isp_digital_gain_max": 1.0,
    "exposure_min_us": 13000,
    "exposure_max_us": 16000000,
    # Anti-banding (fluorescent flicker)
    "aeantibanding": 1,
    # Locks
    "aelock": False,
    "awblock": False,
}


# Enum value → human label
WB_MODES = {
    0: "off (raw)",
    1: "auto",
    2: "incandescent (warm bulbs)",
    3: "fluorescent (gym lights)",
    4: "warm fluorescent",
    5: "daylight",
    6: "cloudy daylight",
    7: "twilight",
    8: "shade",
    9: "manual",
}

TNR_MODES = {0: "off", 1: "fast", 2: "high quality"}
EE_MODES = {0: "off", 1: "fast", 2: "high quality"}
ANTIBANDING = {0: "off", 1: "auto", 2: "50Hz", 3: "60Hz"}
FLIP_METHODS = {
    0: "none",
    1: "ccw 90°",
    2: "rotate 180°",
    3: "cw 90°",
    4: "horizontal flip",
    5: "upper-right ↔ lower-left",
    6: "vertical flip",
    7: "upper-left ↔ lower-right",
}


# ===========================================================================
# Pipeline builder
# ===========================================================================
def build_pipeline(s: dict[str, Any]) -> str:
    """Return the gst-launch-style pipeline string for the given settings."""
    aelock = "true" if s["aelock"] else "false"
    awblock = "true" if s["awblock"] else "false"

    return (
        f"nvarguscamerasrc "
        f"wbmode={int(s['wbmode'])} "
        f"saturation={s['saturation']:.2f} "
        f"exposurecompensation={s['exposurecompensation']:.2f} "
        f"tnr-mode={int(s['tnr_mode'])} "
        f"tnr-strength={s['tnr_strength']:.2f} "
        f"ee-mode={int(s['ee_mode'])} "
        f"ee-strength={s['ee_strength']:.2f} "
        f'gainrange="{s["gain_min"]:.1f} {s["gain_max"]:.1f}" '
        f'ispdigitalgainrange="{s["isp_digital_gain_min"]:.1f} {s["isp_digital_gain_max"]:.1f}" '
        f'exposuretimerange="{int(s["exposure_min_us"])} {int(s["exposure_max_us"])}" '
        f"aeantibanding={int(s['aeantibanding'])} "
        f"aelock={aelock} awblock={awblock} "
        f"! video/x-raw(memory:NVMM), width={int(s['capture_w'])}, "
        f"height={int(s['capture_h'])}, format=NV12, framerate={int(s['fps'])}/1 "
        f"! nvvidconv flip-method={int(s['flip_method'])} "
        f"! video/x-raw, width={int(s['output_w'])}, "
        f"height={int(s['output_h'])}, format=BGRx "
        f"! videoconvert ! video/x-raw, format=BGR "
        f"! appsink drop=true max-buffers=1"
    )


def build_hardware_py_snippet(s: dict[str, Any]) -> str:
    """Generate a drop-in replacement for _gstreamer_pipeline()."""
    aelock = "true" if s["aelock"] else "false"
    awblock = "true" if s["awblock"] else "false"

    return f'''def _gstreamer_pipeline(
    capture_w: int = CAM_W,
    capture_h: int = CAM_H,
    output_w: int = 1920,
    output_h: int = 1080,
    fps: int = CAM_FPS,
) -> str:
    """Jetson CSI pipeline — tuned via camera_tuner.py on {time.strftime("%Y-%m-%d")}."""
    return (
        f"nvarguscamerasrc "
        f"wbmode={int(s['wbmode'])} "
        f"saturation={s['saturation']:.2f} "
        f"exposurecompensation={s['exposurecompensation']:.2f} "
        f"tnr-mode={int(s['tnr_mode'])} tnr-strength={s['tnr_strength']:.2f} "
        f"ee-mode={int(s['ee_mode'])} ee-strength={s['ee_strength']:.2f} "
        f\'gainrange=\\"{s["gain_min"]:.1f} {s["gain_max"]:.1f}\\" \'
        f\'ispdigitalgainrange=\\"{s["isp_digital_gain_min"]:.1f} {s["isp_digital_gain_max"]:.1f}\\" \'
        f\'exposuretimerange=\\"{int(s["exposure_min_us"])} {int(s["exposure_max_us"])}\\" \'
        f"aeantibanding={int(s['aeantibanding'])} "
        f"aelock={aelock} awblock={awblock} "
        f"! video/x-raw(memory:NVMM), width={{capture_w}}, height={{capture_h}}, "
        f"format=NV12, framerate={{fps}}/1 ! "
        f"nvvidconv flip-method={int(s['flip_method'])} ! "
        f"video/x-raw, width={{output_w}}, height={{output_h}}, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! "
        f"appsink drop=true max-buffers=1"
    )
'''


# ===========================================================================
# Main app
# ===========================================================================
class CameraTuner:
    PREVIEW_W = 720
    PREVIEW_H = 405

    def __init__(self) -> None:
        self.settings: dict[str, Any] = DEFAULTS.copy()

        self.cap: cv2.VideoCapture | None = None  # type: ignore[name-defined]
        self._cap_lock = threading.Lock()
        self._frame_lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._capture_thread: threading.Thread | None = None
        self._running = False

        # FPS tracking
        self._fps = 0.0
        self._fps_count = 0
        self._fps_ts = time.time()

        # Tk
        self.root = tk.Tk()
        self.root.title("nvarguscamerasrc Live Tuner — jetson-rlgl")
        self.root.geometry("1500x900")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()
        self._update_pipeline_display()
        self._start_capture()
        self._tick_preview()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        # Two-column root: preview left, controls right.
        root_pane = ttk.PanedWindow(self.root, orient="horizontal")
        root_pane.pack(fill="both", expand=True)

        # ── LEFT: preview + pipeline display ─────────────────────────
        left = ttk.Frame(root_pane, padding=10)
        root_pane.add(left, weight=1)

        ttk.Label(left, text="Live Preview", font=("TkDefaultFont", 14, "bold")).pack(anchor="w")
        self.preview_canvas = tk.Canvas(
            left,
            width=self.PREVIEW_W,
            height=self.PREVIEW_H,
            bg="black",
            highlightthickness=0,
        )
        self.preview_canvas.pack(pady=(4, 8))
        self._preview_image_id = self.preview_canvas.create_image(0, 0, anchor="nw")

        # Status bar
        status_frame = ttk.Frame(left)
        status_frame.pack(fill="x")
        self.status_label = ttk.Label(status_frame, text="Initialising…", foreground="#888")
        self.status_label.pack(side="left")
        self.fps_label = ttk.Label(status_frame, text="-- fps", foreground="#0a0")
        self.fps_label.pack(side="right")

        # Pipeline string
        ttk.Label(left, text="Current pipeline string", font=("TkDefaultFont", 11, "bold")).pack(anchor="w", pady=(12, 2))
        self.pipeline_text = tk.Text(left, height=10, wrap="word", font=("Courier", 9), bg="#1e1e1e", fg="#d4d4d4")
        self.pipeline_text.pack(fill="both", expand=False)

        # Action buttons
        btn_row = ttk.Frame(left)
        btn_row.pack(fill="x", pady=(8, 0))
        ttk.Button(btn_row, text="🔄 Apply (restart camera)", command=self._apply).pack(side="left", padx=2)
        ttk.Button(btn_row, text="📋 Show hardware.py snippet", command=self._show_snippet).pack(side="left", padx=2)
        ttk.Button(btn_row, text="💾 Save preset", command=self._save_preset).pack(side="left", padx=2)
        ttk.Button(btn_row, text="📂 Load preset", command=self._load_preset).pack(side="left", padx=2)
        ttk.Button(btn_row, text="↺ Reset", command=self._reset_defaults).pack(side="left", padx=2)

        # ── RIGHT: tabbed control panel ──────────────────────────────
        right = ttk.Frame(root_pane, padding=10)
        root_pane.add(right, weight=1)

        nb = ttk.Notebook(right)
        nb.pack(fill="both", expand=True)

        # Tab: Colour & White Balance
        tab_color = ttk.Frame(nb, padding=10)
        nb.add(tab_color, text="Colour & WB")
        self._make_dropdown(tab_color, "White balance mode", "wbmode", WB_MODES)
        self._make_slider(tab_color, "Saturation", "saturation", 0.0, 2.0, 0.05, "0=greyscale, 1=neutral, 1.4=pop, 2=cartoonish")
        self._make_slider(tab_color, "Exposure compensation", "exposurecompensation", -2.0, 2.0, 0.1, "Brighten / darken without re-noising")
        self._make_dropdown(tab_color, "Anti-banding", "aeantibanding", ANTIBANDING, help_text="50Hz for non-US fluorescents, 60Hz for US")
        self._make_checkbox(tab_color, "Lock auto white balance (awblock)", "awblock")

        # Tab: Exposure & Gain
        tab_exp = ttk.Frame(nb, padding=10)
        nb.add(tab_exp, text="Exposure & Gain")
        self._make_slider(tab_exp, "Analog gain MIN", "gain_min", 1.0, 16.0, 0.5, "Sensor gain floor — keep at 1")
        self._make_slider(tab_exp, "Analog gain MAX", "gain_max", 1.0, 16.0, 0.5, "Cap before sensor noise dominates (8 = clean)")
        self._make_slider(tab_exp, "ISP digital gain MIN", "isp_digital_gain_min", 1.0, 8.0, 0.5, "Keep at 1")
        self._make_slider(tab_exp, "ISP digital gain MAX", "isp_digital_gain_max", 1.0, 8.0, 0.5, "Cap at 1 to kill amplification noise")
        self._make_slider(tab_exp, "Exposure MIN (µs)", "exposure_min_us", 1000, 50000, 1000)
        self._make_slider(tab_exp, "Exposure MAX (µs)", "exposure_max_us", 100000, 33000000, 100000, "Max for 60fps ≈ 16000000 (16ms)")
        self._make_checkbox(tab_exp, "Lock auto-exposure (aelock)", "aelock")

        # Tab: Noise & Sharpness
        tab_nr = ttk.Frame(nb, padding=10)
        nb.add(tab_nr, text="Noise & Sharpness")
        self._make_dropdown(tab_nr, "Temporal NR mode", "tnr_mode", TNR_MODES, help_text="2=high quality is fine, 0=off if motion artifacts")
        self._make_slider(tab_nr, "TNR strength", "tnr_strength", -1.0, 1.0, 0.05, "0.7 sweet spot. Past 0.85 → motion smudge.")
        self._make_dropdown(tab_nr, "Edge enhance mode", "ee_mode", EE_MODES, help_text="0=off; EE amplifies noise, leave off in low light")
        self._make_slider(tab_nr, "EE strength", "ee_strength", -1.0, 1.0, 0.05)

        # Tab: Capture geometry
        tab_geo = ttk.Frame(nb, padding=10)
        nb.add(tab_geo, text="Geometry")
        self._make_slider(tab_geo, "Capture width", "capture_w", 640, 3280, 160, integer=True)
        self._make_slider(tab_geo, "Capture height", "capture_h", 480, 2464, 80, integer=True)
        self._make_slider(
            tab_geo, "Capture FPS", "fps", 15, 120, 1, integer=True, help_text="60 for game; reduce to 30 if exposure_max needs to be longer"
        )
        self._make_slider(tab_geo, "Output width", "output_w", 640, 1920, 160, integer=True)
        self._make_slider(tab_geo, "Output height", "output_h", 480, 1080, 90, integer=True)
        self._make_dropdown(tab_geo, "Flip method", "flip_method", FLIP_METHODS, help_text="2=rotate 180° matches your current setup")

    def _make_slider(
        self,
        parent: ttk.Frame,
        label: str,
        key: str,
        min_val: float,
        max_val: float,
        resolution: float = 0.01,
        help_text: str = "",
        integer: bool = False,
    ) -> None:
        """Labeled slider that updates self.settings[key] on drag, and triggers
        a camera restart on release."""
        outer = ttk.Frame(parent)
        outer.pack(fill="x", pady=(6, 2))

        top = ttk.Frame(outer)
        top.pack(fill="x")
        ttk.Label(top, text=label, width=24, anchor="w").pack(side="left")

        var = tk.DoubleVar(value=float(self.settings[key]))
        val_text = tk.StringVar(value=self._fmt(self.settings[key], integer))
        ttk.Label(top, textvariable=val_text, width=12, anchor="e", font=("Courier", 10, "bold")).pack(side="right")

        def on_change(v: str) -> None:
            val = float(v)
            if integer:
                val = int(round(val))
            self.settings[key] = val
            val_text.set(self._fmt(val, integer))
            self._update_pipeline_display()

        scale = ttk.Scale(outer, from_=min_val, to=max_val, variable=var, command=on_change, orient="horizontal")
        scale.pack(fill="x", pady=(2, 0))
        scale.bind("<ButtonRelease-1>", lambda e: self._apply())

        if help_text:
            ttk.Label(outer, text=help_text, foreground="#888", font=("TkDefaultFont", 9)).pack(anchor="w")

    def _make_dropdown(
        self,
        parent: ttk.Frame,
        label: str,
        key: str,
        options: dict[int, str],
        help_text: str = "",
    ) -> None:
        outer = ttk.Frame(parent)
        outer.pack(fill="x", pady=(6, 2))

        top = ttk.Frame(outer)
        top.pack(fill="x")
        ttk.Label(top, text=label, width=24, anchor="w").pack(side="left")

        items = [f"{k}  -  {v}" for k, v in options.items()]
        current = self.settings[key]
        var = tk.StringVar(value=f"{current}  -  {options.get(current, '?')}")

        def on_change(_event: Any = None) -> None:
            sel = combo.get()
            n = int(sel.split("  -  ")[0])
            self.settings[key] = n
            self._update_pipeline_display()
            self._apply()

        combo = ttk.Combobox(top, textvariable=var, values=items, state="readonly", width=28)
        combo.pack(side="right")
        combo.bind("<<ComboboxSelected>>", on_change)

        if help_text:
            ttk.Label(outer, text=help_text, foreground="#888", font=("TkDefaultFont", 9)).pack(anchor="w")

    def _make_checkbox(self, parent: ttk.Frame, label: str, key: str) -> None:
        var = tk.BooleanVar(value=bool(self.settings[key]))

        def on_change() -> None:
            self.settings[key] = bool(var.get())
            self._update_pipeline_display()
            self._apply()

        ttk.Checkbutton(parent, text=label, variable=var, command=on_change).pack(anchor="w", pady=(6, 2))

    @staticmethod
    def _fmt(val: Any, integer: bool) -> str:
        if integer:
            return str(int(val))
        return f"{float(val):.2f}"

    # ------------------------------------------------------------------
    # Capture lifecycle
    # ------------------------------------------------------------------
    def _start_capture(self) -> None:
        with self._cap_lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass

            pipeline = build_pipeline(self.settings)

            # Try GStreamer first (Jetson)
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)  # type: ignore[attr-defined]
            backend = "gstreamer"

            if not cap.isOpened():
                print("[WARN] GStreamer pipeline failed — falling back to webcam 0")
                cap = cv2.VideoCapture(0)  # type: ignore[attr-defined]
                backend = "fallback (webcam 0)"

            self.cap = cap
            self.status_label.config(
                text=f"Camera: {backend}  |  " f"{self.settings['capture_w']}×{self.settings['capture_h']}@" f"{self.settings['fps']}fps",
            )

        self._running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True, name="capture")
        self._capture_thread.start()

    def _capture_loop(self) -> None:
        while self._running:
            with self._cap_lock:
                cap = self.cap
            if cap is None:
                time.sleep(0.05)
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            with self._frame_lock:
                self._latest_frame = frame

            self._fps_count += 1
            now = time.time()
            if now - self._fps_ts >= 0.5:
                self._fps = self._fps_count / (now - self._fps_ts)
                self._fps_count = 0
                self._fps_ts = now

    def _stop_capture(self) -> None:
        self._running = False
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=1.5)
        with self._cap_lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

    def _apply(self) -> None:
        """Restart the camera with the current settings."""
        self.status_label.config(text="Restarting camera…")
        self.root.update_idletasks()
        self._stop_capture()
        self._start_capture()

    # ------------------------------------------------------------------
    # Preview ticker (runs on Tk main thread)
    # ------------------------------------------------------------------
    def _tick_preview(self) -> None:
        with self._frame_lock:
            frame = self._latest_frame.copy() if self._latest_frame is not None else None

        if frame is not None:
            disp = cv2.resize(frame, (self.PREVIEW_W, self.PREVIEW_H), interpolation=cv2.INTER_AREA)  # type: ignore[attr-defined]
            disp = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)  # type: ignore[attr-defined]
            img = Image.fromarray(disp)
            self._tk_img = ImageTk.PhotoImage(img)
            self.preview_canvas.itemconfig(self._preview_image_id, image=self._tk_img)

        self.fps_label.config(text=f"{self._fps:5.1f} fps")
        self.root.after(33, self._tick_preview)  # ~30Hz UI

    # ------------------------------------------------------------------
    # Pipeline display + snippet
    # ------------------------------------------------------------------
    def _update_pipeline_display(self) -> None:
        pipeline = build_pipeline(self.settings)
        # Pretty-print: break on " ! " for readability
        pretty = pipeline.replace(" ! ", " !\n    ")
        self.pipeline_text.delete("1.0", "end")
        self.pipeline_text.insert("1.0", pretty)

    def _show_snippet(self) -> None:
        """Open a window with the hardware.py snippet ready to copy."""
        win = tk.Toplevel(self.root)
        win.title("hardware.py snippet — copy this into your code")
        win.geometry("900x600")

        ttk.Label(
            win,
            text="Replace _gstreamer_pipeline() in hardware.py with this:",
            font=("TkDefaultFont", 11, "bold"),
        ).pack(anchor="w", padx=10, pady=(10, 4))

        text = tk.Text(win, wrap="none", font=("Courier", 10), bg="#1e1e1e", fg="#d4d4d4", insertbackground="white")
        text.pack(fill="both", expand=True, padx=10, pady=4)

        snippet = build_hardware_py_snippet(self.settings)
        text.insert("1.0", snippet)

        btn_row = ttk.Frame(win)
        btn_row.pack(fill="x", padx=10, pady=8)

        def copy_to_clipboard() -> None:
            self.root.clipboard_clear()
            self.root.clipboard_append(snippet)
            self.root.update()
            messagebox.showinfo("Copied", "Snippet copied to clipboard!")

        ttk.Button(btn_row, text="📋 Copy to clipboard", command=copy_to_clipboard).pack(side="left")
        ttk.Button(btn_row, text="Close", command=win.destroy).pack(side="right")

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------
    def _save_preset(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON preset", "*.json")],
            initialfile="camera_preset.json",
        )
        if not path:
            return
        with open(path, "w") as f:
            json.dump(self.settings, f, indent=2)
        messagebox.showinfo("Saved", f"Preset saved to:\n{path}")

    def _load_preset(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("JSON preset", "*.json")],
        )
        if not path:
            return
        try:
            with open(path) as f:
                loaded = json.load(f)
            # Merge with defaults so old presets still work after schema changes
            merged = DEFAULTS.copy()
            merged.update({k: v for k, v in loaded.items() if k in DEFAULTS})
            self.settings = merged
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))
            return
        messagebox.showinfo(
            "Loaded",
            "Preset loaded. Close this window — the UI needs a restart " "to reflect the new values in the sliders.",
        )
        # Quick & dirty: rebuild whole UI
        for widget in self.root.winfo_children():
            widget.destroy()
        self._build_ui()
        self._update_pipeline_display()
        self._apply()

    def _reset_defaults(self) -> None:
        self.settings = DEFAULTS.copy()
        for widget in self.root.winfo_children():
            widget.destroy()
        self._build_ui()
        self._update_pipeline_display()
        self._apply()

    # ------------------------------------------------------------------
    # Main loop / shutdown
    # ------------------------------------------------------------------
    def _on_close(self) -> None:
        self._stop_capture()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


# ===========================================================================
# Entry point
# ===========================================================================
def main() -> int:
    if platform.system() == "Windows":
        print("[INFO] Running on Windows — GStreamer pipeline will fail and " "the script will fall back to webcam 0 for UI testing.")

    tuner = CameraTuner()
    tuner.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
