#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Project:      Red Light Green Light (Jetson Orin Nano)
File:         test_lines.py
Description:  Isolated hardware test and HSV Tuner for Tape Detection.
              Provides a GUI to isolate Start/Finish line colors and generates
              the configuration snippet for config.py.
              Uses contour mapping to accurately detect the tilt and angle
              of floor tape.

Author:       Richard Pu (Updated)
Last Updated: April 2026
===============================================================================
"""

from __future__ import annotations

import json
import math
import os
import platform
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Tuple, Optional, Dict

import cv2  # type: ignore
import numpy as np

try:
    from PIL import Image, ImageTk
    _PIL_OK = True
except ImportError:
    _PIL_OK = False
    print("[ERR] Pillow not installed. Run: pip install pillow")
    sys.exit(1)

# Attempt to import your actual camera, fallback to standard OpenCV if it fails
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from hardware import Camera
    HAS_HARDWARE = True
except ImportError:
    HAS_HARDWARE = False


# ===========================================================================
# Defaults
# ===========================================================================
DEFAULTS: dict[str, Any] = {
    "target_line": "START",  # 'START' or 'FINISH'
    "h_min": 35,
    "s_min": 50,
    "v_min": 50,
    "h_max": 85,
    "s_max": 255,
    "v_max": 255,
    "min_area": 500,  # Used to filter out small color noise
}


# ===========================================================================
# Hardware Abstraction / Video Backend
# ===========================================================================
class VisionBackend:
    def __init__(self, settings: dict[str, Any]):
        self.settings = settings
        self.running = False
        self.cap: Any = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_mask: Optional[np.ndarray] = None
        self.latest_metrics: Dict[str, Any] = {
            "status": "NO TAPE",
            "area": 0,
            "angle": 0.0
        }

    def start(self) -> None:
        self.running = True
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _update_loop(self) -> None:
        # 1. Initialize camera in the background thread to satisfy GStreamer
        if HAS_HARDWARE:
            try:
                self.cap = Camera()
                print("[INFO] Using actual hardware.Camera()")
            except Exception as e:
                print(f"[WARN] Camera init failed: {e}. Falling back to standard cv2.")
                self.cap = cv2.VideoCapture(0)  # type: ignore
        else:
            self.cap = cv2.VideoCapture(0)  # type: ignore

        # 2. Main capture loop
        while self.running:
            frame = None
            
            # Explicitly separate the read logic to satisfy linters and avoid unpacking errors
            if self.cap is not None:
                if isinstance(self.cap, cv2.VideoCapture):  # type: ignore
                    ret, frame_read = self.cap.read()  # type: ignore
                    if ret:
                        frame = frame_read
                else:
                    # Custom Camera() class returns the frame directly
                    frame = self.cap.read()

            if frame is None:
                time.sleep(0.01)
                continue

            # Resize for performance and UI consistency
            frame = cv2.resize(frame, (640, 360))  # type: ignore
            
            h_min = self.settings["h_min"]
            s_min = self.settings["s_min"]
            v_min = self.settings["v_min"]
            h_max = self.settings["h_max"]
            s_max = self.settings["s_max"]
            v_max = self.settings["v_max"]

            lower = np.array([h_min, s_min, v_min])
            upper = np.array([h_max, s_max, v_max])

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # type: ignore
            mask = cv2.inRange(hsv, lower, upper)  # type: ignore

            display_frame = frame.copy()
            
            current_status = "NO TAPE"
            current_area = 0
            current_angle = 0.0

            # --- Line Detection Math (Contour & Best Fit Line for Tilt) ---
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # type: ignore
            
            if contours:
                # Find the largest contour (assuming the tape is the biggest object of this color)
                largest_contour = max(contours, key=cv2.contourArea)  # type: ignore
                area = cv2.contourArea(largest_contour)  # type: ignore
                
                if area >= self.settings["min_area"]:
                    current_area = int(area)
                    current_status = "DETECTED"
                    
                    # Fit a 2D line to the contour points to get its exact angle and position
                    [vx, vy, x, y] = cv2.fitLine(largest_contour, cv2.DIST_L2, 0, 0.01, 0.01)  # type: ignore
                    vx, vy, x, y = vx[0], vy[0], x[0], y[0]
                    
                    w = display_frame.shape[1]
                    h = display_frame.shape[0]
                    
                    # Prevent division by zero if the line is perfectly vertical
                    if abs(vx) > 1e-4:
                        m = vy / vx # Calculate slope
                        current_angle = math.degrees(math.atan(m))
                        
                        # Extend line to the far left and right edges of the screen
                        y_left = int(m * (0 - x) + y)
                        y_right = int(m * (w - x) + y)
                        cv2.line(display_frame, (0, y_left), (w, y_right), (0, 255, 0), 3)  # type: ignore
                        
                        # Draw center node
                        cv2.circle(display_frame, (int(x), int(y)), 5, (0, 0, 255), -1)  # type: ignore
                        cv2.putText(display_frame, f"TILT: {current_angle:.1f}deg | AREA: {current_area}",  # type: ignore
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)  # type: ignore
                    else:
                        cv2.line(display_frame, (int(x), 0), (int(x), h), (0, 255, 0), 3)  # type: ignore
                        current_angle = 90.0
                else:
                    current_status = "TAPE TOO SMALL"
                    current_area = int(area)
                    cv2.putText(display_frame, f"TAPE TOO SMALL (Area: {int(area)})", (10, 30),  # type: ignore
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)  # type: ignore
            else:
                cv2.putText(display_frame, "NO TAPE DETECTED", (10, 30),  # type: ignore
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)  # type: ignore

            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # type: ignore

            with self._lock:
                self.latest_frame = display_frame
                self.latest_mask = mask_bgr
                self.latest_metrics = {
                    "status": current_status,
                    "area": current_area,
                    "angle": round(current_angle, 2)
                }

        # 3. Cleanup inside the thread
        if self.cap:
            if hasattr(self.cap, 'release'):
                self.cap.release()
            elif hasattr(self.cap, 'stop'):
                self.cap.stop()

    def get_frames_and_metrics(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
        with self._lock:
            return self.latest_frame, self.latest_mask, self.latest_metrics.copy()


# ===========================================================================
# UI Class
# ===========================================================================
class TapeTuner:
    def __init__(self) -> None:
        if not _PIL_OK:
            sys.exit(1)

        self.settings = DEFAULTS.copy()
        self.backend = VisionBackend(self.settings)

        self.root = tk.Tk()
        self.root.title("RLGL Hardware Test - Tape Vision Tuner")
        # Increased window height to accommodate two 360p video feeds
        self.root.geometry("1100x850")
        self.root.minsize(1000, 820) 
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # Explicitly type variables to fix linter warnings
        self.sliders: dict[str, tk.IntVar] = {}
        self.val_labels: dict[str, ttk.Label] = {}
        
        self._build_ui()
        self._refresh_config_snippet()
        
        self.backend.start()
        self._update_ui_video()

    def _build_ui(self) -> None:
        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # ---------------------------------------------------------
        # LEFT PANE: Video Feeds
        # ---------------------------------------------------------
        left_frame = ttk.Frame(main_pane)
        main_pane.add(left_frame, weight=1)

        lbl_top = ttk.Label(left_frame, text="Live Feed & Line Detection (Tilt-Aware)", font=("Helvetica", 10, "bold"))
        lbl_top.pack(anchor="w", padx=5, pady=(5,0))
        
        self.lbl_video = ttk.Label(left_frame, background="black")
        self.lbl_video.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        lbl_bot = ttk.Label(left_frame, text="Color Mask (TAPE MUST BE WHITE, NOISE MUST BE BLACK)", font=("Helvetica", 10, "bold"))
        lbl_bot.pack(anchor="w", padx=5, pady=(5,0))
        
        self.lbl_mask = ttk.Label(left_frame, background="black")
        self.lbl_mask.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # ---------------------------------------------------------
        # RIGHT PANE: Controls & Tabs
        # ---------------------------------------------------------
        right_frame = ttk.Frame(main_pane, width=420)
        main_pane.add(right_frame, weight=0)

        notebook = ttk.Notebook(right_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        # --- TAB 1: Calibration ---
        tab_calib = ttk.Frame(notebook)
        notebook.add(tab_calib, text="Calibration")

        hsv_frame = ttk.LabelFrame(tab_calib, text="HSV Color Bounds", padding=10)
        hsv_frame.pack(fill=tk.X, padx=5, pady=5)

        row = 0
        for name, label, max_val in [
            ("h_min", "Hue Min", 179), ("h_max", "Hue Max", 179),
            ("s_min", "Sat Min", 255), ("s_max", "Sat Max", 255),
            ("v_min", "Val Min", 255), ("v_max", "Val Max", 255),
        ]:
            ttk.Label(hsv_frame, text=label).grid(row=row, column=0, sticky="w", pady=2)
            
            var = tk.IntVar(value=self.settings[name])
            self.sliders[name] = var
            
            slider = ttk.Scale(hsv_frame, from_=0, to=max_val, orient=tk.HORIZONTAL, variable=var,
                               command=lambda val, n=name: self._on_slider_change(n, val))  # type: ignore
            slider.grid(row=row, column=1, sticky="ew", padx=5)
            
            val_lbl = ttk.Label(hsv_frame, text=str(self.settings[name]), width=4)
            val_lbl.grid(row=row, column=2)
            
            self.val_labels[name] = val_lbl
            row += 1

        hsv_frame.columnconfigure(1, weight=1)
        
        filter_frame = ttk.LabelFrame(tab_calib, text="Noise Filtering", padding=10)
        filter_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(filter_frame, text="Min Area").grid(row=0, column=0, sticky="w", pady=2)
        var_area = tk.IntVar(value=self.settings["min_area"])
        self.sliders["min_area"] = var_area
        area_slider = ttk.Scale(filter_frame, from_=0, to=5000, orient=tk.HORIZONTAL, variable=var_area,
                           command=lambda val, n="min_area": self._on_slider_change(n, val))  # type: ignore
        area_slider.grid(row=0, column=1, sticky="ew", padx=5)
        lbl_area = ttk.Label(filter_frame, text=str(self.settings["min_area"]), width=4)
        lbl_area.grid(row=0, column=2)
        self.val_labels["min_area"] = lbl_area
        
        filter_frame.columnconfigure(1, weight=1)

        # --- TAB 2: Configuration ---
        tab_config = ttk.Frame(notebook)
        notebook.add(tab_config, text="Configuration")

        settings_frame = ttk.LabelFrame(tab_config, text="Line Identity", padding=10)
        settings_frame.pack(fill=tk.X, padx=5, pady=5)

        self.var_target = tk.StringVar(value=self.settings["target_line"])
        ttk.Radiobutton(settings_frame, text="Start Line (Green)", variable=self.var_target, value="START", command=self._on_target_change).pack(anchor="w")
        ttk.Radiobutton(settings_frame, text="Finish Line (Red/Other)", variable=self.var_target, value="FINISH", command=self._on_target_change).pack(anchor="w")

        conf_frame = ttk.LabelFrame(tab_config, text="Config Snippet (config.py)", padding=5)
        conf_frame.pack(fill=tk.BOTH, expand=False, padx=5, pady=5)

        self.txt_config = tk.Text(conf_frame, height=8, width=40, font=("Consolas", 10))
        self.txt_config.pack(fill=tk.BOTH, expand=True)

        # --- TAB 3: Tests ---
        tab_test = ttk.Frame(notebook)
        notebook.add(tab_test, text="Tests")

        metrics_frame = ttk.LabelFrame(tab_test, text="Live Pipeline Metrics", padding=10)
        metrics_frame.pack(fill=tk.X, padx=5, pady=5)

        self.lbl_metric_status = ttk.Label(metrics_frame, text="Status: WAITING...", font=("Helvetica", 10, "bold"))
        self.lbl_metric_status.pack(anchor="w", pady=2)
        
        self.lbl_metric_area = ttk.Label(metrics_frame, text="Largest Contour Area: 0 px")
        self.lbl_metric_area.pack(anchor="w", pady=2)
        
        self.lbl_metric_angle = ttk.Label(metrics_frame, text="Line Tilt Angle: 0.0°")
        self.lbl_metric_angle.pack(anchor="w", pady=2)

        # --- TAB 4: Troubleshooting ---
        tab_troubleshoot = ttk.Frame(notebook)
        notebook.add(tab_troubleshoot, text="Troubleshooting")

        tb_text = (
            "Hardware & Vision Checklist:\n\n"
            "1. Tape Not Detected (Black Mask):\n"
            "   - Ensure the gym lighting is consistent.\n"
            "   - Widen your Saturation (S Min/Max) and Value (V Min/Max) bounds first.\n"
            "   - Adjust Hue (H Min/Max) slowly until tape turns white.\n\n"
            "2. False Positives (Speckles):\n"
            "   - Increase the 'Min Area' slider in the Calibration tab to filter out small reflections or objects.\n\n"
            "3. GStreamer Crashes / Freezes:\n"
            "   - Jetson cameras can timeout if frames are bottlenecked. Close the app and restart `./launcher.sh` if the feed freezes.\n\n"
            "4. Start vs Finish Line:\n"
            "   - If using the same color tape for both, the game relies on geometric Y-coordinates to tell them apart. Make sure the start line is physically further away!"
        )
        ttk.Label(tab_troubleshoot, text=tb_text, justify=tk.LEFT, wraplength=380).pack(padx=10, pady=10, anchor="nw")

        # --- ACTION BUTTONS ---
        btn_frame = ttk.Frame(right_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="Load Preset", command=self._load_preset).pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)
        ttk.Button(btn_frame, text="Save Preset", command=self._save_preset).pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)
        ttk.Button(btn_frame, text="Reset Defaults", command=self._reset_defaults).pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)

    def _on_slider_change(self, name: str, val: str) -> None:
        int_val = int(float(val))
        self.settings[name] = int_val
        self.val_labels[name].config(text=str(int_val))
        self._refresh_config_snippet()

    def _on_target_change(self) -> None:
        self.settings["target_line"] = self.var_target.get()
        self._refresh_config_snippet()

    def _refresh_config_snippet(self) -> None:
        self.txt_config.delete("1.0", tk.END)
        target = self.settings["target_line"]
        
        code = f"# Paste into config.py\n"
        code += f"{target}_LINE_HSV_LOW = ({self.settings['h_min']}, {self.settings['s_min']}, {self.settings['v_min']})\n"
        code += f"{target}_LINE_HSV_HIGH = ({self.settings['h_max']}, {self.settings['s_max']}, {self.settings['v_max']})\n"
        
        self.txt_config.insert("1.0", code)

    def _update_ui_video(self) -> None:
        frame, mask, metrics = self.backend.get_frames_and_metrics()
        
        if frame is not None:
            # Convert BGR to RGB for PIL
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # type: ignore
            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.lbl_video.imgtk = imgtk  # type: ignore
            self.lbl_video.configure(image=imgtk)

        if mask is not None:
            rgb_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)  # type: ignore
            m_img = Image.fromarray(rgb_mask)
            m_imgtk = ImageTk.PhotoImage(image=m_img)
            self.lbl_mask.imgtk = m_imgtk  # type: ignore
            self.lbl_mask.configure(image=m_imgtk)

        # Update Metrics Tab
        if metrics:
            self.lbl_metric_status.config(text=f"Status: {metrics['status']}")
            self.lbl_metric_area.config(text=f"Largest Contour Area: {metrics['area']} px")
            self.lbl_metric_angle.config(text=f"Line Tilt Angle: {metrics['angle']}°")
            
            if metrics['status'] == "DETECTED":
                self.lbl_metric_status.config(foreground="green")
            elif metrics['status'] == "TAPE TOO SMALL":
                self.lbl_metric_status.config(foreground="orange")
            else:
                self.lbl_metric_status.config(foreground="red")

        self.root.after(30, self._update_ui_video)

    # ------------------------------------------------------------------
    # Save / Load / Reset
    # ------------------------------------------------------------------
    def _save_preset(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json")],
            initialfile=f"tape_{self.settings['target_line'].lower()}.json"
        )
        if not path:
            return
        try:
            with open(path, "w") as f:
                json.dump(self.settings, f, indent=2)
            messagebox.showinfo("Saved", f"Saved preset to {path}")
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))

    def _load_preset(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if not path:
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self.settings.update(data)
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))
            return

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
        self.backend.stop()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


# ===========================================================================
# Entry point
# ===========================================================================
def main() -> int:
    if platform.system() != "Linux":
        print("[INFO] Not on Linux - running in fallback camera mode.")
    tuner = TapeTuner()
    tuner.run()
    return 0

if __name__ == "__main__":
    sys.exit(main())