"""
PureScale: Desktop GUI Application.
Standalone native interface using CustomTkinter and OpenCV.
"""

import os
import sys
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageTk

from enhance_image import enhance_image

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class ToolTip:
    """Lightweight tooltip that appears on widget hover with enhanced readable font."""

    def __init__(self, widget, text, delay_ms=200):
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self.tip_window = None
        self.schedule_id = None

        self.widget.bind("<Enter>", self._on_enter, add="+")
        self.widget.bind("<Leave>", self._on_leave, add="+")
        self.widget.bind("<ButtonPress>", self._on_leave, add="+")

    def _on_enter(self, event=None):
        self._cancel_schedule()
        self.schedule_id = self.widget.after(self.delay_ms, self._show_tip)

    def _on_leave(self, event=None):
        self._cancel_schedule()
        self._hide_tip()

    def _cancel_schedule(self):
        if self.schedule_id:
            self.widget.after_cancel(self.schedule_id)
            self.schedule_id = None

    def _show_tip(self):
        if self.tip_window or not self.text:
            return

        x = self.widget.winfo_rootx() + 24
        y = self.widget.winfo_rooty() + 20

        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tw.attributes("-topmost", True)

        frame = tk.Frame(
            tw,
            background="#1e2228",
            highlightbackground="#444c56",
            highlightthickness=1,
            padx=14,
            pady=10,
        )
        frame.pack()

        label = tk.Label(
            frame,
            text=self.text,
            justify=tk.LEFT,
            background="#1e2228",
            foreground="#f0f6fc",
            font=("Segoe UI", 11),
            wraplength=320,
        )
        label.pack()

    def _hide_tip(self):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


def make_info_icon(parent, tooltip_text):
    """Creates a small (?) badge that shows a detailed tooltip on hover."""
    icon = ctk.CTkLabel(
        parent,
        text="?",
        width=18,
        height=18,
        corner_radius=9,
        fg_color="#30363d",
        text_color="#8b949e",
        font=ctk.CTkFont(size=10, weight="bold"),
        cursor="hand2",
    )
    ToolTip(icon, tooltip_text)
    return icon


class PureScaleApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("PureScale Image Enhancer")
        self.geometry("1300x850")
        self.minsize(1080, 720)

        self.input_image_path = None
        self.original_bgr = None
        self.enhanced_bgr = None
        self.original_pil = None
        self.enhanced_pil = None
        self.is_processing = False

        # Zoom & Pan State
        self.is_fit_mode = True
        self.zoom_factor = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.current_tk_img = None

        self._build_layout()
        self._set_default_values()

    def _build_layout(self):
        self.grid_columnconfigure(0, weight=0, minsize=440)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # -------------------------------------------------------------
        # Left Panel (Sidebar - width 440)
        # -------------------------------------------------------------
        self.sidebar = ctk.CTkFrame(self, width=440, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.sidebar.grid_columnconfigure(0, weight=1)
        self.sidebar.grid_rowconfigure(1, weight=1)

        # 1. Top Section: App Title & Open File (Fixed at top)
        top_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        top_frame.grid(row=0, column=0, sticky="ew", padx=18, pady=(16, 8))
        top_frame.grid_columnconfigure(0, weight=1)

        app_title = ctk.CTkLabel(
            top_frame,
            text="PureScale",
            font=ctk.CTkFont(size=24, weight="bold"),
        )
        app_title.grid(row=0, column=0, sticky="w")

        app_sub = ctk.CTkLabel(
            top_frame,
            text="Deterministic Classical Image Enhancer",
            font=ctk.CTkFont(size=12),
            text_color="gray60",
        )
        app_sub.grid(row=1, column=0, sticky="w", pady=(0, 10))

        self.btn_open = ctk.CTkButton(
            top_frame,
            text="Open Image",
            command=self._on_open_image,
            height=38,
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        self.btn_open.grid(row=2, column=0, sticky="ew", pady=(0, 6))

        self.lbl_file_info = ctk.CTkLabel(
            top_frame,
            text="No image selected",
            font=ctk.CTkFont(size=11),
            text_color="gray55",
            anchor="w",
            wraplength=400,
        )
        self.lbl_file_info.grid(row=3, column=0, sticky="w")

        # 2. Middle Section: Scrollable Parameters
        self.scroll_params = ctk.CTkScrollableFrame(self.sidebar, corner_radius=6)
        self.scroll_params.grid(row=1, column=0, sticky="nsew", padx=18, pady=8)
        self.scroll_params.grid_columnconfigure(0, weight=1)

        self._build_parameter_controls(self.scroll_params)

        # 3. Bottom Section: Actions & Status (Fixed at bottom)
        bottom_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        bottom_frame.grid(row=2, column=0, sticky="ew", padx=18, pady=(8, 16))
        bottom_frame.grid_columnconfigure(0, weight=1)

        # Enhanced Export Format Selector
        fmt_header = ctk.CTkFrame(bottom_frame, fg_color="transparent")
        fmt_header.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        fmt_header.grid_columnconfigure(2, weight=1)

        ctk.CTkLabel(
            fmt_header,
            text="Export Format",
            font=ctk.CTkFont(size=12, weight="bold"),
        ).grid(row=0, column=0, sticky="w")

        icon_fmt = make_info_icon(
            fmt_header,
            "Target file container. PNG is lossless and preserves transparency; JPEG is compact for photos (95% quality); WebP offers modern web compression (95% quality).",
        )
        icon_fmt.grid(row=0, column=1, padx=(6, 0), sticky="w")

        self.lbl_format_hint = ctk.CTkLabel(
            fmt_header,
            text="Lossless (Alpha supported)",
            font=ctk.CTkFont(size=11),
            text_color="#58a6ff",
        )
        self.lbl_format_hint.grid(row=0, column=2, sticky="e")

        self.seg_format = ctk.CTkSegmentedButton(
            bottom_frame,
            values=["PNG", "JPEG", "WebP"],
            command=self._on_format_change,
            height=32,
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.seg_format.set("PNG")
        self.seg_format.grid(row=1, column=0, sticky="ew", pady=(0, 10))

        # Main Action Buttons
        self.btn_enhance = ctk.CTkButton(
            bottom_frame,
            text="Enhance Image",
            command=self._on_start_enhance,
            height=42,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#1f6feb",
            hover_color="#1857ba",
        )
        self.btn_enhance.grid(row=2, column=0, sticky="ew", pady=(0, 6))

        self.btn_save = ctk.CTkButton(
            bottom_frame,
            text="Save Enhanced Image",
            command=self._on_save_image,
            height=36,
            font=ctk.CTkFont(size=13, weight="bold"),
            state="disabled",
            fg_color="#238636",
            hover_color="#1f7430",
        )
        self.btn_save.grid(row=3, column=0, sticky="ew", pady=(0, 6))

        self.btn_reset = ctk.CTkButton(
            bottom_frame,
            text="Reset to Defaults",
            command=self._set_default_values,
            height=28,
            fg_color="transparent",
            border_width=1,
            text_color=("gray10", "gray80"),
        )
        self.btn_reset.grid(row=4, column=0, sticky="ew", pady=(0, 8))

        # Progress bar & status
        self.progress_bar = ctk.CTkProgressBar(bottom_frame, height=8)
        self.progress_bar.grid(row=5, column=0, sticky="ew", pady=(0, 4))
        self.progress_bar.set(0)

        self.lbl_status = ctk.CTkLabel(
            bottom_frame,
            text="Status: Ready",
            font=ctk.CTkFont(size=11),
            text_color="gray60",
            anchor="w",
        )
        self.lbl_status.grid(row=6, column=0, sticky="w")

        # -------------------------------------------------------------
        # Right Panel (Preview Area)
        # -------------------------------------------------------------
        self.preview_panel = ctk.CTkFrame(self, corner_radius=0)
        self.preview_panel.grid(row=0, column=1, sticky="nsew", padx=12, pady=12)
        self.preview_panel.grid_columnconfigure(0, weight=1)
        self.preview_panel.grid_rowconfigure(1, weight=1)

        self._build_preview_panel()

    def _build_parameter_controls(self, parent):
        row = 0
        right_pad = 16  # Pushes numbers and sliders away from scrollbar

        # [1] Scale Factor
        scale_hdr = ctk.CTkFrame(parent, fg_color="transparent")
        scale_hdr.grid(row=row, column=0, sticky="ew", padx=(4, right_pad), pady=(4, 2))
        scale_hdr.grid_columnconfigure(2, weight=1)
        ctk.CTkLabel(scale_hdr, text="Upscale Factor", font=ctk.CTkFont(size=12, weight="bold")).grid(row=0, column=0, sticky="w")
        make_info_icon(
            scale_hdr,
            "Resolution multiplier. 1.0x preserves original size; 2.0x doubles dimensions; 3.0x triples resolution; 4.0x quadruples size. Uses 8-lobe Lanczos-4 sinc interpolation.",
        ).grid(row=0, column=1, padx=(6, 0), sticky="w")
        self.lbl_scale_val = ctk.CTkLabel(scale_hdr, text="2.0x", font=ctk.CTkFont(size=12, weight="bold"), text_color="#58a6ff")
        self.lbl_scale_val.grid(row=0, column=2, sticky="e")
        row += 1

        self.seg_scale = ctk.CTkSegmentedButton(
            parent,
            values=["1.0x", "1.5x", "2.0x", "3.0x", "4.0x"],
            command=self._on_scale_change,
            height=30,
        )
        self.seg_scale.set("2.0x")
        self.seg_scale.grid(row=row, column=0, sticky="ew", padx=(4, right_pad), pady=(0, 10))
        row += 1

        # [2] Sharpen Strength
        sharpen_hdr = ctk.CTkFrame(parent, fg_color="transparent")
        sharpen_hdr.grid(row=row, column=0, sticky="ew", padx=(4, right_pad), pady=(4, 2))
        sharpen_hdr.grid_columnconfigure(2, weight=1)
        ctk.CTkLabel(sharpen_hdr, text="Sharpen Strength", font=ctk.CTkFont(size=12, weight="bold")).grid(row=0, column=0, sticky="w")
        make_info_icon(
            sharpen_hdr,
            "Edge clarity gain. 0 is disabled; 1.0-1.4 delivers natural crisp clarity; 2.0+ is aggressive sharpness. Amplifies high-frequency textures without blur.",
        ).grid(row=0, column=1, padx=(6, 0), sticky="w")
        self.lbl_sharpen_val = ctk.CTkLabel(sharpen_hdr, text="1.2", font=ctk.CTkFont(size=12, weight="bold"), text_color="#58a6ff")
        self.lbl_sharpen_val.grid(row=0, column=2, sticky="e")
        row += 1

        self.slider_sharpen_strength = ctk.CTkSlider(
            parent,
            from_=0.0,
            to=3.0,
            number_of_steps=30,
            command=self._on_sharpen_strength_change,
        )
        self.slider_sharpen_strength.grid(row=row, column=0, sticky="ew", padx=(4, right_pad), pady=(0, 8))
        row += 1

        # [3] Sharpen Detail Radius
        radius_hdr = ctk.CTkFrame(parent, fg_color="transparent")
        radius_hdr.grid(row=row, column=0, sticky="ew", padx=(4, right_pad), pady=(4, 2))
        radius_hdr.grid_columnconfigure(2, weight=1)
        ctk.CTkLabel(radius_hdr, text="Detail Radius", font=ctk.CTkFont(size=12, weight="bold")).grid(row=0, column=0, sticky="w")
        make_info_icon(
            radius_hdr,
            "Controls which feature size gets sharpened. 1.0-2.0 targets fine micro-textures (hair strands, fabric weave, small text); 3.0-5.0 targets broad contours and outlines.",
        ).grid(row=0, column=1, padx=(6, 0), sticky="w")
        self.lbl_radius_val = ctk.CTkLabel(radius_hdr, text="2.5", font=ctk.CTkFont(size=12, weight="bold"), text_color="#58a6ff")
        self.lbl_radius_val.grid(row=0, column=2, sticky="e")
        row += 1

        self.slider_sharpen_radius = ctk.CTkSlider(
            parent,
            from_=1.0,
            to=6.0,
            number_of_steps=10,
            command=self._on_sharpen_radius_change,
        )
        self.slider_sharpen_radius.grid(row=row, column=0, sticky="ew", padx=(4, right_pad), pady=(0, 10))
        row += 1

        # [4] Denoise Switch & Intensity
        denoise_hdr = ctk.CTkFrame(parent, fg_color="transparent")
        denoise_hdr.grid(row=row, column=0, sticky="ew", padx=(4, right_pad), pady=(4, 2))
        denoise_hdr.grid_columnconfigure(2, weight=1)
        self.sw_denoise = ctk.CTkSwitch(
            denoise_hdr,
            text="Bilateral Denoise",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._on_denoise_toggle,
        )
        self.sw_denoise.grid(row=0, column=0, sticky="w")
        make_info_icon(
            denoise_hdr,
            "Edge-preserving bilateral smoothing. Eliminates camera sensor grain and JPEG compression artifacts while keeping physical edge gradients crisp.",
        ).grid(row=0, column=1, padx=(6, 0), sticky="w")
        self.lbl_denoise_val = ctk.CTkLabel(denoise_hdr, text="50", font=ctk.CTkFont(size=12, weight="bold"), text_color="#58a6ff")
        self.lbl_denoise_val.grid(row=0, column=2, sticky="e")
        row += 1

        self.slider_denoise_intensity = ctk.CTkSlider(
            parent,
            from_=10,
            to=100,
            number_of_steps=18,
            command=self._on_denoise_intensity_change,
        )
        self.slider_denoise_intensity.grid(row=row, column=0, sticky="ew", padx=(4, right_pad), pady=(0, 10))
        row += 1

        # [5] Contrast Switch & Boost
        contrast_hdr = ctk.CTkFrame(parent, fg_color="transparent")
        contrast_hdr.grid(row=row, column=0, sticky="ew", padx=(4, right_pad), pady=(4, 2))
        contrast_hdr.grid_columnconfigure(2, weight=1)
        self.sw_contrast = ctk.CTkSwitch(
            contrast_hdr,
            text="Adaptive Contrast (CLAHE)",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._on_contrast_toggle,
        )
        self.sw_contrast.grid(row=0, column=0, sticky="w")
        make_info_icon(
            contrast_hdr,
            "Contrast Limited Adaptive Histogram Equalization in decoupled CIE L* lightness space. Locally balances dark shadows and bright skies without altering colors (HDR effect).",
        ).grid(row=0, column=1, padx=(6, 0), sticky="w")
        self.lbl_contrast_val = ctk.CTkLabel(contrast_hdr, text="2.0", font=ctk.CTkFont(size=12, weight="bold"), text_color="#58a6ff")
        self.lbl_contrast_val.grid(row=0, column=2, sticky="e")
        row += 1

        self.slider_contrast_boost = ctk.CTkSlider(
            parent,
            from_=1.0,
            to=4.0,
            number_of_steps=15,
            command=self._on_contrast_boost_change,
        )
        self.slider_contrast_boost.grid(row=row, column=0, sticky="ew", padx=(4, right_pad), pady=(0, 10))
        row += 1

        # [6] Brightness Shift
        bright_hdr = ctk.CTkFrame(parent, fg_color="transparent")
        bright_hdr.grid(row=row, column=0, sticky="ew", padx=(4, right_pad), pady=(4, 2))
        bright_hdr.grid_columnconfigure(2, weight=1)
        ctk.CTkLabel(bright_hdr, text="Brightness Offset", font=ctk.CTkFont(size=12, weight="bold")).grid(row=0, column=0, sticky="w")
        make_info_icon(
            bright_hdr,
            "Overall exposure translation. Negative values dim overly bright or washed-out images; positive values brighten dark, underexposed shots.",
        ).grid(row=0, column=1, padx=(6, 0), sticky="w")
        self.lbl_bright_val = ctk.CTkLabel(bright_hdr, text="0", font=ctk.CTkFont(size=12, weight="bold"), text_color="#58a6ff")
        self.lbl_bright_val.grid(row=0, column=2, sticky="e")
        row += 1

        self.slider_brightness = ctk.CTkSlider(
            parent,
            from_=-50,
            to=50,
            number_of_steps=20,
            command=self._on_brightness_change,
        )
        self.slider_brightness.grid(row=row, column=0, sticky="ew", padx=(4, right_pad), pady=(0, 8))
        row += 1

        # [7] Vibrance Boost
        vibrance_hdr = ctk.CTkFrame(parent, fg_color="transparent")
        vibrance_hdr.grid(row=row, column=0, sticky="ew", padx=(4, right_pad), pady=(4, 2))
        vibrance_hdr.grid_columnconfigure(2, weight=1)
        ctk.CTkLabel(vibrance_hdr, text="Vibrance Boost", font=ctk.CTkFont(size=12, weight="bold")).grid(row=0, column=0, sticky="w")
        make_info_icon(
            vibrance_hdr,
            "Color richness multiplier in cylindrical HSV space. Boosts color saturation naturally without skin tone clipping or chromatic phase distortion.",
        ).grid(row=0, column=1, padx=(6, 0), sticky="w")
        self.lbl_vibrance_val = ctk.CTkLabel(vibrance_hdr, text="1.10", font=ctk.CTkFont(size=12, weight="bold"), text_color="#58a6ff")
        self.lbl_vibrance_val.grid(row=0, column=2, sticky="e")
        row += 1

        self.slider_vibrance = ctk.CTkSlider(
            parent,
            from_=1.0,
            to=1.5,
            number_of_steps=10,
            command=self._on_vibrance_change,
        )
        self.slider_vibrance.grid(row=row, column=0, sticky="ew", padx=(4, right_pad), pady=(0, 8))
        row += 1

        # [8] Color Temperature
        temp_hdr = ctk.CTkFrame(parent, fg_color="transparent")
        temp_hdr.grid(row=row, column=0, sticky="ew", padx=(4, right_pad), pady=(4, 2))
        temp_hdr.grid_columnconfigure(2, weight=1)
        ctk.CTkLabel(temp_hdr, text="Color Temperature", font=ctk.CTkFont(size=12, weight="bold")).grid(row=0, column=0, sticky="w")
        make_info_icon(
            temp_hdr,
            "White balance offset. Negative values add cool daylight blue (fixes warm yellow indoor bulb tint); positive values add golden sunlight warmth.",
        ).grid(row=0, column=1, padx=(6, 0), sticky="w")
        self.lbl_temp_val = ctk.CTkLabel(temp_hdr, text="0", font=ctk.CTkFont(size=12, weight="bold"), text_color="#58a6ff")
        self.lbl_temp_val.grid(row=0, column=2, sticky="e")
        row += 1

        self.slider_temperature = ctk.CTkSlider(
            parent,
            from_=-30,
            to=30,
            number_of_steps=12,
            command=self._on_temperature_change,
        )
        self.slider_temperature.grid(row=row, column=0, sticky="ew", padx=(4, right_pad), pady=(0, 4))
        row += 1

    def _build_preview_panel(self):
        # Top toolbar
        toolbar = ctk.CTkFrame(self.preview_panel, fg_color="transparent")
        toolbar.grid(row=0, column=0, sticky="ew", padx=12, pady=(8, 8))
        toolbar.grid_columnconfigure(2, weight=1)

        # Left: View Toggle
        self.seg_view = ctk.CTkSegmentedButton(
            toolbar,
            values=["Enhanced", "Original"],
            command=self._on_view_mode_change,
            height=30,
        )
        self.seg_view.set("Enhanced")
        self.seg_view.grid(row=0, column=0, sticky="w")

        # Center: Zoom Controls
        zoom_frame = ctk.CTkFrame(toolbar, fg_color="transparent")
        zoom_frame.grid(row=0, column=1, padx=(18, 0), sticky="w")

        self.btn_zoom_out = ctk.CTkButton(
            zoom_frame,
            text="-",
            width=28,
            height=28,
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self._on_zoom_out,
        )
        self.btn_zoom_out.grid(row=0, column=0, padx=2)

        self.lbl_zoom_level = ctk.CTkLabel(
            zoom_frame,
            text="Fit",
            width=48,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#58a6ff",
        )
        self.lbl_zoom_level.grid(row=0, column=1, padx=2)

        self.btn_zoom_in = ctk.CTkButton(
            zoom_frame,
            text="+",
            width=28,
            height=28,
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self._on_zoom_in,
        )
        self.btn_zoom_in.grid(row=0, column=2, padx=2)

        self.btn_zoom_fit = ctk.CTkButton(
            zoom_frame,
            text="Fit",
            width=40,
            height=28,
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self._on_zoom_fit,
        )
        self.btn_zoom_fit.grid(row=0, column=3, padx=2)

        self.btn_zoom_100 = ctk.CTkButton(
            zoom_frame,
            text="100%",
            width=46,
            height=28,
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self._on_zoom_100,
        )
        self.btn_zoom_100.grid(row=0, column=4, padx=2)

        # Right: Image Metadata
        self.lbl_meta = ctk.CTkLabel(
            toolbar,
            text="",
            font=ctk.CTkFont(size=12),
            text_color="gray65",
        )
        self.lbl_meta.grid(row=0, column=2, sticky="e")

        # Preview Container Frame with Canvas
        self.preview_canvas_frame = ctk.CTkFrame(
            self.preview_panel,
            corner_radius=8,
            fg_color="#161b22",
            border_width=1,
            border_color="#30363d",
        )
        self.preview_canvas_frame.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))
        self.preview_canvas_frame.grid_columnconfigure(0, weight=1)
        self.preview_canvas_frame.grid_rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            self.preview_canvas_frame,
            bg="#161b22",
            highlightthickness=0,
            bd=0,
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")

        # Bind events for interaction
        self.canvas.bind("<Configure>", self._on_preview_resize)
        self.canvas.bind("<ButtonPress-1>", self._on_pan_start)
        self.canvas.bind("<B1-Motion>", self._on_pan_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_pan_release)
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Double-Button-1>", lambda e: self._on_zoom_fit())

        self._show_empty_placeholder()

    def _show_empty_placeholder(self):
        self.canvas.delete("all")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 50 or h < 50:
            w, h = 800, 650

        self.canvas.create_text(
            w // 2,
            h // 2 - 10,
            text="No image loaded",
            fill="#8b949e",
            font=("Segoe UI", 15, "bold"),
            justify="center",
        )
        self.canvas.create_text(
            w // 2,
            h // 2 + 22,
            text="Click 'Open Image' to select a photo",
            fill="#6e7681",
            font=("Segoe UI", 11),
            justify="center",
        )

    def _set_default_values(self):
        self.seg_scale.set("2.0x")
        self.lbl_scale_val.configure(text="2.0x")

        self.slider_sharpen_strength.set(1.2)
        self.lbl_sharpen_val.configure(text="1.2")

        self.slider_sharpen_radius.set(2.5)
        self.lbl_radius_val.configure(text="2.5")

        self.sw_denoise.select()
        self.slider_denoise_intensity.set(50)
        self.lbl_denoise_val.configure(text="50")
        self.slider_denoise_intensity.configure(state="normal")

        self.sw_contrast.select()
        self.slider_contrast_boost.set(2.0)
        self.lbl_contrast_val.configure(text="2.0")
        self.slider_contrast_boost.configure(state="normal")

        self.slider_brightness.set(0)
        self.lbl_bright_val.configure(text="0")

        self.slider_vibrance.set(1.10)
        self.lbl_vibrance_val.configure(text="1.10")

        self.slider_temperature.set(0)
        self.lbl_temp_val.configure(text="0")

        self.seg_format.set("PNG")
        self.lbl_format_hint.configure(text="Lossless (Alpha supported)")

    # -------------------------------------------------------------
    # Zoom & Pan Handlers
    # -------------------------------------------------------------
    def _on_zoom_in(self):
        self._zoom_step(1.25)

    def _on_zoom_out(self):
        self._zoom_step(1.0 / 1.25)

    def _on_zoom_fit(self):
        self.is_fit_mode = True
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._update_preview()

    def _on_zoom_100(self):
        self.is_fit_mode = False
        self.zoom_factor = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._update_preview()

    def _zoom_step(self, factor):
        target = self.enhanced_pil if self.seg_view.get() == "Enhanced" and self.enhanced_pil is not None else self.original_pil
        if target is None:
            return

        if self.is_fit_mode:
            # Transition from fit mode to manual zoom
            fit_scale = self._calc_fit_scale(target)
            self.zoom_factor = fit_scale * factor
            self.is_fit_mode = False
        else:
            self.zoom_factor *= factor

        self.zoom_factor = max(0.05, min(15.0, self.zoom_factor))
        self._update_preview()

    def _on_mouse_wheel(self, event):
        target = self.enhanced_pil if self.seg_view.get() == "Enhanced" and self.enhanced_pil is not None else self.original_pil
        if target is None:
            return

        factor = 1.15 if event.delta > 0 else (1.0 / 1.15)
        self._zoom_step(factor)

    def _on_pan_start(self, event):
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        self.canvas.configure(cursor="fleur")

    def _on_pan_drag(self, event):
        target = self.enhanced_pil if self.seg_view.get() == "Enhanced" and self.enhanced_pil is not None else self.original_pil
        if target is None:
            return

        dx = event.x - self.drag_start_x
        dy = event.y - self.drag_start_y
        self.drag_start_x = event.x
        self.drag_start_y = event.y

        self.pan_x += dx
        self.pan_y += dy
        self.is_fit_mode = False
        self._update_preview()

    def _on_pan_release(self, event):
        self.canvas.configure(cursor="")

    def _calc_fit_scale(self, target_pil):
        cw = max(50, self.canvas.winfo_width() - 32)
        ch = max(50, self.canvas.winfo_height() - 32)
        img_w, img_h = target_pil.size
        return min(cw / img_w, ch / img_h)

    # -------------------------------------------------------------
    # Control callbacks
    # -------------------------------------------------------------
    def _on_format_change(self, fmt):
        hints = {
            "PNG": "Lossless (Alpha supported)",
            "JPEG": "High Quality (95% Compression)",
            "WebP": "Modern WebP (95% Compression)",
        }
        self.lbl_format_hint.configure(text=hints.get(fmt, ""))

    def _on_scale_change(self, val):
        self.lbl_scale_val.configure(text=val)

    def _on_sharpen_strength_change(self, val):
        self.lbl_sharpen_val.configure(text=f"{val:.1f}")

    def _on_sharpen_radius_change(self, val):
        self.lbl_radius_val.configure(text=f"{val:.1f}")

    def _on_denoise_toggle(self):
        if self.sw_denoise.get():
            self.slider_denoise_intensity.configure(state="normal")
            self.lbl_denoise_val.configure(text=f"{int(self.slider_denoise_intensity.get())}")
        else:
            self.slider_denoise_intensity.configure(state="disabled")
            self.lbl_denoise_val.configure(text="Off")

    def _on_denoise_intensity_change(self, val):
        self.lbl_denoise_val.configure(text=f"{int(val)}")

    def _on_contrast_toggle(self):
        if self.sw_contrast.get():
            self.slider_contrast_boost.configure(state="normal")
            self.lbl_contrast_val.configure(text=f"{self.slider_contrast_boost.get():.1f}")
        else:
            self.slider_contrast_boost.configure(state="disabled")
            self.lbl_contrast_val.configure(text="Off")

    def _on_contrast_boost_change(self, val):
        self.lbl_contrast_val.configure(text=f"{val:.1f}")

    def _on_brightness_change(self, val):
        self.lbl_bright_val.configure(text=f"{int(val)}")

    def _on_vibrance_change(self, val):
        self.lbl_vibrance_val.configure(text=f"{val:.2f}")

    def _on_temperature_change(self, val):
        self.lbl_temp_val.configure(text=f"{int(val)}")

    def _on_view_mode_change(self, mode):
        self._update_preview()

    def _on_preview_resize(self, event):
        if self.original_pil is not None:
            self._update_preview()
        else:
            self._show_empty_placeholder()

    def _on_open_image(self):
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.webp *.bmp *.tiff"),
            ("All files", "*.*"),
        ]
        path = filedialog.askopenfilename(title="Select an Image", filetypes=filetypes)
        if not path:
            return

        try:
            pil_img = Image.open(path)
            pil_img = ImageOps.exif_transpose(pil_img)
            self.original_pil = pil_img.convert("RGB")

            rgb_arr = np.array(self.original_pil)
            self.original_bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)

            self.input_image_path = Path(path)
            w, h = self.original_pil.size
            file_size_kb = self.input_image_path.stat().st_size / 1024.0

            if file_size_kb >= 1024:
                size_str = f"{file_size_kb / 1024.0:.1f} MB"
            else:
                size_str = f"{file_size_kb:.0f} KB"

            self.lbl_file_info.configure(text=f"{self.input_image_path.name} ({w}x{h} px, {size_str})")

            self.enhanced_bgr = None
            self.enhanced_pil = None
            self.btn_save.configure(state="disabled")
            self.seg_view.set("Original")

            self.lbl_meta.configure(text=f"Original: {w}x{h} px ({size_str})")
            self.lbl_status.configure(text="Status: Image loaded. Ready to enhance.")

            # Reset zoom to Fit
            self.is_fit_mode = True
            self.pan_x = 0.0
            self.pan_y = 0.0
            self._update_preview()

        except Exception as e:
            messagebox.showerror("Error Opening Image", str(e))

    def _on_start_enhance(self):
        if self.original_bgr is None:
            messagebox.showwarning("No Image", "Please open an image first.")
            return

        if self.is_processing:
            return

        self.is_processing = True
        self.btn_enhance.configure(state="disabled")
        self.btn_open.configure(state="disabled")
        self.btn_save.configure(state="disabled")
        self.progress_bar.configure(mode="indeterminate")
        self.progress_bar.start()
        self.lbl_status.configure(text="Status: Processing image...")

        scale = float(self.seg_scale.get().replace("x", ""))
        sharpen_strength = float(self.slider_sharpen_strength.get())
        sharpen_radius = float(self.slider_sharpen_radius.get())
        enable_denoise = bool(self.sw_denoise.get())
        denoise_intensity = int(self.slider_denoise_intensity.get()) if enable_denoise else 0
        enable_clahe = bool(self.sw_contrast.get())
        contrast_boost = float(self.slider_contrast_boost.get()) if enable_clahe else 0.0
        brightness_shift = int(self.slider_brightness.get())
        vibrance_boost = float(self.slider_vibrance.get())
        color_temperature = int(self.slider_temperature.get())

        thread = threading.Thread(
            target=self._worker_enhance,
            args=(
                scale,
                sharpen_strength,
                sharpen_radius,
                enable_denoise,
                denoise_intensity,
                enable_clahe,
                contrast_boost,
                brightness_shift,
                vibrance_boost,
                color_temperature,
            ),
            daemon=True,
        )
        thread.start()

    def _worker_enhance(
        self,
        scale,
        sharpen_strength,
        sharpen_radius,
        enable_denoise,
        denoise_intensity,
        enable_clahe,
        contrast_boost,
        brightness_shift,
        vibrance_boost,
        color_temperature,
    ):
        start_t = time.perf_counter()
        try:
            enhanced = enhance_image(
                self.original_bgr,
                scale=scale,
                sharpen_strength=sharpen_strength,
                sharpen_radius=sharpen_radius,
                enable_denoise=enable_denoise,
                denoise_intensity=denoise_intensity,
                enable_clahe=enable_clahe,
                contrast_boost=contrast_boost,
                brightness_shift=brightness_shift,
                vibrance_boost=vibrance_boost,
                color_temperature=color_temperature,
            )
            elapsed = (time.perf_counter() - start_t) * 1000.0
            self.after(0, self._on_enhance_complete, enhanced, elapsed)
        except Exception as e:
            self.after(0, self._on_enhance_error, str(e))

    def _on_enhance_complete(self, enhanced_bgr, elapsed_ms):
        self.enhanced_bgr = enhanced_bgr
        rgb_arr = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
        self.enhanced_pil = Image.fromarray(rgb_arr)

        self.is_processing = False
        self.btn_enhance.configure(state="normal")
        self.btn_open.configure(state="normal")
        self.btn_save.configure(state="normal")
        self.progress_bar.stop()
        self.progress_bar.configure(mode="determinate")
        self.progress_bar.set(1.0)

        w_orig, h_orig = self.original_pil.size
        w_enh, h_enh = self.enhanced_pil.size
        self.seg_view.set("Enhanced")

        self.lbl_meta.configure(
            text=f"Enhanced: {w_enh}x{h_enh} px ({w_enh/w_orig:.1f}x) | {elapsed_ms:.1f} ms"
        )
        self.lbl_status.configure(
            text=f"Status: Enhanced in {elapsed_ms:.1f} ms ({w_enh}x{h_enh})"
        )
        self._update_preview()

    def _on_enhance_error(self, err_msg):
        self.is_processing = False
        self.btn_enhance.configure(state="normal")
        self.btn_open.configure(state="normal")
        self.progress_bar.stop()
        self.progress_bar.configure(mode="determinate")
        self.progress_bar.set(0)
        self.lbl_status.configure(text="Status: Enhancement failed.")
        messagebox.showerror("Enhancement Error", err_msg)

    def _update_preview(self):
        target_pil = self.enhanced_pil if self.seg_view.get() == "Enhanced" and self.enhanced_pil is not None else self.original_pil

        if target_pil is None:
            self._show_empty_placeholder()
            self.lbl_zoom_level.configure(text="Fit")
            return

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 50 or ch < 50:
            cw, ch = 800, 650

        img_w, img_h = target_pil.size

        if self.is_fit_mode:
            pad = 28
            max_w = max(50, cw - pad)
            max_h = max(50, ch - pad)
            scale = min(max_w / img_w, max_h / img_h)
            self.zoom_factor = scale
            self.pan_x = 0.0
            self.pan_y = 0.0
            self.lbl_zoom_level.configure(text="Fit")
        else:
            pct = int(round(self.zoom_factor * 100))
            self.lbl_zoom_level.configure(text=f"{pct}%")

        scale = self.zoom_factor
        disp_w = int(round(img_w * scale))
        disp_h = int(round(img_h * scale))

        center_x = cw / 2.0 + self.pan_x
        center_y = ch / 2.0 + self.pan_y

        img_left = center_x - disp_w / 2.0
        img_top = center_y - disp_h / 2.0

        self.canvas.delete("all")

        # Viewport intersection to only process visible region for performance
        vis_left = max(0, img_left)
        vis_right = min(cw, img_left + disp_w)
        vis_top = max(0, img_top)
        vis_bottom = min(ch, img_top + disp_h)

        if vis_right > vis_left and vis_bottom > vis_top:
            crop_x1 = max(0.0, (vis_left - img_left) / scale)
            crop_y1 = max(0.0, (vis_top - img_top) / scale)
            crop_x2 = min(float(img_w), (vis_right - img_left) / scale)
            crop_y2 = min(float(img_h), (vis_bottom - img_top) / scale)

            crop_box = (
                int(round(crop_x1)),
                int(round(crop_y1)),
                max(int(round(crop_x1)) + 1, int(round(crop_x2))),
                max(int(round(crop_y1)) + 1, int(round(crop_y2))),
            )

            cropped = target_pil.crop(crop_box)
            out_w = max(1, int(round(vis_right - vis_left)))
            out_h = max(1, int(round(vis_bottom - vis_top)))

            resized = cropped.resize((out_w, out_h), Image.Resampling.LANCZOS if scale <= 1.0 else Image.Resampling.BILINEAR)
            self.current_tk_img = ImageTk.PhotoImage(resized)
            self.canvas.create_image(vis_left, vis_top, image=self.current_tk_img, anchor="nw")

    def _on_save_image(self):
        if self.enhanced_pil is None or self.input_image_path is None:
            messagebox.showwarning("No Enhanced Image", "Please enhance an image before saving.")
            return

        fmt = self.seg_format.get().lower()
        scale_val = self.seg_scale.get().replace("x", "")
        default_name = f"{self.input_image_path.stem}_enhanced_{scale_val}x.{fmt}"

        filetypes = [
            ("PNG file", "*.png"),
            ("JPEG file", "*.jpg *.jpeg"),
            ("WebP file", "*.webp"),
            ("All files", "*.*"),
        ]

        save_path = filedialog.asksaveasfilename(
            title="Save Enhanced Image",
            initialfile=default_name,
            filetypes=filetypes,
            defaultextension=f".{fmt}",
        )

        if not save_path:
            return

        try:
            out_fmt = self.seg_format.get()
            if out_fmt == "JPEG":
                self.enhanced_pil.convert("RGB").save(save_path, format="JPEG", quality=95)
            elif out_fmt == "WebP":
                self.enhanced_pil.save(save_path, format="WEBP", quality=95)
            else:
                self.enhanced_pil.save(save_path, format="PNG", compress_level=3)

            self.lbl_status.configure(text=f"Status: Saved to {Path(save_path).name}")
            messagebox.showinfo("Success", f"Enhanced image saved successfully:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Error Saving Image", str(e))


def main():
    app = PureScaleApp()
    app.mainloop()


if __name__ == "__main__":
    main()
