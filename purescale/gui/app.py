"""Obsidian Studio modern desktop user interface application for PureScale 4.0."""

import logging
import os
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Optional

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image

from purescale.config import (
    DeviceTarget,
    DiagnosticsResult,
    PipelineConfig,
    ProcessingMode,
    ProcessingResult,
    get_preset_config,
)
from purescale.dsp.diagnostics import auto_tune_parameters, diagnose_image
from purescale.dsp.semantic import extract_semantic_masks
from purescale.gui.canvas import InteractiveCanvas
from purescale.gui.components import DiagnosticsHUDCard, LabeledSlider, ParameterCard
from purescale.gui.scrolling import SmoothScrollableFrame
from purescale.pipeline import PureScalePipeline

logger = logging.getLogger(__name__)

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class PureScaleApp(ctk.CTk):
    """Obsidian Studio desktop interface for PureScale 4.0."""

    def __init__(self):
        super().__init__()

        self.title("PureScale 4.0 - Autonomous Multiscale Vision & Edge AI")
        self.geometry("1480x940")
        self.minsize(1080, 720)
        self.configure(fg_color="#090d16")

        # Pipeline and state
        self.pipeline = PureScalePipeline()
        self.current_orig_bgr: Optional[np.ndarray] = None
        self.current_enh_bgr: Optional[np.ndarray] = None
        self.current_alpha: Optional[np.ndarray] = None
        self.current_exif: Optional[bytes] = None
        self.current_icc_profile: Optional[bytes] = None
        self.current_file_path: Optional[str] = None
        self.current_diagnostics: Optional[DiagnosticsResult] = None
        self.is_processing: bool = False

        self._detect_hardware()
        self._build_layout()
        # Keyboard shortcuts: Ctrl+O open, Ctrl+S save, Ctrl+E enhance.
        self.bind("<Control-o>", lambda e: self._open_file_dialog())
        self.bind("<Control-s>", lambda e: self._save_file_dialog())
        self.bind("<Control-e>", lambda e: self._start_enhancement_async())

    def _detect_hardware(self) -> None:
        """Determines active hardware acceleration device name."""
        from purescale.device import cpu_label, has_directml

        if has_directml():
            self.hardware_info = "DirectML GPU"
        else:
            self.hardware_info = cpu_label()

    def _build_layout(self) -> None:
        """Constructs application UI hierarchy."""
        # 1. Top Header Bar
        self.top_bar = ctk.CTkFrame(self, fg_color="#0d131f", height=50, corner_radius=0)
        self.top_bar.pack(side="top", fill="x")

        title_box = ctk.CTkFrame(self.top_bar, fg_color="transparent")
        title_box.pack(side="left", padx=16)

        ctk.CTkLabel(
            title_box,
            text="PURESCALE 4.0",
            font=ctk.CTkFont(family="Consolas", size=15, weight="bold"),
            text_color="#f8fafc",
        ).pack(side="left")

        ctk.CTkLabel(
            title_box,
            text=f"[{self.hardware_info}]",
            font=ctk.CTkFont(family="Consolas", size=11, weight="bold"),
            text_color="#38bdf8",
        ).pack(side="left", padx=12)

        # View Mode Segmented Switcher
        view_box = ctk.CTkFrame(self.top_bar, fg_color="transparent")
        view_box.pack(side="right", padx=16)

        self.view_mode_seg = ctk.CTkSegmentedButton(
            view_box,
            values=["Enhanced", "Original", "Split View"],
            command=self._on_view_mode_change,
            selected_color="#0284c7",
            selected_hover_color="#0369a1",
            unselected_color="#131b2e",
            unselected_hover_color="#1e293b",
            font=ctk.CTkFont(family="Consolas", size=11, weight="bold"),
        )
        self.view_mode_seg.set("Enhanced")
        self.view_mode_seg.pack(side="left", padx=10)

        # Interactive Split Slider Control (shown in Split View)
        self.split_ctrl_frame = ctk.CTkFrame(view_box, fg_color="transparent")

        self.split_label = ctk.CTkLabel(
            self.split_ctrl_frame,
            text="Split: 50%",
            font=ctk.CTkFont(family="Consolas", size=10, weight="bold"),
            text_color="#38bdf8",
        )
        self.split_label.pack(side="left", padx=(4, 6))

        self.split_slider = ctk.CTkSlider(
            self.split_ctrl_frame,
            from_=0.0,
            to=1.0,
            number_of_steps=100,
            width=120,
            height=14,
            command=self._on_split_slider_move,
            fg_color="#1e293b",
            progress_color="#0284c7",
            button_color="#38bdf8",
            button_hover_color="#7dd3fc",
        )
        self.split_slider.set(0.5)
        self.split_slider.pack(side="left", padx=(0, 8))

        # Zoom Controls
        zoom_frame = ctk.CTkFrame(view_box, fg_color="transparent")
        zoom_frame.pack(side="left")

        ctk.CTkButton(zoom_frame, text="-", width=28, height=28, command=lambda: self.canvas.zoom_step(0.85), fg_color="#131b2e", hover_color="#1e293b").pack(side="left", padx=2)
        ctk.CTkButton(zoom_frame, text="+", width=28, height=28, command=lambda: self.canvas.zoom_step(1.15), fg_color="#131b2e", hover_color="#1e293b").pack(side="left", padx=2)
        ctk.CTkButton(zoom_frame, text="100%", width=44, height=28, command=lambda: self.canvas.zoom_100(), fg_color="#131b2e", hover_color="#1e293b").pack(side="left", padx=2)
        ctk.CTkButton(zoom_frame, text="Fit", width=36, height=28, command=lambda: self.canvas.fit_to_window(), fg_color="#131b2e", hover_color="#1e293b").pack(side="left", padx=2)

        # 2. Main Content Split (Sidebar + Viewport)
        content_frame = ctk.CTkFrame(self, fg_color="transparent")
        content_frame.pack(fill="both", expand=True)

        # Left Scrollable Sidebar (kinetic smooth scrolling; same cards API)
        self.sidebar = SmoothScrollableFrame(
            content_frame,
            width=380,
            fg_color="#0d131f",
            corner_radius=0,
            scrollbar_button_color="#1e293b",
            scrollbar_button_hover_color="#334155",
        )
        self.sidebar.pack(side="left", fill="y")

        # Main Viewport & Bottom Status
        viewport_frame = ctk.CTkFrame(content_frame, fg_color="#090d16", corner_radius=0)
        viewport_frame.pack(side="right", fill="both", expand=True)

        self.canvas = InteractiveCanvas(viewport_frame)
        self.canvas.on_split_change = self._on_canvas_split_change
        self.canvas.pack(fill="both", expand=True)

        self.status_bar = ctk.CTkFrame(viewport_frame, height=32, fg_color="#0d131f", corner_radius=0)
        self.status_bar.pack(side="bottom", fill="x")

        self.status_label = ctk.CTkLabel(
            self.status_bar,
            text="Ready. Select an image to enhance.",
            font=ctk.CTkFont(family="Consolas", size=11),
            text_color="#94a3b8",
        )
        self.status_label.pack(side="left", padx=16)

        self.progress_bar = ctk.CTkProgressBar(self.status_bar, width=200, height=8, fg_color="#131b2e", progress_color="#38bdf8")
        self.progress_bar.set(0.0)
        self.progress_bar.pack(side="right", padx=16)

        self._populate_sidebar()

    def _populate_sidebar(self) -> None:
        """Fills sidebar with action buttons, diagnostics HUD, and parameter cards."""
        # Primary Action Buttons
        btn_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        btn_frame.pack(fill="x", padx=12, pady=(12, 6))

        ctk.CTkButton(
            btn_frame,
            text="Open Image",
            command=self._open_file_dialog,
            fg_color="#1e293b",
            hover_color="#334155",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            height=34,
        ).pack(fill="x", pady=2)

        # Autonomous Auto-Enhance Button
        self.btn_auto = ctk.CTkButton(
            btn_frame,
            text="Auto-Enhance (CV Engine)",
            command=self._on_auto_enhance_click,
            fg_color="#059669",
            hover_color="#047857",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            height=36,
        )
        self.btn_auto.pack(fill="x", pady=3)

        self.btn_enhance = ctk.CTkButton(
            btn_frame,
            text="Enhance Image",
            command=self._start_enhancement_async,
            fg_color="#0284c7",
            hover_color="#0369a1",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            height=36,
        )
        self.btn_enhance.pack(fill="x", pady=3)

        self.btn_save = ctk.CTkButton(
            btn_frame,
            text="Save Result",
            command=self._save_file_dialog,
            fg_color="#131b2e",
            border_color="#38bdf8",
            border_width=1,
            hover_color="#1e293b",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            height=32,
        )
        self.btn_save.pack(fill="x", pady=2)

        # 0. Live Signal Diagnostics HUD Card
        self.hud_card = DiagnosticsHUDCard(self.sidebar)
        self.hud_card.pack(fill="x", padx=12, pady=4)

        # 1. Mode Selector Card
        mode_card = ParameterCard(
            self.sidebar,
            title="EXECUTION ENGINE",
            badge="MODE",
            help_text=(
                "Processing engine.\n\n"
                "PureDSP: 100% deterministic math DSP, no model, fastest.\n"
                "Neural AI: Real-ESRGAN edge synthesis on GPU/CPU.\n"
                "Hybrid: neural upscaling + DSP color, contrast and clarity."
            ),
        )
        mode_card.pack(fill="x", padx=12, pady=4)

        self.mode_seg = ctk.CTkSegmentedButton(
            mode_card,
            values=["PureDSP", "Neural AI", "Hybrid"],
            command=self._on_mode_change,
            selected_color="#0284c7",
            unselected_color="#090d16",
            font=ctk.CTkFont(family="Consolas", size=10, weight="bold"),
        )
        self.mode_seg.set("PureDSP")
        self.mode_seg.pack(fill="x", padx=8, pady=(4, 8))

        # 2. Preset Profiles Card (OptionMenu wraps to any preset count;
        # a 6-value SegmentedButton crowds a 380px sidebar and clips).
        # Borderless card: the dropdown field itself carries the visual weight,
        # so an extra 1px frame around it looks doubled and ugly.
        preset_card = ParameterCard(
            self.sidebar,
            title="OPTIMIZED PROFILES",
            badge="PRESETS",
            border_width=0,
            help_text=(
                "One-click parameter sets.\n\n"
                "Balanced / Portrait / Landscape / Low-Light / Art tune "
                "sharpening, denoise, contrast and color for the scene.\n"
                "Fast disables diagnostics, semantic guidance, pyramid and "
                "denoising for maximum speed."
            ),
        )
        preset_card.pack(fill="x", padx=12, pady=4)

        self.preset_opt = ctk.CTkOptionMenu(
            preset_card,
            values=["Balanced", "Portrait", "Landscape", "Low-Light", "Art", "Fast"],
            command=self._on_preset_change,
            fg_color="#131b2e",
            button_color="#0284c7",
            button_hover_color="#0369a1",
            text_color="#e2e8f0",
            dropdown_fg_color="#131b2e",
            dropdown_hover_color="#1e293b",
            dropdown_text_color="#e2e8f0",
            corner_radius=6,
            height=32,
            anchor="center",
            font=ctk.CTkFont(family="Consolas", size=11, weight="bold"),
            dropdown_font=ctk.CTkFont(family="Segoe UI", size=11),
        )
        self.preset_opt.set("Balanced")
        self.preset_opt.pack(fill="x", padx=8, pady=(4, 8))
        # Backward-compat alias (older code/tests reference preset_seg).
        self.preset_seg = self.preset_opt

        # 2b. Performance / Telemetry Card (CLI parity: --no-pyramid is the
        # pyramid switch, --no-semantic/--diagnostics/--no-contrast below).
        perf_card = ParameterCard(
            self.sidebar,
            title="PERFORMANCE & TELEMETRY",
            badge="FAST",
            help_text=(
                "Toggles for the heavy stages (CLI parity).\n\n"
                "Signal Diagnostics: wavelet/blur/entropy/haze analysis. "
                "Off skips the HUD input and auto-tune data.\n"
                "Semantic Region Guidance: sky/foliage/skin masks that steer "
                "detail and denoise. Off is faster; detail falls back to global.\n"
                "BIMEF Dynamic Range Fusion: HDR midtone recovery. Off keeps "
                "original exposure curve."
            ),
        )
        perf_card.pack(fill="x", padx=12, pady=4)
        self.diagnostics_switch = ctk.CTkSwitch(perf_card, text="Signal Diagnostics (HUD + auto-tune input)", font=ctk.CTkFont(family="Segoe UI", size=11), progress_color="#38bdf8")
        self.diagnostics_switch.select()
        self.diagnostics_switch.pack(anchor="w", padx=12, pady=(4, 2))
        self.semantic_switch = ctk.CTkSwitch(perf_card, text="Semantic Region Guidance", font=ctk.CTkFont(family="Segoe UI", size=11), progress_color="#38bdf8")
        self.semantic_switch.select()
        self.semantic_switch.pack(anchor="w", padx=12, pady=(2, 2))
        self.contrast_switch = ctk.CTkSwitch(perf_card, text="BIMEF Dynamic Range Fusion", font=ctk.CTkFont(family="Segoe UI", size=11), progress_color="#38bdf8")
        self.contrast_switch.select()
        self.contrast_switch.pack(anchor="w", padx=12, pady=(2, 4))

        # 3. Multiscale Local Laplacian Pyramid Card
        pyramid_card = ParameterCard(
            self.sidebar,
            title="MULTISCALE LAPLACIAN PYRAMID",
            badge="OCTAVE",
            help_text=(
                "4-octave detail synthesis (L1 micro-texture, L2 contours, "
                "G3 base illumination).\n\n"
                "Micro-Texture Detail (0.5–2.0): fabric, foliage, pore gain.\n"
                "Structural Contours (0.8–1.8): edge and shape volume gain.\n"
                "Dynamic Range Compression (0.0–1.0): base illumination "
                "squeeze for harsh light. Higher values can look flat."
            ),
        )
        pyramid_card.pack(fill="x", padx=12, pady=4)
        self.pyramid_switch = ctk.CTkSwitch(pyramid_card, text="Enable Multiscale Pyramid", font=ctk.CTkFont(family="Segoe UI", size=11), progress_color="#38bdf8")
        self.pyramid_switch.select()
        self.pyramid_switch.pack(anchor="w", padx=12, pady=(4, 2))
        self.pyramid_detail_slider = LabeledSlider(pyramid_card, label="Micro-Texture Detail (L1)", from_=0.5, to=2.0, default_val=1.20, step=0.05, is_float=True)
        self.pyramid_detail_slider.pack(fill="x", padx=8)
        self.pyramid_struct_slider = LabeledSlider(pyramid_card, label="Structural Contours (L2)", from_=0.8, to=1.8, default_val=1.10, step=0.05, is_float=True)
        self.pyramid_struct_slider.pack(fill="x", padx=8)
        self.pyramid_dynrange_slider = LabeledSlider(pyramid_card, label="Dynamic Range Compression (G3)", from_=0.0, to=1.0, default_val=0.0, step=0.05, is_float=True)
        self.pyramid_dynrange_slider.pack(fill="x", padx=8)

        # 4. Atmospheric Dehazing Card
        dehaze_card = ParameterCard(
            self.sidebar,
            title="ATMOSPHERIC DEHAZING",
            badge="DCP",
            help_text=(
                "Dark Channel Prior fog/haze removal.\n\n"
                "Strength 0.0–1.0: 0.3–0.6 suits light haze, 0.7+ dense fog. "
                "Too high darkens skies and boosts noise. Off by default."
            ),
        )
        dehaze_card.pack(fill="x", padx=12, pady=4)
        self.dehaze_switch = ctk.CTkSwitch(dehaze_card, text="Enable Dark Channel Dehazing", font=ctk.CTkFont(family="Segoe UI", size=11), progress_color="#38bdf8")
        self.dehaze_switch.pack(anchor="w", padx=12, pady=(4, 2))
        self.dehaze_slider = LabeledSlider(dehaze_card, label="Dehazing Strength", from_=0.0, to=1.0, default_val=0.50, step=0.05, is_float=True)
        self.dehaze_slider.pack(fill="x", padx=8)

        # 5. Pre-Restoration Conditioning Card
        restore_card = ParameterCard(
            self.sidebar,
            title="PRE-RESTORATION CONDITIONING",
            badge="RESTORE",
            help_text=(
                "Cleanup before upscaling.\n\n"
                "Subpixel Anti-Aliasing (0–100): dissolves pixel blocks and "
                "JPEG seams. 20–50 subtle, 60+ for blocky avatars.\n"
                "Shock Deblur (0–100): steepens soft lens/motion edges. "
                "20–40 subtle, 50+ for out-of-focus shots. Both 0 = off."
            ),
        )
        restore_card.pack(fill="x", padx=12, pady=4)
        self.depixel_slider = LabeledSlider(restore_card, label="Subpixel Anti-Aliasing", from_=0, to=100, default_val=0, step=5, is_float=False)
        self.depixel_slider.pack(fill="x", padx=8)
        self.deblur_slider = LabeledSlider(restore_card, label="Structure Tensor Shock Deblur", from_=0, to=100, default_val=0, step=5, is_float=False)
        self.deblur_slider.pack(fill="x", padx=8)

        # 6. Spatial Scaling Card
        scale_card = ParameterCard(
            self.sidebar,
            title="SPATIAL SUPER-RESOLUTION",
            badge="SCALE",
            help_text=(
                "Output magnification.\n\n"
                "1.0 keeps original size (quality pass only), 2.0 doubles "
                "width and height, up to 4.0. Larger scales need more RAM "
                "and time; the OOM guard caps output at 40 MP."
            ),
        )
        scale_card.pack(fill="x", padx=12, pady=4)
        self.scale_slider = LabeledSlider(scale_card, label="Magnification Scale", from_=1.0, to=4.0, default_val=2.0, step=0.5, is_float=True)
        self.scale_slider.pack(fill="x", padx=8)

        # 7. Detail Clarity Card
        cas_card = ParameterCard(
            self.sidebar,
            title="DETAIL CLARITY",
            badge="CAS",
            help_text=(
                "Contrast-Adaptive Sharpening (halo-free).\n\n"
                "0.0 off, 0.8–1.4 natural crispness, 2.0+ aggressive. "
                "Gain is clamped by local contrast so strong edges never "
                "grow halos; noisy shots prefer ≤1.0."
            ),
        )
        cas_card.pack(fill="x", padx=12, pady=4)
        self.cas_slider = LabeledSlider(cas_card, label="Contrast-Adaptive Sharpening", from_=0.0, to=3.0, default_val=1.1, step=0.1, is_float=True)
        self.cas_slider.pack(fill="x", padx=8)

        # 8. Structural Denoise Card
        swf_card = ParameterCard(
            self.sidebar,
            title="STRUCTURAL DENOISE",
            badge="SWF",
            help_text=(
                "Side Window Filter: edge-preserving denoise.\n\n"
                "Cleaning Power 10–40 keeps film grain, 50–60 balances phone "
                "noise, 70+ smooths heavily. Corners and text stay sharp. "
                "Clean shots auto-skip it (diagnosed sigma < 2.0)."
            ),
        )
        swf_card.pack(fill="x", padx=12, pady=4)
        self.denoise_switch = ctk.CTkSwitch(swf_card, text="Enable Side Window Filter", font=ctk.CTkFont(family="Segoe UI", size=11), progress_color="#38bdf8")
        self.denoise_switch.select()
        self.denoise_switch.pack(anchor="w", padx=12, pady=(4, 2))
        self.denoise_slider = LabeledSlider(swf_card, label="Cleaning Power", from_=10, to=100, default_val=40, step=5, is_float=False)
        self.denoise_slider.pack(fill="x", padx=8)

        # 9. Dynamic Range & Exposure Card
        bimef_card = ParameterCard(
            self.sidebar,
            title="DYNAMIC RANGE & EXPOSURE",
            badge="BIMEF",
            help_text=(
                "Tone and exposure.\n\n"
                "HDR Boost 1.0–4.0: midtone recovery with pinned blacks "
                "(~1.5 subtle, ~2.0 balanced, 3.0+ dramatic).\n"
                "Exposure Offset −50…+50: linear brightness shift. Negative "
                "protects highlights, positive lifts dark shots."
            ),
        )
        bimef_card.pack(fill="x", padx=12, pady=4)
        self.contrast_slider = LabeledSlider(bimef_card, label="HDR Dynamic Range Boost", from_=1.0, to=4.0, default_val=1.8, step=0.1, is_float=True)
        self.contrast_slider.pack(fill="x", padx=8)
        self.bright_slider = LabeledSlider(bimef_card, label="Radiometric Exposure Offset", from_=-50, to=50, default_val=0, step=5, is_float=False)
        self.bright_slider.pack(fill="x", padx=8)

        # 10. Perceptual Color & White Balance Card
        color_card = ParameterCard(
            self.sidebar,
            title="PERCEPTUAL COLOR",
            badge="OKLAB / CAT16",
            help_text=(
                "Color science.\n\n"
                "Oklab Vibrance 1.0–1.5: saturation along straight hue lines "
                "(1.05–1.15 natural, no blue-to-purple shift; shadows gated).\n"
                "CAT16 Temperature −30…+30: white balance. Negative cools "
                "yellow indoor light, positive warms daylight."
            ),
        )
        color_card.pack(fill="x", padx=12, pady=4)
        self.vibrance_slider = LabeledSlider(color_card, label="Oklab LMS Vibrance", from_=1.0, to=1.5, default_val=1.10, step=0.05, is_float=True)
        self.vibrance_slider.pack(fill="x", padx=8)
        self.temp_slider = LabeledSlider(color_card, label="CAT16 Color Temperature", from_=-30, to=30, default_val=0, step=5, is_float=False)
        self.temp_slider.pack(fill="x", padx=8)

        # 11. Synchronized Portrait Retouching Card
        portrait_card = ParameterCard(
            self.sidebar,
            title="PORTRAIT RETOUCHING",
            badge="YUNET / FGF",
            help_text=(
                "Face retouching (YuNet detection + guided filter).\n\n"
                "Skin Smoothing 0–100: blemish softening inside faces only "
                "(20–40 natural, eyes excluded). 0 disables.\n"
                "Eye Clarity 1.0–2.0: iris/catchlight pop (1.2–1.5 lifelike). "
                "No faces detected = no change."
            ),
        )
        portrait_card.pack(fill="x", padx=12, pady=4)
        self.smooth_slider = LabeledSlider(portrait_card, label="Fast Guided Skin Smoothing", from_=0, to=100, default_val=35, step=5, is_float=False)
        self.smooth_slider.pack(fill="x", padx=8)
        self.eye_slider = LabeledSlider(portrait_card, label="Eye Catchlight Sharpness", from_=1.0, to=2.0, default_val=1.3, step=0.1, is_float=True)
        self.eye_slider.pack(fill="x", padx=8)

        # 12. Output Format Card
        fmt_card = ParameterCard(
            self.sidebar,
            title="CONTAINER FORMAT",
            badge="EXPORT",
            help_text=(
                "Output file container.\n\n"
                "PNG keeps alpha and is lossless. JPEG/WebP save at 95% "
                "quality (alpha dropped for JPEG). EXIF and ICC profile "
                "carry over when present."
            ),
        )
        fmt_card.pack(fill="x", padx=12, pady=(4, 16))
        self.fmt_seg = ctk.CTkSegmentedButton(
            fmt_card,
            values=["PNG", "JPEG", "WebP"],
            selected_color="#0284c7",
            unselected_color="#090d16",
            font=ctk.CTkFont(family="Consolas", size=10, weight="bold"),
        )
        self.fmt_seg.set("PNG")
        self.fmt_seg.pack(fill="x", padx=8, pady=(4, 8))

    def _on_mode_change(self, mode_str: str) -> None:
        """Handles execution engine mode change, enabling/disabling widgets appropriately."""
        is_neural = (mode_str == "Neural AI")

        # CAS Sharpen: enabled in PureDSP and Hybrid, disabled in Neural AI
        self.cas_slider.set_enabled(not is_neural)

        # Contrast & Brightness (BIMEF): enabled in PureDSP and Hybrid, disabled in Neural AI
        self.contrast_slider.set_enabled(not is_neural)
        self.bright_slider.set_enabled(not is_neural)

        # Perceptual Vibrance & Color Temp: enabled in PureDSP and Hybrid, disabled in Neural AI
        self.vibrance_slider.set_enabled(not is_neural)
        self.temp_slider.set_enabled(not is_neural)

        # SWF Denoise: enabled in PureDSP and Hybrid, disabled in Neural AI
        self.denoise_switch.configure(state="normal" if not is_neural else "disabled")
        self.denoise_slider.set_enabled(not is_neural)

    def _on_preset_change(self, preset_name: str) -> None:
        """Loads preset configuration into slider widgets."""
        # CTkOptionMenu may pass a tkinter Event on some bindings; guard.
        if not isinstance(preset_name, str):
            return
        try:
            cfg = get_preset_config(preset_name)
        except ValueError:
            return
        self.scale_slider.set(cfg.scale)
        self.cas_slider.set(cfg.sharpen_strength)
        self.denoise_slider.set(cfg.denoise_intensity)
        if cfg.enable_denoise:
            self.denoise_switch.select()
        else:
            self.denoise_switch.deselect()
        self.contrast_slider.set(cfg.contrast_boost)
        self.bright_slider.set(cfg.brightness_shift)
        self.vibrance_slider.set(cfg.vibrance_boost)
        self.temp_slider.set(cfg.color_temperature)
        self.smooth_slider.set(cfg.portrait_smooth)
        self.eye_slider.set(cfg.eye_clarity)
        self.depixel_slider.set(getattr(cfg, "depixel_strength", 0))
        self.deblur_slider.set(getattr(cfg, "deblur_strength", 0))
        if hasattr(cfg, "enable_dehaze"):
            if cfg.enable_dehaze:
                self.dehaze_switch.select()
            else:
                self.dehaze_switch.deselect()
            self.dehaze_slider.set(cfg.dehaze_strength)
        if hasattr(cfg, "enable_pyramid"):
            if cfg.enable_pyramid:
                self.pyramid_switch.select()
            else:
                self.pyramid_switch.deselect()
            self.pyramid_detail_slider.set(cfg.pyramid_micro_texture)
            self.pyramid_struct_slider.set(cfg.pyramid_structure_boost)
            self.pyramid_dynrange_slider.set(getattr(cfg, "pyramid_dynamic_range", 0.0))
        # Performance/telemetry parity with CLI --fast / --no-* flags.
        if getattr(cfg, "enable_diagnostics", True):
            self.diagnostics_switch.select()
        else:
            self.diagnostics_switch.deselect()
        if getattr(cfg, "enable_semantic_guidance", True):
            self.semantic_switch.select()
        else:
            self.semantic_switch.deselect()
        if getattr(cfg, "enable_contrast", True):
            self.contrast_switch.select()
        else:
            self.contrast_switch.deselect()

    def _on_view_mode_change(self, mode_str: str) -> None:
        """Updates interactive canvas rendering mode."""
        mapping = {"Enhanced": "enhanced", "Original": "original", "Split View": "split"}
        view = mapping.get(mode_str, "enhanced")
        self.canvas.set_view_mode(view)
        if view == "split":
            self.split_ctrl_frame.pack(side="left", padx=(0, 10))
            self.split_slider.set(self.canvas.split_pos)
            self.split_label.configure(text=f"Split: {int(round(self.canvas.split_pos * 100))}%")
        else:
            self.split_ctrl_frame.pack_forget()

    def _on_canvas_split_change(self, pos: float) -> None:
        """Synchronizes toolbar slider when divider is dragged on canvas."""
        self.split_slider.set(pos)
        self.split_label.configure(text=f"Split: {int(round(pos * 100))}%")

    def _on_split_slider_move(self, val: float) -> None:
        """Synchronizes canvas divider when toolbar slider is moved."""
        self.split_label.configure(text=f"Split: {int(round(val * 100))}%")
        self.canvas.set_split_pos(val, notify=False)

    def _run_quick_diagnostics_async(self, bgr: np.ndarray) -> None:
        """Runs fast diagnostic signal analysis and updates HUD."""
        def diag_worker():
            try:
                sem = extract_semantic_masks(bgr)
                diag = diagnose_image(bgr, semantic_breakdown=sem.breakdown)
                self.current_diagnostics = diag
                self.after(0, lambda d=diag: self.hud_card.update_diagnostics(d))
            except (cv2.error, ValueError, TypeError, RuntimeError) as e:
                logger.warning("Quick diagnostics background analysis failed: %s", e)

        threading.Thread(target=diag_worker, daemon=True).start()

    def _open_file_dialog(self) -> None:
        """Prompts user to select image file and loads it."""
        path = filedialog.askopenfilename(
            title="Select Image to Enhance",
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.webp *.bmp *.tiff"),
                ("All Files", "*.*"),
            ],
        )
        if not path:
            return

        try:
            from purescale.cli import load_image_with_alpha
            bgr, alpha, exif, icc = load_image_with_alpha(path, return_meta=True)
            self.current_file_path = path
            self.current_orig_bgr = bgr
            self.current_alpha = alpha
            self.current_exif = exif
            self.current_icc_profile = icc
            self.current_enh_bgr = None

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            orig_pil = Image.fromarray(rgb)
            self.canvas.set_images(orig_pil, None)

            h, w = bgr.shape[:2]
            self.status_label.configure(text=f"Loaded: {os.path.basename(path)} ({w}x{h})")
            self.progress_bar.set(0.0)

            # Analyze image signal immediately
            self._run_quick_diagnostics_async(bgr)
        except (OSError, IOError, ValueError, cv2.error) as err:
            messagebox.showerror("Image Load Error", f"Could not load image:\n{err}")

    def _apply_diagnostics_to_ui(self, diag: DiagnosticsResult) -> None:
        """Synchronizes UI sliders with auto-tuned parameters."""
        rec = diag.recommended_parameters or auto_tune_parameters(diag)
        if "denoise_intensity" in rec:
            self.denoise_slider.set(rec["denoise_intensity"])
        if "enable_denoise" in rec:
            if rec["enable_denoise"]:
                self.denoise_switch.select()
            else:
                self.denoise_switch.deselect()
        if "sharpen_strength" in rec:
            self.cas_slider.set(rec["sharpen_strength"])
        if "deblur_strength" in rec:
            self.deblur_slider.set(rec["deblur_strength"])
        if "contrast_boost" in rec:
            self.contrast_slider.set(rec["contrast_boost"])
        if "brightness_shift" in rec:
            self.bright_slider.set(rec["brightness_shift"])
        if "color_temperature" in rec:
            self.temp_slider.set(rec["color_temperature"])
        if "vibrance_boost" in rec:
            self.vibrance_slider.set(rec["vibrance_boost"])
        if "enable_dehaze" in rec:
            if rec["enable_dehaze"]:
                self.dehaze_switch.select()
            else:
                self.dehaze_switch.deselect()
        if "dehaze_strength" in rec:
            self.dehaze_slider.set(rec["dehaze_strength"])
        if "pyramid_micro_texture" in rec:
            self.pyramid_detail_slider.set(rec["pyramid_micro_texture"])
        if "pyramid_structure_boost" in rec:
            self.pyramid_struct_slider.set(rec["pyramid_structure_boost"])

    def _on_auto_enhance_click(self) -> None:
        """Executes autonomous auto-tuning and enhancement."""
        if self.current_orig_bgr is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        if self.current_diagnostics is None:
            self.current_diagnostics = diagnose_image(self.current_orig_bgr)
            self.hud_card.update_diagnostics(self.current_diagnostics)

        self._apply_diagnostics_to_ui(self.current_diagnostics)
        self._start_enhancement_async()

    def _get_current_config(self) -> PipelineConfig:
        """Gathers parameters from UI controls into a PipelineConfig."""
        mode_map = {
            "PureDSP": ProcessingMode.PURE_DSP,
            "Neural AI": ProcessingMode.NEURAL_AI,
            "Hybrid": ProcessingMode.HYBRID,
        }
        cfg = PipelineConfig(
            mode=mode_map.get(self.mode_seg.get(), ProcessingMode.PURE_DSP),
            device=DeviceTarget.AUTO,
            enable_diagnostics=bool(self.diagnostics_switch.get()),
            auto_tune=False,
            enable_dehaze=bool(self.dehaze_switch.get()),
            dehaze_strength=self.dehaze_slider.get(),
            enable_pyramid=bool(self.pyramid_switch.get()),
            pyramid_micro_texture=self.pyramid_detail_slider.get(),
            pyramid_structure_boost=self.pyramid_struct_slider.get(),
            pyramid_dynamic_range=self.pyramid_dynrange_slider.get(),
            enable_semantic_guidance=bool(self.semantic_switch.get()),
            scale=self.scale_slider.get(),
            sharpen_strength=self.cas_slider.get(),
            enable_denoise=bool(self.denoise_switch.get()),
            denoise_intensity=int(self.denoise_slider.get()),
            enable_contrast=bool(self.contrast_switch.get()),
            contrast_boost=self.contrast_slider.get(),
            brightness_shift=int(self.bright_slider.get()),
            vibrance_boost=self.vibrance_slider.get(),
            color_temperature=int(self.temp_slider.get()),
            depixel_strength=int(self.depixel_slider.get()),
            deblur_strength=int(self.deblur_slider.get()),
            portrait_smooth=int(self.smooth_slider.get()),
            eye_clarity=self.eye_slider.get(),
            output_format=self.fmt_seg.get(),
        )
        return cfg

    def _start_enhancement_async(self) -> None:
        """Spawns enhancement worker thread."""
        if self.current_orig_bgr is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        if self.is_processing:
            return

        cfg = self._get_current_config()

        # Pre-flight OOM guard: fail fast before the worker thread starts.
        try:
            h0, w0 = self.current_orig_bgr.shape[:2]
            dest_mp = (w0 * cfg.scale) * (h0 * cfg.scale) / 1e6
            in_mp = w0 * h0 / 1e6
            if max(in_mp, dest_mp) > cfg.max_megapixels:
                messagebox.showerror(
                    "Image Too Large",
                    f"Output would be {dest_mp:.1f} MP (limit {cfg.max_megapixels:.1f} MP).\n"
                    "Lower the scale or raise max_megapixels.",
                )
                return
        except (AttributeError, ValueError, TypeError):
            pass

        self.is_processing = True
        self.btn_enhance.configure(state="disabled", text="Processing...")
        self.btn_auto.configure(state="disabled")
        self.progress_bar.set(0.05)
        self.status_label.configure(text="Enhancement in progress...")

        def worker():
            try:
                def on_progress(stage: str, ratio: float):
                    self.after(0, lambda s=stage, r=ratio: self._update_progress(s, r))

                res = self.pipeline.enhance(
                    self.current_orig_bgr,
                    config=cfg,
                    alpha=self.current_alpha,
                    progress_callback=on_progress,
                )
                self.after(0, lambda r=res: self._on_enhancement_complete(r))
            except (cv2.error, RuntimeError, ValueError, MemoryError, OSError) as err:
                self.after(0, lambda e=err: self._on_enhancement_error(e))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def _update_progress(self, stage_name: str, ratio: float) -> None:
        self.progress_bar.set(ratio)
        self.status_label.configure(text=f"Stage: {stage_name}")

    def _on_enhancement_complete(self, res: ProcessingResult) -> None:
        self.is_processing = False
        self.btn_enhance.configure(state="normal", text="Enhance Image")
        self.btn_auto.configure(state="normal")
        self.progress_bar.set(1.0)
        self.current_enh_bgr = res.image

        if res.diagnostics:
            self.current_diagnostics = res.diagnostics
            self.hud_card.update_diagnostics(res.diagnostics)

        # Convert to PIL and update canvas
        rgb = cv2.cvtColor(res.image, cv2.COLOR_BGR2RGB)
        enh_pil = Image.fromarray(rgb)
        self.canvas.set_images(self.canvas.orig_pil, enh_pil)

        h, w = res.image.shape[:2]
        faces_str = f" | {res.faces_detected} faces" if res.faces_detected > 0 else ""
        self.status_label.configure(
            text=f"Done in {res.latency_ms:.1f} ms ({res.backend_name}) | {w}x{h}{faces_str}"
        )

        # Completion dialog: success is otherwise silent (status bar only).
        # Failures already surface via _on_enhancement_error's error dialog.
        if self.current_orig_bgr is not None:
            oh, ow = self.current_orig_bgr.shape[:2]
        else:
            ow, oh = w, h
        detail = f"Finished in {res.latency_ms:.1f} ms ({res.backend_name}).\nResolution: {ow}x{oh} → {w}x{h}."
        if res.faces_detected > 0:
            detail += f"\nFaces retouched: {res.faces_detected}."
        messagebox.showinfo("Enhancement Complete", detail)

    def _on_enhancement_error(self, err: Exception) -> None:
        self.is_processing = False
        self.btn_enhance.configure(state="normal", text="Enhance Image")
        self.btn_auto.configure(state="normal")
        self.progress_bar.set(0.0)
        self.status_label.configure(text=f"Error: {err}")
        messagebox.showerror("Enhancement Error", f"Processing failed:\n{err}")

    def _save_file_dialog(self) -> None:
        """Prompts user to save enhanced image."""
        if self.current_enh_bgr is None:
            messagebox.showwarning("No Result", "No enhanced image available to save.")
            return

        fmt = self.fmt_seg.get()
        ext = ".png" if fmt == "PNG" else (".jpg" if fmt == "JPEG" else ".webp")

        init_file = "enhanced" + ext
        if self.current_file_path:
            base = os.path.splitext(os.path.basename(self.current_file_path))[0]
            init_file = f"{base}_enhanced{ext}"

        dest_path = filedialog.asksaveasfilename(
            title="Save Enhanced Image",
            initialfile=init_file,
            defaultextension=ext,
            filetypes=[
                ("PNG Image", "*.png"),
                ("JPEG Image", "*.jpg *.jpeg"),
                ("WebP Image", "*.webp"),
                ("All Files", "*.*"),
            ],
        )
        if not dest_path:
            return

        try:
            from purescale.cli import save_image_with_alpha
            save_image_with_alpha(
                self.current_enh_bgr, self.current_alpha, dest_path, fmt,
                exif=self.current_exif, icc_profile=self.current_icc_profile,
            )
            self.status_label.configure(text=f"Saved: {os.path.basename(dest_path)}")
        except (OSError, IOError, ValueError, cv2.error) as err:
            messagebox.showerror("Save Error", f"Could not save file:\n{err}")


def launch_gui() -> None:
    """Entrypoint function to instantiate and run GUI."""
    app = PureScaleApp()
    app.mainloop()


if __name__ == "__main__":
    launch_gui()
