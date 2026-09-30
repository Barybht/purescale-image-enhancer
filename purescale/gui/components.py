"""Modular UI card components, sliders, and badges for Obsidian Studio."""

from typing import Callable, Optional
import customtkinter as ctk
from purescale.gui.theme import PALETTE, font
from tkinter import messagebox

from purescale.config import AUTO_STYLE_CONFIDENCE, DiagnosticsResult


class ParameterCard(ctk.CTkFrame):
    """Card container with 1px border, header label, technical badge, and (?) help."""

    def __init__(self, master, title: str, badge: str = "", help_text: str = "", **kwargs):
        kwargs.setdefault("fg_color", PALETTE["surface"])
        kwargs.setdefault("border_width", 1)
        kwargs.setdefault("border_color", PALETTE["border"])
        kwargs.setdefault("corner_radius", 6)
        super().__init__(master, **kwargs)

        self.header_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.header_frame.pack(fill="x", padx=12, pady=(8, 4))

        self.title_label = ctk.CTkLabel(
            self.header_frame,
            text=title,
            font=font("mono11b"),
            text_color=PALETTE["text"],
        )
        self.title_label.pack(side="left")

        if help_text:
            self.help_text = help_text
            self.help_title = title
            self.help_btn = ctk.CTkButton(
                self.header_frame,
                text="?",
                width=20,
                height=20,
                corner_radius=10,
                fg_color="transparent",
                border_width=1,
                border_color=PALETTE["accent_bright"],
                text_color=PALETTE["accent_bright"],
                hover_color=PALETTE["surface_hover"],
                font=font("mono10b"),
                command=self._show_help,
            )
            self.help_btn.pack(side="right", padx=(6, 0))

        if badge:
            self.badge_label = ctk.CTkLabel(
                self.header_frame,
                text=f"[{badge}]",
                font=font("mono9b"),
                text_color=PALETTE["accent_bright"],
            )
            self.badge_label.pack(side="right")

    def _show_help(self) -> None:
        """Shows the card's help dialog."""
        messagebox.showinfo(self.help_title, self.help_text)


class LabeledSlider(ctk.CTkFrame):
    """Precision slider with title, unit, and dynamic numerical readout."""

    def __init__(
        self,
        master,
        label: str,
        from_: float,
        to: float,
        default_val: float,
        step: float = 1.0,
        is_float: bool = False,
        command: Optional[Callable[[float], None]] = None,
        **kwargs
    ):
        kwargs.setdefault("fg_color", "transparent")
        super().__init__(master, **kwargs)

        self.command = command
        self.is_float = is_float

        top_row = ctk.CTkFrame(self, fg_color="transparent")
        top_row.pack(fill="x", padx=4, pady=(2, 0))

        self.name_label = ctk.CTkLabel(
            top_row,
            text=label,
            font=font("ui11"),
            text_color=PALETTE["text_muted"],
        )
        self.name_label.pack(side="left")

        self.val_label = ctk.CTkLabel(
            top_row,
            text=f"{default_val:.2f}" if is_float else str(int(default_val)),
            font=font("mono11b"),
            text_color=PALETTE["accent_bright"],
        )
        self.val_label.pack(side="right")

        span = to - from_
        steps = int(round(span / step)) if step and step > 0 and span > 0 else 0
        steps = max(1, steps)
        self.slider = ctk.CTkSlider(
            self,
            from_=from_,
            to=to,
            number_of_steps=steps,
            command=self._on_change,
            fg_color=PALETTE["surface_hover"],
            progress_color=PALETTE["accent"],
            button_color=PALETTE["accent_bright"],
            button_hover_color=PALETTE["accent_lightest"],
            height=16,
        )
        self.slider.set(default_val)
        self.slider.pack(fill="x", padx=4, pady=(2, 6))

    def _on_change(self, val: float) -> None:
        if self.is_float:
            self.val_label.configure(text=f"{val:.2f}")
        else:
            self.val_label.configure(text=str(int(round(val))))
        if self.command:
            self.command(val)

    def get(self) -> float:
        val = self.slider.get()
        return float(val) if self.is_float else float(int(round(val)))

    def set(self, val: float) -> None:
        self.slider.set(val)
        if self.is_float:
            self.val_label.configure(text=f"{val:.2f}")
        else:
            self.val_label.configure(text=str(int(round(val))))

    def set_enabled(self, enabled: bool) -> None:
        """Enables or disables the slider and dims text."""
        state = "normal" if enabled else "disabled"
        self.slider.configure(state=state)
        self.name_label.configure(text_color=PALETTE["text_muted"] if enabled else PALETTE["text_faint"])
        self.val_label.configure(text_color=PALETTE["accent_bright"] if enabled else PALETTE["text_faint"])


class DiagnosticsHUDCard(ParameterCard):
    """Live Signal Diagnostics and Quality Telemetry card."""

    def __init__(self, master, **kwargs):
        super().__init__(
            master,
            title="SIGNAL DIAGNOSTICS",
            badge="LIVE HUD",
            help_text=(
                "Physical signal measurements of the loaded image.\n\n"
                "Noise Floor (MAD): sensor noise sigma from Haar wavelets.\n"
                "Optical Blur Index: defocus estimate from Laplacian variance.\n"
                "Dynamic Entropy: histogram spread and clipping.\n"
                "Atmospheric Haze: fog detection from the dark channel.\n"
                "Content Style: photo / anime / manga classification.\n"
                "Scene Semantics: soft sky / foliage / skin / shadow share.\n\n"
                "Auto-Enhance copies these into the sliders below."
            ),
            **kwargs,
        )

        hud_frame = ctk.CTkFrame(self, fg_color="transparent")
        hud_frame.pack(fill="x", padx=12, pady=(2, 8))

        def create_metric_row(label_text: str):
            row = ctk.CTkFrame(hud_frame, fg_color="transparent")
            row.pack(fill="x", pady=1)
            lbl = ctk.CTkLabel(row, text=label_text, font=font("ui10"), text_color=PALETTE["text_muted"])
            lbl.pack(side="left")
            val = ctk.CTkLabel(row, text="--", font=font("mono10b"), text_color=PALETTE["accent_bright"])
            val.pack(side="right")
            return val

        self.noise_val = create_metric_row("Noise Floor (MAD)")
        self.blur_val = create_metric_row("Optical Blur Index")
        self.entropy_val = create_metric_row("Dynamic Entropy")
        self.cast_val = create_metric_row("White Balance")
        self.light_val = create_metric_row("Lighting Geometry")
        self.haze_val = create_metric_row("Atmospheric Haze")
        self.style_val = create_metric_row("Content Style")
        self.scene_val = create_metric_row("Scene Semantics")

    def update_diagnostics(self, diag: Optional[DiagnosticsResult]) -> None:
        """Refreshes telemetry values on HUD."""
        if diag is None:
            self.noise_val.configure(text="--", text_color=PALETTE["text_faint"])
            self.blur_val.configure(text="--", text_color=PALETTE["text_faint"])
            self.entropy_val.configure(text="--", text_color=PALETTE["text_faint"])
            self.cast_val.configure(text="--", text_color=PALETTE["text_faint"])
            self.light_val.configure(text="--", text_color=PALETTE["text_faint"])
            self.haze_val.configure(text="--", text_color=PALETTE["text_faint"])
            self.style_val.configure(text="--", text_color=PALETTE["text_faint"])
            self.scene_val.configure(text="--", text_color=PALETTE["text_faint"])
            return

        # Noise
        noise_color = PALETTE["ok"] if diag.noise_sigma < 3.0 else (PALETTE["warn"] if diag.noise_sigma < 8.0 else PALETTE["err"])
        chroma_sigma = float(getattr(diag, "chroma_noise_sigma", 0.0) or 0.0)
        # Same >= 2.0 display gate as the summary table (Clean band edge).
        if chroma_sigma >= 2.0:
            self.noise_val.configure(text=f"{diag.noise_sigma:.1f}Y/{chroma_sigma:.1f}C [{diag.noise_category}]", text_color=noise_color)
        else:
            self.noise_val.configure(text=f"{diag.noise_sigma:.1f} [{diag.noise_category}]", text_color=noise_color)

        # Blur
        blur_color = PALETTE["ok"] if diag.blur_score < 0.3 else (PALETTE["warn"] if diag.blur_score < 0.6 else PALETTE["err"])
        self.blur_val.configure(text=f"{diag.blur_score:.2f} [{diag.blur_category}]", text_color=blur_color)

        # Entropy
        self.entropy_val.configure(text=f"{diag.entropy:.1f} bits ({diag.dynamic_range} lvls)", text_color=PALETTE["accent_bright"])

        # White Balance (CAT16 Temp / Tint)
        tint_val = int(getattr(diag, "color_tint_offset", 0) or 0)
        cast_color = PALETTE["ok"] if (diag.color_cast_kelvin == 0 and tint_val == 0) else PALETTE["accent_bright"]
        self.cast_val.configure(text=f"{diag.color_cast_kelvin:+d}K/{tint_val:+d}T [{diag.color_cast_name}]", text_color=cast_color)

        # Lighting Geometry (Backlight ratio)
        backlight = bool(getattr(diag, "backlight_detected", False))
        ratio = float(getattr(diag, "backlight_ratio", 1.0) or 1.0)
        light_color = PALETTE["warn"] if backlight else PALETTE["ok"]
        self.light_val.configure(text=f"{'Backlit' if backlight else 'Balanced'} [{ratio:.1f}x]", text_color=light_color)

        # Haze
        haze_color = PALETTE["err"] if diag.haze_detected else PALETTE["ok"]
        haze_text = f"{diag.haze_index:.2f} [{'HAZE' if diag.haze_detected else 'CLEAR'}]"
        self.haze_val.configure(text=haze_text, text_color=haze_color)

        # Style (amber when below the auto-routing threshold: shown but not applied)
        style_conf = float(getattr(diag, "style_confidence", 0.0) or 0.0)
        style_name = str(getattr(diag, "suggested_style", "photo") or "photo")
        style_color = PALETTE["accent_bright"] if style_conf >= AUTO_STYLE_CONFIDENCE else PALETTE["warn"]
        self.style_val.configure(text=f"{style_name.title()} [{style_conf:.2f}]", text_color=style_color)

        # Scene breakdown (same 5% threshold as the CLI summary table;
        # wrapped HUD label shows top entries instead of truncating at 3).
        if diag.semantic_breakdown:
            top_classes = [f"{k.capitalize()[:4]}:{int(v*100)}%" for k, v in diag.semantic_breakdown.items() if v > 0.05]
            self.scene_val.configure(text=", ".join(top_classes[:4]) if top_classes else "General", text_color=PALETTE["text_soft"])
        else:
            self.scene_val.configure(text="General", text_color=PALETTE["text_soft"])

