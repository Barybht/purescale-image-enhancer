"""Modular UI card components, sliders, badges, and pickers for Obsidian Studio."""

from typing import Callable, List, Optional
import tkinter as tk
from tkinter import ttk
import customtkinter as ctk
from purescale.gui.theme import PALETTE, font
from tkinter import messagebox
import math

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
        self.step = step if step and step > 0 else 1.0

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
            text=self._fmt(default_val),
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

    def _fmt(self, val: float) -> str:
        """Formats readout to the slider's step precision (1.1 not 1.10)."""
        if not self.is_float:
            return str(int(round(val)))
        decimals = max(0, -int(math.floor(math.log10(self.step) + 1e-9)))
        return f"{val:.{decimals}f}"

    def _on_change(self, val: float) -> None:
        self.val_label.configure(text=self._fmt(val))
        if self.command:
            self.command(val)

    def get(self) -> float:
        val = self.slider.get()
        return float(val) if self.is_float else float(int(round(val)))

    def set(self, val: float) -> None:
        self.slider.set(val)
        self.val_label.configure(text=self._fmt(val))

    def set_enabled(self, enabled: bool) -> None:
        """Enables or disables the slider and dims text."""
        state = "normal" if enabled else "disabled"
        self.slider.configure(state=state)
        self.name_label.configure(text_color=PALETTE["text_muted"] if enabled else PALETTE["text_faint"])
        self.val_label.configure(text_color=PALETTE["accent_bright"] if enabled else PALETTE["text_faint"])


class PresetPicker(ctk.CTkFrame):
    """Themed preset picker with an owned popup list.

    Replaces CTkComboBox, whose system popup cannot be themed (white frame)
    and always expands to the full item count. This popup shows exactly
    ROWS_VISIBLE rows with a themed scrollbar, a live filter entry, and a
    1px themed border. Keyboard: type to filter, Up/Down to move,
    Enter to apply, Escape to dismiss.
    """

    ROWS_VISIBLE = 5
    ROW_HEIGHT_PX = 30

    def __init__(
        self,
        master,
        values: List[str],
        command: Optional[Callable[[str], None]] = None,
        **kwargs,
    ):
        kwargs.setdefault("fg_color", "transparent")
        super().__init__(master, **kwargs)
        self._values = list(values)
        self._command = command
        self._current = self._values[0] if self._values else ""
        self._popup: Optional[tk.Toplevel] = None
        self._wheel_ids: list = []

        self._field = ctk.CTkButton(
            self,
            text=self._current,
            command=self.toggle,
            anchor="center",
            height=32,
            corner_radius=6,
            fg_color=PALETTE["surface"],
            hover_color=PALETTE["surface_hover"],
            text_color=PALETTE["text"],
            font=font("mono11b"),
        )
        self._field.pack(fill="x")

    def get(self) -> str:
        """Returns the current preset label."""
        return self._current

    def set(self, label: str) -> None:
        """Sets the current preset label without firing the callback."""
        self._current = label
        self._field.configure(text=label)

    def toggle(self) -> None:
        """Opens the popup, or closes it when already open."""
        if self._popup is not None:
            self.close()
        else:
            self.open()

    def open(self) -> None:
        """Shows the popup under the field (flips above near screen bottom)."""
        if self._popup is not None:
            return
        root = self.winfo_toplevel()
        pop = tk.Toplevel(root)
        pop.withdraw()
        pop.overrideredirect(True)
        pop.configure(bg=PALETTE["border"])
        try:
            pop.attributes("-topmost", True)
        except tk.TclError:
            pass

        row_h = self.ROW_HEIGHT_PX
        list_h = row_h * self.ROWS_VISIBLE
        chrome_h = 46
        w = max(160, self._field.winfo_width())
        h = chrome_h + list_h
        x = self._field.winfo_rootx()
        y = self._field.winfo_rooty() + self._field.winfo_height() + 4
        if y + h > root.winfo_screenheight() - 40:
            y = max(0, self._field.winfo_rooty() - h - 4)
        pop.geometry(f"{w}x{h}+{x}+{y}")

        frame = ctk.CTkFrame(pop, fg_color=PALETTE["surface"], corner_radius=6)
        frame.pack(fill="both", expand=True, padx=1, pady=1)

        self._filter_entry = ctk.CTkEntry(
            frame,
            placeholder_text="Type to filter...",
            height=30,
            corner_radius=6,
            fg_color=PALETTE["bg_app"],
            border_color=PALETTE["border"],
            text_color=PALETTE["text"],
            placeholder_text_color=PALETTE["text_faint"],
            font=font("ui11"),
        )
        self._filter_entry.pack(fill="x", padx=6, pady=(6, 4))
        self._filter_entry.bind("<KeyRelease>", self._on_filter)

        list_frame = ctk.CTkFrame(frame, fg_color="transparent")
        list_frame.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        self._listbox = tk.Listbox(
            list_frame,
            height=self.ROWS_VISIBLE,
            activestyle="none",
            highlightthickness=0,
            borderwidth=0,
            relief="flat",
            bg=PALETTE["surface"],
            fg=PALETTE["text"],
            selectbackground=PALETTE["accent"],
            selectforeground=PALETTE["text"],
            font=("Consolas", 11),
        )
        self._listbox.pack(side="left", fill="both", expand=True)
        # ttk/club scrollbar: native tk.Scrollbar renders its trough in
        # system colors on Windows (white at rest). A clam-styled ttk bar
        # with the arrows removed stays fully themed.
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style_name = "PresetPicker.Vertical.TScrollbar"
        style.layout(style_name, [(
            "Vertical.Scrollbar.trough",
            {"sticky": "ns", "children": [
                ("Vertical.Scrollbar.thumb", {"expand": "1", "sticky": "nswe"}),
            ]},
        )])
        style.configure(
            style_name,
            background=PALETTE["surface_hover"],
            troughcolor=PALETTE["surface"],
            borderwidth=0,
            arrowsize=0,
            relief="flat",
        )
        style.map(
            style_name,
            background=[("active", PALETTE["accent"]), ("pressed", PALETTE["accent"])],
        )
        scroll = ttk.Scrollbar(
            list_frame,
            orient="vertical",
            command=self._listbox.yview,
            style=style_name,
        )
        scroll.pack(side="right", fill="y")
        self._listbox.configure(yscrollcommand=scroll.set)

        self._refill("")
        self._listbox.bind("<<ListboxSelect>>", lambda _e: self._choose())
        self._listbox.bind("<Return>", lambda _e: self._choose())
        pop.bind("<Escape>", lambda _e: self.close())
        pop.bind("<FocusOut>", lambda _e: self._close_if_unfocused(pop))
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self._wheel_ids.append((seq, root.bind(seq, lambda _e: self.close(), add=True)))

        self._popup = pop
        pop.deiconify()
        self._filter_entry.focus_set()

    def _refill(self, query: str) -> None:
        items = [v for v in self._values if query.lower() in v.lower()]
        self._listbox.delete(0, "end")
        for item in items:
            self._listbox.insert("end", item)
        try:
            idx = items.index(self._current)
        except ValueError:
            idx = 0 if items else None
        if idx is not None:
            self._listbox.selection_set(idx)
            self._listbox.see(idx)

    def _on_filter(self, _event=None) -> None:
        if self._popup is None:
            return
        self._refill(self._filter_entry.get())

    def _choose(self) -> None:
        if self._popup is None:
            return
        sel = self._listbox.curselection()
        if not sel:
            return
        label = self._listbox.get(sel[0])
        self.close()
        self.set(label)
        if self._command:
            self._command(label)

    def _close_if_unfocused(self, pop: tk.Toplevel) -> None:
        if self._popup is not pop:
            return

        def check():
            if self._popup is not pop:
                return
            try:
                fw = pop.focus_get()
            except tk.TclError:
                fw = None
            if fw is None or not str(fw).startswith(str(pop) + "."):
                self.close()

        self.after(20, check)

    def close(self) -> None:
        """Destroys the popup and releases global bindings."""
        pop, self._popup = self._popup, None
        if self._wheel_ids:
            try:
                root = self.winfo_toplevel()
                for seq, funcid in self._wheel_ids:
                    root.unbind(seq, funcid)
            except tk.TclError:
                pass
            self._wheel_ids = []
        if pop is not None:
            try:
                pop.destroy()
            except tk.TclError:
                pass
            try:
                self._field.focus_set()
            except tk.TclError:
                pass


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

