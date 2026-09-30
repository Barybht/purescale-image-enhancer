"""Interactive zoomable, pannable canvas with draggable split-view comparison."""

import tkinter as tk
from typing import Callable, Optional
import numpy as np
from PIL import Image, ImageTk


class InteractiveCanvas(tk.Canvas):
    """
    High-performance viewport supporting pan, zoom, fit-to-window,
    and side-by-side split-view divider compositing with interactive drag.
    """

    def __init__(self, master, **kwargs):
        kwargs.setdefault("bg", "#090d16")
        kwargs.setdefault("highlightthickness", 0)
        super().__init__(master, **kwargs)

        self.orig_pil: Optional[Image.Image] = None
        self.enh_pil: Optional[Image.Image] = None
        self.view_mode: str = "enhanced"  # "enhanced", "original", "split"
        self.split_pos: float = 0.5       # 0.0 to 1.0
        self.dragging_divider: bool = False
        self.on_split_change: Optional[Callable[[float], None]] = None

        # Viewport transform
        self.zoom_level: float = 1.0
        self.pan_x: float = 0.0
        self.pan_y: float = 0.0
        self.drag_start_x: float = 0.0
        self.drag_start_y: float = 0.0
        self.is_fit_mode: bool = True

        self._tk_img: Optional[ImageTk.PhotoImage] = None
        self._redraw_after: Optional[str] = None

        # Bind event handlers
        self.bind("<Configure>", self._on_resize)
        self.bind("<ButtonPress-1>", self._on_drag_start)
        self.bind("<B1-Motion>", self._on_drag_motion)
        self.bind("<ButtonRelease-1>", self._on_drag_release)
        self.bind("<Motion>", self._on_mouse_move)
        self.bind("<MouseWheel>", self._on_mouse_wheel)
        # Linux wheel (X11 sends Button-4/5 instead of MouseWheel)
        self.bind("<Button-4>", lambda e: self._zoom_at(e.x, e.y, 1.15))
        self.bind("<Button-5>", lambda e: self._zoom_at(e.x, e.y, 0.85))
        self.bind("<Double-Button-1>", lambda e: self.fit_to_window())

    def _active_ref_image(self) -> Optional[Image.Image]:
        """Returns the PIL image that defines the active display geometry."""
        if self.view_mode == "original":
            return self.orig_pil
        return self.enh_pil or self.orig_pil

    def set_images(
        self,
        orig: Optional[Image.Image],
        enh: Optional[Image.Image],
        force_fit: Optional[bool] = None,
    ) -> None:
        """Sets active images and updates viewport to fit if requested or in fit mode."""
        prev_orig = self.orig_pil
        prev_enh = self.enh_pil

        self.orig_pil = orig
        self.enh_pil = enh

        # A brand new source image is being loaded if orig is provided and either:
        # - there was no previous image
        # - orig is a different image instance or has different dimensions
        # - enh is None (loading a fresh un-enhanced original)
        is_new_source = (
            orig is not None
            and (
                prev_orig is None
                or orig is not prev_orig
                or orig.size != prev_orig.size
                or enh is None
            )
        )

        if force_fit is True or is_new_source or self.is_fit_mode:
            self.is_fit_mode = True
            self.fit_to_window()
        else:
            # If not in fit mode, keep visual alignment when resolution changes
            if prev_enh is None and enh is not None and prev_orig is not None and enh.size[0] > 0:
                scale_change = float(orig.size[0]) / float(enh.size[0])
                self.zoom_level = max(0.01, self.zoom_level * scale_change)
            self.redraw()

    def set_view_mode(self, mode: str) -> None:
        """Updates display mode: 'enhanced', 'original', or 'split'."""
        self.view_mode = mode.lower()
        if self.is_fit_mode:
            self.fit_to_window()
        else:
            self.redraw()

    def set_split_pos(self, pos: float, notify: bool = True) -> None:
        """Sets divider position (0.0 to 1.0) and redraws."""
        self.split_pos = float(np.clip(pos, 0.0, 1.0))
        if notify and self.on_split_change:
            self.on_split_change(self.split_pos)
        self.redraw()

    def fit_to_window(self) -> None:
        """Calculates optimal zoom factor to fit active image inside viewport."""
        ref = self._active_ref_image()
        if ref is None:
            return

        try:
            self.update_idletasks()
        except (RuntimeError, tk.TclError):
            pass

        cw = self.winfo_width()
        ch = self.winfo_height()
        if cw <= 10 or ch <= 10:
            try:
                cfg_w = int(self.cget("width"))
                cfg_h = int(self.cget("height"))
                if cfg_w > 10 and cfg_h > 10:
                    cw, ch = cfg_w, cfg_h
            except (ValueError, TypeError, tk.TclError):
                pass

        if cw <= 10 or ch <= 10:
            # Widget not laid out yet (e.g. right after set_images in __init__);
            # retry once geometry is available instead of zooming to ~0.
            try:
                self.after(50, self.fit_to_window)
            except (RuntimeError, tk.TclError):
                pass
            return

        iw, ih = ref.size
        pad = 40
        scale_w = max(10, cw - pad) / float(iw)
        scale_h = max(10, ch - pad) / float(ih)
        self.zoom_level = max(0.01, min(scale_w, scale_h))

        self.pan_x = (cw - iw * self.zoom_level) / 2.0
        self.pan_y = (ch - ih * self.zoom_level) / 2.0
        self.is_fit_mode = True
        self.redraw()

    def zoom_step(self, factor: float) -> None:
        """Zooms centered on viewport."""
        self.is_fit_mode = False
        ref = self._active_ref_image()
        if ref is None:
            return

        cw = self.winfo_width() / 2.0
        ch = self.winfo_height() / 2.0

        new_zoom = max(0.01, min(20.0, self.zoom_level * factor))
        ratio = new_zoom / self.zoom_level
        self.zoom_level = new_zoom

        self.pan_x = cw - (cw - self.pan_x) * ratio
        self.pan_y = ch - (ch - self.pan_y) * ratio
        self._clamp_pan()
        self.redraw()

    def zoom_100(self) -> None:
        """Sets zoom to exactly 1.0 (100% native pixels)."""
        self.is_fit_mode = False
        cw = self.winfo_width() / 2.0
        ch = self.winfo_height() / 2.0
        ratio = 1.0 / self.zoom_level
        self.zoom_level = 1.0
        self.pan_x = cw - (cw - self.pan_x) * ratio
        self.pan_y = ch - (ch - self.pan_y) * ratio
        self._clamp_pan()
        self.redraw()

    def _on_resize(self, event) -> None:
        if self.is_fit_mode and self._active_ref_image() is not None:
            self.fit_to_window()
        else:
            self._request_redraw()

    def _get_divider_screen_x(self) -> Optional[int]:
        """Returns the current screen X coordinate of the split divider, if visible."""
        if self.view_mode != "split" or self.orig_pil is None or self.enh_pil is None:
            return None
        ref = self._active_ref_image()
        if ref is None:
            return None
        disp_w = max(1, int(round(ref.size[0] * self.zoom_level)))
        x1 = int(round(self.pan_x))
        return x1 + int(round(disp_w * self.split_pos))


    def _on_mouse_move(self, event) -> None:
        """Changes cursor to resize cursor when hovering near the split divider."""
        div_x = self._get_divider_screen_x()
        if div_x is not None and abs(event.x - div_x) <= 15:
            self.config(cursor="sb_h_double_arrow")
        else:
            self.config(cursor="")

    def _on_drag_start(self, event) -> None:
        div_x = self._get_divider_screen_x()
        if div_x is not None and abs(event.x - div_x) <= 18:
            self.dragging_divider = True
            return

        self.dragging_divider = False
        self.drag_start_x = event.x
        self.drag_start_y = event.y

    def _on_drag_motion(self, event) -> None:
        if self.dragging_divider:
            ref = self._active_ref_image()
            if ref:
                disp_w = max(1, int(round(ref.size[0] * self.zoom_level)))
                x1 = int(round(self.pan_x))
                rel = (event.x - x1) / float(disp_w)
                self.split_pos = float(np.clip(rel, 0.0, 1.0))
                if self.on_split_change:
                    self.on_split_change(self.split_pos)
                self._request_redraw()
            return

        dx = event.x - self.drag_start_x
        dy = event.y - self.drag_start_y
        if abs(dx) > 1 or abs(dy) > 1:
            self.is_fit_mode = False
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        self.pan_x += dx
        self.pan_y += dy
        self._clamp_pan()
        self._request_redraw()

    def _request_redraw(self) -> None:
        """Coalesces rapid motion/resize events into one redraw (~60fps max)."""
        if self._redraw_after is not None:
            try:
                self.after_cancel(self._redraw_after)
            except (ValueError, RuntimeError, tk.TclError):
                pass
        try:
            self._redraw_after = self.after(16, self._do_deferred_redraw)
        except (RuntimeError, tk.TclError):
            self._redraw_after = None

    def _do_deferred_redraw(self) -> None:
        self._redraw_after = None
        try:
            self.redraw()
        except (RuntimeError, tk.TclError):
            pass

    def _clamp_pan(self) -> None:
        """Keeps at least a margin of the image visible so it can't get lost."""
        ref = self._active_ref_image()
        if ref is None:
            return
        try:
            cw = max(1, self.winfo_width())
            ch = max(1, self.winfo_height())
        except (RuntimeError, tk.TclError):
            return
        disp_w = max(1, int(round(ref.size[0] * self.zoom_level)))
        disp_h = max(1, int(round(ref.size[1] * self.zoom_level)))
        margin = 80
        self.pan_x = min(cw - margin, max(margin - disp_w, self.pan_x))
        self.pan_y = min(ch - margin, max(margin - disp_h, self.pan_y))

    def _on_drag_release(self, event) -> None:
        self.dragging_divider = False

    def _on_mouse_wheel(self, event) -> None:
        factor = 1.15 if event.delta > 0 else 0.85
        self._zoom_at(event.x, event.y, factor)

    def _zoom_at(self, cx: float, cy: float, factor: float) -> None:
        self.is_fit_mode = False
        ref = self._active_ref_image()
        if ref is None:
            return
        new_zoom = max(0.01, min(20.0, self.zoom_level * factor))
        ratio = new_zoom / self.zoom_level
        self.zoom_level = new_zoom

        self.pan_x = cx - (cx - self.pan_x) * ratio
        self.pan_y = cy - (cy - self.pan_y) * ratio
        self._clamp_pan()
        self._request_redraw()

    @staticmethod
    def _resample_filter(zoom: float):
        # Downscaling: LANCZOS preserves detail; upscaling past 2x: NEAREST
        # keeps pixels crisp and is far cheaper than BILINEAR on huge sizes.
        if zoom < 1.0:
            return Image.Resampling.LANCZOS
        if zoom >= 2.0:
            return Image.Resampling.NEAREST
        return Image.Resampling.BILINEAR

    def redraw(self) -> None:
        """Composites and renders active view mode."""
        self.delete("all")
        ref = self._active_ref_image()
        if ref is None:
            cw = self.winfo_width() / 2
            ch = self.winfo_height() / 2
            self.create_text(
                cw, ch,
                text="Load an image to begin (Ctrl+O)",
                fill="#475569",
                font=("Consolas", 14, "bold")
            )
            return

        cw = self.winfo_width()
        ch = self.winfo_height()

        # Cap preview allocation: zooming a 4K image to 1500% would otherwise
        # allocate gigapixel temporaries. Clamp display size; pan stays exact.
        max_side = 4096
        disp_w = max(1, int(round(ref.size[0] * self.zoom_level)))
        disp_h = max(1, int(round(ref.size[1] * self.zoom_level)))
        if max(disp_w, disp_h) > max_side:
            fit = max_side / float(max(disp_w, disp_h))
            disp_w = max(1, int(round(disp_w * fit)))
            disp_h = max(1, int(round(disp_h * fit)))
        resample = self._resample_filter(self.zoom_level)

        x1 = int(round(self.pan_x))
        y1 = int(round(self.pan_y))

        # View Mode: Single Image
        if self.view_mode != "split" or self.orig_pil is None or self.enh_pil is None:
            active_pil = self._active_ref_image()
            if active_pil is not None:
                resized = active_pil.resize((disp_w, disp_h), resample)
                self._tk_img = ImageTk.PhotoImage(resized)
                self.create_image(x1, y1, anchor=tk.NW, image=self._tk_img)
            return

        # View Mode: Split Comparison
        # Harmonize original to enhanced display geometry
        enh_disp = self.enh_pil.resize((disp_w, disp_h), resample)
        orig_disp = self.orig_pil.resize((disp_w, disp_h), resample)

        divider_x = int(disp_w * self.split_pos)

        # Composite left half (Original) and right half (Enhanced)
        composite = Image.new("RGB", (disp_w, disp_h))
        if divider_x > 0:
            left_crop = orig_disp.crop((0, 0, divider_x, disp_h))
            composite.paste(left_crop, (0, 0))
        if divider_x < disp_w:
            right_crop = enh_disp.crop((divider_x, 0, disp_w, disp_h))
            composite.paste(right_crop, (divider_x, 0))

        self._tk_img = ImageTk.PhotoImage(composite)
        self.create_image(x1, y1, anchor=tk.NW, image=self._tk_img)

        # Draw Cyan Divider Line & Interactive Handle
        div_screen_x = x1 + divider_x
        top_y = max(0, y1)
        bottom_y = min(ch, y1 + disp_h)

        if 0 <= div_screen_x <= cw and bottom_y > top_y:
            # Main Divider Line with glow
            self.create_line(div_screen_x, top_y, div_screen_x, bottom_y, fill="#0284c7", width=4)
            self.create_line(div_screen_x, top_y, div_screen_x, bottom_y, fill="#38bdf8", width=2)

            # Center Interactive Circular Handle
            center_y = (top_y + bottom_y) // 2
            self.create_oval(
                div_screen_x - 16, center_y - 16,
                div_screen_x + 16, center_y + 16,
                fill="#0d131f", outline="#38bdf8", width=2
            )
            self.create_text(
                div_screen_x, center_y,
                text="< | >",
                fill="#38bdf8",
                font=("Consolas", 8, "bold")
            )

            # Badges with pill backgrounds
            badge_y = max(24, top_y + 24)

            # Left Badge (Original)
            self.create_rectangle(
                div_screen_x - 90, badge_y - 12,
                div_screen_x - 10, badge_y + 12,
                fill="#0f172a", outline="#334155", width=1
            )
            self.create_text(
                div_screen_x - 50, badge_y,
                text="ORIGINAL",
                fill="#94a3b8",
                font=("Consolas", 9, "bold")
            )

            # Right Badge (Enhanced)
            self.create_rectangle(
                div_screen_x + 10, badge_y - 12,
                div_screen_x + 90, badge_y + 12,
                fill="#0f172a", outline="#0284c7", width=1
            )
            self.create_text(
                div_screen_x + 50, badge_y,
                text="ENHANCED",
                fill="#38bdf8",
                font=("Consolas", 9, "bold")
            )
