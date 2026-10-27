"""Interactive zoomable, pannable canvas with draggable split-view comparison."""

import tkinter as tk
from typing import Callable, Optional
import numpy as np
from PIL import Image, ImageTk

from purescale.gui.session import ImageState
from purescale.gui.theme import PALETTE, font


class InteractiveCanvas(tk.Canvas):
    """
    High-performance viewport supporting pan, zoom, fit-to-window,
    and side-by-side split-view comparison with interactive drag.

    The canvas does not own display state — it reads and writes through the
    shared ``ImageState`` session, which is also owned by ``PureScaleApp``.
    This kills the cross-layer reach-ins where app and canvas each held
    overlapping copies of the same image/viewport state.
    """

    def __init__(self, master, session: Optional[ImageState] = None, **kwargs):
        kwargs.setdefault("bg", PALETTE["bg_app"])
        kwargs.setdefault("highlightthickness", 0)
        super().__init__(master, **kwargs)

        self.session = session if session is not None else ImageState()
        self.on_split_change: Optional[Callable[[float], None]] = None
        self.on_zoom_change: Optional[Callable[[float], None]] = None

        # Transient UI state (not session state — owned by the canvas alone).
        self.dragging_divider: bool = False
        self.drag_start_x: float = 0.0
        self.drag_start_y: float = 0.0
        self._wheel_delta: float = 0.0
        self._WHEEL_NOTCH: float = 120.0

        self._tk_img: Optional[ImageTk.PhotoImage] = None
        self._redraw_after: Optional[str] = None

        # Display-size resizes, cached across redraws.
        self._disp_cache: dict = {}

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
        self.bind("<Double-Button-1>", self._on_double_click)

    # -- session pass-through (keeps a small public surface for tests) --------

    @property
    def view_mode(self) -> str:
        return self.session.view_mode

    @view_mode.setter
    def view_mode(self, value: str) -> None:
        self.session.view_mode = value

    @property
    def orig_pil(self) -> Optional[Image.Image]:
        return self.session.orig_pil

    @orig_pil.setter
    def orig_pil(self, value: Optional[Image.Image]) -> None:
        self.session.orig_pil = value

    @property
    def enh_pil(self) -> Optional[Image.Image]:
        return self.session.enh_pil

    @enh_pil.setter
    def enh_pil(self, value: Optional[Image.Image]) -> None:
        self.session.enh_pil = value

    @property
    def is_fit_mode(self) -> bool:
        return self.session.is_fit_mode

    @is_fit_mode.setter
    def is_fit_mode(self, value: bool) -> None:
        self.session.is_fit_mode = value

    @property
    def zoom_level(self) -> float:
        return self.session.zoom_level

    @zoom_level.setter
    def zoom_level(self, value: float) -> None:
        self.session.zoom_level = value

    # -- display state -------------------------------------------------------

    def _active_ref_image(self) -> Optional[Image.Image]:
        """Returns the PIL image that defines the active display geometry."""
        s = self.session
        if s.view_mode == "original":
            return s.orig_pil
        return s.enh_pil or s.orig_pil

    def set_images(
        self,
        orig: Optional[Image.Image],
        enh: Optional[Image.Image],
    ) -> None:
        """Sets active images and fits the viewport on a new source or in fit mode."""
        s = self.session
        prev_orig = s.orig_pil
        prev_enh = s.enh_pil

        s.orig_pil = orig
        s.enh_pil = enh
        # New displayed images invalidate every cached resize (entries may
        # also simply fall out via their identity key, but keeping the cache
        # bounded starts here).
        self._disp_cache.clear()

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

        if is_new_source or s.is_fit_mode:
            s.is_fit_mode = True
            self.fit_to_window()
        else:
            # If not in fit mode, keep visual alignment when resolution changes
            if prev_enh is None and enh is not None and prev_orig is not None and enh.size[0] > 0:
                scale_change = float(orig.size[0]) / float(enh.size[0])
                s.zoom_level = max(0.01, s.zoom_level * scale_change)
                self._notify_zoom_change()
            self.redraw()

    def set_view_mode(self, mode: str) -> None:
        """Updates display mode: 'enhanced', 'original', or 'split'."""
        self.session.view_mode = mode.lower()
        if self.session.is_fit_mode:
            self.fit_to_window()
        else:
            self.redraw()

    def set_split_pos(self, pos: float, notify: bool = True) -> None:
        """Sets divider position (0.0 to 1.0) and redraws."""
        self.session.split_pos = float(np.clip(pos, 0.0, 1.0))
        if notify and self.on_split_change:
            self.on_split_change(self.session.split_pos)
        self.redraw()

    def _notify_zoom_change(self) -> None:
        """Notifies registered listener of zoom change."""
        if self.on_zoom_change:
            try:
                self.on_zoom_change(self.session.zoom_level)
            except Exception:
                pass

    def fit_to_window(self) -> None:
        """Calculates optimal zoom factor to fit active image inside viewport."""
        s = self.session
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
        s.zoom_level = max(0.01, min(scale_w, scale_h))

        s.pan_x = (cw - iw * s.zoom_level) / 2.0
        s.pan_y = (ch - ih * s.zoom_level) / 2.0
        s.is_fit_mode = True
        self._notify_zoom_change()
        self.redraw()

    def zoom_step(self, factor: float) -> None:
        """Zooms centered on viewport."""
        s = self.session
        s.is_fit_mode = False
        ref = self._active_ref_image()
        if ref is None:
            return

        cw = self.winfo_width() / 2.0
        ch = self.winfo_height() / 2.0

        new_zoom = max(0.01, min(20.0, s.zoom_level * factor))
        ratio = new_zoom / s.zoom_level
        s.zoom_level = new_zoom

        s.pan_x = cw - (cw - s.pan_x) * ratio
        s.pan_y = ch - (ch - s.pan_y) * ratio
        self._clamp_pan()
        self._notify_zoom_change()
        self.redraw()

    def zoom_100(self) -> None:
        """Sets zoom to exactly 1.0 (100% native pixels)."""
        s = self.session
        s.is_fit_mode = False
        cw = self.winfo_width() / 2.0
        ch = self.winfo_height() / 2.0
        ratio = 1.0 / s.zoom_level
        s.zoom_level = 1.0
        s.pan_x = cw - (cw - s.pan_x) * ratio
        s.pan_y = ch - (ch - s.pan_y) * ratio
        self._clamp_pan()
        self._notify_zoom_change()
        self.redraw()

    def _on_resize(self, event) -> None:
        s = self.session
        if s.is_fit_mode and self._active_ref_image() is not None:
            self.fit_to_window()
        else:
            self._request_redraw()

    def _get_divider_screen_x(self) -> Optional[int]:
        """Returns the current screen X coordinate of the split divider, if visible."""
        s = self.session
        if s.view_mode != "split" or s.orig_pil is None or s.enh_pil is None:
            return None
        ref = self._active_ref_image()
        if ref is None:
            return None
        disp_w = max(1, int(round(ref.size[0] * s.zoom_level)))
        x1 = int(round(s.pan_x))
        return x1 + int(round(disp_w * s.split_pos))

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
        s = self.session
        if self.dragging_divider:
            ref = self._active_ref_image()
            if ref:
                disp_w = max(1, int(round(ref.size[0] * s.zoom_level)))
                x1 = int(round(s.pan_x))
                rel = (event.x - x1) / float(disp_w)
                s.split_pos = float(np.clip(rel, 0.0, 1.0))
                if self.on_split_change:
                    self.on_split_change(s.split_pos)
                self._request_redraw()
            return

        dx = event.x - self.drag_start_x
        dy = event.y - self.drag_start_y
        if abs(dx) > 1 or abs(dy) > 1:
            s.is_fit_mode = False
            # Leaving fit mode changes the label basis; refresh the header
            # immediately (it used to go stale until the next zoom action).
            self._notify_zoom_change()
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        s.pan_x += dx
        s.pan_y += dy
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
        s = self.session
        ref = self._active_ref_image()
        if ref is None:
            return
        try:
            cw = max(1, self.winfo_width())
            ch = max(1, self.winfo_height())
        except (RuntimeError, tk.TclError):
            return
        disp_w = max(1, int(round(ref.size[0] * s.zoom_level)))
        disp_h = max(1, int(round(ref.size[1] * s.zoom_level)))
        margin = 80
        s.pan_x = min(cw - margin, max(margin - disp_w, s.pan_x))
        s.pan_y = min(ch - margin, max(margin - disp_h, s.pan_y))

    def _on_drag_release(self, event) -> None:
        self.dragging_divider = False

    def _on_double_click(self, event) -> None:
        """Recenters the split divider (near it), else fits to window."""
        div_x = self._get_divider_screen_x()
        if div_x is not None and abs(event.x - div_x) <= 18:
            self.set_split_pos(0.5)
        else:
            self.fit_to_window()

    def _on_mouse_wheel(self, event) -> None:
        """Zoom on wheel scroll, with platform-normalized delta.

        Windows emits ~±120 per wheel notch while macOS trackpads emit tiny
        deltas per scroll event. Accumulate until one notch has built up,
        then apply a single zoom step so the canvas feels consistent across
        platforms instead of hypersensitive on macOS.
        """
        self._wheel_delta += float(event.delta)
        if abs(self._wheel_delta) < self._WHEEL_NOTCH:
            return
        factor = 1.15 if self._wheel_delta > 0 else 0.85
        self._wheel_delta = 0.0
        self._zoom_at(event.x, event.y, factor)

    def _zoom_at(self, cx: float, cy: float, factor: float) -> None:
        s = self.session
        s.is_fit_mode = False
        ref = self._active_ref_image()
        if ref is None:
            return
        new_zoom = max(0.01, min(20.0, s.zoom_level * factor))
        ratio = new_zoom / s.zoom_level
        s.zoom_level = new_zoom

        s.pan_x = cx - (cx - s.pan_x) * ratio
        s.pan_y = cy - (cy - s.pan_y) * ratio
        self._clamp_pan()
        self._notify_zoom_change()
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

    def _display_image(self, pil: Image.Image, disp_w: int, disp_h: int, resample) -> Image.Image:
        """Returns ``pil`` resized to display size, memoized across redraws.

        Same pixels as an uncached ``pil.resize(...)`` — the key is the source
        image identity plus the exact target size and resampling filter.
        """
        key = (id(pil), disp_w, disp_h, resample)
        entry = self._disp_cache.get(key)
        if entry is not None and entry[0] is pil:
            return entry[1]
        resized = pil.resize((disp_w, disp_h), resample)
        if len(self._disp_cache) >= 4:
            self._disp_cache.clear()
        # Strong reference to the source alongside the resize: keeps the id()
        # in the key from being reused by a different image while cached.
        self._disp_cache[key] = (pil, resized)
        return resized

    def redraw(self) -> None:
        """Composites and renders active view mode."""
        s = self.session
        self.delete("all")
        ref = self._active_ref_image()
        if ref is None:
            cw = self.winfo_width() / 2
            ch = self.winfo_height() / 2
            self.create_text(
                cw, ch,
                text="Drop an image here or press Ctrl+O",
                fill=PALETTE["text_faint"],
                font=font("mono14b"),
            )
            self.create_text(
                cw, ch + 28,
                text="PNG / JPEG / WebP / BMP / TIFF",
                fill=PALETTE["text_faint"],
                font=font("mono10b"),
            )
            return

        cw = self.winfo_width()
        ch = self.winfo_height()

        # Cap preview allocation: zooming a 4K image to 1500% would otherwise
        # allocate gigapixel temporaries. Clamp display size; pan stays exact.
        max_side = 4096
        disp_w = max(1, int(round(ref.size[0] * s.zoom_level)))
        disp_h = max(1, int(round(ref.size[1] * s.zoom_level)))
        if max(disp_w, disp_h) > max_side:
            fit = max_side / float(max(disp_w, disp_h))
            disp_w = max(1, int(round(disp_w * fit)))
            disp_h = max(1, int(round(disp_h * fit)))
        resample = self._resample_filter(s.zoom_level)

        x1 = int(round(s.pan_x))
        y1 = int(round(s.pan_y))

        # View Mode: Single Image
        if s.view_mode != "split" or s.orig_pil is None or s.enh_pil is None:
            active_pil = self._active_ref_image()
            if active_pil is not None:
                resized = self._display_image(active_pil, disp_w, disp_h, resample)
                self._tk_img = ImageTk.PhotoImage(resized)
                self.create_image(x1, y1, anchor=tk.NW, image=self._tk_img)
            return

        # View Mode: Split Comparison
        # Harmonize original to enhanced display geometry (cached across
        # divider-drag frames: pan/zoom unchanged -> pure paste work).
        enh_disp = self._display_image(s.enh_pil, disp_w, disp_h, resample)
        orig_disp = self._display_image(s.orig_pil, disp_w, disp_h, resample)

        divider_x = int(disp_w * s.split_pos)

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
            self.create_line(div_screen_x, top_y, div_screen_x, bottom_y, fill=PALETTE["accent"], width=4)
            self.create_line(div_screen_x, top_y, div_screen_x, bottom_y, fill=PALETTE["accent_bright"], width=2)

            # Center Interactive Circular Handle
            center_y = (top_y + bottom_y) // 2
            self.create_oval(
                div_screen_x - 16, center_y - 16,
                div_screen_x + 16, center_y + 16,
                fill=PALETTE["bg_bar"], outline=PALETTE["accent_bright"], width=2
            )
            self.create_text(
                div_screen_x, center_y,
                text="< | >",
                fill=PALETTE["accent_bright"],
                font=font("mono8b")
            )

            # Badges pinned to the image corners (never collide with the
            # divider at extreme split positions). Skipped on narrow views.
            badge_y = max(24, top_y + 24)
            if disp_w >= 220:
                # Left Badge (Original)
                self.create_rectangle(
                    x1 + 10, badge_y - 12,
                    x1 + 90, badge_y + 12,
                    fill=PALETTE["bg_bar"], outline=PALETTE["hover_light"], width=1
                )
                self.create_text(
                    x1 + 50, badge_y,
                    text="ORIGINAL",
                    fill=PALETTE["text_muted"],
                    font=font("mono9b"),
                )

                # Right Badge (Enhanced)
                self.create_rectangle(
                    x1 + disp_w - 90, badge_y - 12,
                    x1 + disp_w - 10, badge_y + 12,
                    fill=PALETTE["bg_bar"], outline=PALETTE["accent"], width=1
                )
                self.create_text(
                    x1 + disp_w - 50, badge_y,
                    text="ENHANCED",
                    fill=PALETTE["accent_bright"],
                    font=font("mono9b"),
                )
