"""Smooth kinetic scrolling for the PureScale sidebar.

`CTkScrollableFrame` scrolls in instant jumps (20px/notch on Windows,
30px on Linux) which feels choppy next to the animated viewport canvas.
`SmoothScrollableFrame` keeps the stock widget (and its child-widget wheel
routing, e.g. hovering a slider still adjusts the slider) and only replaces
the motion: wheel input accumulates into a target position that the frame
eases toward at ~60fps.
"""

import tkinter as tk
from typing import Optional

import customtkinter as ctk

from purescale.gui.scroll_math import clamp_fraction, wheel_pixels

_ANIMATION_INTERVAL_MS = 12
_EASING_FACTOR = 0.35
_SNAP_EPSILON = 0.0005

# Re-exported for backward compatibility (moved to scroll_math.py).
__all__ = ["SmoothScrollableFrame", "clamp_fraction", "wheel_pixels"]


class SmoothScrollableFrame(ctk.CTkScrollableFrame):
    """`CTkScrollableFrame` with animated, accumulative wheel scrolling.

    Drop-in replacement: same constructor, same child-routing rules
    (sliders/textboxes keep their own wheel behavior). Only the motion
    changes from instant jumps to eased animation.
    """

    def __init__(self, master, wheel_step_px: int = 56, **kwargs):
        self._wheel_step_px = max(8, int(wheel_step_px))
        self._smooth_target: Optional[float] = None
        self._smooth_after_id: Optional[str] = None
        super().__init__(master, **kwargs)

    # -- motion core (pure-ish, unit tested) ---------------------------------
    def _scroll_target_for(self, pixels: int) -> Optional[float]:
        """Converts a pixel delta into an absolute clamped target fraction."""
        try:
            first, last = self._parent_canvas.yview()
        except (tk.TclError, AttributeError):
            return None
        if (first, last) == (0.0, 1.0):
            return None  # Content fits; nothing to scroll.
        window = last - first
        try:
            total = float(self._parent_canvas.cget("scrollregion").split()[3])
        except (tk.TclError, AttributeError, IndexError, ValueError):
            return None
        if total <= 0:
            return None
        base = self._smooth_target if self._smooth_target is not None else first
        return clamp_fraction(base + pixels / total, window)

    def _smooth_step(self) -> None:
        """Advances one animation frame toward the target fraction."""
        self._smooth_after_id = None
        if self._smooth_target is None:
            return
        try:
            first, _ = self._parent_canvas.yview()
        except (tk.TclError, AttributeError):
            self._smooth_target = None
            return
        nxt = first + (self._smooth_target - first) * _EASING_FACTOR
        if abs(self._smooth_target - nxt) < _SNAP_EPSILON:
            nxt = self._smooth_target
            done = True
        else:
            done = False
        try:
            self._parent_canvas.yview_moveto(nxt)
        except (tk.TclError, AttributeError):
            self._smooth_target = None
            return
        if done:
            self._smooth_target = None
        else:
            self._schedule_step()

    def _schedule_step(self) -> None:
        if self._smooth_after_id is not None:
            return
        try:
            self._smooth_after_id = self.after(_ANIMATION_INTERVAL_MS, self._smooth_step)
        except (tk.TclError, AttributeError):
            self._smooth_after_id = None

    # -- event entry point (overrides base instant jumps) --------------------
    def _mouse_wheel_all(self, event) -> None:  # noqa: N802 (matches base name)
        if not self._check_if_valid_scroll(event.widget):
            return
        if getattr(self, "_shift_pressed", False):
            return  # No horizontal scrolling in the sidebar; ignore.
        pixels = wheel_pixels(
            delta=getattr(event, "delta", 0) or 0,
            button_num=getattr(event, "num", 0) or 0,
            step_px=self._wheel_step_px,
        )
        if not pixels:
            return
        target = self._scroll_target_for(pixels)
        if target is None:
            return
        self._smooth_target = target
        self._schedule_step()

    def destroy(self) -> None:  # noqa: D102
        if self._smooth_after_id is not None:
            try:
                self.after_cancel(self._smooth_after_id)
            except (tk.TclError, ValueError, AttributeError):
                pass
            self._smooth_after_id = None
        super().destroy()
