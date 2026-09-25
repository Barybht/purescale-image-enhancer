"""Pure scrolling math: no GUI imports, safe on headless/minimal installs.

Kept separate from `scrolling.py` (which needs customtkinter) so unit tests
can run anywhere, including CI runners without tkinter.
"""

import sys


def wheel_pixels(delta: int, button_num: int, step_px: int) -> int:
    """Normalizes a wheel event to a signed pixel distance (down positive).

    Args:
        delta: ``event.delta`` (Windows: +-120 per notch, macOS: small ints).
        button_num: ``event.num`` for X11 Button-4/5 events (0 otherwise).
        step_px: Pixels travelled per full mouse-wheel notch.
    """
    if button_num == 4:
        return -step_px
    if button_num == 5:
        return step_px
    if sys.platform == "darwin":
        return int(-delta * 2)
    # Windows (and anything reporting 120-unit deltas): one notch = step.
    notches = delta / 120.0 if delta else 0.0
    return int(round(-notches * step_px))


def clamp_fraction(first: float, window: float) -> float:
    """Clamps a ``yview_moveto`` fraction to the valid [0, 1 - window] range."""
    return max(0.0, min(max(0.0, 1.0 - window), first))
