"""Central theme for Obsidian Studio: palette roles and font factories.

All GUI colors and fonts live here. Call sites reference roles
(``PALETTE["surface"]``), never hex literals, so palette unification and
future theming touch exactly one file. Values below reproduce the
long-standing Obsidian palette 1:1.

Font families are chosen per-platform: Consolas/Segoe UI are Windows-only,
so macOS and Linux fall back to system monospace/sans families that Tk can
actually resolve instead of rendering in an ugly fallback.
"""

import sys
import customtkinter as ctk

if sys.platform == "win32":
    _MONO = "Consolas"
    _UI = "Segoe UI"
elif sys.platform == "darwin":
    _MONO = "Menlo"
    _UI = ".AppleSystemUIFont"
else:
    _MONO = "monospace"
    _UI = "sans-serif"

PALETTE = {
    # Backgrounds
    "bg_app": "#090d16",      # main window, viewport, segmented wells
    "bg_bar": "#0d131f",      # top/side/bottom bars, canvas overlay panels
    # Surfaces
    "surface": "#131b2e",     # cards, buttons at rest, slider tracks
    "surface_hover": "#1e293b",  # button/row hover, secondary buttons
    "hover_light": "#334155",    # scrollbar hover, tertiary hover
    "border": "#202b3f",         # card outline
    # Accent ramp (blue)
    "accent": "#0284c7",         # selection, primary actions, progress fill
    "accent_hover": "#0369a1",   # primary hover
    "accent_bright": "#38bdf8",  # telemetry text, badges, slider thumbs, borders
    "accent_lightest": "#7dd3fc",  # thumb hover
    # Go (auto-enhance)
    "go": "#059669",
    "go_hover": "#047857",
    # Text ramp
    "text": "#e2e8f0",         # primary text, titles
    "text_soft": "#cbd5e1",    # secondary values
    "text_muted": "#94a3b8",   # labels
    "text_faint": "#64748b",   # placeholders, disabled, de-emphasized
    # Status
    "ok": "#4ade80",
    "warn": "#facc15",
    "err": "#f87171",
}

_FONTS = {
    "mono15b": (_MONO, 15, "bold"),
    "mono14b": (_MONO, 14, "bold"),
    "mono11b": (_MONO, 11, "bold"),
    "mono11": (_MONO, 11, "normal"),
    "mono10b": (_MONO, 10, "bold"),
    "mono9b": (_MONO, 9, "bold"),
    "mono8b": (_MONO, 8, "bold"),
    "ui13b": (_UI, 13, "bold"),
    "ui12b": (_UI, 12, "bold"),
    "ui11": (_UI, 11, "normal"),
    "ui10": (_UI, 10, "normal"),
}


def font(name: str) -> ctk.CTkFont:
    """Returns a cached-role font (e.g. ``font("mono11b")``)."""
    family, size, weight = _FONTS[name]
    return ctk.CTkFont(family=family, size=size, weight=weight)
