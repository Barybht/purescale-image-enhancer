"""Central theme for Obsidian Studio: palette roles and font factories.

All GUI colors and fonts live here. Call sites reference roles
(``PALETTE["surface"]``), never hex literals, so palette unification and
future theming touch exactly one file. Values below reproduce the
long-standing Obsidian palette 1:1.
"""

import customtkinter as ctk

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
    "mono15b": ("Consolas", 15, "bold"),
    "mono14b": ("Consolas", 14, "bold"),
    "mono11b": ("Consolas", 11, "bold"),
    "mono11": ("Consolas", 11, "normal"),
    "mono10b": ("Consolas", 10, "bold"),
    "mono9b": ("Consolas", 9, "bold"),
    "mono8b": ("Consolas", 8, "bold"),
    "ui13b": ("Segoe UI", 13, "bold"),
    "ui12b": ("Segoe UI", 12, "bold"),
    "ui11": ("Segoe UI", 11, "normal"),
    "ui10": ("Segoe UI", 10, "normal"),
}


def font(name: str) -> ctk.CTkFont:
    """Returns a cached-role font (e.g. ``font("mono11b")``)."""
    family, size, weight = _FONTS[name]
    return ctk.CTkFont(family=family, size=size, weight=weight)
