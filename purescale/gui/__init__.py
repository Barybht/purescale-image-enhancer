"""Obsidian Studio desktop user interface for PureScale 4.0."""

__all__ = ["PureScaleApp", "launch_gui"]


def __getattr__(name: str):
    # Lazy import: `purescale.gui.app` pulls in tkinter + customtkinter, which
    # are absent on headless/minimal installs. Deferring keeps dependency-free
    # submodules (e.g. `purescale.gui.scroll_math`) importable everywhere.
    if name in __all__:
        from purescale.gui import app as _app

        return getattr(_app, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
