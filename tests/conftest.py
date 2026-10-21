"""Shared test infrastructure for the PureScale suite.

Imported by pytest before any test module, so it is the right place for
process-global hardening and for helpers shared across test classes.

Tk hardening:
    Creating many Tk interpreters in one process (the GUI test classes
    create/destroy several roots per class) occasionally fails on Windows
    with "Can't find a usable init.tcl in the following directories:" and
    an empty search list — Tcl's library discovery consults relative paths
    and the environment, and churn between interpreters can leave that
    search coming up empty. Pinning TCL_LIBRARY/TK_LIBRARY to the
    interpreter's bundled Tcl/Tk disables the search entirely and makes
    every Tk() deterministic. Both are honored by every CPython/Tk build;
    with them unset the behavior is identical (same directories, found by
    search instead of pin).
"""

import os
import sys

import pytest


def _pin_tcl_tk(force: bool = False) -> None:
    """Sets TCL_LIBRARY/TK_LIBRARY when discoverable.

    With ``force`` the existing values are overwritten (used by the Tk
    init retry below in case the environment was clobbered mid-run).
    """
    if not force and "TCL_LIBRARY" in os.environ:
        return
    try:
        prefix = sys.prefix
    except AttributeError:
        return

    def _has_init_tcl(directory: str, marker: str) -> bool:
        return os.path.isfile(os.path.join(directory, marker))

    base = os.path.join(prefix, "tcl")
    if not os.path.isdir(base):
        return
    for name in sorted(os.listdir(base)):
        path = os.path.join(base, name)
        if not os.path.isdir(path):
            continue
        if not name.startswith("tcl") and not name.startswith("tk"):
            continue
        if _has_init_tcl(path, "init.tcl"):
            os.environ["TCL_LIBRARY"] = path
        elif _has_init_tcl(path, "tk.tcl"):
            os.environ["TK_LIBRARY"] = path


_pin_tcl_tk()

# Tk init retry --------------------------------------------------------------
#
# Even with directories pinned, repeated Tk interpreter creation/teardown in
# one long-lived process can hit "_tkinter.TclError: Can't find a usable
# init.tcl in the following directories:" with an EMPTY search list — meaning
# Tcl could not even read the environment candidates it normally has. That
# points at process environment corruption from some earlier test, not at the
# machine's Tcl install. The hook below intercepts every interpreter creation
# (plain tk.Tk() AND customtkinter's CTk(), which both funnel through
# _tkinter.create) and, on exactly this library-path failure, force-restores
# the pinned directories once and retries. Display errors and anything else
# propagate unchanged so headless runners keep auto-skipping.


def _create_tk_interpreter_intercepted(*args, **kwargs):
    import _tkinter as _real_tkinter
    import tkinter as tk

    last_error = None
    for attempt in range(2):
        try:
            return _real_tkinter.create(*args, **kwargs)
        except tk.TclError as err:
            message = str(err)
            # Two observed symptoms of the same partial-init failure family
            # (see module docstring): Tcl cannot find its library paths, or
            # the library script ran against a half-initialized interp and
            # never defined tcl_findLibrary. Both leave a fresh interpreter
            # creation as the recovery path; everything else propagates.
            retriable = (
                "Can't find a usable init.tcl" in message
                or "Can't find a usable tk.tcl" in message
                or "tcl_findLibrary" in message
            )
            if not retriable:
                raise
            last_error = err
            import sys as _sys

            print(
                f"\n[conftest] Tcl init failed on Tk init attempt {attempt + 1} "
                f"({message.splitlines()[0]}); force-restoring TCL_LIBRARY/TK_LIBRARY "
                "and retrying.",
                file=_sys.stderr,
                flush=True,
            )
            _pin_tcl_tk(force=True)
    raise last_error


class _TkinterModuleProxy:
    """Forwards everything to the real _tkinter except interpreter creation."""

    def __getattr__(self, name):
        import _tkinter as _real_tkinter

        return getattr(_real_tkinter, name)

    def create(self, *args, **kwargs):
        return _create_tk_interpreter_intercepted(*args, **kwargs)



def make_deterministic_image(width: int, height: int, seed: int = 7):
    """Builds a deterministic photo-like BGR uint8 fixture (bench-identical generator)."""
    import cv2
    import numpy as np

    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :].repeat(height, axis=0)
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None].repeat(width, axis=1)
    base = np.stack([x * 180 + 30, y * 150 + 40, (1 - x) * y * 160 + 30], axis=-1)
    grain = rng.normal(0, 6.0, (height, width, 1)).astype(np.float32)
    img = base + grain
    cv2.circle(img, (width // 3, height // 3), min(width, height) // 8, (200, 150, 100), -1)
    cv2.rectangle(img, (width // 2, height // 2), (width // 2 + width // 5, height // 2 + height // 5), (40, 40, 40), -1)
    return np.clip(img, 0, 255).astype(np.uint8)


@pytest.fixture(autouse=True)
def _tkinit_retry_guard(monkeypatch):
    """Wraps every Tk interpreter creation in the init-retry for the test run.

    Applies to unittest.TestCase tests as well (pytest runs autouse fixtures
    for them). On the plain `python -m unittest` path conftest is not loaded,
    leaving the original behavior there.
    """
    try:
        import tkinter  # noqa: F401
    except ImportError:
        yield
        return
    monkeypatch.setattr("tkinter._tkinter", _TkinterModuleProxy(), raising=False)
    yield

