"""ImageState / Session: single source of truth for image data + display state.

Kills the cross-layer reach-ins between ``PureScaleApp`` and
``InteractiveCanvas``. Previously pixel data lived on the app
(``current_*_bgr``) while display state lived on the canvas
(``orig_pil``, ``zoom_level``, ``split_pos``, ...). Both objects mutated
overlapping state, which is how loading mid-job could mix old results with
new images, and why the app had to reach into canvas internals
(``canvas.split_pos``, ``canvas.orig_pil``).

One ``ImageState`` owns everything; both app and canvas reference it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

from purescale.config import DiagnosticsResult


@dataclass
class ImageState:
    """All mutable session state: pixel data, metadata, and display transform.

    Fields are grouped by domain but there is no hard boundary — the session
    is the single owner, so any reader (app or canvas) sees consistent state.
    """

    # ---- Pixel data (analytical / pipeline domain) ----
    orig_bgr: Optional[np.ndarray] = None
    enh_bgr: Optional[np.ndarray] = None
    alpha: Optional[np.ndarray] = None
    exif: Optional[bytes] = None
    icc_profile: Optional[bytes] = None
    file_path: Optional[str] = None
    result_grayscale: bool = False
    diagnostics: Optional[DiagnosticsResult] = None

    # ---- Display domain (PIL images + viewport transform) ----
    orig_pil: Optional[Image.Image] = None
    enh_pil: Optional[Image.Image] = None
    zoom_level: float = 1.0
    pan_x: float = 0.0
    pan_y: float = 0.0
    split_pos: float = 0.5
    view_mode: str = "enhanced"  # "enhanced", "original", "split"
    is_fit_mode: bool = True

    # ---- Job lifecycle ----
    is_processing: bool = False

    def reset_for_new_source(self) -> None:
        """Clear enhancement result and display transform for a fresh source."""
        self.enh_bgr = None
        self.enh_pil = None
        self.result_grayscale = False
        self.diagnostics = None
        self.view_mode = "original"
        self.split_pos = 0.5
        self.is_fit_mode = True
