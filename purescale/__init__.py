"""PureScale 4.0: Autonomous Multiscale Computational Vision and Edge AI Engine.

Combines deterministic analytical signal processing (PureDSP), autonomous
image quality diagnostics, Dark Channel Prior dehazing, multiscale Local Laplacian
pyramid filtering, and edge neural super-resolution (Real-ESRGAN Compact).
"""

# Single source of truth for the version is pyproject.toml; when the package
# is installed (pip install -e . included) the metadata wins, and the static
# fallback covers interpreter runs straight from a repo checkout with no
# install metadata present.
try:
    from importlib.metadata import PackageNotFoundError, version as _md_version

    try:
        __version__ = _md_version("purescale")
    except PackageNotFoundError:
        __version__ = "4.0.1"
except ImportError:  # importlib.metadata requires Python 3.8+; 3.9 floor makes this moot
    __version__ = "4.0.1"

__author__ = "PureScale Architecture Team"

# Suppress OpenCV 5.x DNN new-graph-engine WARN spam
# ("Targets are not supported by the new graph engine for now").
# The warning is emitted from C++ on every Net::setPreferableTarget call
# and from FaceDetectorYN internals; default CPU behavior is already correct.
# Override with PURESCALE_OPENCV_LOG_LEVEL=SILENT/FATAL/ERROR/WARNING/INFO.
try:
    import os as _os
    import cv2 as _cv2

    if hasattr(_cv2, "utils") and hasattr(_cv2.utils, "logging"):
        _level_name = _os.environ.get("PURESCALE_OPENCV_LOG_LEVEL", "ERROR").upper()
        _level = getattr(_cv2.utils.logging, f"LOG_LEVEL_{_level_name}", _cv2.utils.logging.LOG_LEVEL_ERROR)
        _cv2.utils.logging.setLogLevel(_level)
except Exception:
    pass

from purescale.config import (
    DeviceTarget,
    DiagnosticsResult,
    PipelineConfig,
    ProcessingMode,
    ProcessingResult,
    get_preset_config,
)
from purescale.pipeline import PureScalePipeline

__all__ = [
    "__version__",
    "DeviceTarget",
    "DiagnosticsResult",
    "PipelineConfig",
    "ProcessingMode",
    "ProcessingResult",
    "PureScalePipeline",
    "get_preset_config",
]
