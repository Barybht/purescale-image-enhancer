"""Host device detection and execution-provider selection for PureScale.

Single source of truth for human-readable backend names. Previously the
strings "Zen 4 CPU (AVX-512 / AVX2)", "AMD Ryzen 5 (Zen 4 AVX-512)" and
"AMD Radeon 760M (DirectML GPU)" were hardcoded in ``pipeline.py``,
``neural/engine.py`` and ``gui/app.py`` regardless of the actual host.
These helpers report what the machine really has.
"""

import logging
import platform
from typing import List, Tuple

from purescale.config import DeviceTarget

logger = logging.getLogger(__name__)


def cpu_label() -> str:
    """Returns a short human-readable CPU identifier, e.g. ``"AMD64 CPU"``."""
    name = (platform.processor() or "").strip()
    lowered = name.lower()
    verbose = (
        not name
        or len(name) > 24
        or "family" in lowered
        or "stepping" in lowered
        or lowered in {"x86_64", "amd64", "arm64", "aarch64"}
    )
    if verbose:
        machine = (platform.machine() or "").strip()
        name = machine if machine else "CPU"
    if not name.lower().endswith("cpu"):
        name = f"{name} CPU"
    return name


def available_providers() -> List[str]:
    """Returns ONNX Runtime execution providers, or ``[]`` if unavailable."""
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        return list(providers) if providers else []
    except (ImportError, AttributeError, RuntimeError, OSError) as err:
        logger.debug("ONNX Runtime provider query skipped: %s", err)
        return []


def has_directml() -> bool:
    """Returns True when the DirectML execution provider is available."""
    return "DmlExecutionProvider" in available_providers()


def gpu_label() -> str:
    """Returns a human-readable GPU identifier for DirectML-capable hosts."""
    return "DirectML GPU"


def select_providers(target: DeviceTarget) -> Tuple[List[str], str]:
    """Maps a device target to ``(providers, backend_name)``.

    Args:
        target: Device target (``AUTO``, ``DIRECTML_GPU``, ``CPU``,
            ``OPENCV_DNN``). ``OPENCV_DNN`` raises ``ImportError`` so callers
            fall through to the OpenCV DNN backend.

    Returns:
        Tuple of (provider list for ONNX Runtime, display name).
    """
    cpu = cpu_label()
    if target == DeviceTarget.DIRECTML_GPU:
        if has_directml():
            return ["DmlExecutionProvider", "CPUExecutionProvider"], f"{gpu_label()} (DirectML)"
        return ["CPUExecutionProvider"], f"{cpu} (DML Unavailable)"
    if target == DeviceTarget.CPU:
        return ["CPUExecutionProvider"], f"{cpu} (ONNX Runtime)"
    if target == DeviceTarget.OPENCV_DNN:
        raise ImportError("OpenCV DNN fallback requested")
    # auto
    if has_directml():
        return ["DmlExecutionProvider", "CPUExecutionProvider"], f"{gpu_label()} (DirectML)"
    return ["CPUExecutionProvider"], f"{cpu} (ONNX Runtime)"
