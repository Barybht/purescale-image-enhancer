"""Modern Edge AI Neural Engine.

Provides lightweight ONNX super-resolution, overlap tiling, multi-backend
hardware acceleration (DirectML GPU / CPU AVX-512), and portrait facial retouching.
"""

from purescale.neural.models import ModelManager, get_default_model_path
from purescale.neural.engine import NeuralSuperResEngine
from purescale.neural.tiling import tile_process
from purescale.neural.portrait import (
    detect_and_enhance_portrait,
    fast_guided_filter,
)

__all__ = [
    "ModelManager",
    "get_default_model_path",
    "NeuralSuperResEngine",
    "tile_process",
    "detect_and_enhance_portrait",
    "fast_guided_filter",
]
