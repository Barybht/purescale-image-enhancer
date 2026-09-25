"""Configuration structures, diagnostic types, and presets for PureScale 4.0."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional
import numpy as np

# Canonical definition lives in purescale.dsp.diagnostics (re-exported here
# for backward compatibility: `from purescale.config import DiagnosticsResult`
# keeps working, but there is exactly one class object).
from purescale.dsp.diagnostics import DiagnosticsResult

__all__ = [
    "ProcessingMode",
    "DeviceTarget",
    "DiagnosticsResult",
    "PipelineConfig",
    "ProcessingResult",
    "PRESETS",
    "get_preset_config",
]


class ProcessingMode(str, Enum):
    """Execution mode selection."""
    PURE_DSP = "puredsp"       # 100% deterministic, zero neural weights, ~45ms
    NEURAL_AI = "neural"       # Compact ~4.87MB ONNX model edge super-resolution
    HYBRID = "hybrid"          # Neural AI edge synthesis + Multiscale Laplacian & Perceptual DSP


class DeviceTarget(str, Enum):
    """Hardware acceleration target."""
    AUTO = "auto"              # Auto-detect best: CUDA GPU -> DirectML GPU -> CPU -> OpenCV DNN
    DIRECTML_GPU = "directml"  # Microsoft DirectML (AMD Radeon 760M / Intel / NVIDIA GPUs)
    CUDA_GPU = "cuda"          # NVIDIA CUDA execution provider (falls back to CPU)
    CPU = "cpu"                # Pure CPU execution (Zen 4 AVX-512 / AVX2 vector SIMD)
    OPENCV_DNN = "opencv"      # Built-in OpenCV DNN module (zero external dependencies)


@dataclass
class PipelineConfig:
    """Strongly-typed configuration for PureScale 4.0 processing pipeline."""
    # Operating Mode
    mode: ProcessingMode = ProcessingMode.PURE_DSP
    device: DeviceTarget = DeviceTarget.AUTO

    # Autonomous Diagnostics & Auto-Tuning
    # Heavy stages (~35ms on VGA combined). The `fast` preset disables both;
    # `--auto` re-enables diagnostics regardless to compute recommendations.
    enable_diagnostics: bool = True
    auto_tune: bool = False

    # Atmospheric Dehazing (Dark Channel Prior & Guided Filter)
    enable_dehaze: bool = False
    dehaze_strength: float = 0.50

    # Multiscale Local Laplacian Pyramid Filtering
    enable_pyramid: bool = True
    pyramid_micro_texture: float = 1.20
    pyramid_structure_boost: float = 1.10
    pyramid_dynamic_range: float = 0.0

    # Multi-Cue Semantic Region Guidance
    # Heavy stage (~17ms on VGA: 5 guided-filter refinements). The `fast`
    # preset disables it; the pipeline also skips it when no consumer
    # (pyramid + SWF) needs the guidance maps.
    enable_semantic_guidance: bool = True

    # Spatial Scaling (EASU or Neural AI)
    scale: float = 2.0

    # Detail Clarity (Contrast-Adaptive Sharpening)
    sharpen_strength: float = 1.10

    # Structural Denoising (Side Window Filter)
    enable_denoise: bool = True
    denoise_intensity: int = 40

    # Dynamic Range Fusion (BIMEF with Black-Point Pinning)
    enable_contrast: bool = True
    contrast_boost: float = 1.80
    brightness_shift: int = 0

    # Perceptual Color & White Balance (Oklab & Bradford CAT16)
    vibrance_boost: float = 1.10
    color_temperature: int = 0

    # Pre-Restoration Conditioning (De-pixelate & Deblur)
    depixel_strength: int = 0
    deblur_strength: int = 0

    # Facial Retouching (YuNet + Fast Guided Filter)
    portrait_smooth: int = 35
    eye_clarity: float = 1.30

    # Neural AI Execution Settings
    tile_size: int = 256
    tile_overlap: int = 32

    # Memory and Dimension Limits (8K OOM Guard)
    max_megapixels: float = 40.0

    # Output Container Format
    output_format: str = "PNG"

    def __post_init__(self) -> None:
        """Validates configuration parameters."""
        if not (0.5 <= float(self.scale) <= 4.0):
            raise ValueError(
                f"Invalid scale factor: {self.scale}. Must satisfy 0.5 <= scale <= 4.0."
            )
        if self.tile_size < 32:
            raise ValueError(f"Invalid tile_size: {self.tile_size}. Must be >= 32.")
        if self.tile_overlap < 0 or self.tile_overlap >= self.tile_size:
            raise ValueError(
                f"Invalid tile_overlap: {self.tile_overlap}. Must be 0 <= tile_overlap < tile_size."
            )
        if self.max_megapixels <= 0:
            raise ValueError(
                f"Invalid max_megapixels: {self.max_megapixels}. Must be > 0."
            )
        if not (0.0 <= float(self.pyramid_dynamic_range) <= 1.0):
            raise ValueError(
                f"Invalid pyramid_dynamic_range: {self.pyramid_dynamic_range}. Must satisfy 0.0 <= x <= 1.0."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Converts config to dictionary representation."""
        return {
            "mode": self.mode.value if isinstance(self.mode, ProcessingMode) else str(self.mode),
            "device": self.device.value if isinstance(self.device, DeviceTarget) else str(self.device),
            "enable_diagnostics": self.enable_diagnostics,
            "auto_tune": self.auto_tune,
            "enable_dehaze": self.enable_dehaze,
            "dehaze_strength": self.dehaze_strength,
            "enable_pyramid": self.enable_pyramid,
            "pyramid_micro_texture": self.pyramid_micro_texture,
            "pyramid_structure_boost": self.pyramid_structure_boost,
            "pyramid_dynamic_range": self.pyramid_dynamic_range,
            "enable_semantic_guidance": self.enable_semantic_guidance,
            "scale": self.scale,
            "sharpen_strength": self.sharpen_strength,
            "enable_denoise": self.enable_denoise,
            "denoise_intensity": self.denoise_intensity,
            "enable_contrast": self.enable_contrast,
            "contrast_boost": self.contrast_boost,
            "brightness_shift": self.brightness_shift,
            "vibrance_boost": self.vibrance_boost,
            "color_temperature": self.color_temperature,
            "depixel_strength": self.depixel_strength,
            "deblur_strength": self.deblur_strength,
            "portrait_smooth": self.portrait_smooth,
            "eye_clarity": self.eye_clarity,
            "tile_size": self.tile_size,
            "tile_overlap": self.tile_overlap,
            "max_megapixels": self.max_megapixels,
            "output_format": self.output_format,
        }


@dataclass
class ProcessingResult:
    """Execution output, diagnostics, and performance telemetry."""
    image: np.ndarray
    alpha: Optional[np.ndarray] = None
    latency_ms: float = 0.0
    backend_name: str = "CPU"
    mode_name: str = "puredsp"
    faces_detected: int = 0
    diagnostics: Optional[DiagnosticsResult] = None
    semantic_breakdown: Dict[str, float] = field(default_factory=dict)
    stage_latencies: Dict[str, float] = field(default_factory=dict)


PRESETS: Dict[str, Dict[str, Any]] = {
    "balanced": {
        "scale": 2.0,
        "sharpen_strength": 1.1,
        "enable_denoise": True,
        "denoise_intensity": 35,
        "enable_contrast": True,
        "contrast_boost": 1.7,
        "brightness_shift": 0,
        "vibrance_boost": 1.10,
        "color_temperature": 0,
        "portrait_smooth": 35,
        "eye_clarity": 1.3,
        "enable_dehaze": False,
        "dehaze_strength": 0.5,
        "enable_pyramid": True,
        "pyramid_micro_texture": 1.2,
        "pyramid_structure_boost": 1.1,
        "enable_semantic_guidance": True,
    },
    "portrait": {
        "scale": 2.0,
        "sharpen_strength": 0.9,
        "enable_denoise": True,
        "denoise_intensity": 40,
        "enable_contrast": True,
        "contrast_boost": 1.5,
        "brightness_shift": 2,
        "vibrance_boost": 1.08,
        "color_temperature": 5,
        "portrait_smooth": 45,
        "eye_clarity": 1.4,
        "enable_dehaze": False,
        "dehaze_strength": 0.0,
        "enable_pyramid": True,
        "pyramid_micro_texture": 1.1,
        "pyramid_structure_boost": 1.05,
        "enable_semantic_guidance": True,
    },
    "landscape": {
        "scale": 2.0,
        "sharpen_strength": 1.3,
        "enable_denoise": True,
        "denoise_intensity": 30,
        "enable_contrast": True,
        "contrast_boost": 2.1,
        "brightness_shift": 0,
        "vibrance_boost": 1.15,
        "color_temperature": 0,
        "portrait_smooth": 0,
        "eye_clarity": 1.0,
        "enable_dehaze": True,
        "dehaze_strength": 0.6,
        "enable_pyramid": True,
        "pyramid_micro_texture": 1.35,
        "pyramid_structure_boost": 1.2,
        "enable_semantic_guidance": True,
    },
    "low-light": {
        "scale": 2.0,
        "sharpen_strength": 0.9,
        "enable_denoise": True,
        "denoise_intensity": 65,
        "enable_contrast": True,
        "contrast_boost": 2.0,
        "brightness_shift": 8,
        "vibrance_boost": 1.05,
        "color_temperature": -5,
        "portrait_smooth": 0,
        "eye_clarity": 1.0,
        "enable_dehaze": False,
        "dehaze_strength": 0.0,
        "enable_pyramid": True,
        "pyramid_micro_texture": 1.1,
        "pyramid_structure_boost": 1.1,
        "enable_semantic_guidance": True,
    },
    "art": {
        "scale": 3.0,
        "sharpen_strength": 1.4,
        "enable_denoise": True,
        "denoise_intensity": 25,
        "enable_contrast": True,
        "contrast_boost": 1.3,
        "brightness_shift": 0,
        "vibrance_boost": 1.12,
        "color_temperature": 0,
        "portrait_smooth": 0,
        "eye_clarity": 1.0,
        "enable_dehaze": False,
        "dehaze_strength": 0.0,
        "enable_pyramid": True,
        "pyramid_micro_texture": 1.3,
        "pyramid_structure_boost": 1.2,
        "enable_semantic_guidance": True,
    },
    "fast": {
        "scale": 2.0,
        "sharpen_strength": 0.8,
        "enable_denoise": False,
        "denoise_intensity": 25,
        "enable_contrast": True,
        "contrast_boost": 1.5,
        "brightness_shift": 0,
        "vibrance_boost": 1.05,
        "color_temperature": 0,
        "portrait_smooth": 0,
        "eye_clarity": 1.0,
        "enable_diagnostics": False,
        "auto_tune": False,
        "enable_dehaze": False,
        "dehaze_strength": 0.0,
        "enable_pyramid": False,
        "pyramid_micro_texture": 1.0,
        "pyramid_structure_boost": 1.0,
        "pyramid_dynamic_range": 0.0,
        "enable_semantic_guidance": False,
    },
}


def get_preset_config(preset_name: str, base_config: Optional[PipelineConfig] = None) -> PipelineConfig:
    """Returns a PipelineConfig populated with parameters from named preset.

    Raises:
        ValueError: If ``preset_name`` is not a known preset. Use
            ``sorted(PRESETS)`` to list valid names.
    """
    cfg = PipelineConfig() if base_config is None else PipelineConfig(**base_config.to_dict())
    key = preset_name.lower().strip()
    if key not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name!r}. Available: {sorted(PRESETS)}")
    for param, val in PRESETS[key].items():
        setattr(cfg, param, val)
    return cfg
