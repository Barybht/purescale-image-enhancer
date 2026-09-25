"""Analytical Digital Signal Processing (DSP) and Advanced Computer Vision primitives.

Pure mathematical formulations running deterministically in milliseconds
on modern Zen 4 CPU SIMD architectures.
"""

from purescale.dsp.filters import side_window_filter, fast_guided_filter
from purescale.dsp.upsample import edge_adaptive_upsample
from purescale.dsp.contrast import bimef_exposure_fusion, contrast_adaptive_sharpen
from purescale.dsp.color import (
    srgb_to_oklab,
    oklab_to_srgb,
    oklab_vibrance,
    bradford_cat16_white_balance,
)
from purescale.dsp.restore import (
    directional_subpixel_depixelate,
    tensor_steered_shock_filter,
    pre_restoration_conditioning,
)
from purescale.dsp.diagnostics import (
    DiagnosticsResult,
    diagnose_image,
    auto_tune_parameters,
    estimate_wavelet_noise_mad,
    estimate_optical_blur,
    estimate_dynamic_range_entropy,
    estimate_shades_of_gray_illuminant,
    estimate_atmospheric_haze,
)
from purescale.dsp.pyramid import (
    multiscale_laplacian_filter,
    build_gaussian_pyramid,
    build_laplacian_pyramid,
    reconstruct_laplacian_pyramid,
)
from purescale.dsp.semantic import (
    SemanticMasks,
    extract_semantic_masks,
    compute_spatial_guidance_maps,
)
from purescale.dsp.dehaze import (
    atmospheric_dehaze,
    compute_dark_channel,
    estimate_atmospheric_light,
)

__all__ = [
    "side_window_filter",
    "edge_adaptive_upsample",
    "bimef_exposure_fusion",
    "contrast_adaptive_sharpen",
    "srgb_to_oklab",
    "oklab_to_srgb",
    "oklab_vibrance",
    "bradford_cat16_white_balance",
    "directional_subpixel_depixelate",
    "tensor_steered_shock_filter",
    "pre_restoration_conditioning",
    "DiagnosticsResult",
    "diagnose_image",
    "auto_tune_parameters",
    "estimate_wavelet_noise_mad",
    "estimate_optical_blur",
    "estimate_dynamic_range_entropy",
    "estimate_shades_of_gray_illuminant",
    "estimate_atmospheric_haze",
    "multiscale_laplacian_filter",
    "build_gaussian_pyramid",
    "build_laplacian_pyramid",
    "reconstruct_laplacian_pyramid",
    "SemanticMasks",
    "extract_semantic_masks",
    "compute_spatial_guidance_maps",
    "fast_guided_filter",
    "atmospheric_dehaze",
    "compute_dark_channel",
    "estimate_atmospheric_light",
]
