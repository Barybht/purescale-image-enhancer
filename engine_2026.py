"""Backward compatibility module re-exporting computational photography primitives from purescale.dsp."""

from purescale.dsp import (
    side_window_filter,
    edge_adaptive_upsample,
    bimef_exposure_fusion,
    contrast_adaptive_sharpen,
    srgb_to_oklab,
    oklab_to_srgb,
    oklab_vibrance,
    bradford_cat16_white_balance,
)
from purescale.neural.portrait import fast_guided_filter

__all__ = [
    "side_window_filter",
    "edge_adaptive_upsample",
    "bimef_exposure_fusion",
    "contrast_adaptive_sharpen",
    "fast_guided_filter",
    "srgb_to_oklab",
    "oklab_to_srgb",
    "oklab_vibrance",
    "bradford_cat16_white_balance",
]
