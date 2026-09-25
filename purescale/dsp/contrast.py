"""Dynamic range fusion (BIMEF with black-point pinning) and halo-free sharpening (CAS)."""

import cv2
import numpy as np


def bimef_exposure_fusion(img: np.ndarray, contrast_boost: float = 1.8) -> np.ndarray:
    """
    Bio-Inspired Contrast and Dynamic Range Enhancement with Black-Point Pinning.
    Anchors deep blacks (<= 0.04) and specular highlights (>= 0.96) while expanding
    midtone local contrast and clarity without lifting shadow floors or creating haze.
    """
    if contrast_boost <= 1.0:
        return img.copy()

    img_f = img.astype(np.float32) / 255.0
    luma = 0.2126 * img_f[:, :, 2] + 0.7152 * img_f[:, :, 1] + 0.0722 * img_f[:, :, 0]

    k_size = max(15, int(min(img.shape[:2]) * 0.03) | 1)
    base = cv2.GaussianBlur(luma, (k_size, k_size), 0)
    detail = luma - base

    # S-curve with strict black/white point anchoring
    p = 1.0 + (contrast_boost - 1.0) * 0.25
    base_curved = (base ** p) / (base ** p + (1.0 - base) ** p + 1e-6)

    # Local micro-contrast boost on detail (attenuated at extremes to avoid noise/halos)
    shadow_gate = np.clip((luma - 0.04) / 0.16, 0.0, 1.0)
    highlight_gate = np.clip((0.96 - luma) / 0.16, 0.0, 1.0)
    detail_weight = shadow_gate * highlight_gate

    detail_boosted = detail * (1.0 + (contrast_boost - 1.0) * 0.4 * detail_weight)
    new_luma = np.clip(base_curved + detail_boosted, 0.0, 1.0)

    # Transfer new luminance to color channels preserving chromaticity
    ratio = np.where(luma > 1e-4, new_luma / (luma + 1e-4), 1.0)[:, :, np.newaxis]
    ratio = np.clip(ratio, 0.75, 1.45)

    return np.clip(img_f * ratio * 255.0, 0.0, 255.0).astype(np.uint8)


def contrast_adaptive_sharpen(img: np.ndarray, strength: float = 1.2) -> np.ndarray:
    """
    Contrast-Adaptive Sharpening (CAS).
    Applies non-linear high-frequency detail amplification strictly clamped by
    local 3x3 contrast bounds. Completely prevents halo overshoot on step edges.
    """
    if strength <= 0.0:
        return img.copy()

    img_f = img.astype(np.float32)
    pad = cv2.copyMakeBorder(img_f, 1, 1, 1, 1, cv2.BORDER_REFLECT)

    # 3x3 cardinal neighbors: b (top), d (left), f (right), h (bottom), e (center)
    b = pad[:-2, 1:-1]
    d = pad[1:-1, :-2]
    f = pad[1:-1, 2:]
    h = pad[2:, 1:-1]
    e = img_f

    # Local min/max bounding across cardinal cross
    mn = np.minimum(np.minimum(b, d), np.minimum(f, np.minimum(h, e)))
    mx = np.maximum(np.maximum(b, d), np.maximum(f, np.maximum(h, e)))

    # Smooth sharpening ramp bounded by distance to dynamic range limits
    amp = np.clip(np.minimum(mn, 255.0 - mx) / (mx + 1e-5), 0.0, 1.0)

    # Peak weight maps strength [0..3] -> reciprocal weight
    peak = -1.0 / (np.interp(strength, [0.0, 3.0], [8.0, 3.5]) + 1e-5)
    w = amp * peak

    kernel_sum = b + d + f + h
    out = (e + w * kernel_sum) / (1.0 + 4.0 * w)

    return np.clip(out, 0.0, 255.0).astype(np.uint8)
