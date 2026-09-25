"""Reference-based image quality metrics for PureScale.

Implemented with OpenCV + NumPy only (no new dependencies):
- PSNR (peak signal-to-noise ratio, dB; ``inf`` for identical images)
- SSIM (Wang et al. 2004 structural similarity, 11x11 Gaussian, [-1, 1])

Used by the CLI ``compare`` command and the golden-fixture regression
tests that gate future output-changing optimizations (e.g. dehaze proxy).
"""

import math
from typing import Dict, Tuple

import cv2
import numpy as np


def _to_gray_float(img: np.ndarray) -> np.ndarray:
    """Converts BGR/gray uint8 image to float32 grayscale."""
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    return gray.astype(np.float32)


def _check_pair(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Validates a comparison pair and returns grayscale float32 views."""
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch for quality metrics: {a.shape} vs {b.shape}")
    if a.dtype != np.uint8 or b.dtype != np.uint8:
        raise ValueError("Quality metrics require uint8 images")
    return _to_gray_float(a), _to_gray_float(b)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB (``inf`` when identical)."""
    ga, gb = _check_pair(a, b)
    mse = float(np.mean((ga - gb) ** 2))
    if mse <= 0.0:
        return math.inf
    return 10.0 * math.log10(255.0 * 255.0 / mse)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean structural similarity index in [-1, 1] (1.0 when identical)."""
    ga, gb = _check_pair(a, b)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    mu1 = cv2.GaussianBlur(ga, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(gb, (11, 11), 1.5)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(ga * ga, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(gb * gb, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(ga * gb, (11, 11), 1.5) - mu12

    num = (2.0 * mu12 + c1) * (2.0 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = num / np.maximum(den, 1e-12)
    return float(np.mean(ssim_map))


def compare_images(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Compares two uint8 images of equal shape.

    Returns:
        Dict with ``psnr_db`` (inf when identical) and ``ssim`` (1.0 when
        identical).
    """
    return {"psnr_db": psnr(a, b), "ssim": ssim(a, b)}
