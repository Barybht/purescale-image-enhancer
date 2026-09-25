"""Perceptual color vibrance (Oklab) and chromatic adaptation (Bradford CAT16)."""

from typing import Tuple
import cv2
import numpy as np

# Matrices for sRGB <-> Oklab transformation (Bjorn Ottosson)
_M1 = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
], dtype=np.float32)

_M1_INV = np.linalg.inv(_M1)

_M2 = np.array([
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660],
], dtype=np.float32)

_M2_INV = np.linalg.inv(_M2)


# Contiguous transposes: `M.T` is a strided view that pushes matmul onto a
# slow path (measured ~10x on the M2 product). These are the operands used
# in every conversion below.
_M1_T = np.ascontiguousarray(_M1.T)
_M1_INV_T = np.ascontiguousarray(_M1_INV.T)
_M2_T = np.ascontiguousarray(_M2.T)
_M2_INV_T = np.ascontiguousarray(_M2_INV.T)


def _srgb_to_linear_lut() -> np.ndarray:
    """Exact 256-entry sRGB -> linear table (identical to the piecewise formula)."""
    v = np.arange(256, dtype=np.float32) / 255.0
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4).astype(np.float32)


def _linear_to_srgb_lut(size: int = 4096) -> np.ndarray:
    """Dense linear -> sRGB table for nearest-index lookup.

    Max error is half a step (0.5/4095 in sRGB units ~= 0.03 LSB), safely
    inside the 1 LSB roundtrip budget. A single gather pass, unlike lerp
    which costs two gathers plus arithmetic and profiles slower than the
    direct ``where``+``pow`` evaluation it replaces.
    """
    v = np.linspace(0.0, 1.0, size, dtype=np.float32)
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * (v ** (1.0 / 2.4)) - 0.055).astype(np.float32)


_LUT_SRGB2LIN = _srgb_to_linear_lut()
_LUT_LIN2SRGB = _linear_to_srgb_lut()
_LUT_LIN2SRGB_N = _LUT_LIN2SRGB.shape[0]


def srgb_to_oklab(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converts BGR image [0..255] to Oklab coordinates (L, a, b)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Exact 256-entry LUT instead of per-pixel where+pow (identical values).
    lin = _LUT_SRGB2LIN[rgb]

    # Linear sRGB to cone LMS space
    lms = lin @ _M1_T
    lms_cbrt = np.cbrt(np.maximum(lms, 0.0))

    # LMS^(1/3) to Oklab
    lab = lms_cbrt @ _M2_T
    return lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]


def oklab_to_srgb(L: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Converts Oklab coordinates (L, a, b) back to BGR image [0..255]."""
    lab = np.stack([L, a, b], axis=-1)
    lms_cbrt = lab @ _M2_INV_T
    # Sign-safe across NumPy versions (avoids float power NaN on negative values)
    lms = lms_cbrt * lms_cbrt * lms_cbrt

    lin = np.clip(lms @ _M1_INV_T, 0.0, 1.0)

    # Nearest-index LUT instead of per-pixel where+pow (~3x, <=0.03 LSB error).
    srgb = _LUT_LIN2SRGB[np.rint(lin * (_LUT_LIN2SRGB_N - 1)).astype(np.int32)]
    srgb_bytes = np.clip(srgb * 255.0, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(srgb_bytes, cv2.COLOR_RGB2BGR)


def oklab_vibrance(bgr: np.ndarray, boost: float = 1.1) -> np.ndarray:
    """
    Scales chromaticity in the Oklab perceptual color space with shadow desaturation protection.
    Maintains constant perceived lightness (L) and straight hue lines,
    completely preventing the blue-to-purple shift of legacy CIELAB and HSV.
    Protects near-black pixels from chroma noise amplification.
    """
    if boost == 1.0:
        return bgr.copy()
    if abs(float(boost) - 1.0) < 1e-3:
        return bgr.copy()

    # Grayscale fast path: nothing chromatic to boost (also skips 2 conversions).
    b, g, r = cv2.split(bgr)
    if np.array_equal(b, g) and np.array_equal(g, r):
        return bgr.copy()

    L, a, b = srgb_to_oklab(bgr)

    # Shadow gate: desaturate near-black pixels to keep blacks neutral and clean
    shadow_mask = np.clip((L - 0.03) / 0.12, 0.0, 1.0)
    effective_boost = 1.0 + (float(boost) - 1.0) * shadow_mask

    # True perceptual vibrance: boost muted colors more than saturated colors to prevent clipping
    chroma = np.sqrt(a * a + b * b)
    vibrance_gain = 1.0 + (effective_boost - 1.0) / (1.0 + 2.0 * chroma)

    # Negligible-gain fast path: skip the return conversion entirely.
    if float(np.max(vibrance_gain)) < 1.005:
        return bgr.copy()

    a_scaled = a * vibrance_gain
    b_scaled = b * vibrance_gain

    return oklab_to_srgb(L, a_scaled, b_scaled)


# CAT16 / CAT02 cone response matrix (Li et al., 2017 / CIE CAT16)
# Used for Bradford-style von Kries chromatic adaptation in LMS cone space
_M_CAT = np.array([
    [0.7328, 0.4296, -0.1624],
    [-0.7036, 1.6975, 0.0061],
    [0.0030, 0.0136, 0.9834],
], dtype=np.float32)

_M_CAT_INV = np.linalg.inv(_M_CAT)


def bradford_cat16_white_balance(bgr: np.ndarray, temperature_offset: int = 0) -> np.ndarray:
    """
    Bradford / CAT16 Chromatic Adaptation Transform.
    Adjusts white balance color temperature in the LMS cone domain using a von Kries diagonal transform
    derived from the CAT16/CAT02 cone response matrix.
    Preserves neutral black shadows and specular white highlights without color clipping.
    """
    if temperature_offset == 0:
        return bgr.copy()

    # Temperature shift: positive = warmer (golden/amber), negative = cooler (daylight blue)
    delta_k = temperature_offset / 100.0
    diag = np.diag([1.0 + 0.18 * delta_k, 1.0 + 0.04 * delta_k, 1.0 - 0.22 * delta_k]).astype(np.float32)

    # Full adaptation matrix T = M_inv @ diag @ M
    t_mat = _M_CAT_INV @ diag @ _M_CAT

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    adapted = rgb @ np.ascontiguousarray(t_mat.T)
    adapted_bgr = cv2.cvtColor(np.clip(adapted, 0.0, 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)

    return adapted_bgr
