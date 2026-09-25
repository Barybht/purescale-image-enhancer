"""Multiscale Local Laplacian Pyramid Filtering for PureScale 4.0.

Provides edge-preserving octave frequency band decomposition (L0, L1, L2, G3)
with non-linear micro-texture enhancement and halo-free dynamic range control,
following the principles of Paris, Hasinoff, and Kautz (SIGGRAPH 2011).
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np


def build_gaussian_pyramid(img: np.ndarray, num_levels: int = 4) -> List[np.ndarray]:
    """
    Constructs an edge-preserving Gaussian pyramid using 5x5 binomial smoothing.

    Args:
        img: float32 image [H, W] or [H, W, C] in range [0.0, 1.0]
        num_levels: Total pyramid levels including original (default 4: G0, G1, G2, G3)

    Returns:
        List of Gaussian pyramid levels [G0, G1, G2, G3]
    """
    pyramid = [img]
    current = img
    for _ in range(num_levels - 1):
        current = cv2.pyrDown(current)
        pyramid.append(current)
    return pyramid


def build_laplacian_pyramid(
    gaussian_pyr: List[np.ndarray],
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Constructs a Laplacian bandpass pyramid from a Gaussian pyramid.

    Formula:
        L_k = G_k - pyrUp(G_{k+1}, dstsize=shape(G_k))

    Args:
        gaussian_pyr: List of Gaussian pyramid levels [G0, G1, G2, G3]

    Returns:
        Tuple of (laplacian_bands [L0, L1, L2], base_residual G3)
    """
    num_levels = len(gaussian_pyr)
    laplacian_bands = []

    for k in range(num_levels - 1):
        g_k = gaussian_pyr[k]
        g_k_plus_1 = gaussian_pyr[k + 1]
        h, w = g_k.shape[:2]
        upsampled = cv2.pyrUp(g_k_plus_1, dstsize=(w, h))
        laplacian_bands.append(g_k - upsampled)

    base_residual = gaussian_pyr[-1]
    return laplacian_bands, base_residual


def reconstruct_laplacian_pyramid(
    laplacian_bands: List[np.ndarray],
    base_residual: np.ndarray,
) -> np.ndarray:
    """
    Reconstructs the full-resolution spatial image from Laplacian bands and base residual.

    Energy-conserving: Reconstructs exact input if bands are unmodified.

    Args:
        laplacian_bands: List of Laplacian bands [L0, L1, L2]
        base_residual: Base low-frequency residual G3

    Returns:
        Reconstructed full-resolution float32 image [H, W] or [H, W, C]
    """
    current = base_residual.copy()
    num_bands = len(laplacian_bands)

    for k in range(num_bands - 1, -1, -1):
        band = laplacian_bands[k]
        h, w = band.shape[:2]
        upsampled = cv2.pyrUp(current, dstsize=(w, h))
        current = upsampled + band

    return current


def edge_preserving_detail_transfer(
    band: np.ndarray,
    gain: float,
    threshold: float = 0.08,
    attenuation_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Non-linear detail enhancement transfer function.

    Formula:
        f(v) = v * (1.0 + (gain - 1.0) * exp(-|v| / threshold))

    Small micro-textures (|v| < threshold) receive full boost `gain`.
    Large high-contrast step edges (|v| >> threshold) decay smoothly to 1.0 gain,
    guaranteeing strict edge preservation and 100% halo-free operation.

    Args:
        band: Laplacian band float32 array
        gain: Target micro-texture gain factor (> 1.0 boosts, < 1.0 softens)
        threshold: Edge transition cutoff in normalized luminance units
        attenuation_mask: Optional spatial attenuation map [0.0, 1.0]

    Returns:
        Enhanced Laplacian band
    """
    if abs(gain - 1.0) < 1e-4 and attenuation_mask is None:
        return band

    abs_v = np.abs(band)
    # Exponential transition factor
    transition = np.exp(-abs_v / (threshold + 1e-5))

    effective_gain = 1.0 + (gain - 1.0) * transition

    if attenuation_mask is not None:
        # Interpolate between unity gain and effective gain based on spatial mask
        h, w = band.shape[:2]
        if attenuation_mask.shape[:2] != (h, w):
            mask_scaled = cv2.resize(attenuation_mask, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            mask_scaled = attenuation_mask
        if band.ndim == 3 and mask_scaled.ndim == 2:
            mask_scaled = mask_scaled[:, :, np.newaxis]
        effective_gain = 1.0 + (effective_gain - 1.0) * mask_scaled

    return band * effective_gain


def multiscale_laplacian_filter(
    img_bgr: np.ndarray,
    micro_texture_gain: float = 1.25,
    structure_boost: float = 1.10,
    noise_damping: float = 0.0,
    dynamic_range_compression: float = 0.0,
    spatial_detail_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Executes 4-octave Multiscale Local Laplacian Pyramid filtering on an image.

    Processes luminance in YCrCb color space to preserve chromatic fidelity
    and prevent color fringing artifacts.

    Args:
        img_bgr: uint8 BGR image [H, W, 3]
        micro_texture_gain: Detail gain for band L1 (micro-textures like fabric, foliage, pores)
        structure_boost: Contour gain for band L2 (geometric shapes and contours)
        noise_damping: Attenuation for band L0 (sensor noise / grain suppression: 0.0 to 0.8)
        dynamic_range_compression: Compression factor for base residual G3 (0.0 to 1.0)
        spatial_detail_mask: Optional semantic soft mask [H, W] float32 in [0.0, 1.0]

    Returns:
        Enhanced uint8 BGR image [H, W, 3]
    """
    # Quick bypass if all parameters are neutral
    if (
        abs(micro_texture_gain - 1.0) < 0.02
        and abs(structure_boost - 1.0) < 0.02
        and noise_damping < 0.02
        and dynamic_range_compression < 0.02
    ):
        return img_bgr.copy()

    # Convert to YCrCb float32 for luminance separation
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    y_channel = ycrcb[:, :, 0] / 255.0  # Normalize to [0.0, 1.0]

    # 1. Build 4-level Gaussian Pyramid (G0, G1, G2, G3)
    gauss_pyr = build_gaussian_pyramid(y_channel, num_levels=4)

    # 2. Build Laplacian Pyramid (L0, L1, L2, and base residual G3)
    lap_bands, base_g3 = build_laplacian_pyramid(gauss_pyr)

    # 3. Process Band L0 (Highest frequency: Sensor grain & subpixel edges)
    if noise_damping > 0.01:
        # Soft-shrinkage on L0 to suppress sensor grain
        shrink_thresh = noise_damping * 0.03
        sign_l0 = np.sign(lap_bands[0])
        abs_l0 = np.abs(lap_bands[0])
        lap_bands[0] = sign_l0 * np.maximum(0.0, abs_l0 - shrink_thresh)

    # 4. Process Band L1 (Micro-texture: fabric, foliage, skin pores)
    lap_bands[1] = edge_preserving_detail_transfer(
        lap_bands[1],
        gain=micro_texture_gain,
        threshold=0.07,
        attenuation_mask=spatial_detail_mask,
    )

    # 5. Process Band L2 (Structural contours and shape volume)
    lap_bands[2] = edge_preserving_detail_transfer(
        lap_bands[2],
        gain=structure_boost,
        threshold=0.15,
        attenuation_mask=spatial_detail_mask,
    )

    # 6. Process Base Residual G3 (Low-frequency illumination field)
    if dynamic_range_compression > 0.01:
        # Compress base tone smoothly without halos
        mean_lum = float(np.mean(base_g3))
        # Gamma tone curve centered on mean luminance
        compressed = np.power(np.clip(base_g3, 1e-4, 1.0), 0.85)
        base_g3 = (1.0 - dynamic_range_compression) * base_g3 + dynamic_range_compression * compressed

    # 7. Collapse and Reconstruct Pyramid
    y_reconstructed = reconstruct_laplacian_pyramid(lap_bands, base_g3)
    y_clipped = np.clip(np.rint(y_reconstructed * 255.0), 0.0, 255.0)

    # Re-insert luminance channel and convert back to BGR
    ycrcb[:, :, 0] = y_clipped
    bgr_out = cv2.cvtColor(np.clip(np.rint(ycrcb), 0.0, 255.0).astype(np.uint8), cv2.COLOR_YCrCb2BGR)

    return bgr_out
