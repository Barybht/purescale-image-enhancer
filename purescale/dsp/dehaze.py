"""Dark Channel Prior (DCP) Atmospheric Dehazing for PureScale 4.0.

Recovers true scene radiance, atmospheric contrast, and color saturation in hazy,
foggy, or backlit imagery following the Dark Channel Prior formulation
(He, Sun, and Tang, CVPR 2009 / TPAMI 2011) with Fast Guided Filter refinement.
"""

from typing import Tuple
import cv2
import numpy as np

from purescale.dsp.semantic import fast_guided_filter


def compute_dark_channel(img_norm: np.ndarray, patch_radius: int = 7) -> np.ndarray:
    """
    Computes the dark channel of an image over a square local neighborhood.

    Formula:
        J^{dark}(x) = min_{y in Omega(x)} ( min_{c} I^c(y) )

    Args:
        img_norm: float32 BGR image [H, W, 3] in [0.0, 1.0]
        patch_radius: Neighborhood patch radius (kernel size = 2*r + 1)

    Returns:
        Dark channel array [H, W] float32 in [0.0, 1.0]
    """
    min_channel = np.min(img_norm, axis=2)
    ksize = 2 * patch_radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.erode(min_channel, kernel)


def estimate_atmospheric_light(img_norm: np.ndarray, dark_channel: np.ndarray) -> np.ndarray:
    """
    Estimates the atmospheric airlight vector A [B, G, R] from the top 0.1%
    brightest pixels in the dark channel.

    Args:
        img_norm: float32 BGR image [H, W, 3] in [0.0, 1.0]
        dark_channel: Dark channel array [H, W] float32

    Returns:
        Atmospheric airlight vector A of shape (3,) in float32
    """
    h, w = dark_channel.shape
    total_pixels = h * w
    num_top = max(10, int(total_pixels * 0.001))

    # Indices of top 0.1% in dark channel
    flat_dark = dark_channel.ravel()
    top_indices = np.argpartition(flat_dark, -num_top)[-num_top:]

    # Select the candidate pixel with maximum luminance
    flat_img = img_norm.reshape(-1, 3)
    candidates = flat_img[top_indices]
    luminances = 0.299 * candidates[:, 2] + 0.587 * candidates[:, 1] + 0.114 * candidates[:, 0]
    best_idx = np.argmax(luminances)
    airlight = candidates[best_idx]

    # Ensure airlight is physically plausible (not completely dark)
    airlight = np.maximum(airlight, 0.40)
    return airlight


def estimate_transmission_map(
    img_norm: np.ndarray,
    airlight: np.ndarray,
    omega: float = 0.95,
    patch_radius: int = 7,
) -> np.ndarray:
    """
    Estimates the coarse transmission map t_tilde(x) normalized by atmospheric airlight.

    Formula:
        t_tilde(x) = 1.0 - omega * min_{Omega} ( min_{c} (I^c(y) / A^c) )

    Args:
        img_norm: float32 BGR image [H, W, 3] in [0.0, 1.0]
        airlight: Atmospheric airlight vector [B, G, R]
        omega: Aerial perspective constant (preserves subtle distance cues)
        patch_radius: Local neighborhood radius

    Returns:
        Coarse transmission map [H, W] float32 in [0.0, 1.0]
    """
    # Normalize channels by atmospheric airlight
    norm_by_a = img_norm / (airlight[np.newaxis, np.newaxis, :] + 1e-6)
    dark_norm = compute_dark_channel(norm_by_a, patch_radius=patch_radius)
    transmission = 1.0 - omega * dark_norm
    return np.clip(transmission, 0.0, 1.0)


def atmospheric_dehaze(
    img_bgr: np.ndarray,
    strength: float = 0.65,
    transmission_floor: float = 0.10,
    patch_radius: int = 7,
    proxy_max_dim: int = 640,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies Dark Channel Prior dehazing to restore scene radiance.

    Args:
        img_bgr: uint8 BGR image [H, W, 3]
        strength: Dehazing intensity factor in range [0.0, 1.0]
        transmission_floor: Minimum transmission t_0 to prevent noise explosion in distant sky
        patch_radius: Patch radius for dark channel erosion
        proxy_max_dim: Long-edge cap for dark-channel/airlight/coarse-transmission
            estimation. Larger images use a downsampled proxy (~9x fewer pixels
            at 1080p); the coarse map is upsampled and refined with the
            full-resolution guide, preserving edge alignment. Set to 0 to
            always run the full-resolution path.

    Returns:
        Tuple of (dehazed_bgr uint8, refined_transmission_map float32)
    """
    if strength <= 0.01:
        return img_bgr, np.ones(img_bgr.shape[:2], dtype=np.float32)

    h, w = img_bgr.shape[:2]
    if proxy_max_dim > 0 and max(h, w) > proxy_max_dim:
        scale = proxy_max_dim / float(max(h, w))
        proxy = cv2.resize(img_bgr, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        proxy_norm = proxy.astype(np.float32) / 255.0
        # Keep the physical patch size roughly constant across scales.
        proxy_radius = max(2, int(round(patch_radius * scale)))

        dark_ch = compute_dark_channel(proxy_norm, patch_radius=proxy_radius)
        airlight = estimate_atmospheric_light(proxy_norm, dark_ch)

        omega = 0.95 * np.clip(strength, 0.1, 1.0)
        coarse_small = estimate_transmission_map(
            proxy_norm, airlight, omega=omega, patch_radius=proxy_radius
        )
        coarse_t = cv2.resize(coarse_small, (w, h), interpolation=cv2.INTER_LINEAR)
        img_norm = img_bgr.astype(np.float32) / 255.0
    else:
        img_norm = img_bgr.astype(np.float32) / 255.0

        # 1. Dark Channel & Atmospheric Airlight
        dark_ch = compute_dark_channel(img_norm, patch_radius=patch_radius)
        airlight = estimate_atmospheric_light(img_norm, dark_ch)

        # 2. Coarse Transmission Map with strength-scaled omega
        omega = 0.95 * np.clip(strength, 0.1, 1.0)
        coarse_t = estimate_transmission_map(img_norm, airlight, omega=omega, patch_radius=patch_radius)

    # 3. Refine Transmission with Fast Guided Filter using grayscale guide
    gray_guide = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    refined_t = fast_guided_filter(gray_guide, coarse_t, radius=12, eps=1e-3, subsample=2)

    # Enforce minimum transmission floor
    t_clamped = np.maximum(refined_t, transmission_floor)[:, :, np.newaxis]

    # 4. Radiance Recovery: J(x) = (I(x) - A) / max(t(x), t0) + A
    radiance = (img_norm - airlight[np.newaxis, np.newaxis, :]) / t_clamped + airlight[np.newaxis, np.newaxis, :]
    radiance_clipped = np.clip(np.rint(radiance * 255.0), 0.0, 255.0).astype(np.uint8)

    # Blend with original according to strength
    if strength < 1.0:
        result = cv2.addWeighted(radiance_clipped, strength, img_bgr, 1.0 - strength, 0.0)
    else:
        result = radiance_clipped

    return result, refined_t
