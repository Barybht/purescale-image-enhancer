"""Pre-restoration conditioning: Structure tensor shock deblurring and directional subpixel de-pixelation."""

import cv2
import numpy as np


def directional_subpixel_depixelate(img: np.ndarray, strength: int = 40) -> np.ndarray:
    """
    Directional Subpixel Anti-Aliasing (De-pixelation and Deblocking).

    Detects axis-aligned staircase pixel steps and compression grid boundaries,
    interpolating along edge tangents to reconstruct smooth geometric curves
    prior to super-resolution scaling.

    Args:
        img: uint8 BGR image [H, W, 3]
        strength: Intensity from 0 to 100

    Returns:
        De-pixelated uint8 BGR image [H, W, 3]
    """
    if strength <= 0:
        return img.copy()

    h, w = img.shape[:2]
    alpha = (float(strength) / 100.0) * 0.85
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # First derivatives
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)

    # Dominant edge orientation angle via structure tensor
    theta = 0.5 * np.arctan2(2.0 * gx * gy, (gx**2 - gy**2 + 1e-5))
    # Tangent angle (orthogonal to gradient normal)
    phi = theta + (np.pi / 2.0)

    # Subpixel grid coordinates along tangent (+1 and -1 pixel)
    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = np.cos(phi)
    dy = np.sin(phi)

    map_x1 = np.clip(grid_x + dx, 0, w - 1).astype(np.float32)
    map_y1 = np.clip(grid_y + dy, 0, h - 1).astype(np.float32)
    map_x2 = np.clip(grid_x - dx, 0, w - 1).astype(np.float32)
    map_y2 = np.clip(grid_y - dy, 0, h - 1).astype(np.float32)

    sample1 = cv2.remap(img, map_x1, map_y1, interpolation=cv2.INTER_LINEAR)
    sample2 = cv2.remap(img, map_x2, map_y2, interpolation=cv2.INTER_LINEAR)

    tangent_blend = 0.5 * sample1.astype(np.float32) + 0.5 * sample2.astype(np.float32)

    # Apply selectively along high-frequency pixel steps and block borders
    gate = np.clip((mag - 10.0) / 26.0, 0.0, 1.0)[:, :, np.newaxis]
    weight = alpha * gate

    out = (1.0 - weight) * img.astype(np.float32) + weight * tangent_blend
    return np.clip(np.rint(out), 0.0, 255.0).astype(np.uint8)


def tensor_steered_shock_filter(
    img: np.ndarray,
    strength: int = 40,
    iterations: int = 2,
) -> np.ndarray:
    """
    Structure Tensor-Steered Morphological Shock Filter (Alvarez & Mazorra formulation).

    Reverses optical diffusion and lens defocus by calculating the second directional
    derivative I_{eta eta} along the gradient normal, steered by the 2D structure tensor
    eigen-coherence to prevent ringing along corners and noise textures.

    Args:
        img: uint8 BGR image [H, W, 3]
        strength: Deblur intensity (0 to 100)
        iterations: Number of shock iterations (1 to 3)

    Returns:
        Sharpened/deblurred uint8 BGR image [H, W, 3]
    """
    if strength <= 0:
        return img.copy()

    alpha = (float(strength) / 100.0) * 0.70
    cur = img.copy()
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    iters = max(1, min(iterations, 3 if strength > 60 else 2))

    for _ in range(iters):
        dil = cv2.dilate(cur, k)
        ero = cv2.erode(cur, k)

        gray = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # Spatial smoothing prior to differentiation
        blur = cv2.GaussianBlur(gray, (3, 3), 0.7)

        # 1st derivatives
        ix = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        iy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        grad_norm_sq = ix**2 + iy**2 + 1e-5

        # 2nd derivatives
        ixx = cv2.Sobel(ix, cv2.CV_32F, 1, 0, ksize=3)
        iyy = cv2.Sobel(iy, cv2.CV_32F, 0, 1, ksize=3)
        ixy = cv2.Sobel(ix, cv2.CV_32F, 0, 1, ksize=3)

        # Directional curvature along gradient direction: I_{eta eta}
        i_eta_eta = (ix**2 * ixx + 2.0 * ix * iy * ixy + iy**2 * iyy) / grad_norm_sq

        # Structure tensor coherence: lambda1 - lambda2 / lambda1 + lambda2
        j_xx = cv2.GaussianBlur(ix**2, (3, 3), 1.0)
        j_yy = cv2.GaussianBlur(iy**2, (3, 3), 1.0)
        j_xy = cv2.GaussianBlur(ix * iy, (3, 3), 1.0)

        tr = j_xx + j_yy
        det = j_xx * j_yy - j_xy**2
        discriminant = np.maximum(0.0, tr**2 - 4.0 * det)
        sqrt_disc = np.sqrt(discriminant)
        lambda1 = 0.5 * (tr + sqrt_disc)
        lambda2 = 0.5 * (tr - sqrt_disc)

        coherence = np.clip((lambda1 - lambda2) / (lambda1 + lambda2 + 1e-4), 0.0, 1.0)
        coherence = coherence[:, :, np.newaxis]

        # Convex side (i_eta_eta < 0, bright slope) -> propagate dilation
        # Concave side (i_eta_eta > 0, dark slope) -> propagate erosion
        choice = np.where(i_eta_eta[:, :, np.newaxis] < 0, dil, ero)

        # Gated by local coherence so flat noise regions are unaffected
        effective_weight = alpha * coherence

        blend = (1.0 - effective_weight) * cur.astype(np.float32) + effective_weight * choice.astype(np.float32)
        cur = np.clip(np.rint(blend), 0.0, 255.0).astype(np.uint8)

    return cur


def pre_restoration_conditioning(
    img: np.ndarray,
    depixel_strength: int = 0,
    deblur_strength: int = 0,
) -> np.ndarray:
    """
    Applies de-pixelation and tensor-steered deblurring prior to super-resolution.

    Args:
        img: uint8 BGR image [H, W, 3]
        depixel_strength: Anti-aliasing strength (0-100)
        deblur_strength: Tensor shock deblur strength (0-100)

    Returns:
        Conditioned uint8 BGR image [H, W, 3]
    """
    out = img
    if depixel_strength > 0:
        out = directional_subpixel_depixelate(out, strength=depixel_strength)
    if deblur_strength > 0:
        out = tensor_steered_shock_filter(out, strength=deblur_strength)
    return out
