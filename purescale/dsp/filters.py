"""Edge- and corner-preserving Side Window Filter (SWF) denoising."""

import cv2
import numpy as np
from typing import Optional


def side_window_filter(img: np.ndarray, radius: int = 2, iterations: int = 1,
                        noise_sigma: Optional[float] = None) -> np.ndarray:
    """
    Side Window Filter (SWF) for edge- and corner-preserving denoising.
    Decomposes the kernel into 8 directional half-windows (L, R, U, D, NW, NE, SW, SE)
    and selects the window with minimal variance per pixel.
    Preserves acute corners, text terminals, and fine hair boundaries without rounding.

    Early exits (bitwise-identical copy, ~1ms instead of ~75-1000ms):
    - ``noise_sigma < 2.0`` (diagnosed Clean) with default strength skips entirely.
    - Near-uniform images (downsampled gray std < 0.75) skip entirely.
    """
    if radius <= 0 or iterations <= 0:
        return img.copy()

    # Diagnosed-clean skip: matches auto_tune Clean -> enable_denoise=False,
    # but also applies when diagnostics ran without auto_tune.
    if noise_sigma is not None and noise_sigma < 2.0 and radius <= 2 and iterations <= 1:
        return img.copy()

    # Near-uniform fast path on a small proxy (cheap: single resize + std).
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        h0, w0 = gray.shape[:2]
        scale = min(1.0, 160.0 / max(h0, w0))
        proxy = cv2.resize(gray, (max(1, int(w0 * scale)), max(1, int(h0 * scale))),
                           interpolation=cv2.INTER_AREA) if scale < 1.0 else gray
        if float(np.std(proxy.astype(np.float32))) < 0.75:
            return img.copy()
    except (cv2.error, ValueError):
        pass

    cur = img.astype(np.float32)
    h, w, c = cur.shape
    r = radius

    # 8 directional window slices relative to center (0,0)
    # Each window bounds: (row_start, row_end, col_start, col_end)
    windows = [
        (-r, 0, -r, 0),    # NW
        (-r, 0, 0, r),     # NE
        (0, r, -r, 0),     # SW
        (0, r, 0, r),      # SE
        (-r, 0, -r, r),    # U
        (0, r, -r, r),     # D
        (-r, r, -r, 0),    # L
        (-r, r, 0, r),     # R
    ]

    for _ in range(iterations):
        padded = cv2.copyMakeBorder(cur, r, r, r, r, cv2.BORDER_REFLECT)
        sq_padded = padded * padded

        best_diff = np.full((h, w, c), 1e9, dtype=np.float32)
        best_mean = cur.copy()

        for r1, r2, c1, c2 in windows:
            kh = r2 - r1 + 1
            kw = c2 - c1 + 1

            p_r1 = r + r1
            p_r2 = p_r1 + h + kh - 1
            p_c1 = r + c1
            p_c2 = p_c1 + w + kw - 1

            sub_p = padded[p_r1 : p_r2 + 1, p_c1 : p_c2 + 1]
            sub_sq = sq_padded[p_r1 : p_r2 + 1, p_c1 : p_c2 + 1]

            mean = cv2.boxFilter(sub_p, cv2.CV_32F, (kw, kh))[:h, :w]
            sq_mean = cv2.boxFilter(sub_sq, cv2.CV_32F, (kw, kh))[:h, :w]
            var = np.maximum(0.0, sq_mean - mean * mean)

            total_var = np.sum(var, axis=2, keepdims=True)
            diff = np.abs(mean - cur) + 0.1 * total_var

            mask = diff < best_diff
            best_diff = np.where(mask, diff, best_diff)
            best_mean = np.where(mask, mean, best_mean)

        cur = best_mean

    return np.clip(cur, 0.0, 255.0).astype(np.uint8)


def fast_guided_filter(
    guide: np.ndarray,
    src: np.ndarray,
    radius: int = 8,
    eps: float = 1e-3,
    subsample: int = 4,
) -> np.ndarray:
    """
    Subsampled Fast Guided Filter (He et al., ECCV 2010 / TPAMI 2013).

    Edge-preserving smoothing filter in O(N) runtime.
    Supports single-channel (grayscale) or multi-channel (BGR/RGB) inputs in both
    float32 [0.0, 1.0] and uint8 [0, 255] representations.

    Args:
        guide: Guidance image [H, W] or [H, W, C], float32 or uint8.
        src: Source image to filter [H, W] or [H, W, C], float32 or uint8.
        radius: Local window radius.
        eps: Regularization parameter.
        subsample: Spatial subsampling factor for O(N/s^2) acceleration.

    Returns:
        Filtered image with identical shape and dtype as `src`.
    """
    if radius <= 0:
        return src.copy()

    h, w = guide.shape[:2]
    if h == 0 or w == 0:
        return src.copy()

    is_uint8 = (src.dtype == np.uint8)

    # Normalize guide to float32 in [0.0, 1.0]
    if guide.dtype == np.uint8:
        g = guide.astype(np.float32) / 255.0
    else:
        g = guide.astype(np.float32)
        if float(np.max(g)) > 1.5:
            g = g / 255.0

    # Normalize src to float32 in [0.0, 1.0]
    if is_uint8:
        p = src.astype(np.float32) / 255.0
    else:
        p = src.astype(np.float32)
        if float(np.max(p)) > 1.5:
            p = p / 255.0

    s = max(1, subsample)
    small_size = (max(1, w // s), max(1, h // s))

    if s > 1:
        g_sub = cv2.resize(g, small_size, interpolation=cv2.INTER_AREA if is_uint8 else cv2.INTER_NEAREST)
        p_sub = cv2.resize(p, small_size, interpolation=cv2.INTER_AREA if is_uint8 else cv2.INTER_NEAREST)
    else:
        g_sub = g
        p_sub = p

    r_sub = max(1, radius // s)
    ksize = (2 * r_sub + 1, 2 * r_sub + 1)

    # Prepare broadcastable guidance if guide is 2D and src is 3D
    if g_sub.ndim == 2 and p_sub.ndim == 3:
        g_sub_corr = g_sub[:, :, np.newaxis]
    else:
        g_sub_corr = g_sub

    mean_g = cv2.boxFilter(g_sub, cv2.CV_32F, ksize)
    mean_p = cv2.boxFilter(p_sub, cv2.CV_32F, ksize)
    corr_gp = cv2.boxFilter(g_sub_corr * p_sub, cv2.CV_32F, ksize)
    var_g = cv2.boxFilter(g_sub * g_sub, cv2.CV_32F, ksize) - mean_g * mean_g

    if g_sub.ndim == 2 and p_sub.ndim == 3:
        mean_g_b = mean_g[:, :, np.newaxis]
        var_g_b = var_g[:, :, np.newaxis]
    else:
        mean_g_b = mean_g
        var_g_b = var_g

    a = (corr_gp - mean_g_b * mean_p) / (var_g_b + eps)
    b = mean_p - a * mean_g_b

    mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize)

    if s > 1:
        mean_a_up = cv2.resize(mean_a, (w, h), interpolation=cv2.INTER_LINEAR)
        mean_b_up = cv2.resize(mean_b, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        mean_a_up = mean_a
        mean_b_up = mean_b

    if g.ndim == 2 and p.ndim == 3:
        g_full = g[:, :, np.newaxis]
    else:
        g_full = g

    q = mean_a_up * g_full + mean_b_up

    if is_uint8:
        return np.clip(np.rint(q * 255.0), 0.0, 255.0).astype(np.uint8)
    else:
        if float(np.max(src)) > 1.5:
            return np.clip(q * 255.0, 0.0, 255.0).astype(src.dtype)
        return np.clip(q, 0.0, 1.0).astype(src.dtype)
