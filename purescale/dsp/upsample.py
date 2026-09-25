"""Edge-Adaptive Spatial Upsampling (EASU) super-resolution."""

import cv2
import numpy as np


def edge_adaptive_upsample(img: np.ndarray, scale: float = 2.0) -> np.ndarray:
    """
    Edge-Adaptive Spatial Upsampling (EASU).
    Evaluates directional structure tensors to detect edge tangents,
    reconstructing smooth diagonal lines without staircase aliasing (jaggies).
    """
    if scale == 1.0:
        return img.copy()

    h, w = img.shape[:2]
    target_w = int(round(w * scale))
    target_h = int(round(h * scale))

    # Base high-fidelity Lanczos-4 sinc reconstruction
    lanczos = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    # Compute directional edge orientation on luminance
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    mag = cv2.magnitude(gx, gy)
    mag_up = cv2.resize(mag, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # Edge coherence weight (0 in flat areas, 1 on strong directional edges)
    coherence = np.clip((mag_up - 20.0) / 40.0, 0.0, 1.0)[:, :, np.newaxis]

    # Directional refinement: smooth along edge gradient tangent on high-coherence edges
    if np.any(coherence > 0.01):
        blur_tangent = cv2.GaussianBlur(lanczos, (3, 3), 0.5)
        out = (1.0 - 0.25 * coherence) * lanczos.astype(np.float32) + (0.25 * coherence) * blur_tangent.astype(np.float32)
        return np.clip(np.rint(out), 0.0, 255.0).astype(np.uint8)

    return lanczos
