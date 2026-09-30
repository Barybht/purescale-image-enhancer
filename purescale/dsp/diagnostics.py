"""Autonomous image quality diagnostics and physical signal analysis for PureScale 4.0.

Provides deterministic mathematical estimation of sensor noise, optical blur,
dynamic range entropy, color cast illuminant bias, and atmospheric haze veiling,
along with an autonomous auto-tuner for optimal pipeline parameter synthesis.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np


@dataclass
class DiagnosticsResult:
    """Strongly-typed physical signal diagnostics and quality metrics."""
    noise_sigma: float = 0.0          # Estimated Gaussian noise sigma (0.0 to 100.0)
    noise_category: str = "Clean"     # "Clean", "Low", "Moderate", "Heavy"
    chroma_noise_sigma: float = 0.0   # Estimated chrominance noise sigma (Cb/Cr)
    blur_score: float = 0.0           # Optical/motion blur index (0.0=sharp, 1.0=heavily blurred)
    blur_category: str = "Sharp"      # "Sharp", "Acceptable", "Soft", "Blurred"
    entropy: float = 0.0              # Shannon luminance entropy (0.0 to 8.0 bits)
    dynamic_range: int = 255          # 99th minus 1st percentile luminance spread
    shadow_clipping: float = 0.0      # Shadow under-exposure percentage (Y < 5)
    highlight_clipping: float = 0.0   # Highlight blowout percentage (Y > 250)
    mean_luminance: float = 128.0     # Mean luminance level (0.0 to 255.0)
    backlight_ratio: float = 1.0      # Peripheral-to-center luminance ratio
    backlight_detected: bool = False  # True if central subject is backlit
    color_cast_kelvin: int = 0        # Recommended white-balance temperature shift (+-50 estimated, +-30 after auto-tune)
    color_tint_offset: int = 0        # Recommended green-magenta tint shift (+-40 estimated, +-30 after auto-tune)
    color_cast_name: str = "Neutral"  # "Neutral", "Warm", "Cool", "Green", "Magenta"
    haze_index: float = 0.0           # Atmospheric veiling index (0.0=clear, 1.0=dense haze)
    haze_detected: bool = False       # True if haze index exceeds atmospheric threshold
    jpeg_blocking_score: float = 0.0  # 8x8 DCT boundary step ratio
    jpeg_blocking_detected: bool = False # True if compression blocking exceeds threshold
    suggested_style: str = "photo"    # Content classification: "photo", "anime", or "manga"
    style_confidence: float = 0.0     # Classification confidence in [0.0, 1.0]
    semantic_breakdown: Dict[str, float] = field(default_factory=dict)
    recommended_parameters: Dict[str, Any] = field(default_factory=dict)
    explanations: list[str] = field(default_factory=list)

    def summary_table(self) -> str:
        """Returns a formatted ASCII summary of diagnostic telemetry."""
        width = 64
        border = "+" + "-" * (width - 2) + "+"

        def row(text: str) -> str:
            return f"| {text:<{width - 4}} |"

        noise_str = f"{self.noise_sigma:5.2f}Y"
        if self.chroma_noise_sigma >= 2.0:
            noise_str += f"/{self.chroma_noise_sigma:4.1f}C"
        noise_str += f" [{self.noise_category:<8}]"

        wb_str = f"{self.color_cast_kelvin:+4d}K/{self.color_tint_offset:+3d}T [{self.color_cast_name:<8}]"
        light_str = f"{'BACKLIT' if self.backlight_detected else 'BALANCED':<8} [{self.backlight_ratio:4.2f}x]"

        lines = [
            border,
            row("PURESCALE 4.0 SIGNAL DIAGNOSTICS"),
            border,
            row(f"Sensor Noise Sigma  : {noise_str:<26}"),
            row(f"Optical Blur Score  : {self.blur_score:6.3f}       [{self.blur_category:<10}]"),
            row(f"Dynamic Entropy     : {self.entropy:6.2f} bits    [Range: {self.dynamic_range:<3} levels]"),
            row(f"Shadow/High Clipping: {self.shadow_clipping:5.1f}% / {self.highlight_clipping:4.1f}%"),
            row(f"Lighting Geometry   : {light_str:<26}"),
            row(f"Color Cast / Tint   : {wb_str:<26}"),
            row(f"Atmospheric Haze    : {self.haze_index:6.3f}       [{'HAZY' if self.haze_detected else 'CLEAR':<8}]"),
            row(f"Content Style       : {self.suggested_style:<8} [{self.style_confidence:4.2f} confidence]"),
        ]
        if self.jpeg_blocking_detected:
            lines.append(row(f"Compression Blocking: {self.jpeg_blocking_score:6.2f}       [DETECTED  ]"))
        if self.semantic_breakdown:
            items = [f"{k}: {v * 100:.0f}%" for k, v in self.semantic_breakdown.items() if v > 0.05]
            current = "Scene Semantics     : "
            for item in items:
                candidate = item if current.endswith(": ") else current + ", " + item
                if len(candidate) > width - 4:
                    lines.append(row(current))
                    current = "                      " + item
                else:
                    current = candidate
            lines.append(row(current if items else "Scene Semantics     : General"))
        if self.explanations:
            lines.append(border)
            lines.append(row("DIAGNOSTIC AUTO-TUNE EXPLANATIONS:"))
            for exp in self.explanations:
                if len(exp) > width - 6:
                    lines.append(row("- " + exp[:width - 6]))
                else:
                    lines.append(row("- " + exp))
        lines.append(border)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Machine-readable telemetry (single source for CLI JSON output)."""
        import dataclasses

        return dataclasses.asdict(self)



def estimate_wavelet_noise_mad(img_gray: np.ndarray) -> Tuple[float, str]:
    """
    Estimates additive Gaussian sensor noise standard deviation using the
    Donoho & Johnstone (1994) Median Absolute Deviation (MAD) on the high-frequency
    diagonal subband (HH1) of a 2D Haar wavelet transform.

    Formula:
        sigma_noise = median(|HH1 - median(HH1)|) / 0.6745

    Args:
        img_gray: Single-channel uint8 or float32 image [H, W]

    Returns:
        Tuple of (sigma_estimate, category_string)
    """
    gray = img_gray.astype(np.float32)
    h, w = gray.shape[:2]

    # Crop to even dimensions for exact 2x2 Haar decomposition
    h_even = h - (h % 2)
    w_even = w - (w % 2)
    g = gray[:h_even, :w_even]

    # 2D Haar diagonal subband HH1:
    hh1 = 0.5 * (
        g[0::2, 0::2]
        - g[0::2, 1::2]
        - g[1::2, 0::2]
        + g[1::2, 1::2]
    )

    # Median Absolute Deviation
    hh_abs = np.abs(hh1 - np.median(hh1))
    mad = float(np.median(hh_abs))
    sigma = mad / 0.6745

    if sigma < 2.0:
        cat = "Clean"
    elif sigma < 5.0:
        cat = "Low"
    elif sigma < 10.0:
        cat = "Moderate"
    else:
        cat = "Heavy"

    return float(round(sigma, 2)), cat


def estimate_chroma_wavelet_noise_mad(img_bgr: np.ndarray) -> Tuple[float, str]:
    """
    Estimates chrominance sensor noise standard deviation across Cb and Cr channels
    using the Donoho & Johnstone (1994) 2D Haar diagonal subband (HH1) MAD.

    Args:
        img_bgr: uint8 BGR image [H, W, 3]

    Returns:
        Tuple of (chroma_sigma_estimate, category_string)
    """
    if img_bgr is None or img_bgr.size == 0 or img_bgr.ndim == 2:
        return 0.0, "Clean"
    if img_bgr.ndim == 3 and img_bgr.shape[2] == 1:
        return 0.0, "Clean"
    if img_bgr.ndim == 3 and img_bgr.shape[2] == 4:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)

    h, w = img_bgr.shape[:2]
    if h < 4 or w < 4:
        return 0.0, "Clean"

    # Fast 512px proxy for high-resolution images
    max_dim = 512
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        small = cv2.resize(img_bgr, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = img_bgr

    ycrcb = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:, :, 1].astype(np.float32)
    cb = ycrcb[:, :, 2].astype(np.float32)

    def _channel_mad(ch: np.ndarray) -> float:
        ch_h, ch_w = ch.shape[:2]
        ch_h_even = ch_h - (ch_h % 2)
        ch_w_even = ch_w - (ch_w % 2)
        g = ch[:ch_h_even, :ch_w_even]
        hh1 = 0.5 * (
            g[0::2, 0::2]
            - g[0::2, 1::2]
            - g[1::2, 0::2]
            + g[1::2, 1::2]
        )
        hh_abs = np.abs(hh1 - np.median(hh1))
        return float(np.median(hh_abs)) / 0.6745

    sigma_cr = _channel_mad(cr)
    sigma_cb = _channel_mad(cb)
    sigma_chroma = float(np.sqrt(0.5 * (sigma_cr * sigma_cr + sigma_cb * sigma_cb)))

    if sigma_chroma < 2.0:
        cat = "Clean"
    elif sigma_chroma < 5.0:
        cat = "Low"
    elif sigma_chroma < 10.0:
        cat = "Moderate"
    else:
        cat = "Heavy"

    return float(round(sigma_chroma, 2)), cat


def estimate_optical_blur(img_gray: np.ndarray) -> Tuple[float, str]:
    """
    Calculates an optical/motion blur index using spectral energy roll-off
    and edge-profile acutance.

    Combines global Laplacian high-frequency variance with localized edge-transition
    gradient steepness to ensure sharp images with sparse textures (e.g. minimalist art,
    clean anime, clear sky compositions) are not falsely scored as blurred.

    Values range from 0.0 (pin-sharp edge transitions) to 1.0 (severe optical defocus).

    Args:
        img_gray: Single-channel uint8 or float32 image [H, W]

    Returns:
        Tuple of (blur_score, category_string)
    """
    if img_gray is None or img_gray.size == 0:
        return 0.0, "Sharp"

    gray = img_gray.astype(np.float32)
    h, w = gray.shape[:2]
    if h < 4 or w < 4:
        return 0.0, "Sharp"

    # 1. Global Laplace operator variance
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    lap_var = float(np.var(lap))
    lap_blur = float(np.clip(1.0 / (1.0 + (lap_var / 1200.0) ** 0.5), 0.0, 1.0))

    # 2. Localized edge acutance: evaluate sharpness along strong edge transitions
    # Use top 0.1% strongest Laplacian magnitudes (invariant to flat background fraction)
    abs_lap = np.abs(lap)
    total_pixels = gray.size
    k = max(16, int(total_pixels * 0.001))
    if k < total_pixels:
        top_k = np.partition(abs_lap.ravel(), -k)[-k:]
        top_lap = float(np.mean(top_k))
    else:
        top_lap = float(np.mean(abs_lap))

    # Edge acutance blur index:
    # Sharp edge transition has top_lap > 300 (edge_blur < 0.10)
    # Soft/blurred edge has top_lap < 30 (edge_blur > 0.65)
    edge_blur = float(np.clip(1.0 / (1.0 + (top_lap / 60.0) ** 1.4), 0.0, 1.0))

    # Composite blur score: if edges are razor sharp, sparse texture should not penalize it
    if top_lap > 15.0:
        blur_score = float(min(lap_blur, edge_blur))
    else:
        blur_score = lap_blur

    blur_score = float(round(blur_score, 3))

    if blur_score < 0.25:
        cat = "Sharp"
    elif blur_score < 0.45:
        cat = "Acceptable"
    elif blur_score < 0.70:
        cat = "Soft"
    else:
        cat = "Blurred"

    return blur_score, cat



def estimate_dynamic_range_entropy(img_gray: np.ndarray) -> Tuple[float, int, float, float, float]:
    """
    Computes Shannon luminance entropy, dynamic range spread, and clipping percentages.

    Args:
        img_gray: Single-channel uint8 image [H, W]

    Returns:
        Tuple of (entropy_bits, dynamic_range_levels, shadow_clip_pct, highlight_clip_pct, mean_lum)
    """
    gray = img_gray if img_gray.dtype == np.uint8 else np.clip(np.rint(img_gray), 0, 255).astype(np.uint8)
    total_pixels = float(gray.size)

    # Histogram calculation over 256 bins
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    prob = hist / (total_pixels + 1e-12)
    prob_nonzero = prob[prob > 0]
    entropy = -float(np.sum(prob_nonzero * np.log2(prob_nonzero)))

    # Percentiles for robust dynamic range
    p1 = float(np.percentile(gray, 1))
    p99 = float(np.percentile(gray, 99))
    dynamic_range = int(round(p99 - p1))

    # Clipping statistics
    shadow_clip = float(np.sum(gray < 5)) / total_pixels * 100.0
    highlight_clip = float(np.sum(gray > 250)) / total_pixels * 100.0
    mean_lum = float(np.mean(gray))

    return round(entropy, 2), dynamic_range, round(shadow_clip, 2), round(highlight_clip, 2), round(mean_lum, 1)


def estimate_shades_of_gray_illuminant(
    img_bgr: np.ndarray,
    p: int = 6,
    return_tint: bool = False,
) -> Any:
    """
    Estimates illuminant color cast using the Minkowski p-norm (Finlayson & Trezzi, 2004
    Shades-of-Gray hypothesis).

    Returns a recommended Bradford CAT16 temperature offset (clamped to
    -50..+50 here; auto-tune narrows to -30..+30 per the PipelineConfig
    range), optional tint offset (clamped to -40..+40 here, -30..+30 after
    auto-tune), and cast name.

    Args:
        img_bgr: uint8 BGR image [H, W, 3]
        p: Minkowski norm order (p=6 balances gray-world and max-RGB)
        return_tint: If True, returns (temperature_offset, tint_offset, cast_name);
                     otherwise returns (temperature_offset, cast_name) for backward compatibility.

    Returns:
        Tuple of (temperature_offset, cast_name) or (temperature_offset, tint_offset, cast_name)
    """
    if img_bgr is None or img_bgr.size == 0:
        if return_tint:
            return 0, 0, "Neutral"
        return 0, "Neutral"
    if img_bgr.ndim == 2:
        if return_tint:
            return 0, 0, "Neutral"
        return 0, "Neutral"

    h, w = img_bgr.shape[:2]
    max_dim = 256
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        small = cv2.resize(img_bgr, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = img_bgr

    img_f = small.astype(np.float32) / 255.0

    img_pow = np.power(img_f + 1e-5, p)
    mean_pow = np.mean(img_pow, axis=(0, 1))
    illuminant = np.power(mean_pow, 1.0 / p)

    norm = np.linalg.norm(illuminant) + 1e-6
    e_b, e_g, e_r = illuminant / norm

    rb_ratio = e_r / (e_b + 1e-5)
    gm_ratio = e_g / ((e_r + e_b) * 0.5 + 1e-5)

    temp_offset = 0
    tint_offset = 0

    if rb_ratio > 1.15:
        cast_name = "Warm"
        temp_offset = int(np.clip(-1 * (rb_ratio - 1.0) * 35, -50, -5))
    elif rb_ratio < 0.85:
        cast_name = "Cool"
        temp_offset = int(np.clip((1.0 - rb_ratio) * 35, 5, 50))
    elif gm_ratio > 1.15:
        cast_name = "Green"
        temp_offset = 0
    elif gm_ratio < 0.85:
        cast_name = "Magenta"
        temp_offset = 0
    else:
        cast_name = "Neutral"
        temp_offset = 0

    # Orthogonal green-magenta tint offset:
    # gm_ratio > 1.08 indicates green cast (corrected by magenta tint, delta > 0)
    # gm_ratio < 0.92 indicates magenta cast (corrected by green tint, delta < 0)
    if gm_ratio > 1.08:
        tint_offset = int(np.clip((gm_ratio - 1.0) * 45.0, 5, 40))
        if cast_name == "Neutral":
            cast_name = "Green"
    elif gm_ratio < 0.92:
        tint_offset = int(np.clip(-1 * (1.0 - gm_ratio) * 45.0, -40, -5))
        if cast_name == "Neutral":
            cast_name = "Magenta"

    if return_tint:
        return temp_offset, tint_offset, cast_name
    return temp_offset, cast_name


def estimate_spatial_lighting_geometry(img_gray: np.ndarray) -> Tuple[float, bool]:
    """
    Analyzes spatial luminance distribution between central subject area and outer periphery
    to detect backlit compositions (e.g. portraits against bright windows, sunsets, or sky).

    Args:
        img_gray: Single-channel uint8 or float32 image [H, W]

    Returns:
        Tuple of (backlight_ratio, is_backlit_flag)
    """
    if img_gray is None or img_gray.size == 0:
        return 1.0, False

    gray = img_gray if img_gray.dtype == np.uint8 else np.clip(np.rint(img_gray), 0, 255).astype(np.uint8)
    h, w = gray.shape[:2]
    if h < 16 or w < 16:
        return 1.0, False

    # Fast 160px proxy
    max_dim = 160
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        small = cv2.resize(gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = gray

    sh, sw = small.shape[:2]
    y1, y2 = int(sh * 0.25), int(sh * 0.75)
    x1, x2 = int(sw * 0.25), int(sw * 0.75)

    center = small[y1:y2, x1:x2].astype(np.float32)
    mean_center = float(np.mean(center))

    border_pixels = []
    if y1 > 0:
        border_pixels.append(small[:y1, :].ravel())
    if y2 < sh:
        border_pixels.append(small[y2:, :].ravel())
    if x1 > 0:
        border_pixels.append(small[y1:y2, :x1].ravel())
    if x2 < sw:
        border_pixels.append(small[y1:y2, x2:].ravel())

    if border_pixels:
        periph = np.concatenate(border_pixels).astype(np.float32)
        mean_periph = float(np.mean(periph))
    else:
        mean_periph = mean_center

    ratio = float((mean_periph + 1.0) / (mean_center + 1.0))
    ratio = float(round(np.clip(ratio, 0.1, 10.0), 2))

    is_backlit = bool(ratio >= 1.65 and mean_center < 95.0 and mean_periph > 130.0)
    return ratio, is_backlit


def estimate_jpeg_blocking(img_gray: np.ndarray) -> Tuple[float, bool]:
    """
    Estimates 8x8 discrete cosine transform (DCT) grid boundary discontinuity ratio (Wang et al.).
    Measures the ratio of step discontinuities across 8-pixel block boundaries relative to
    interior gradient steps to detect heavy JPEG compression artifacts.

    Args:
        img_gray: Single-channel uint8 image [H, W]

    Returns:
        Tuple of (blocking_ratio, is_blocked_flag)
    """
    if img_gray is None or img_gray.size == 0:
        return 1.0, False

    gray = img_gray.astype(np.float32)
    h, w = gray.shape[:2]

    if h < 32 or w < 32:
        return 1.0, False

    h8 = (h // 8) * 8
    w8 = (w // 8) * 8
    g = gray[:h8, :w8]

    # Horizontal boundary differences (|I[:, 8k] - I[:, 8k-1]|)
    boundary_cols = np.arange(8, w8, 8)
    diff_h_boundary = np.abs(g[:, boundary_cols] - g[:, boundary_cols - 1])

    # Horizontal interior differences (|I[:, 8k+4] - I[:, 8k+3]|)
    interior_cols = np.arange(4, w8 - 4, 8)
    diff_h_interior = np.abs(g[:, interior_cols] - g[:, interior_cols - 1])

    mean_b = float(np.mean(diff_h_boundary))
    mean_i = float(np.mean(diff_h_interior))

    blocking_score = float((mean_b + 1e-4) / (mean_i + 1e-4))
    blocking_score = float(round(np.clip(blocking_score, 0.5, 5.0), 3))

    is_blocked = bool(blocking_score > 1.25 and mean_b > 2.0)
    return blocking_score, is_blocked


def estimate_atmospheric_haze(img_bgr: np.ndarray) -> Tuple[float, bool]:
    """
    Estimates atmospheric haze veiling luminance via Dark Channel Prior analysis.

    Clear atmospheric conditions exhibit low median dark channel radiance (< 0.18),
    whereas atmospheric fog/haze/smoke elevates the dark channel significantly.

    Args:
        img_bgr: uint8 BGR image [H, W, 3]

    Returns:
        Tuple of (haze_index, is_hazy_flag)
    """
    h, w = img_bgr.shape[:2]
    max_dim = 320
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        small = cv2.resize(img_bgr, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = img_bgr

    img_f = small.astype(np.float32) / 255.0
    dark_channel = np.min(img_f, axis=2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dark_filtered = cv2.erode(dark_channel, kernel)

    # Atmospheric haze index: median of eroded dark channel
    haze_index = float(np.percentile(dark_filtered, 50))
    haze_index = float(round(np.clip(haze_index, 0.0, 1.0), 3))

    is_hazy = haze_index > 0.18

    return haze_index, is_hazy



def classify_content_style(img_bgr: np.ndarray) -> Tuple[str, float]:
    """
    Classifies image content into a processing style ("photo", "anime",
    "manga") from cheap proxy signals, deterministically.

    Signals (160px proxy): mean HSV saturation, quantized unique-color
    fraction (5 bits/channel), 95%- and 80%-coverage color concentration
    (fraction of distinct colors covering 95%/80% of pixels — robust to
    the JPEG noise that defeats raw distinct counts on real files),
    high-pass residual energy ratio (dot/line-scale structure vs broadband
    variance), extreme tonal occupancy (paper-white/ink-black fraction),
    and smooth-area fraction (5x5 local std < 5.0).

    Decision order (first match wins):
    - manga: near-zero saturation with few quantized colors, genuine
      structure, and extreme tonal occupancy > 0.50 (B&W ink line-art and
      halftones; the occupancy gate rejects continuous-tone B&W photos
      where midtones dominate), or dot-scale energy above 0.60
      (B&W energetic) or above 0.75 with few colors regardless of
      color (color halftones). Noisy photos fail the dot-energy gate
      (resid ~0.48).
    - anime: very few distinct colors with genuine ink-edge energy (a
      0.20 residual floor rejects clean/blurred gradients, whose
      smoothness would otherwise mimic flat color) and real color, or
      concentrated color usage (n80 < 0.20 or n95 < 0.22) with structure,
      color, and smooth-filled areas (painterly illustration, whose
      gradients defeat raw color counts).
    - photo: fast path for saturated, unconcentrated, textured content,
      else fallback for natural imagery and continuous-tone B&W photography.

    Returns:
        Tuple of (style, confidence in [0.0, 1.0]).
    """
    if img_bgr is None or img_bgr.size == 0:
        return "photo", 0.50
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    elif img_bgr.ndim == 3 and img_bgr.shape[2] == 1:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    elif img_bgr.ndim == 3 and img_bgr.shape[2] == 4:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
    h, w = img_bgr.shape[:2]
    if h < 4 or w < 4:
        return "photo", 0.50

    scale = min(1.0, 160.0 / max(h, w))
    if scale < 1.0:
        small = cv2.resize(img_bgr, (0, 0), fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
    else:
        small = img_bgr

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    sat = float(hsv[:, :, 1].astype(np.float32).mean() / 255.0)

    q = (small.astype(np.uint16) >> 5).reshape(-1, 3)
    keys = q[:, 0] * 1024 + q[:, 1] * 32 + q[:, 2]
    n_pixels = float(q.shape[0])
    _vals, counts = np.unique(keys, return_counts=True)
    if len(counts) == 0:
        return "photo", 0.50
    counts = np.sort(counts)[::-1]
    uniq = float(len(counts)) / n_pixels
    cdf = np.cumsum(counts) / float(counts.sum())
    n95 = float(np.searchsorted(cdf, 0.95) + 1) / float(len(counts))
    n80 = float(np.searchsorted(cdf, 0.80) + 1) / float(len(counts))

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    resid = float((gray - cv2.GaussianBlur(gray, (9, 9), 2.0)).std() / (gray.std() + 1e-6))
    extreme_tones = float(np.mean((gray > 215.0) | (gray < 45.0)))

    mean5 = cv2.blur(gray, (5, 5))
    mean_sq5 = cv2.blur(gray**2, (5, 5))
    local_std = np.sqrt(np.maximum(0.0, mean_sq5 - mean5**2))
    smooth_frac = float(np.mean(local_std < 5.0))

    if sat < 0.05 and uniq < 0.0035 and resid > 0.15 and extreme_tones > 0.50:
        m1 = float(np.clip((0.0035 - uniq) / 0.0035, 0.0, 1.0))
        m2 = float(np.clip((resid - 0.15) / 0.30, 0.0, 1.0))
        m3 = float(np.clip((extreme_tones - 0.50) / 0.30, 0.0, 1.0))
        return "manga", float(round(0.55 + 0.45 * min(m1, m2, m3), 3))

    if sat < 0.05 and resid > 0.60:
        m1 = float(np.clip((resid - 0.60) / 0.40, 0.0, 1.0))
        m2 = float(np.clip((0.05 - sat) / 0.05, 0.0, 1.0))
        return "manga", float(round(0.55 + 0.45 * min(m1, m2), 3))

    if resid > 0.75 and uniq < 0.001:
        m1 = float(np.clip((resid - 0.75) / 0.25, 0.0, 1.0))
        return "manga", float(round(0.55 + 0.45 * m1, 3))

    if uniq < 0.001 and 0.20 < resid < 0.70 and sat >= 0.05:
        m1 = float(np.clip((0.001 - uniq) / 0.001, 0.0, 1.0))
        m2 = float(np.clip((0.70 - resid) / 0.70, 0.0, 1.0))
        m3 = float(np.clip((resid - 0.20) / 0.20, 0.0, 1.0))
        return "anime", float(round(0.55 + 0.45 * min(m1, m2, m3), 3))

    if (n80 < 0.20 or n95 < 0.22) and resid > 0.14 and sat > 0.15 and smooth_frac > 0.18:
        m1 = float(np.clip((0.22 - min(n80, n95)) / 0.20, 0.0, 1.0))
        m2 = float(np.clip((resid - 0.14) / 0.25, 0.0, 1.0))
        m3 = float(np.clip((sat - 0.15) / 0.35, 0.0, 1.0))
        m4 = float(np.clip((smooth_frac - 0.18) / 0.25, 0.0, 1.0))
        return "anime", float(round(0.60 + 0.40 * min(m1, m2, m3, m4), 3))

    if sat > 0.15 and n80 >= 0.25 and smooth_frac < 0.20:
        return "photo", 0.70

    return "photo", 0.60


def diagnose_image(
    img_bgr: np.ndarray,
    semantic_breakdown: Optional[Dict[str, float]] = None,
) -> DiagnosticsResult:
    """
    Performs comprehensive autonomous diagnostic signal analysis on an input image.

    Args:
        img_bgr: uint8 BGR image [H, W, 3]
        semantic_breakdown: Optional semantic region breakdown dictionary

    Returns:
        DiagnosticsResult populated with all physical signal metrics and auto-tune parameters

    Raises:
        ValueError: If ``img_bgr`` is None, empty, or not a 2D/3-channel image.
    """
    if img_bgr is None or not isinstance(img_bgr, np.ndarray) or img_bgr.size == 0:
        raise ValueError("diagnose_image requires a non-empty image array.")
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    if img_bgr.ndim != 3 or img_bgr.shape[2] not in (3, 4):
        raise ValueError(f"diagnose_image requires BGR/BGRA input, got shape {img_bgr.shape}.")
    if img_bgr.shape[2] == 4:
        img_bgr = img_bgr[:, :, :3]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Wavelet MAD Noise Estimation (Luma + Chroma)
    noise_sigma, noise_cat = estimate_wavelet_noise_mad(gray)
    chroma_noise_sigma, _ = estimate_chroma_wavelet_noise_mad(img_bgr)

    # 2. Optical Blur Estimation
    blur_score, blur_cat = estimate_optical_blur(gray)

    # 3. Dynamic Range & Entropy
    entropy, dyn_range, shadow_clip, highlight_clip, mean_lum = estimate_dynamic_range_entropy(gray)

    # 3b. Spatial Illumination Geometry & Backlight Detection
    backlight_ratio, is_backlit = estimate_spatial_lighting_geometry(gray)

    # 4. Color Cast / Illuminant Shift (Temp + Tint)
    temp_offset, tint_offset, cast_name = estimate_shades_of_gray_illuminant(img_bgr, return_tint=True)

    # 5. Atmospheric Haze
    haze_index, is_hazy = estimate_atmospheric_haze(img_bgr)

    # 5b. Compression Blocking Detection
    blocking_score, is_blocked = estimate_jpeg_blocking(gray)

    # 6. Content Style Classification (photo / anime / manga)
    suggested_style, style_confidence = classify_content_style(img_bgr)

    result = DiagnosticsResult(
        noise_sigma=noise_sigma,
        noise_category=noise_cat,
        chroma_noise_sigma=chroma_noise_sigma,
        blur_score=blur_score,
        blur_category=blur_cat,
        entropy=entropy,
        dynamic_range=dyn_range,
        shadow_clipping=shadow_clip,
        highlight_clipping=highlight_clip,
        mean_luminance=mean_lum,
        backlight_ratio=backlight_ratio,
        backlight_detected=is_backlit,
        color_cast_kelvin=temp_offset,
        color_tint_offset=tint_offset,
        color_cast_name=cast_name,
        haze_index=haze_index,
        haze_detected=is_hazy,
        jpeg_blocking_score=blocking_score,
        jpeg_blocking_detected=is_blocked,
        suggested_style=suggested_style,
        style_confidence=style_confidence,
        semantic_breakdown=semantic_breakdown or {},
    )

    # 7. Autonomous Parameter Synthesis
    result.recommended_parameters = auto_tune_parameters(result)

    return result


def auto_tune_parameters(diag: DiagnosticsResult) -> Dict[str, Any]:
    """
    Synthesizes mathematically optimal pipeline parameters for the diagnosed image.

    Args:
        diag: DiagnosticsResult from diagnose_image()

    Returns:
        Dictionary of recommended parameter overrides for PipelineConfig
    """
    params: Dict[str, Any] = {}
    explanations: list[str] = []

    # 1. Denoising intensity based on Wavelet MAD noise (Luma + Chroma)
    effective_noise = max(diag.noise_sigma, getattr(diag, "chroma_noise_sigma", 0.0) * 0.7)
    if effective_noise < 2.0:
        params["enable_denoise"] = False
        params["denoise_intensity"] = 15
        explanations.append(f"Sensor noise is minimal (sigma={diag.noise_sigma:.1f}): bypassed SWF denoiser to preserve micro-textures.")
    elif effective_noise < 5.0:
        params["enable_denoise"] = True
        params["denoise_intensity"] = 35
        explanations.append(f"Low noise detected (sigma={effective_noise:.1f}): engaged mild SWF denoiser (35).")
    elif effective_noise < 10.0:
        params["enable_denoise"] = True
        params["denoise_intensity"] = 55
        explanations.append(f"Moderate noise detected (sigma={effective_noise:.1f}): applied balanced SWF denoiser (55).")
    else:
        params["enable_denoise"] = True
        params["denoise_intensity"] = int(np.clip(effective_noise * 5.5, 60, 95))
        explanations.append(f"Heavy noise detected (sigma={effective_noise:.1f}): engaged aggressive SWF denoiser ({params['denoise_intensity']}).")

    if getattr(diag, "chroma_noise_sigma", 0.0) >= 5.0 and diag.noise_sigma < 3.0:
        explanations.append(f"Elevated chrominance noise (sigma={diag.chroma_noise_sigma:.1f}): boosted color denoise filtering.")

    # 2. Deblurring & Sharpening tuning (tempered by noise and JPEG blocking)
    is_blocked = getattr(diag, "jpeg_blocking_detected", False)
    if diag.noise_sigma > 7.0 or is_blocked:
        base_sharpen = 0.8
        base_pyramid_detail = 0.9
        if is_blocked:
            explanations.append(f"JPEG compression blocking detected (score={getattr(diag, 'jpeg_blocking_score', 0.0):.2f}): tempered sharpening to avoid grid artifacts.")
    elif diag.noise_sigma < 3.0:
        base_sharpen = 1.3
        base_pyramid_detail = 1.25
    else:
        base_sharpen = 1.1
        base_pyramid_detail = 1.1

    if diag.blur_score > 0.45:
        params["deblur_strength"] = int(np.clip((diag.blur_score - 0.4) * 100, 20, 60))
        params["sharpen_strength"] = min(1.6, base_sharpen + 0.3)
        params["pyramid_micro_texture"] = min(1.5, base_pyramid_detail + 0.2)
        params["pyramid_structure_boost"] = 1.2
        explanations.append(f"Optical softness detected (score={diag.blur_score:.2f}): activated structure tensor deblur ({params['deblur_strength']}).")
    else:
        params["deblur_strength"] = 0
        params["sharpen_strength"] = base_sharpen
        params["pyramid_micro_texture"] = base_pyramid_detail
        params["pyramid_structure_boost"] = 1.05

    # 3. Dynamic Range & Contrast Fusion (BIMEF)
    if diag.dynamic_range < 170 or diag.entropy < 6.5:
        params["enable_contrast"] = True
        params["contrast_boost"] = 2.2
    elif diag.shadow_clipping > 5.0:
        params["enable_contrast"] = True
        params["contrast_boost"] = 2.0
    else:
        params["enable_contrast"] = True
        params["contrast_boost"] = 1.6

    # 4. Brightness offset for under-exposed scenes
    if diag.mean_luminance < 75.0:
        params["brightness_shift"] = int(np.clip((75.0 - diag.mean_luminance) * 0.25, 2, 12))
    elif diag.mean_luminance > 180.0:
        params["brightness_shift"] = -3
    else:
        params["brightness_shift"] = 0

    # 5. Atmospheric Dehazing (single 0.18 threshold lives in the estimator)
    if diag.haze_detected:
        params["enable_dehaze"] = True
        params["dehaze_strength"] = float(round(np.clip((diag.haze_index - 0.15) * 1.8, 0.3, 0.85), 2))
        explanations.append(f"Atmospheric haze detected (index={diag.haze_index:.2f}): enabled DCP dehaze ({params['dehaze_strength']:.2f}).")
    else:
        params["enable_dehaze"] = False
        params["dehaze_strength"] = 0.0

    # 6. Local Tone-Mapping, Backlight & Highlight Reconstruction
    is_backlit = getattr(diag, "backlight_detected", False)
    if is_backlit:
        params["enable_local_tone"] = True
        params["local_tone_strength"] = 0.70
        b_ratio = getattr(diag, "backlight_ratio", 2.0)
        params["shadow_boost"] = float(round(np.clip(0.50 + (b_ratio - 1.6) * 0.2, 0.60, 0.90), 2))
        params["highlight_recovery"] = float(round(np.clip(diag.highlight_clipping * 0.12, 0.35, 0.85), 2))
        explanations.append(f"Backlit subject detected (ratio={b_ratio:.1f}x): engaged Local Tone Mapping with Shadow Toe Boost ({params['shadow_boost']:.2f}).")
    elif diag.shadow_clipping > 3.0 or diag.highlight_clipping > 2.0:
        params["enable_local_tone"] = True
        params["local_tone_strength"] = float(round(np.clip(
            0.35 + (diag.shadow_clipping + diag.highlight_clipping) * 0.03, 0.30, 0.85
        ), 2))
        params["highlight_recovery"] = float(round(np.clip(
            diag.highlight_clipping * 0.12, 0.20, 0.90
        ), 2))
        params["shadow_boost"] = float(round(np.clip(
            diag.shadow_clipping * 0.10, 0.20, 0.85
        ), 2))
        explanations.append(f"Dynamic clipping detected (shadow={diag.shadow_clipping:.1f}%, highlight={diag.highlight_clipping:.1f}%): enabled tone recovery.")
    else:
        params["enable_local_tone"] = False

    # 7. Perceptual White Balance (Temp + Tint)
    params["color_temperature"] = int(np.clip(diag.color_cast_kelvin, -30, 30))
    params["color_tint"] = int(np.clip(getattr(diag, "color_tint_offset", 0), -30, 30))
    params["vibrance_boost"] = 1.10

    if params["color_temperature"] != 0 or params["color_tint"] != 0:
        explanations.append(f"Color cast detected ({diag.color_cast_name}): applied CAT16 adaptation (Temp {params['color_temperature']:+d}, Tint {params['color_tint']:+d}).")

    diag.explanations = explanations
    return params

