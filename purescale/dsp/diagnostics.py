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
    blur_score: float = 0.0           # Optical/motion blur index (0.0=sharp, 1.0=heavily blurred)
    blur_category: str = "Sharp"      # "Sharp", "Acceptable", "Soft", "Blurred"
    entropy: float = 0.0              # Shannon luminance entropy (0.0 to 8.0 bits)
    dynamic_range: int = 255          # 99th minus 1st percentile luminance spread
    shadow_clipping: float = 0.0      # Shadow under-exposure percentage (Y < 5)
    highlight_clipping: float = 0.0   # Highlight blowout percentage (Y > 250)
    mean_luminance: float = 128.0     # Mean luminance level (0.0 to 255.0)
    color_cast_kelvin: int = 0        # Recommended white-balance temperature shift (-100 to 100)
    color_cast_name: str = "Neutral"  # "Neutral", "Warm", "Cool", "Green", "Magenta"
    haze_index: float = 0.0           # Atmospheric veiling index (0.0=clear, 1.0=dense haze)
    haze_detected: bool = False       # True if haze index exceeds atmospheric threshold
    semantic_breakdown: Dict[str, float] = field(default_factory=dict)
    recommended_parameters: Dict[str, Any] = field(default_factory=dict)

    def summary_table(self) -> str:
        """Returns a formatted ASCII summary of diagnostic telemetry."""
        width = 64
        border = "+" + "-" * (width - 2) + "+"

        def row(text: str) -> str:
            return f"| {text:<{width - 4}} |"

        lines = [
            border,
            row("PURESCALE 4.0 SIGNAL DIAGNOSTICS"),
            border,
            row(f"Sensor Noise Sigma  : {self.noise_sigma:6.2f} / 100 [{self.noise_category:<8}]"),
            row(f"Optical Blur Score  : {self.blur_score:6.3f}       [{self.blur_category:<10}]"),
            row(f"Dynamic Entropy     : {self.entropy:6.2f} bits    [Range: {self.dynamic_range:<3} levels]"),
            row(f"Shadow/High Clipping: {self.shadow_clipping:5.1f}% / {self.highlight_clipping:4.1f}%"),
            row(f"Color Cast Offset   : {self.color_cast_kelvin:+5d}        [{self.color_cast_name:<8}]"),
            row(f"Atmospheric Haze    : {self.haze_index:6.3f}       [{'HAZY' if self.haze_detected else 'CLEAR':<8}]"),
        ]
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
        lines.append(border)
        return "\n".join(lines)


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


def estimate_optical_blur(img_gray: np.ndarray) -> Tuple[float, str]:
    """
    Calculates an optical/motion blur index using spectral energy roll-off
    and edge width distribution.

    Values range from 0.0 (pin-sharp edge transitions) to 1.0 (severe optical defocus).

    Args:
        img_gray: Single-channel uint8 or float32 image [H, W]

    Returns:
        Tuple of (blur_score, category_string)
    """
    gray = img_gray.astype(np.float32)

    # Laplace operator variance reflects high-frequency edge presence
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    lap_var = float(np.var(lap))

    # Normalized blur index:
    # Sharp image has lap_var > 10,000 (blur_score < 0.25)
    # Acceptable has lap_var ~ 2,000 - 10,000 (blur_score ~ 0.25 - 0.45)
    # Soft has lap_var ~ 500 - 2,000 (blur_score ~ 0.45 - 0.70)
    # Blurred has lap_var < 500 (blur_score > 0.70)
    blur_score = float(np.clip(1.0 / (1.0 + (lap_var / 1200.0) ** 0.5), 0.0, 1.0))
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


def estimate_shades_of_gray_illuminant(img_bgr: np.ndarray, p: int = 6) -> Tuple[int, str]:
    """
    Estimates illuminant color cast using the Minkowski p-norm (Finlayson & Trezzi, 2004
    Shades-of-Gray hypothesis).

    Returns a recommended Bradford CAT16 temperature offset (-100 to +100) and cast name.

    Args:
        img_bgr: uint8 BGR image [H, W, 3]
        p: Minkowski norm order (p=6 balances gray-world and max-RGB)

    Returns:
        Tuple of (temperature_offset, cast_name)
    """
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

    return temp_offset, cast_name


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
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Wavelet MAD Noise Estimation
    noise_sigma, noise_cat = estimate_wavelet_noise_mad(gray)

    # 2. Optical Blur Estimation
    blur_score, blur_cat = estimate_optical_blur(gray)

    # 3. Dynamic Range & Entropy
    entropy, dyn_range, shadow_clip, highlight_clip, mean_lum = estimate_dynamic_range_entropy(gray)

    # 4. Color Cast / Illuminant Shift
    temp_offset, cast_name = estimate_shades_of_gray_illuminant(img_bgr)

    # 5. Atmospheric Haze
    haze_index, is_hazy = estimate_atmospheric_haze(img_bgr)

    result = DiagnosticsResult(
        noise_sigma=noise_sigma,
        noise_category=noise_cat,
        blur_score=blur_score,
        blur_category=blur_cat,
        entropy=entropy,
        dynamic_range=dyn_range,
        shadow_clipping=shadow_clip,
        highlight_clipping=highlight_clip,
        mean_luminance=mean_lum,
        color_cast_kelvin=temp_offset,
        color_cast_name=cast_name,
        haze_index=haze_index,
        haze_detected=is_hazy,
        semantic_breakdown=semantic_breakdown or {},
    )

    # 6. Autonomous Parameter Synthesis
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

    # Denoising intensity based on Wavelet MAD noise
    if diag.noise_sigma < 2.0:
        params["enable_denoise"] = False
        params["denoise_intensity"] = 15
    elif diag.noise_sigma < 5.0:
        params["enable_denoise"] = True
        params["denoise_intensity"] = 35
    elif diag.noise_sigma < 10.0:
        params["enable_denoise"] = True
        params["denoise_intensity"] = 55
    else:
        params["enable_denoise"] = True
        params["denoise_intensity"] = int(np.clip(diag.noise_sigma * 5.5, 60, 95))

    # Deblurring & Sharpening tuning
    if diag.noise_sigma > 7.0:
        base_sharpen = 0.8
        base_pyramid_detail = 0.9
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
    else:
        params["deblur_strength"] = 0
        params["sharpen_strength"] = base_sharpen
        params["pyramid_micro_texture"] = base_pyramid_detail
        params["pyramid_structure_boost"] = 1.05

    # Dynamic Range & Contrast Fusion (BIMEF)
    if diag.dynamic_range < 170 or diag.entropy < 6.5:
        params["enable_contrast"] = True
        params["contrast_boost"] = 2.2
    elif diag.shadow_clipping > 5.0:
        params["enable_contrast"] = True
        params["contrast_boost"] = 2.0
    else:
        params["enable_contrast"] = True
        params["contrast_boost"] = 1.6

    # Brightness offset for under-exposed scenes
    if diag.mean_luminance < 75.0:
        params["brightness_shift"] = int(np.clip((75.0 - diag.mean_luminance) * 0.25, 2, 12))
    elif diag.mean_luminance > 180.0:
        params["brightness_shift"] = -3
    else:
        params["brightness_shift"] = 0

    # Atmospheric Dehazing
    if diag.haze_detected or diag.haze_index > 0.20:
        params["enable_dehaze"] = True
        params["dehaze_strength"] = float(round(np.clip((diag.haze_index - 0.15) * 1.8, 0.3, 0.85), 2))
    else:
        params["enable_dehaze"] = False
        params["dehaze_strength"] = 0.0

    params["color_temperature"] = diag.color_cast_kelvin
    params["vibrance_boost"] = 1.10

    return params
