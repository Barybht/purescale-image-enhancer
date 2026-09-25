"""Multi-cue analytical semantic region parsing and spatial weighting for PureScale 4.0.

Segments images into soft continuous probability masks for Sky, Foliage,
Skin/Portrait, Deep Shadow, and Structural Geometry using chromatic opponent
signatures, gradient coherence, and edge-preserving Guided Filter boundary refinement.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import cv2
import numpy as np


@dataclass
class SemanticMasks:
    """Continuous soft probability masks in range [0.0, 1.0]."""
    sky: np.ndarray          # Soft mask for sky / open atmosphere
    foliage: np.ndarray      # Soft mask for vegetation, trees, grass
    skin: np.ndarray         # Soft mask for human skin tones
    shadow: np.ndarray       # Soft mask for underexposed deep shadows
    structure: np.ndarray    # Soft mask for high-coherence architectural edges
    breakdown: Dict[str, float]  # Percentage coverage per semantic class


from purescale.dsp.filters import fast_guided_filter


def extract_semantic_masks(
    img_bgr: np.ndarray,
    face_boxes: Optional[np.ndarray] = None,
) -> SemanticMasks:
    """
    Extracts multi-cue continuous soft semantic probability masks.

    Args:
        img_bgr: uint8 BGR image [H, W, 3]
        face_boxes: Optional Nx4 or Nx14 array of detected face bounding boxes

    Returns:
        SemanticMasks container with soft masks [H, W] float32 in [0.0, 1.0]
    """
    h, w = img_bgr.shape[:2]
    total_pixels = float(h * w)

    # Downsample guide for high-performance segmentation pass
    proc_w = min(640, w)
    proc_h = int(round(h * (proc_w / float(w))))
    small = cv2.resize(img_bgr, (proc_w, proc_h), interpolation=cv2.INTER_AREA)

    # Color spaces
    b, g, r = cv2.split(small.astype(np.float32) / 255.0)
    ycrcb = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
    y_lum = ycrcb[:, :, 0].astype(np.float32) / 255.0
    cr = ycrcb[:, :, 1].astype(np.float32)
    cb = ycrcb[:, :, 2].astype(np.float32)

    # Gradient magnitude of luminance
    gx = cv2.Sobel(y_lum, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y_lum, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    smooth_ness = np.exp(-grad_mag * 8.0)  # High where texture is flat/smooth

    # 1. SKY MASK
    # Vertical prior: higher in the upper half of the frame
    y_coords = np.linspace(1.0, 0.0, proc_h, dtype=np.float32)[:, np.newaxis]
    vertical_prior = np.power(y_coords, 0.65)

    # Blue/Cyan chromatic dominance
    blue_dom = np.clip((b - np.maximum(r, g * 0.95)) * 3.5, 0.0, 1.0)
    # Brightness constraint: sky is rarely dark
    bright_prior = np.clip((y_lum - 0.35) * 3.0, 0.0, 1.0)

    raw_sky = blue_dom * smooth_ness * vertical_prior * bright_prior
    sky_mask = cv2.GaussianBlur(raw_sky, (11, 11), 3.0)
    sky_mask = np.clip(sky_mask * 1.5, 0.0, 1.0)

    # 2. FOLIAGE / VEGETATION MASK
    # Chlorophyll chromatic signature: Green channel exceeds Red and Blue
    green_dom = np.clip((g - np.maximum(r * 0.9, b)) * 4.0, 0.0, 1.0)
    # Texture energy: foliage is rich in fine micro-textures (not smooth like sky)
    texture_energy = np.clip(grad_mag * 4.0, 0.0, 1.0)
    raw_foliage = green_dom * (0.4 + 0.6 * texture_energy) * (1.0 - sky_mask)
    foliage_mask = cv2.GaussianBlur(raw_foliage, (9, 9), 2.5)
    foliage_mask = np.clip(foliage_mask * 1.4, 0.0, 1.0)

    # 3. SKIN / PORTRAIT MASK
    # Inclusive YCrCb skin chromatic locus across Fitzpatrick skin phototypes I-VI
    skin_cr_dist = np.abs(cr - 152.0) / 32.0
    skin_cb_dist = np.abs(cb - 105.0) / 30.0
    skin_chroma = np.exp(-(skin_cr_dist ** 2 + skin_cb_dist ** 2))
    # Skin luminance prior: include dark skin tones down to Y=15/255 (y_lum >= 0.06)
    skin_lum = np.clip((y_lum - 0.06) * 5.0, 0.0, 1.0) * np.clip((0.98 - y_lum) * 5.0, 0.0, 1.0)
    raw_skin = skin_chroma * skin_lum * (1.0 - sky_mask) * (1.0 - foliage_mask)
    skin_mask = cv2.GaussianBlur(raw_skin, (7, 7), 2.0)
    skin_mask = np.clip(skin_mask * 1.3, 0.0, 1.0)

    # 4. DEEP SHADOW MASK
    # Areas with very low luminance where noise tends to hide
    shadow_mask = np.clip((0.18 - y_lum) * 6.0, 0.0, 1.0)

    # 5. STRUCTURE / ARCHITECTURE MASK
    # High edge coherence and geometric contours
    structure_mask = np.clip(grad_mag * 3.5, 0.0, 1.0) * (1.0 - sky_mask)

    # Upsample masks back to full resolution
    sky_full = cv2.resize(sky_mask, (w, h), interpolation=cv2.INTER_LINEAR)
    foliage_full = cv2.resize(foliage_mask, (w, h), interpolation=cv2.INTER_LINEAR)
    skin_full = cv2.resize(skin_mask, (w, h), interpolation=cv2.INTER_LINEAR)
    shadow_full = cv2.resize(shadow_mask, (w, h), interpolation=cv2.INTER_LINEAR)
    structure_full = cv2.resize(structure_mask, (w, h), interpolation=cv2.INTER_LINEAR)

    # Semantic coverage breakdown
    breakdown = {
        "sky": float(round(np.sum(sky_full > 0.3) / total_pixels, 3)),
        "foliage": float(round(np.sum(foliage_full > 0.3) / total_pixels, 3)),
        "skin": float(round(np.sum(skin_full > 0.3) / total_pixels, 3)),
        "shadow": float(round(np.sum(shadow_full > 0.3) / total_pixels, 3)),
        "structure": float(round(np.sum(structure_full > 0.3) / total_pixels, 3)),
    }

    return SemanticMasks(
        sky=sky_full,
        foliage=foliage_full,
        skin=skin_full,
        shadow=shadow_full,
        structure=structure_full,
        breakdown=breakdown,
    )


def compute_spatial_guidance_maps(
    masks: SemanticMasks,
    guide: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes spatially-varying gain maps for detail enhancement and denoising.

    Args:
        masks: Continuous SemanticMasks container.
        guide: Optional single-channel grayscale guidance image [H, W] float32 or uint8.
               When provided, refines both guidance maps with Fast Guided Filter boundary preservation.

    Returns:
        Tuple of (detail_weight_map, denoise_weight_map)
        - detail_weight_map: Attenuates sharpening in sky/shadow/skin; boosts in foliage/structure
        - denoise_weight_map: Increases denoising in sky/shadow/skin; preserves micro-textures in foliage/structure
    """
    # Detail Map: Base is 1.0.
    # Suppress in sky (-0.85) and shadow (-0.60); protect skin pores (-0.40);
    # boost in foliage (+0.25) and sharp architectural structures (+0.30).
    detail_map = (
        1.0
        - 0.85 * masks.sky
        - 0.60 * masks.shadow
        - 0.40 * masks.skin
        + 0.25 * masks.foliage
        + 0.30 * masks.structure
    )
    detail_map = np.clip(detail_map, 0.15, 1.45)

    # Denoise Map: Base is 1.0.
    # Boost in sky (+0.60), shadow (+0.70), and skin (+0.35);
    # protect high-frequency texture in foliage (-0.30) and structural contours (-0.25).
    denoise_map = (
        1.0
        + 0.60 * masks.sky
        + 0.70 * masks.shadow
        + 0.35 * masks.skin
        - 0.30 * masks.foliage
        - 0.25 * masks.structure
    )
    denoise_map = np.clip(denoise_map, 0.40, 1.85)

    if guide is not None:
        detail_map = fast_guided_filter(guide, detail_map, radius=6, eps=1e-3, subsample=2)
        denoise_map = fast_guided_filter(guide, denoise_map, radius=6, eps=1e-3, subsample=2)

    return detail_map, denoise_map
