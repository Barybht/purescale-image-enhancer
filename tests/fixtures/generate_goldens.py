"""Generates golden reference outputs for tests/fixtures/.

Run from repo root:  python tests/fixtures/generate_goldens.py
Regenerate ONLY when an intentional output-changing optimization lands,
and document the measured quality delta in the commit message.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from purescale.config import PipelineConfig, ProcessingMode
from purescale.pipeline import PureScalePipeline

OUT_DIR = os.path.join(os.path.dirname(__file__))


def make_inputs():
    rng = np.random.default_rng(20260925)
    # 1. Photo-like: gradient + shapes + grain.
    x = np.linspace(0, 1, 96, dtype=np.float32)[None, :].repeat(96, axis=0)
    y = np.linspace(0, 1, 96, dtype=np.float32)[:, None].repeat(96, axis=1)
    photo = np.stack([x * 180 + 30, y * 150 + 40, (1 - x) * y * 160 + 30], axis=-1)
    photo += rng.normal(0, 6.0, photo.shape)
    cv2.circle(photo, (32, 32), 12, (200, 150, 100), -1)
    # 2. Step edge + noise (edge preservation probe).
    edge = np.full((64, 64, 3), 128.0)
    edge[:, 32:] = 200.0
    edge += rng.normal(0, 12.0, edge.shape)
    # 3. High-frequency texture (checker + noise).
    yy, xx = np.mgrid[0:80, 0:80]
    tex = np.where(((xx // 8) + (yy // 8)) % 2 == 0, 90.0, 170.0)
    tex = np.repeat(tex[:, :, None], 3, axis=2)
    tex += rng.normal(0, 8.0, tex.shape)
    return {
        "photo": np.clip(photo, 0, 255).astype(np.uint8),
        "edge": np.clip(edge, 0, 255).astype(np.uint8),
        "texture": np.clip(tex, 0, 255).astype(np.uint8),
    }


def golden_config() -> PipelineConfig:
    # Hermetic: PureDSP, no model-dependent portrait stage.
    return PipelineConfig(
        mode=ProcessingMode.PURE_DSP,
        scale=1.0,
        enable_diagnostics=True,
        enable_semantic_guidance=True,
        enable_pyramid=True,
        enable_denoise=True,
        denoise_intensity=40,
        enable_contrast=True,
        contrast_boost=1.8,
        sharpen_strength=1.1,
        vibrance_boost=1.10,
        portrait_smooth=0,
        eye_clarity=1.0,
    )


def main() -> None:
    pipeline = PureScalePipeline()
    cfg = golden_config()
    for name, img in make_inputs().items():
        res = pipeline.enhance(img, config=cfg)
        path = os.path.join(OUT_DIR, f"golden_{name}.png")
        cv2.imwrite(path, res.image)
        print(f"Wrote {path} {res.image.shape}")


if __name__ == "__main__":
    main()
