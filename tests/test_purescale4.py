"""Automated verification test suite for PureScale 4.0 Advanced Computer Vision."""

import os
import sys
import unittest
import numpy as np
import cv2
import tkinter as tk

from purescale.config import (
    DeviceTarget,
    DiagnosticsResult,
    PipelineConfig,
    ProcessingMode,
)
from purescale.dsp.diagnostics import (
    auto_tune_parameters,
    diagnose_image,
    estimate_atmospheric_haze,
    estimate_optical_blur,
    estimate_wavelet_noise_mad,
)
from purescale.dsp.pyramid import (
    build_gaussian_pyramid,
    build_laplacian_pyramid,
    multiscale_laplacian_filter,
    reconstruct_laplacian_pyramid,
)
from purescale.dsp.semantic import (
    compute_spatial_guidance_maps,
    extract_semantic_masks,
    fast_guided_filter,
)
from purescale.dsp.dehaze import atmospheric_dehaze, compute_dark_channel
from purescale.dsp.restore import (
    directional_subpixel_depixelate,
    tensor_steered_shock_filter,
)
from purescale.pipeline import PureScalePipeline


class TestPureScale4Diagnostics(unittest.TestCase):
    """Test suite for autonomous signal quality diagnostics."""

    def test_wavelet_noise_mad(self):
        # Create pure synthetic image with known Gaussian noise sigma=15.0
        np.random.seed(42)
        base = np.full((128, 128), 128.0, dtype=np.float32)
        noise = np.random.normal(0, 15.0, (128, 128)).astype(np.float32)
        noisy = np.clip(base + noise, 0, 255).astype(np.uint8)

        sigma, cat = estimate_wavelet_noise_mad(noisy)
        # Should be within +/- 3.0 of true sigma 15.0
        self.assertAlmostEqual(sigma, 15.0, delta=3.0)
        self.assertEqual(cat, "Heavy")

        # Clean image
        clean = np.full((128, 128), 128, dtype=np.uint8)
        sigma_clean, cat_clean = estimate_wavelet_noise_mad(clean)
        self.assertLess(sigma_clean, 1.0)
        self.assertEqual(cat_clean, "Clean")

    def test_optical_blur_estimation(self):
        # Sharp synthetic checkerboard
        sharp = np.zeros((128, 128), dtype=np.uint8)
        sharp[::16, :] = 255
        sharp[:, ::16] = 255

        blur_sharp, cat_sharp = estimate_optical_blur(sharp)

        # Apply heavy Gaussian blur
        blurred = cv2.GaussianBlur(sharp, (21, 21), 6.0)
        blur_soft, cat_soft = estimate_optical_blur(blurred)

        self.assertGreater(blur_soft, blur_sharp)
        self.assertGreaterEqual(blur_soft, 0.45)

    def test_atmospheric_haze_estimation(self):
        # Clear synthetic landscape (dark shadows present)
        clear = np.full((128, 128, 3), 40, dtype=np.uint8)
        clear[:64, :] = [200, 150, 80]  # Ground
        clear[64:, :] = [20, 20, 20]    # Dark foreground

        haze_clear, is_hazy_clear = estimate_atmospheric_haze(clear)
        self.assertFalse(is_hazy_clear)

        # Synthetic haze: add veiling luminance
        hazy = np.clip(clear.astype(np.float32) * 0.4 + 160.0, 0, 255).astype(np.uint8)
        haze_score, is_hazy = estimate_atmospheric_haze(hazy)
        self.assertTrue(is_hazy)
        self.assertGreater(haze_score, 0.25)


class TestPureScale4Pyramids(unittest.TestCase):
    """Test suite for Multiscale Local Laplacian Pyramid Filtering."""

    def test_pyramid_reconstruction_invariance(self):
        # Unity gain should reconstruct exact original floating-point image
        np.random.seed(123)
        sample = np.random.uniform(0.1, 0.9, (128, 128)).astype(np.float32)

        gauss_pyr = build_gaussian_pyramid(sample, num_levels=4)
        lap_bands, base_res = build_laplacian_pyramid(gauss_pyr)
        reconstructed = reconstruct_laplacian_pyramid(lap_bands, base_res)

        max_err = float(np.max(np.abs(sample - reconstructed)))
        # Burt-Adelson pyramid reconstruction error should be floating point negligible
        self.assertLess(max_err, 1e-4)

    def test_multiscale_laplacian_filter_execution(self):
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        cv2.circle(img, (64, 64), 30, (200, 150, 100), -1)

        filtered = multiscale_laplacian_filter(
            img,
            micro_texture_gain=1.3,
            structure_boost=1.1,
            noise_damping=0.0,
        )
        self.assertEqual(filtered.shape, img.shape)
        self.assertEqual(filtered.dtype, np.uint8)

    def test_multiscale_laplacian_filter_bypass_alias(self):
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        out = multiscale_laplacian_filter(
            img,
            micro_texture_gain=1.0,
            structure_boost=1.0,
            noise_damping=0.0,
            dynamic_range_compression=0.0,
        )
        self.assertIsNot(img, out)
        out[0, 0, 0] = 255
        self.assertEqual(img[0, 0, 0], 128, "Bypassed return must not alias input buffer")


class TestPureScale4SemanticAndDehaze(unittest.TestCase):
    """Test suite for semantic region parsing and Dark Channel Prior dehaze."""

    def test_semantic_masks_bounds(self):
        img = np.zeros((160, 160, 3), dtype=np.uint8)
        img[:80, :] = [220, 180, 100]   # Sky-like blue (BGR)
        img[80:, :] = [30, 140, 40]     # Foliage-like green (BGR)

        masks = extract_semantic_masks(img)
        self.assertEqual(masks.sky.shape, (160, 160))
        self.assertTrue(np.all((masks.sky >= 0.0) & (masks.sky <= 1.0)))
        self.assertTrue(np.all((masks.foliage >= 0.0) & (masks.foliage <= 1.0)))

        det_map, den_map = compute_spatial_guidance_maps(masks)
        self.assertTrue(np.all((det_map >= 0.1) & (det_map <= 1.5)))

    def test_dcp_dehaze(self):
        img = np.full((128, 128, 3), 180, dtype=np.uint8)
        cv2.rectangle(img, (20, 20), (60, 60), (40, 40, 40), -1)

        dehazed, trans = atmospheric_dehaze(img, strength=0.7)
        self.assertEqual(dehazed.shape, img.shape)
        self.assertEqual(trans.shape, (128, 128))
        self.assertTrue(np.all(trans >= 0.09))


class TestPureScale4Pipeline(unittest.TestCase):
    """Test suite for end-to-end PureScale 4.0 Pipeline determinism."""

    def test_puredsp_determinism(self):
        # Create synthetic test pattern
        np.random.seed(99)
        test_img = np.random.randint(40, 220, (100, 100, 3), dtype=np.uint8)

        cfg = PipelineConfig(
            mode=ProcessingMode.PURE_DSP,
            scale=1.5,
            enable_diagnostics=True,
            enable_dehaze=True,
            dehaze_strength=0.3,
            enable_pyramid=True,
            pyramid_micro_texture=1.2,
            pyramid_structure_boost=1.1,
            enable_semantic_guidance=True,
        )

        pipeline = PureScalePipeline(config=cfg)
        res1 = pipeline.enhance(test_img, config=cfg)
        res2 = pipeline.enhance(test_img, config=cfg)

        # Output must be 100% bitwise identical
        diff = np.max(np.abs(res1.image.astype(np.int32) - res2.image.astype(np.int32)))
        self.assertEqual(diff, 0, "PureDSP must be 100% bitwise deterministic")
        self.assertIsNotNone(res1.diagnostics)

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            PipelineConfig(scale=0.0)
        with self.assertRaises(ValueError):
            PipelineConfig(scale=5.0)
        with self.assertRaises(ValueError):
            PipelineConfig(tile_size=16)
        with self.assertRaises(ValueError):
            PipelineConfig(tile_size=256, tile_overlap=-1)
        with self.assertRaises(ValueError):
            PipelineConfig(tile_size=256, tile_overlap=256)

    def test_pipeline_grayscale_and_rgba(self):
        pipeline = PureScalePipeline()
        cfg = PipelineConfig(scale=1.5)

        # 2D Grayscale input
        gray_in = np.full((60, 60), 120, dtype=np.uint8)
        res_gray = pipeline.enhance(gray_in, config=cfg)
        self.assertEqual(res_gray.image.ndim, 2)
        self.assertEqual(res_gray.image.shape, (90, 90))

        # 4-Channel BGRA input
        bgra_in = np.full((60, 60, 4), 180, dtype=np.uint8)
        bgra_in[:, :, 3] = 200
        res_bgra = pipeline.enhance(bgra_in, config=cfg)
        self.assertEqual(res_bgra.image.shape[:2], (90, 90))
        self.assertIsNotNone(res_bgra.alpha)
        self.assertEqual(res_bgra.alpha.shape, (90, 90))

    def test_oklab_srgb_roundtrip(self):
        from purescale.dsp.color import srgb_to_oklab, oklab_to_srgb
        test_pixels = np.array([
            [[0, 0, 0], [255, 255, 255], [128, 128, 128]],
            [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            [[10, 50, 200], [200, 150, 30], [80, 220, 140]],
        ], dtype=np.uint8)
        L, a, b = srgb_to_oklab(test_pixels)
        reconstructed = oklab_to_srgb(L, a, b)
        max_diff = np.max(np.abs(test_pixels.astype(np.int32) - reconstructed.astype(np.int32)))
        self.assertLessEqual(max_diff, 1, "Oklab roundtrip must be within 1 LSB rounding")

    def test_tiling_smooth_cosine_partition_of_unity(self):
        from purescale.neural.tiling import tile_process
        y, x = np.mgrid[0:256, 0:256]
        grad = ((x + y) * 255.0 / 510.0).astype(np.uint8)
        grad_bgr = np.repeat(grad[:, :, np.newaxis], 3, axis=2)

        def identity_fn(patch):
            return patch

        tiled = tile_process(grad_bgr, process_fn=identity_fn, scale=1, tile_size=96, overlap=32)
        diff = np.max(np.abs(grad_bgr.astype(np.int32) - tiled.astype(np.int32)))
        self.assertLessEqual(diff, 1, "Cosine feathering partition of unity must produce zero seam artifacts")


class TestPureScale4CliAndGui(unittest.TestCase):
    """Test suite for CLI arguments and Headless GUI controls."""

    def setUp(self):
        self.temp_in = os.path.abspath("test_temp_in.png")
        self.temp_out = os.path.abspath("test_temp_out.png")
        sample = np.zeros((80, 80, 3), dtype=np.uint8)
        sample[:40, :] = [200, 100, 50]
        sample[40:, :] = [50, 180, 70]
        cv2.imwrite(self.temp_in, sample)

    def tearDown(self):
        for f in (self.temp_in, self.temp_out):
            if os.path.exists(f):
                os.remove(f)

    def test_cli_diagnostics_flag(self):
        from purescale.cli import main
        code = main([self.temp_in, "--diagnostics"])
        self.assertEqual(code, 0)

    def test_cli_auto_flag(self):
        from purescale.cli import main
        code = main([self.temp_in, "-o", self.temp_out, "--auto", "--scale", "1.0"])
        self.assertEqual(code, 0)
        self.assertTrue(os.path.exists(self.temp_out))

    def test_headless_gui(self):
        from purescale.gui.app import PureScaleApp
        try:
            app = PureScaleApp()
            self.assertIn("PureScale 4.0", app.title())

            # Test preset synchronization
            app._on_preset_change("Landscape")
            self.assertTrue(app.dehaze_switch.get())
            self.assertAlmostEqual(app.dehaze_slider.get(), 0.6)

            # Test HUD update
            from purescale.dsp.diagnostics import diagnose_image
            diag = diagnose_image(cv2.imread(self.temp_in))
            app.hud_card.update_diagnostics(diag)
            app._apply_diagnostics_to_ui(diag)

            app.destroy()
        except (tk.TclError, RuntimeError, OSError) as e:
            # If no display available in CI environment, skip cleanly
            if "no display" in str(e).lower() or "display name" in str(e).lower():
                self.skipTest(f"Headless display not available: {e}")
            else:
                raise e


class TestPureScale4MemoryGuard(unittest.TestCase):
    """Tests 8K OOM guard against runaway spatial memory allocations."""

    def test_config_max_megapixels_validation(self):
        with self.assertRaises(ValueError):
            PipelineConfig(max_megapixels=0.0)
        with self.assertRaises(ValueError):
            PipelineConfig(max_megapixels=-10.0)

    def test_oom_guard_input_resolution(self):
        pipeline = PureScalePipeline()
        # 100x100 = 10,000 pixels. Guard set to 0.005 MP = 5,000 pixels.
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cfg = PipelineConfig(max_megapixels=0.005)
        with self.assertRaises(ValueError) as ctx:
            pipeline.enhance(img, config=cfg)
        self.assertIn("exceeds safety threshold", str(ctx.exception))

    def test_oom_guard_output_resolution(self):
        pipeline = PureScalePipeline()
        # 40x40 = 1600 pixels. At 4x scale -> 160x160 = 25,600 pixels.
        # Guard set to 0.010 MP = 10,000 pixels.
        img = np.zeros((40, 40, 3), dtype=np.uint8)
        cfg = PipelineConfig(scale=4.0, max_megapixels=0.010)
        with self.assertRaises(ValueError) as ctx:
            pipeline.enhance(img, config=cfg)
        self.assertIn("Output target resolution", str(ctx.exception))


class TestPureScale4NeuralIntegration(unittest.TestCase):
    """Integration tests executing real neural inference models."""

    def setUp(self):
        self.patch = np.zeros((32, 32, 3), dtype=np.uint8)
        self.patch[8:24, 8:24] = 200

    def test_neural_sr_cpu_forward_pass(self):
        from purescale.neural.engine import NeuralSuperResEngine
        from purescale.neural.models import get_default_model_path

        model_path = get_default_model_path(auto_download=False)
        if not model_path or not os.path.exists(model_path):
            self.skipTest("Neural super-resolution model weights not downloaded locally.")

        engine = NeuralSuperResEngine(model_path=model_path, target_device=DeviceTarget.CPU)
        out = engine.upscale(self.patch, target_scale=2.0, tile_size=32, tile_overlap=4)

        self.assertEqual(out.shape, (64, 64, 3))
        self.assertEqual(out.dtype, np.uint8)
        self.assertGreater(float(np.mean(out)), 0.0)

    def test_portrait_retoucher_inference(self):
        from purescale.neural.portrait import PortraitRetoucher
        from purescale.neural.models import ModelManager

        manager = ModelManager()
        model_path = manager.get_model_path("yunet-face", auto_download=False)
        if not model_path or not os.path.exists(model_path):
            self.skipTest("YuNet face detection model weights not downloaded locally.")

        retoucher = PortraitRetoucher(model_path=model_path)
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[16:48, 16:48] = [180, 150, 130]  # Synthetic skin tone patch
        enhanced, faces = retoucher.enhance(img, portrait_smooth=35, eye_clarity=1.3)

        self.assertEqual(enhanced.shape, (64, 64, 3))
        self.assertEqual(enhanced.dtype, np.uint8)
        self.assertIsInstance(faces, int)

    def test_hybrid_pipeline_execution(self):
        from purescale.neural.models import get_default_model_path

        model_path = get_default_model_path(auto_download=False)
        if not model_path or not os.path.exists(model_path):
            self.skipTest("Neural super-resolution model weights not downloaded locally.")

        pipeline = PureScalePipeline()
        cfg = PipelineConfig(
            mode=ProcessingMode.HYBRID,
            scale=2.0,
            tile_size=32,
            tile_overlap=4,
        )
        res = pipeline.enhance(self.patch, config=cfg)

        self.assertEqual(res.image.shape, (64, 64, 3))
        self.assertEqual(res.image.dtype, np.uint8)
        self.assertEqual(res.mode_name, "hybrid")
        self.assertGreater(res.latency_ms, 0.0)


class TestPureScale4CodeIntegrity(unittest.TestCase):
    """Verifies complete absence of non-ASCII emojis across all codebase files."""

    def test_zero_emojis(self):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        disallowed_emoji_count = 0
        offending_lines = []

        import re
        emoji_pattern = re.compile(r"[\U00010000-\U0010ffff]", flags=re.UNICODE)

        for root, dirs, files in os.walk(repo_root):
            if any(p in root for p in [".git", "__pycache__", ".ipynb_checkpoints", "models"]):
                continue
            for f in files:
                if f.endswith((".py", ".md", ".json", ".yml", ".yaml")):
                    fpath = os.path.join(root, f)
                    try:
                        with open(fpath, "r", encoding="utf-8") as fp:
                            for idx, line in enumerate(fp, 1):
                                matches = emoji_pattern.findall(line)
                                if matches:
                                    disallowed_emoji_count += len(matches)
                                    offending_lines.append(f"{f}:{idx}: {matches}")
                    except (UnicodeDecodeError, OSError, IOError):
                        pass

        msg = f"Found {disallowed_emoji_count} non-ASCII emojis: " + "; ".join(offending_lines[:5])
        self.assertEqual(disallowed_emoji_count, 0, msg)


if __name__ == "__main__":
    unittest.main()
