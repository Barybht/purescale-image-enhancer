"""Automated verification test suite for PureScale 4.0 Advanced Computer Vision."""

import math
import os
import sys
import unittest
import numpy as np
import cv2

try:
    import tkinter as tk
    _TK_AVAILABLE = True
except ImportError:
    tk = None  # type: ignore[assignment]
    _TK_AVAILABLE = False


def _is_display_error(err: BaseException) -> bool:
    """True for missing-display failures (skipped) as opposed to real bugs."""
    msg = str(err).lower()
    return "no display" in msg or "display name" in msg or "$display" in msg

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
    estimate_chroma_wavelet_noise_mad,
    estimate_jpeg_blocking,
    estimate_optical_blur,
    estimate_shades_of_gray_illuminant,
    estimate_spatial_lighting_geometry,
    estimate_wavelet_noise_mad,
)
from purescale.dsp.color import bradford_cat16_white_balance

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
from purescale.dsp.tone import local_tone_mapping
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

    def test_chroma_wavelet_noise_mad(self):
        # Pure clean canvas
        clean = np.full((128, 128, 3), 128, dtype=np.uint8)
        sigma_clean, cat_clean = estimate_chroma_wavelet_noise_mad(clean)
        self.assertEqual(sigma_clean, 0.0)
        self.assertEqual(cat_clean, "Clean")

        # Inject pure chroma noise on Cr/Cb without changing luma
        ycrcb = cv2.cvtColor(clean, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        np.random.seed(42)
        ycrcb[:, :, 1] += np.random.normal(0, 10.0, (128, 128))
        ycrcb[:, :, 2] += np.random.normal(0, 10.0, (128, 128))
        noisy_chroma = cv2.cvtColor(np.clip(ycrcb, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR)

        sigma_c, cat_c = estimate_chroma_wavelet_noise_mad(noisy_chroma)
        self.assertAlmostEqual(sigma_c, 10.0, delta=2.5)
        self.assertIn(cat_c, ("Moderate", "Heavy"))

    def test_optical_blur_sparse_edges(self):
        # A sharp minimalist square on a large background should NOT score as blurred
        canvas = np.full((400, 400), 200, dtype=np.uint8)
        canvas[150:250, 150:250] = 50
        blur_score, cat = estimate_optical_blur(canvas)
        self.assertLess(blur_score, 0.30)
        self.assertEqual(cat, "Sharp")

        # Blurring the square should increase blur score to soft/blurred
        blurred_canvas = cv2.GaussianBlur(canvas, (21, 21), 6.0)
        blur_soft, cat_soft = estimate_optical_blur(blurred_canvas)
        self.assertGreater(blur_soft, blur_score)
        self.assertGreaterEqual(blur_soft, 0.50)

    def test_spatial_lighting_geometry_backlight(self):
        # Balanced lighting
        balanced = np.full((160, 160), 128, dtype=np.uint8)
        ratio_bal, is_backlit_bal = estimate_spatial_lighting_geometry(balanced)
        self.assertFalse(is_backlit_bal)
        self.assertAlmostEqual(ratio_bal, 1.0, delta=0.2)

        # Backlit subject: dark center silhouette with blown highlights on periphery
        backlit = np.full((160, 160), 220, dtype=np.uint8)
        backlit[40:120, 40:120] = 35
        ratio_bl, is_backlit_bl = estimate_spatial_lighting_geometry(backlit)
        self.assertTrue(is_backlit_bl)
        self.assertGreater(ratio_bl, 2.0)

    def test_jpeg_blocking_detection(self):
        # Smooth uncompressed gradient
        x = np.linspace(40, 220, 256, dtype=np.uint8)
        smooth = np.tile(x, (256, 1))
        score_smooth, is_blocked_smooth = estimate_jpeg_blocking(smooth)
        self.assertFalse(is_blocked_smooth)

        # Heavy JPEG compression
        _, buf = cv2.imencode(".jpg", smooth, [cv2.IMWRITE_JPEG_QUALITY, 20])
        jpg = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        score_jpg, is_blocked_jpg = estimate_jpeg_blocking(jpg)
        self.assertTrue(is_blocked_jpg)
        self.assertGreater(score_jpg, 1.25)

    def test_shades_of_gray_with_tint(self):
        base = np.full((100, 100, 3), 128, dtype=np.uint8)
        # Green tint (elevated G)
        green = base.copy()
        green[:, :, 1] = 165
        t_off, tint_off, cast = estimate_shades_of_gray_illuminant(green, return_tint=True)
        self.assertEqual(cast, "Green")
        self.assertGreater(tint_off, 0)

        # Magenta tint (depressed G)
        magenta = base.copy()
        magenta[:, :, 1] = 95
        t_off, tint_off, cast = estimate_shades_of_gray_illuminant(magenta, return_tint=True)
        self.assertEqual(cast, "Magenta")
        self.assertLess(tint_off, 0)

    def test_bradford_cat16_with_tint(self):
        img = np.full((32, 32, 3), 128, dtype=np.uint8)
        # Identity
        identity = bradford_cat16_white_balance(img, temperature_offset=0, tint_offset=0)
        self.assertTrue(np.array_equal(img, identity))

        # Magenta correction shifts green channel down and red/blue up
        mag_adapted = bradford_cat16_white_balance(img, temperature_offset=0, tint_offset=20)
        self.assertGreater(int(mag_adapted[16, 16, 2]), 128) # Red boosted
        self.assertLess(int(mag_adapted[16, 16, 1]), 128)    # Green attenuated



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

    def test_dcp_dehaze_proxy_consistency(self):
        # Proxy estimation must stay near-identical to the full-res path
        # while actually removing haze (contrast gain), deterministically.
        from purescale.quality import compare_images

        rng = np.random.default_rng(21)
        h, w = 320, 480
        x = np.linspace(0, 1, w, dtype=np.float32)[None, :].repeat(h, axis=0)
        base = np.stack([x * 180 + 30, x * 150 + 40, x * 160 + 30], axis=-1)
        hazy = np.clip(base * 0.55 + 90 + rng.normal(0, 4.0, base.shape), 0, 255).astype(np.uint8)

        full, _ = atmospheric_dehaze(hazy, strength=0.65, proxy_max_dim=0)
        proxy, _ = atmospheric_dehaze(hazy, strength=0.65, proxy_max_dim=160)
        metrics = compare_images(full, proxy)
        self.assertGreater(metrics["ssim"], 0.98)
        self.assertGreater(metrics["psnr_db"], 35.0)

        def contrast(im):
            g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
            return float(cv2.Laplacian(g, cv2.CV_32F, ksize=3).std())

        self.assertGreater(contrast(proxy), contrast(hazy))

        again, _ = atmospheric_dehaze(hazy, strength=0.65, proxy_max_dim=160)
        self.assertTrue(np.array_equal(proxy, again))


class TestPureScale4SideWindowFilter(unittest.TestCase):
    """Tests luminance-selection SWF: determinism, denoising, edge preservation."""

    def test_swf_determinism_and_denoising(self):
        from purescale.dsp.filters import side_window_filter

        rng = np.random.default_rng(11)
        img = np.full((200, 300, 3), 128.0)
        img[:, 150:] = 200.0
        img += rng.normal(0, 12.0, img.shape)
        img = np.clip(img, 0, 255).astype(np.uint8)

        out1 = side_window_filter(img, radius=2, iterations=1)
        out2 = side_window_filter(img, radius=2, iterations=1)
        self.assertTrue(np.array_equal(out1, out2))
        self.assertEqual(out1.shape, img.shape)
        self.assertEqual(out1.dtype, np.uint8)

        flat_in = img[20:180, 20:130].astype(np.float32).std()
        flat_out = out1[20:180, 20:130].astype(np.float32).std()
        self.assertLess(flat_out, flat_in * 0.5)

        edge = out1[20:180, 140:160].mean(axis=(0, 2))
        self.assertGreater(float(edge[-1] - edge[0]), 60.0)

    def test_swf_clean_skip(self):
        from purescale.dsp.filters import side_window_filter

        clean = np.full((64, 64, 3), 128, dtype=np.uint8)
        out = side_window_filter(clean, radius=2, iterations=1, noise_sigma=0.5)
        self.assertTrue(np.array_equal(out, clean))

    def test_swf_proxy_consistency(self):
        # Proxy path must stay near-identical to the full-res path while
        # still removing noise and preserving step edges, deterministically.
        # SSIM bar is lower than dehaze's 0.98: two valid denoises differ in
        # residual grain texture (proxy smooths more), not structure.
        from purescale.dsp.filters import side_window_filter
        from purescale.quality import compare_images

        rng = np.random.default_rng(11)
        img = np.full((360, 640, 3), 128.0)
        img[:, 320:] = 200.0
        img += rng.normal(0, 12.0, img.shape)
        img = np.clip(img, 0, 255).astype(np.uint8)

        full = side_window_filter(img, radius=2, iterations=1, proxy_max_dim=0)
        proxy = side_window_filter(img, radius=2, iterations=1, proxy_max_dim=480)
        metrics = compare_images(full, proxy)
        self.assertGreater(metrics["psnr_db"], 35.0)
        self.assertGreater(metrics["ssim"], 0.90)

        flat_in = img[40:300, 40:260].astype(np.float32).std()
        flat_out = proxy[40:300, 40:260].astype(np.float32).std()
        self.assertLess(flat_out, flat_in * 0.5)

        edge = proxy[40:300, 300:340].mean(axis=(0, 2))
        self.assertGreater(float(edge[-1] - edge[0]), 60.0)

        again = side_window_filter(img, radius=2, iterations=1, proxy_max_dim=480)
        self.assertTrue(np.array_equal(proxy, again))


class TestPureScale4ToneMapping(unittest.TestCase):
    """Test suite for Local Tone-Mapping and Highlight Reconstruction."""

    def test_local_tone_determinism(self):
        rng = np.random.default_rng(42)
        img = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
        out1 = local_tone_mapping(img, strength=0.6, highlight_recovery=0.5, shadow_boost=0.5)
        out2 = local_tone_mapping(img, strength=0.6, highlight_recovery=0.5, shadow_boost=0.5)
        self.assertTrue(np.array_equal(out1, out2))
        self.assertEqual(out1.shape, img.shape)
        self.assertEqual(out1.dtype, np.uint8)

    def test_local_tone_proxy_consistency(self):
        from purescale.quality import compare_images
        from bench.bench import make_fixture

        img = make_fixture(640, 480)
        full = local_tone_mapping(img, strength=0.5, highlight_recovery=0.5, shadow_boost=0.5, proxy_max_dim=0)
        proxy = local_tone_mapping(img, strength=0.5, highlight_recovery=0.5, shadow_boost=0.5, proxy_max_dim=320)
        metrics = compare_images(full, proxy)
        self.assertGreater(metrics["psnr_db"], 35.0)
        self.assertGreater(metrics["ssim"], 0.98)

        again = local_tone_mapping(img, strength=0.5, highlight_recovery=0.5, shadow_boost=0.5, proxy_max_dim=320)
        self.assertTrue(np.array_equal(proxy, again))

    def test_local_tone_shadow_and_highlight_efficacy(self):
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        img[:40, :40] = 15
        img[60:, 60:] = 250

        out = local_tone_mapping(img, strength=0.7, highlight_recovery=0.8, shadow_boost=0.8)
        shadow_in = float(img[:40, :40].mean())
        shadow_out = float(out[:40, :40].mean())
        self.assertGreater(shadow_out, shadow_in)

        hl_in = float(img[60:, 60:].mean())
        hl_out = float(out[60:, 60:].mean())
        self.assertLess(hl_out, hl_in)

    def test_local_tone_bypass(self):
        img = np.full((64, 64, 3), 120, dtype=np.uint8)
        out = local_tone_mapping(img, strength=0.0, highlight_recovery=0.0, shadow_boost=0.0)
        self.assertTrue(np.array_equal(img, out))


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
        # Documented numeric ranges (README CLI Parameters Reference).
        for kwargs in (
            {"dehaze_strength": 1.5}, {"dehaze_strength": -0.1},
            {"pyramid_micro_texture": 0.4}, {"pyramid_micro_texture": 2.1},
            {"pyramid_structure_boost": 0.7}, {"pyramid_structure_boost": 1.9},
            {"sharpen_strength": -0.1}, {"sharpen_strength": 3.1},
            {"denoise_intensity": 9}, {"denoise_intensity": 101},
            {"contrast_boost": 0.9}, {"contrast_boost": 4.1},
            {"brightness_shift": 51}, {"brightness_shift": -51},
            {"vibrance_boost": 0.9}, {"vibrance_boost": 1.6},
            {"color_temperature": 31}, {"color_temperature": -31},
            {"depixel_strength": -1}, {"depixel_strength": 101},
            {"deblur_strength": 101}, {"portrait_smooth": 101},
            {"eye_clarity": 0.9}, {"eye_clarity": 2.1},
            {"local_tone_strength": -0.1}, {"local_tone_strength": 1.1},
            {"highlight_recovery": -0.1}, {"highlight_recovery": 1.1},
            {"shadow_boost": -0.1}, {"shadow_boost": 1.1},
            {"output_format": "TIFF"},
            {"enable_fast_2x": "invalid"},
            {"enable_local_tone": "invalid"},
            {"sr_style": "unknown"},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    PipelineConfig(**kwargs)

    def test_auto_tune_output_validates(self):
        # Auto-tuner recommendations must always satisfy PipelineConfig ranges
        # (diagnostics illuminant can report beyond the CAT16 +-30 UI range).
        from purescale.dsp.diagnostics import DiagnosticsResult, auto_tune_parameters

        diag = DiagnosticsResult(noise_sigma=12.0, blur_score=0.6, entropy=5.0,
                                 dynamic_range=120, mean_luminance=60.0,
                                 color_cast_kelvin=50, haze_index=0.5,
                                 haze_detected=True)
        params = auto_tune_parameters(diag)
        cfg = PipelineConfig(**{k: v for k, v in params.items() if hasattr(PipelineConfig(), k)})
        cfg.validate()

        # Clipping-driven local tone mapping activation
        diag_clipped = DiagnosticsResult(
            noise_sigma=1.0, blur_score=0.2, entropy=7.0,
            dynamic_range=200, mean_luminance=120.0,
            color_cast_kelvin=0, haze_index=0.05,
            haze_detected=False, shadow_clipping=6.5, highlight_clipping=4.2,
        )
        params_clipped = auto_tune_parameters(diag_clipped)
        self.assertTrue(params_clipped.get("enable_local_tone"))
        self.assertGreater(params_clipped.get("highlight_recovery"), 0.3)
        self.assertGreater(params_clipped.get("shadow_boost"), 0.3)
        cfg_clipped = PipelineConfig(**{k: v for k, v in params_clipped.items() if hasattr(PipelineConfig(), k)})
        cfg_clipped.validate()

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

    def test_manga_preset_halftone_preservation(self):
        from purescale.config import get_preset_config

        h, w = 120, 120
        ht = np.full((h, w), 250, dtype=np.uint8)
        dot_pitch = 4
        dot_radius = 1
        for y in range(4, h // 2, dot_pitch):
            for x in range(4, w - 4, dot_pitch):
                cv2.circle(ht, (x, y), dot_radius, 40, -1)
        cv2.line(ht, (0, h // 2), (w, h // 2), 10, thickness=2)
        ht[h // 2 + 10:h - 10, w // 2:w - 10] = 10

        rng = np.random.default_rng(42)
        ht_noisy = np.clip(ht.astype(np.float32) + rng.normal(0, 3.5, ht.shape), 0, 255).astype(np.uint8)

        pipeline = PureScalePipeline()

        # Manga preset (DSP)
        cfg_manga = get_preset_config("manga")
        cfg_manga.mode = ProcessingMode.PURE_DSP
        cfg_manga.scale = 1.0
        out_manga = pipeline.enhance(ht_noisy, config=cfg_manga).image

        # Balanced preset (DSP)
        cfg_bal = get_preset_config("balanced")
        cfg_bal.mode = ProcessingMode.PURE_DSP
        cfg_bal.scale = 1.0
        out_bal = pipeline.enhance(ht_noisy, config=cfg_bal).image

        region_slice = (slice(4, 56), slice(4, 116))
        var_in = float(np.var(ht_noisy[region_slice].astype(np.float32)))
        var_manga = float(np.var(out_manga[region_slice].astype(np.float32)))
        var_bal = float(np.var(out_bal[region_slice].astype(np.float32)))

        manga_retention = var_manga / var_in
        self.assertGreater(manga_retention, 0.85, f"Manga preset smoothed screentone: {manga_retention:.1%}")

        bal_retention = var_bal / var_in
        self.assertLess(bal_retention, 0.35, f"Balanced preset did not smooth noise: {bal_retention:.1%}")

    def test_anime_style_grayscale_handling(self):
        pipeline = PureScalePipeline()
        for mode in (ProcessingMode.NEURAL_AI, ProcessingMode.HYBRID):
            cfg = PipelineConfig(mode=mode, scale=2.0, sr_style="anime")

            # 2D Grayscale [H, W]
            gray_2d = np.full((64, 64), 200, dtype=np.uint8)
            cv2.line(gray_2d, (10, 10), (54, 54), 20, thickness=2)
            res_2d = pipeline.enhance(gray_2d, config=cfg)
            self.assertEqual(res_2d.image.ndim, 2, f"Expected 2D image for mode {mode}")
            self.assertEqual(res_2d.image.shape, (128, 128))
            self.assertEqual(res_2d.image.dtype, np.uint8)

            # 3D 1-channel [H, W, 1]
            gray_3d = gray_2d[:, :, np.newaxis]
            res_3d = pipeline.enhance(gray_3d, config=cfg)
            self.assertEqual(res_3d.image.ndim, 2)
            self.assertEqual(res_3d.image.shape, (128, 128))

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

        # LUT fast paths must hold the same bound on dense random data.
        rng = np.random.default_rng(7)
        field = rng.integers(0, 256, (64, 64, 3)).astype(np.uint8)
        L, a, b = srgb_to_oklab(field)
        rec = oklab_to_srgb(L, a, b)
        self.assertLessEqual(int(np.max(np.abs(field.astype(np.int32) - rec.astype(np.int32)))), 1)

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

    def test_cli_fast_2x_flags(self):
        from purescale.cli import main
        code = main([self.temp_in, "-o", self.temp_out, "--fast-2x", "--scale", "2.0"])
        self.assertEqual(code, 0)
        self.assertTrue(os.path.exists(self.temp_out))

        code = main([self.temp_in, "-o", self.temp_out, "--preset", "fast", "--no-fast-2x"])
        self.assertEqual(code, 0)

    def test_cli_style_flags(self):
        from purescale.cli import main
        code = main([self.temp_in, "-o", self.temp_out, "--preset", "art", "--style", "photo", "--scale", "2.0"])
        self.assertEqual(code, 0)
        self.assertTrue(os.path.exists(self.temp_out))

        code = main([self.temp_in, "-o", self.temp_out, "--style", "anime", "--scale", "2.0"])
        self.assertEqual(code, 0)

    def test_cli_local_tone_flags(self):
        from purescale.cli import main
        code = main([
            self.temp_in, "-o", self.temp_out,
            "--local-tone", "--tone-strength", "0.6",
            "--highlight-recovery", "0.7", "--shadow-boost", "0.4",
            "--scale", "1.0",
        ])
        self.assertEqual(code, 0)
        self.assertTrue(os.path.exists(self.temp_out))

        code = main([
            self.temp_in, "-o", self.temp_out,
            "--preset", "landscape", "--no-local-tone",
            "--scale", "1.0",
        ])
        self.assertEqual(code, 0)

    def test_headless_gui(self):
        if not _TK_AVAILABLE:
            self.skipTest("tkinter not available on this runner")
        try:
            import customtkinter  # noqa: F401
        except ImportError as e:
            self.skipTest(f"GUI dependencies not installed: {e}")
        from purescale.gui.app import PureScaleApp
        try:
            app = PureScaleApp()
            self.assertIn("PureScale 4.0", app.title())

            # Test preset synchronization
            app._on_preset_change("Landscape")
            self.assertTrue(app.dehaze_switch.get())
            self.assertAlmostEqual(app.dehaze_slider.get(), 0.6)
            self.assertTrue(app.local_tone_switch.get())
            self.assertAlmostEqual(app.tone_strength_slider.get(), 0.45)
            self.assertFalse(app.fast_2x_switch.get())
            self.assertEqual(app.style_seg.get(), "Photo")

            app._on_preset_change("Art")
            self.assertEqual(app.style_seg.get(), "Anime")

            app._on_preset_change("Manga")
            self.assertFalse(app.denoise_switch.get())
            self.assertEqual(app.style_seg.get(), "Anime")
            self.assertAlmostEqual(app.cas_slider.get(), 0.7)

            app._on_preset_change("Fast")
            self.assertTrue(app.fast_2x_switch.get())
            self.assertEqual(app.style_seg.get(), "Photo")

            # Test mode change toggles style_seg state
            app._on_mode_change("PureDSP")
            self.assertEqual(app.style_seg.cget("state"), "disabled")
            app._on_mode_change("Neural AI")
            self.assertEqual(app.style_seg.cget("state"), "normal")
            app._on_mode_change("Hybrid")
            self.assertEqual(app.style_seg.cget("state"), "normal")

            # Test HUD update
            from purescale.dsp.diagnostics import diagnose_image
            diag = diagnose_image(cv2.imread(self.temp_in))
            app.hud_card.update_diagnostics(diag)
            app._apply_diagnostics_to_ui(diag)

            app.destroy()
        except Exception as e:
            # If no display available in CI environment, skip cleanly
            if _is_display_error(e):
                self.skipTest(f"Headless display not available: {e}")
            else:
                raise e

    def test_gui_style_routing_display(self):
        if not _TK_AVAILABLE:
            self.skipTest("tkinter not available on this runner")
        try:
            import customtkinter  # noqa: F401
        except ImportError as e:
            self.skipTest(f"GUI dependencies not installed: {e}")
        from purescale.gui.app import PureScaleApp
        try:
            app = PureScaleApp()
            from purescale.config import DiagnosticsResult

            # HUD shows classified style with confidence.
            diag = DiagnosticsResult(suggested_style="anime", style_confidence=0.85)
            app.hud_card.update_diagnostics(diag)
            self.assertIn("Anime", app.hud_card.style_val.cget("text"))
            self.assertIn("0.85", app.hud_card.style_val.cget("text"))

            # Routed weights style syncs the selector...
            app.style_seg.set("Photo")
            app._sync_routed_style(diag)
            self.assertEqual(app.style_seg.get(), "Anime")
            self.assertEqual(app._routed_style_label(diag), "Anime")

            # ...but manga keeps photo weights: selector stays, HUD reports.
            diag_manga = DiagnosticsResult(suggested_style="manga", style_confidence=0.90)
            app._sync_routed_style(diag_manga)
            self.assertEqual(app.style_seg.get(), "Anime")
            self.assertEqual(app._routed_style_label(diag_manga), "Manga")

            # Photo / low confidence: no label, no sync.
            self.assertIsNone(app._routed_style_label(DiagnosticsResult()))
            low = DiagnosticsResult(suggested_style="anime", style_confidence=0.30)
            self.assertIsNone(app._routed_style_label(low))

            # Auto-style switch toggle syncs selector on demand.
            app.current_diagnostics = diag
            app.style_seg.set("Photo")
            app.auto_style_switch.select()
            app._on_auto_style_toggle()
            self.assertEqual(app.style_seg.get(), "Anime")

            app.destroy()
        except Exception as e:
            if _is_display_error(e):
                self.skipTest(f"Headless display not available: {e}")
            else:
                raise e

    def test_gui_diagnostics_tint_and_local_tone_sync(self):
        if not _TK_AVAILABLE:
            self.skipTest("tkinter not available on this runner")
        try:
            import customtkinter  # noqa: F401
        except ImportError as e:
            self.skipTest(f"GUI dependencies not installed: {e}")
        from purescale.gui.app import PureScaleApp
        try:
            app = PureScaleApp()
            from purescale.config import DiagnosticsResult
            diag = DiagnosticsResult(
                shadow_clipping=8.0,
                highlight_clipping=5.0,
                color_cast_kelvin=-15,
                color_tint_offset=10,
            )
            diag.recommended_parameters = auto_tune_parameters(diag)
            app._apply_diagnostics_to_ui(diag)
            self.assertTrue(bool(app.local_tone_switch.get()))
            self.assertEqual(int(app.temp_slider.get()), -15)
            self.assertEqual(int(app.tint_slider.get()), 10)
            self.assertGreater(float(app.shadow_boost_slider.get()), 0.3)

            app.destroy()
        except Exception as e:
            if _is_display_error(e):
                self.skipTest(f"Headless display not available: {e}")
            else:
                raise e



class TestPureScale4GuiScrolling(unittest.TestCase):
    """Tests kinetic sidebar scrolling math (display-free) and routing."""

    def test_wheel_pixels(self):
        from purescale.gui.scroll_math import wheel_pixels

        self.assertEqual(wheel_pixels(-120, 0, 56), 56)   # one notch down
        self.assertEqual(wheel_pixels(120, 0, 56), -56)   # one notch up
        self.assertEqual(wheel_pixels(-240, 0, 56), 112)  # fast flick accumulates
        self.assertEqual(wheel_pixels(0, 4, 56), -56)     # Linux Button-4 (up)
        self.assertEqual(wheel_pixels(0, 5, 56), 56)      # Linux Button-5 (down)
        self.assertEqual(wheel_pixels(0, 0, 56), 0)       # no input, no motion

    def test_clamp_fraction(self):
        from purescale.gui.scroll_math import clamp_fraction

        self.assertEqual(clamp_fraction(0.5, 0.4), 0.5)
        self.assertEqual(clamp_fraction(-0.2, 0.4), 0.0)
        self.assertEqual(clamp_fraction(0.9, 0.4), 0.6)   # 1 - window
        self.assertEqual(clamp_fraction(0.3, 1.0), 0.0)   # fits entirely

    def test_sidebar_smooth_scroll_routing(self):
        if not _TK_AVAILABLE:
            self.skipTest("tkinter not available on this runner")
        try:
            import customtkinter  # noqa: F401
        except ImportError as e:
            self.skipTest(f"GUI dependencies not installed: {e}")
        from purescale.gui.app import PureScaleApp
        from purescale.gui.scrolling import SmoothScrollableFrame
        try:
            import types
            app = PureScaleApp()
            self.assertIsInstance(app.sidebar, SmoothScrollableFrame)
            app.update_idletasks()
            first, last = app.sidebar._parent_canvas.yview()
            if (first, last) == (0.0, 1.0):
                app.destroy()
                self.skipTest("Sidebar content fits; nothing to scroll.")

            canvas = app.sidebar._parent_canvas

            # Wheel over the canvas scrolls toward the bottom.
            app.sidebar._mouse_wheel_all(types.SimpleNamespace(widget=canvas, delta=-480, num=0))
            for _ in range(60):
                app.sidebar._smooth_step()
            moved, _ = canvas.yview()
            self.assertGreater(moved, first)

            # Wheel over a slider must NOT scroll (slider keeps the event).
            app.sidebar._smooth_target = None
            before, _ = canvas.yview()
            app.sidebar._mouse_wheel_all(types.SimpleNamespace(widget=app.cas_slider.slider, delta=-480, num=0))
            self.assertIsNone(app.sidebar._smooth_target)
            after, _ = canvas.yview()
            self.assertEqual(before, after)

            app.destroy()
        except Exception as e:
            if _is_display_error(e):
                self.skipTest(f"Headless display not available: {e}")
            else:
                raise e


class TestPureScale4GuiCanvas(unittest.TestCase):
    """Tests InteractiveCanvas viewport fit mode, multi-image transitions, and enhancement scaling."""

    def setUp(self):
        if not _TK_AVAILABLE:
            self.skipTest("tkinter not available on this runner")
        try:
            self.root = tk.Tk()
            self.root.geometry("800x600")
            self.root.withdraw()
        except Exception as e:
            if _is_display_error(e):
                self.skipTest(f"Headless display not available: {e}")
            else:
                raise e

    def tearDown(self):
        if hasattr(self, "root") and self.root:
            try:
                self.root.destroy()
            except Exception:
                pass

    def test_canvas_initial_fit_and_subsequent_image_fit(self):
        from PIL import Image
        from purescale.gui.canvas import InteractiveCanvas

        canvas = InteractiveCanvas(self.root, width=800, height=600)
        canvas.pack(fill="both", expand=True)
        self.root.update_idletasks()

        # Image 1 (1000x500)
        img1 = Image.new("RGB", (1000, 500), color=(100, 100, 100))
        canvas.set_images(img1, None)

        self.assertTrue(canvas.is_fit_mode)
        expected_zoom1 = (800 - 40) / 1000.0
        self.assertAlmostEqual(canvas.zoom_level, expected_zoom1, delta=0.05)

        # Image 2 (400x300, subsequent image load)
        img2 = Image.new("RGB", (400, 300), color=(50, 50, 50))
        canvas.set_images(img2, None)

        # Must automatically re-fit to img2 dimensions and stay in fit mode
        self.assertTrue(canvas.is_fit_mode)
        expected_zoom2 = min((800 - 40) / 400.0, (600 - 40) / 300.0)
        self.assertAlmostEqual(canvas.zoom_level, expected_zoom2, delta=0.05)

    def test_canvas_enhancement_stays_in_fit_mode(self):
        from PIL import Image
        from purescale.gui.canvas import InteractiveCanvas

        canvas = InteractiveCanvas(self.root, width=800, height=600)
        canvas.pack(fill="both", expand=True)
        self.root.update_idletasks()

        # Load initial original image
        orig = Image.new("RGB", (1000, 800), color=(80, 80, 80))
        canvas.set_images(orig, None)
        self.assertTrue(canvas.is_fit_mode)
        orig_disp_w = orig.size[0] * canvas.zoom_level

        # Now 4x enhanced image arrives
        enh = Image.new("RGB", (4000, 3200), color=(120, 120, 120))
        canvas.set_images(orig, enh)

        # Remains in fit mode and adjusts zoom so on-screen size does not explode
        self.assertTrue(canvas.is_fit_mode)
        enh_disp_w = enh.size[0] * canvas.zoom_level
        self.assertAlmostEqual(enh_disp_w, orig_disp_w, delta=5.0)

    def test_canvas_manual_zoom_exits_fit_mode_and_preserves_scale_on_enhance(self):
        from PIL import Image
        from purescale.gui.canvas import InteractiveCanvas

        canvas = InteractiveCanvas(self.root, width=800, height=600)
        canvas.pack(fill="both", expand=True)
        self.root.update_idletasks()

        orig = Image.new("RGB", (1000, 800), color=(80, 80, 80))
        canvas.set_images(orig, None)

        # User zooms in manually
        canvas.zoom_step(1.5)
        self.assertFalse(canvas.is_fit_mode)

        canvas.zoom_level = 2.0
        disp_w_before = orig.size[0] * canvas.zoom_level

        # Enhanced image arrives while user is zoomed in
        enh = Image.new("RGB", (4000, 3200), color=(120, 120, 120))
        canvas.set_images(orig, enh)

        self.assertFalse(canvas.is_fit_mode)
        # Zoom adjusted for 4x resolution: 2.0 * (1000/4000) = 0.5
        self.assertAlmostEqual(canvas.zoom_level, 0.5, delta=0.01)
        disp_w_after = enh.size[0] * canvas.zoom_level
        self.assertAlmostEqual(disp_w_after, disp_w_before, delta=1.0)

    def test_canvas_resize_refits_when_in_fit_mode(self):
        from PIL import Image
        from purescale.gui.canvas import InteractiveCanvas

        canvas = InteractiveCanvas(self.root, width=800, height=600)
        canvas.pack(fill="both", expand=True)
        self.root.update_idletasks()

        orig = Image.new("RGB", (1000, 800), color=(80, 80, 80))
        canvas.set_images(orig, None)
        self.assertTrue(canvas.is_fit_mode)

        # In fit mode, _on_resize triggers fit_to_window
        canvas._on_resize(None)
        self.assertTrue(canvas.is_fit_mode)

        # When not in fit mode, _on_resize does not force fit
        canvas.zoom_step(1.5)
        self.assertFalse(canvas.is_fit_mode)
        saved_zoom = canvas.zoom_level
        canvas._on_resize(None)
        self.assertFalse(canvas.is_fit_mode)
        self.assertAlmostEqual(canvas.zoom_level, saved_zoom, delta=0.001)


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

    def test_neural_fast_2x_proxy_consistency(self):
        from purescale.neural.engine import NeuralSuperResEngine
        from purescale.neural.models import get_default_model_path
        from purescale.quality import compare_images

        model_path = get_default_model_path(auto_download=False)
        if not model_path or not os.path.exists(model_path):
            self.skipTest("Neural super-resolution model weights not downloaded locally.")

        engine = NeuralSuperResEngine(model_path=model_path, target_device=DeviceTarget.CPU)
        ref = engine.upscale(self.patch, target_scale=2.0, tile_size=32, tile_overlap=4, fast_2x=False)
        fast = engine.upscale(self.patch, target_scale=2.0, tile_size=32, tile_overlap=4, fast_2x=True)

        self.assertEqual(ref.shape, (64, 64, 3))
        self.assertEqual(fast.shape, (64, 64, 3))

        metrics = compare_images(ref, fast)
        self.assertGreater(metrics["psnr_db"], 25.0)
        self.assertGreater(metrics["ssim"], 0.90)

        again = engine.upscale(self.patch, target_scale=2.0, tile_size=32, tile_overlap=4, fast_2x=True)
        self.assertTrue(np.array_equal(fast, again))

    def test_neural_sr_anime_style(self):
        from purescale.neural.engine import NeuralSuperResEngine
        from purescale.neural.models import ModelManager

        manager = ModelManager()
        model_path = manager.get_model_path("realesr-animevideov3-x4", auto_download=False)
        if not model_path or not os.path.exists(model_path):
            self.skipTest("Anime super-resolution model weights not downloaded locally.")

        engine = NeuralSuperResEngine(model_key="realesr-animevideov3-x4", target_device=DeviceTarget.CPU)
        out = engine.upscale(self.patch, target_scale=2.0, tile_size=32, tile_overlap=4)
        self.assertEqual(out.shape, (64, 64, 3))
        self.assertEqual(out.dtype, np.uint8)

        # Pipeline integration with sr_style="anime"
        pipeline = PureScalePipeline()
        cfg = PipelineConfig(mode=ProcessingMode.NEURAL_AI, scale=2.0, sr_style="anime", tile_size=32, tile_overlap=4)
        res = pipeline.enhance(self.patch, config=cfg)
        self.assertEqual(res.image.shape, (64, 64, 3))

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


class TestPureScale4StyleProfiles(unittest.TestCase):
    """Tests style-bound DSP profiles (photo/anime/manga)."""

    def test_profile_fills_defaults(self):
        from purescale.config import apply_style_profile

        anime = apply_style_profile(PipelineConfig(sr_style="anime"))
        self.assertEqual(anime.sharpen_strength, 0.8)
        self.assertEqual(anime.denoise_intensity, 25)
        self.assertEqual(anime.portrait_smooth, 0)
        self.assertEqual(anime.eye_clarity, 1.0)
        self.assertEqual(anime.pyramid_micro_texture, 1.05)

        manga = apply_style_profile(PipelineConfig(style_profile="manga"))
        self.assertFalse(manga.enable_denoise)
        self.assertFalse(manga.enable_contrast)
        self.assertEqual(manga.sharpen_strength, 0.7)

    def test_explicit_values_win(self):
        from purescale.config import apply_style_profile

        cfg = apply_style_profile(PipelineConfig(sr_style="anime", sharpen_strength=2.0))
        self.assertEqual(cfg.sharpen_strength, 2.0)
        # Unmentioned fields still filled from the profile.
        self.assertEqual(cfg.denoise_intensity, 25)

        # Explicit style_profile beats sr_style routing.
        cfg = apply_style_profile(PipelineConfig(sr_style="photo", style_profile="manga"))
        self.assertFalse(cfg.enable_denoise)

    def test_unknown_profile_raises(self):
        with self.assertRaises(ValueError):
            PipelineConfig(style_profile="nope")

    def test_photo_profile_byte_identical(self):
        pipeline = PureScalePipeline()
        np.random.seed(7)
        img = np.random.randint(40, 220, (64, 64, 3), dtype=np.uint8)
        base = PipelineConfig(mode=ProcessingMode.PURE_DSP, scale=1.0)
        styled = PipelineConfig(mode=ProcessingMode.PURE_DSP, scale=1.0, sr_style="photo")
        self.assertTrue(np.array_equal(
            pipeline.enhance(img, config=base).image,
            pipeline.enhance(img, config=styled).image,
        ))

    def test_manga_preserves_halftone_direction(self):
        pipeline = PureScalePipeline()
        # Fine halftone dots (4px period, print-realistic pitch): photo DSP
        # reads them as noise and smooths them, manga profile must keep them.
        yy, xx = np.mgrid[0:96, 0:96]
        dots = (((xx % 4) < 2) & ((yy % 4) < 2)).astype(np.float32) * 120 + 60
        rng = np.random.default_rng(9)
        halftone = np.clip(dots + rng.normal(0, 5.0, dots.shape), 0, 255).astype(np.uint8)
        halftone = np.repeat(halftone[:, :, np.newaxis], 3, axis=2)

        def dot_energy(out: np.ndarray) -> float:
            g = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32)
            return float((g - cv2.GaussianBlur(g, (9, 9), 2.0)).std())

        photo_cfg = PipelineConfig(mode=ProcessingMode.PURE_DSP, scale=1.0)
        manga_cfg = PipelineConfig(mode=ProcessingMode.PURE_DSP, scale=1.0, style_profile="manga")
        out_photo = pipeline.enhance(halftone, config=photo_cfg).image
        out_manga = pipeline.enhance(halftone, config=manga_cfg).image
        # Manga profile must preserve more dot-scale energy than photo DSP.
        self.assertGreater(dot_energy(out_manga), dot_energy(out_photo) * 1.2)

    def test_style_determinism(self):
        pipeline = PureScalePipeline()
        np.random.seed(11)
        img = np.random.randint(40, 220, (64, 64, 3), dtype=np.uint8)
        cfg = PipelineConfig(mode=ProcessingMode.PURE_DSP, scale=1.0, sr_style="anime")
        self.assertTrue(np.array_equal(
            pipeline.enhance(img, config=cfg).image,
            pipeline.enhance(img, config=cfg).image,
        ))

    def test_cli_style_profile_flag(self):
        import tempfile
        from purescale.cli import main

        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "in.png")
            dst = os.path.join(d, "out.png")
            cv2.imwrite(src, np.full((48, 48, 3), 128, dtype=np.uint8))
            code = main([src, "-o", dst, "--style-profile", "manga",
                         "--scale", "1.0", "--preset", "fast"])
            self.assertEqual(code, 0)
            self.assertTrue(os.path.exists(dst))


def _style_probe(kind: str) -> np.ndarray:
    """Deterministic synthetic fixtures per content style."""
    rng = np.random.default_rng(5)
    if kind == "cartoon":
        img = np.full((128, 128, 3), [255, 200, 150], np.uint8)
        img[10:60, 10:60] = [80, 120, 220]
        img[58:62, :] = 20
        img[:, 58:62] = 20
        return img
    if kind == "manga":
        yy, xx = np.mgrid[0:128, 0:128]
        dots = (((xx % 4) < 2) & ((yy % 4) < 2)).astype(np.float32) * 150 + 50
        gray = np.clip(dots + rng.normal(0, 4.0, dots.shape), 0, 255).astype(np.uint8)
        return np.repeat(gray[:, :, np.newaxis], 3, axis=2)
    if kind == "bw-photo":
        wave = 120 + 60 * np.sin(np.linspace(0, 6, 128))[None, :, None]
        return np.clip(wave + rng.normal(0, 5.0, (128, 128, 3)), 0, 255).astype(np.uint8)
    if kind == "photo-clean":
        x = np.linspace(0, 1, 128, dtype=np.float32)[None, :].repeat(128, axis=0)
        return np.clip(np.stack([x * 180 + 30, x * 150 + 40,
                                 (1 - x) * 120 + 30], axis=-1), 0, 255).astype(np.uint8)
    if kind == "photo-blur":
        return cv2.GaussianBlur(_style_probe("photo-clean"), (15, 15), 4.0)
    if kind == "manga-color":
        base = _style_probe("manga").astype(np.float32)
        return np.clip(base * np.array([1.0, 0.7, 0.4], np.float32), 0, 255).astype(np.uint8)
    if kind == "bw-ink":
        # B&W ink regions without fine dots (large flats + structure).
        ink = np.full((128, 128), 230, np.uint8)
        ink[20:60, 20:100] = 20
        ink[70:110, 30:90] = 120
        rng = np.random.default_rng(17)
        ink = np.clip(ink.astype(np.float32) + rng.normal(0, 2.0, ink.shape), 0, 255).astype(np.uint8)
        return np.repeat(ink[:, :, np.newaxis], 3, axis=2)
    if kind == "webtoon":
        webtoon = np.full((128, 128, 3), [245, 245, 245], dtype=np.uint8)
        webtoon[15:60, 15:110] = [60, 130, 220]
        cv2.rectangle(webtoon, (15, 15), (110, 60), (20, 20, 20), 2)
        webtoon[70:115, 15:110] = [80, 200, 120]
        cv2.rectangle(webtoon, (15, 70), (110, 115), (20, 20, 20), 2)
        return webtoon
    # noisy photo gradient
    x = np.linspace(0, 1, 128, dtype=np.float32)[None, :].repeat(128, axis=0)
    base = np.stack([x * 180 + 30, x * 150 + 40, (1 - x) * 120 + 30], axis=-1)
    return np.clip(base + rng.normal(0, 6.0, base.shape), 0, 255).astype(np.uint8)


class TestPureScale4AutoStyle(unittest.TestCase):
    """Tests content-aware style auto-detect (photo/anime/manga)."""

    def test_classifier_probes(self):
        from purescale.dsp.diagnostics import classify_content_style

        style, conf = classify_content_style(_style_probe("cartoon"))
        self.assertEqual(style, "anime")
        self.assertGreaterEqual(conf, 0.60)

        style, conf = classify_content_style(_style_probe("webtoon"))
        self.assertEqual(style, "anime")
        self.assertGreaterEqual(conf, 0.60)

        style, conf = classify_content_style(_style_probe("manga"))
        self.assertEqual(style, "manga")
        self.assertGreaterEqual(conf, 0.60)

        style, conf = classify_content_style(_style_probe("photo"))
        self.assertEqual(style, "photo")

        # B&W photo without dots must not route to manga.
        style, _ = classify_content_style(_style_probe("bw-photo"))
        self.assertEqual(style, "photo")

        # Clean/blurred gradients must not mimic flat anime color.
        for kind in ("photo-clean", "photo-blur"):
            style, _ = classify_content_style(_style_probe(kind))
            self.assertEqual(style, "photo")

        # Color halftones route to manga despite saturation.
        style, conf = classify_content_style(_style_probe("manga-color"))
        self.assertEqual(style, "manga")
        self.assertGreaterEqual(conf, 0.60)

        # 2D grayscale input must not crash.
        gray = cv2.cvtColor(_style_probe("manga"), cv2.COLOR_BGR2GRAY)
        style, _ = classify_content_style(gray)
        self.assertEqual(style, "manga")

        # B&W ink flats route to manga (no color, few tones, structure).
        style, conf = classify_content_style(_style_probe("bw-ink"))
        self.assertEqual(style, "manga")
        self.assertGreaterEqual(conf, 0.60)

    def test_classifier_grayscale_photos(self):
        """Ensures continuous-tone B&W photos do not get misclassified as manga."""
        from purescale.dsp.diagnostics import classify_content_style

        fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures")
        for fname in ("golden_photo.png", "golden_edge.png", "golden_texture.png"):
            path = os.path.join(fixtures_dir, fname)
            if not os.path.exists(path):
                continue
            bgr = cv2.imread(path)
            gray_2d = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            gray_3d = cv2.cvtColor(gray_2d, cv2.COLOR_GRAY2BGR)

            style_2d, _ = classify_content_style(gray_2d)
            self.assertEqual(style_2d, "photo", f"2D grayscale {fname} misclassified as {style_2d}")

            style_3d, _ = classify_content_style(gray_3d)
            self.assertEqual(style_3d, "photo", f"3D grayscale {fname} misclassified as {style_3d}")

    def test_classifier_edge_cases(self):
        """Tests that irregular/edge-case inputs do not raise errors."""
        from purescale.dsp.diagnostics import classify_content_style

        # None / empty / tiny inputs
        self.assertEqual(classify_content_style(None)[0], "photo")
        self.assertEqual(classify_content_style(np.zeros((0, 0, 3), dtype=np.uint8))[0], "photo")
        self.assertEqual(classify_content_style(np.full((2, 2, 3), 128, dtype=np.uint8))[0], "photo")

        # 3D 1-channel grayscale
        gray_3d_1ch = np.zeros((64, 64, 1), dtype=np.uint8)
        self.assertEqual(classify_content_style(gray_3d_1ch)[0], "photo")

        # 4-channel BGRA
        bgra = np.full((64, 64, 4), 180, dtype=np.uint8)
        self.assertEqual(classify_content_style(bgra)[0], "photo")

    def test_classifier_painterly_illustration(self):
        """Tests that digital illustrations with background lighting/gradients route to anime."""
        from purescale.dsp.diagnostics import classify_content_style

        # Digital character in front of atmospheric sky gradient
        sky = np.linspace(240, 60, 128, dtype=np.float32)[:, None].repeat(128, axis=1)
        ill = np.stack([sky, sky * 0.75, sky * 0.15], axis=-1).astype(np.uint8)
        ill[40:100, 40:90] = [40, 160, 230] # character body
        cv2.rectangle(ill, (40, 40), (90, 100), (15, 15, 15), 2) # outline

        style, conf = classify_content_style(ill)
        self.assertEqual(style, "anime")
        self.assertGreaterEqual(conf, 0.60)

    def test_classifier_determinism(self):
        from purescale.dsp.diagnostics import classify_content_style

        for kind in ("cartoon", "webtoon", "manga", "bw-ink", "photo"):
            img = _style_probe(kind)
            self.assertEqual(classify_content_style(img), classify_content_style(img))

    def test_pipeline_manga_routing(self):
        pipeline = PureScalePipeline()
        cfg = PipelineConfig(mode=ProcessingMode.PURE_DSP, scale=1.0, auto_style=True)
        res = pipeline.enhance(_style_probe("manga"), config=cfg)
        self.assertEqual(res.diagnostics.suggested_style, "manga")
        # Manga DSP profile applied: denoise + semantic stages skipped.
        self.assertNotIn("swf_denoise", res.stage_latencies)
        self.assertNotIn("semantic_parsing", res.stage_latencies)

    def test_pipeline_anime_routing(self):
        pipeline = PureScalePipeline()
        cfg = PipelineConfig(mode=ProcessingMode.PURE_DSP, scale=1.0, auto_style=True)
        res = pipeline.enhance(_style_probe("cartoon"), config=cfg)
        self.assertEqual(res.diagnostics.suggested_style, "anime")

    def test_explicit_style_pin_wins(self):
        pipeline = PureScalePipeline()
        cfg = PipelineConfig(mode=ProcessingMode.PURE_DSP, scale=1.0,
                             sr_style="anime", auto_style=True)
        res = pipeline.enhance(_style_probe("manga"), config=cfg)
        self.assertEqual(res.diagnostics.suggested_style, "manga")
        # Explicit anime pin kept for weights; manga DSP profile still applied.
        self.assertNotIn("swf_denoise", res.stage_latencies)

    def test_cli_auto_style_flag(self):
        import tempfile
        from purescale.cli import main

        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "in.png")
            dst = os.path.join(d, "out.png")
            cv2.imwrite(src, _style_probe("manga"))
            code = main([src, "-o", dst, "--auto-style", "--scale", "1.0"])
            self.assertEqual(code, 0)
            self.assertTrue(os.path.exists(dst))


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


class TestPureScale4Packaging(unittest.TestCase):
    """Tests version centralization, preset strictness, and device detection."""

    def test_version_consistency(self):
        import re
        import purescale

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        with open(os.path.join(repo_root, "pyproject.toml"), encoding="utf-8") as fp:
            pyproject = fp.read()
        match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, flags=re.MULTILINE)
        self.assertIsNotNone(match, "pyproject.toml must declare a version")
        self.assertEqual(purescale.__version__, match.group(1))

    def test_unknown_preset_raises(self):
        from purescale.config import get_preset_config

        with self.assertRaises(ValueError):
            get_preset_config("not-a-preset")

    def test_diagnostics_result_single_source(self):
        from purescale.config import DiagnosticsResult as FromConfig
        from purescale.dsp.diagnostics import DiagnosticsResult as FromDsp

        self.assertIs(FromConfig, FromDsp)

    def test_device_selection(self):
        from purescale.device import cpu_label, has_directml, select_providers

        self.assertTrue(cpu_label())
        self.assertIsInstance(has_directml(), bool)
        providers, name = select_providers(DeviceTarget.CPU)
        self.assertEqual(providers, ["CPUExecutionProvider"])
        self.assertTrue(name)
        with self.assertRaises(ImportError):
            select_providers(DeviceTarget.OPENCV_DNN)


class TestPureScale4Golden(unittest.TestCase):
    """Golden regression tests: output-changing optimizations must not
    silently degrade quality. Thresholds tolerate cross-platform numeric
    noise but fail on real regressions (regenerate fixtures intentionally
    via tests/fixtures/generate_goldens.py and document the delta)."""

    def test_golden_outputs(self):
        from tests.fixtures.generate_goldens import golden_config, make_inputs
        from purescale.quality import compare_images

        fixture_dir = os.path.join(os.path.dirname(__file__), "fixtures")
        pipeline = PureScalePipeline()
        cfg = golden_config()
        for name, img in make_inputs().items():
            with self.subTest(fixture=name):
                ref = cv2.imread(os.path.join(fixture_dir, f"golden_{name}.png"))
                self.assertIsNotNone(ref, f"Missing golden fixture: {name}")
                res = pipeline.enhance(img, config=cfg)
                metrics = compare_images(ref, res.image)
                self.assertGreater(metrics["ssim"], 0.999, f"SSIM regression on {name}")
                self.assertGreater(metrics["psnr_db"], 60.0, f"PSNR regression on {name}")


class TestPureScale4Quality(unittest.TestCase):
    """Tests reference-based quality metrics and the compare command."""

    def test_psnr_ssim_identities(self):
        from purescale.quality import compare_images

        rng = np.random.default_rng(5)
        img = rng.integers(0, 256, (48, 48, 3)).astype(np.uint8)
        m = compare_images(img, img.copy())
        self.assertEqual(m["psnr_db"], math.inf)
        self.assertAlmostEqual(m["ssim"], 1.0, places=5)

    def test_psnr_ssim_sensitivity(self):
        from purescale.quality import compare_images

        base = np.full((48, 48, 3), 128, dtype=np.uint8)
        noisy = np.clip(base.astype(np.int16) + 25, 0, 255).astype(np.uint8)
        m = compare_images(base, noisy)
        self.assertLess(m["psnr_db"], 25.0)
        self.assertLess(m["ssim"], 0.99)

    def test_quality_shape_mismatch(self):
        from purescale.quality import compare_images

        with self.assertRaises(ValueError):
            compare_images(np.zeros((8, 8, 3), dtype=np.uint8),
                           np.zeros((16, 16, 3), dtype=np.uint8))

    def test_cli_compare(self):
        from purescale.cli import main_compare

        d = os.path.abspath("test_compare_tmp")
        os.makedirs(d, exist_ok=True)
        try:
            a = os.path.join(d, "a.png")
            b = os.path.join(d, "b.png")
            cv2.imwrite(a, np.full((32, 32, 3), 100, dtype=np.uint8))
            cv2.imwrite(b, np.full((32, 32, 3), 110, dtype=np.uint8))
            self.assertEqual(main_compare([a, b]), 0)
            self.assertEqual(main_compare([a, b, "--json"]), 0)
            self.assertEqual(main_compare([a, os.path.join(d, "missing.png")]), 1)
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)


class TestPureScale4Models(unittest.TestCase):
    """Tests model registry integrity, scale-aware routing, and device listing."""

    def test_registry_schema(self):
        from purescale.neural.models import MODEL_REGISTRY

        self.assertGreaterEqual(len(MODEL_REGISTRY), 2)
        for key, info in MODEL_REGISTRY.items():
            for field in ("filename", "url", "size_bytes", "scale", "task", "description"):
                self.assertIn(field, info, f"{key} missing {field}")
            self.assertTrue(info["url"].startswith("https://"), key)
            if info.get("task") == "super-resolution":
                self.assertIn("style", info, f"{key} missing style")
                self.assertIn(info["style"], ("photo", "anime"))

    def test_resolve_sr_model_prefers_smallest_covering_scale(self):
        from purescale.neural.models import resolve_sr_model

        fake = {
            "sr-photo-x2": {"task": "super-resolution", "scale": 2, "style": "photo"},
            "sr-photo-x4": {"task": "super-resolution", "scale": 4, "style": "photo"},
            "sr-anime-x4": {"task": "super-resolution", "scale": 4, "style": "anime"},
            "det": {"task": "face-detection", "scale": 1},
        }
        self.assertEqual(resolve_sr_model(1.5, style="photo", registry=fake), "sr-photo-x2")
        self.assertEqual(resolve_sr_model(2.0, style="photo", registry=fake), "sr-photo-x2")
        self.assertEqual(resolve_sr_model(3.0, style="photo", registry=fake), "sr-photo-x4")
        self.assertEqual(resolve_sr_model(2.0, style="anime", registry=fake), "sr-anime-x4")
        with self.assertRaises(ValueError):
            resolve_sr_model(2.0, style="fantasy", registry=fake)
        with self.assertRaises(ValueError):
            resolve_sr_model(2.0, registry={"det": {"task": "face-detection", "scale": 1}})

    def test_resolve_sr_model_defaults(self):
        from purescale.neural.models import resolve_sr_model

        self.assertEqual(resolve_sr_model(2.0), "realesr-general-x4v3")
        self.assertEqual(resolve_sr_model(2.0, style="photo"), "realesr-general-x4v3")
        self.assertEqual(resolve_sr_model(2.0, style="anime"), "realesr-animevideov3-x4")
        with self.assertRaises(ValueError):
            resolve_sr_model(2.0, style="nonexistent")

    def test_cached_models_verify(self):
        from purescale.neural.models import MODEL_REGISTRY, ModelManager

        manager = ModelManager()
        checked = 0
        for key in MODEL_REGISTRY:
            path = manager.get_model_path(key, auto_download=False)
            if not path or not os.path.exists(path):
                continue
            checked += 1
            self.assertTrue(manager.verify_model(key), f"{key} failed SHA256 check")
        if checked == 0:
            self.skipTest("No model weights downloaded locally.")

    def test_list_devices(self):
        from purescale.cli import main

        self.assertEqual(main(["--list-devices"]), 0)
        self.assertEqual(main(["--list-devices", "--json"]), 0)

    def test_cuda_target_selection(self):
        from purescale.device import has_cuda, select_providers
        from purescale.config import DeviceTarget

        providers, name = select_providers(DeviceTarget.CUDA_GPU)
        if has_cuda():
            self.assertIn("CUDAExecutionProvider", providers)
        else:
            self.assertEqual(providers, ["CPUExecutionProvider"])
        self.assertTrue(name)


if __name__ == "__main__":
    unittest.main()
