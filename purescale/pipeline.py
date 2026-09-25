"""Unified multi-mode enhancement pipeline orchestrator for PureScale 4.0.

Integrates Autonomous Signal Diagnostics, Dark Channel Prior Dehazing,
Multi-Cue Semantic Region Guidance, Multiscale Local Laplacian Pyramids,
and Hybrid Neural Edge Synthesis.
"""

import time
from typing import Callable, Dict, Optional, Tuple
import cv2
import numpy as np

from purescale.config import (
    DeviceTarget,
    DiagnosticsResult,
    PipelineConfig,
    ProcessingMode,
    ProcessingResult,
)
from purescale.dsp.color import bradford_cat16_white_balance, oklab_vibrance
from purescale.dsp.contrast import bimef_exposure_fusion, contrast_adaptive_sharpen
from purescale.dsp.dehaze import atmospheric_dehaze
from purescale.dsp.diagnostics import auto_tune_parameters, diagnose_image
from purescale.dsp.filters import side_window_filter
from purescale.dsp.pyramid import multiscale_laplacian_filter
from purescale.dsp.restore import directional_subpixel_depixelate, tensor_steered_shock_filter
from purescale.dsp.semantic import compute_spatial_guidance_maps, extract_semantic_masks
from purescale.dsp.upsample import edge_adaptive_upsample
from purescale.device import cpu_label
from purescale.neural.engine import NeuralSuperResEngine
from purescale.neural.models import resolve_sr_model
from purescale.neural.portrait import PortraitRetoucher


class PureScalePipeline:
    """
    Main enhancement orchestrator for PureScale 4.0.
    Executes autonomous signal diagnostics, physical conditioning,
    multiscale frequency filtering, and neural edge synthesis.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        model_path: Optional[str] = None,
    ):
        self.config = config or PipelineConfig()
        self.model_path = model_path
        self._neural_engines: Dict[Tuple[str, str], NeuralSuperResEngine] = {}
        self._portrait_retoucher: Optional[PortraitRetoucher] = None

    def _get_neural_engine(
        self,
        target_device: DeviceTarget,
        target_scale: float = 4.0,
    ) -> NeuralSuperResEngine:
        # Scale-aware routing: a native x2 model serves target_scale <= 2
        # directly instead of 4x inference followed by downsampling. Engines
        # are cached per (model, device); an explicit model_path keeps the
        # legacy single-engine behavior.
        if self.model_path:
            cache_key = ("explicit-path", str(target_device))
            if cache_key not in self._neural_engines:
                self._neural_engines[cache_key] = NeuralSuperResEngine(
                    model_path=self.model_path,
                    target_device=target_device,
                )
            return self._neural_engines[cache_key]
        model_key = resolve_sr_model(target_scale)
        cache_key = (model_key, str(target_device))
        if cache_key not in self._neural_engines:
            self._neural_engines[cache_key] = NeuralSuperResEngine(
                target_device=target_device,
                model_key=model_key,
            )
        return self._neural_engines[cache_key]

    def _get_portrait_retoucher(self) -> PortraitRetoucher:
        if self._portrait_retoucher is None:
            self._portrait_retoucher = PortraitRetoucher()
        return self._portrait_retoucher

    def enhance(
        self,
        img: np.ndarray,
        config: Optional[PipelineConfig] = None,
        alpha: Optional[np.ndarray] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> ProcessingResult:
        """
        Executes enhancement across configured mode with PureScale 4.0 computer vision.

        Args:
            img: BGR uint8 input image [H, W, 3]
            config: Optional PipelineConfig override
            alpha: Optional alpha channel mask [H, W] uint8
            progress_callback: Optional callback(stage_name, progress_ratio)

        Returns:
            ProcessingResult containing enhanced image, diagnostics, and telemetry
        """
        cfg = PipelineConfig(**(config or self.config).to_dict())
        t_start = time.perf_counter()
        stage_latencies = {}
        faces_detected = 0
        active_backend = f"{cpu_label()} (DSP)"

        cur = img.copy()
        is_grayscale = (cur.ndim == 2)
        if is_grayscale:
            cur = cv2.cvtColor(cur, cv2.COLOR_GRAY2BGR)
        elif cur.ndim == 3 and cur.shape[2] == 4:
            if alpha is None:
                alpha = cur[:, :, 3].copy()
            cur = cur[:, :, :3].copy()

        orig_h, orig_w = cur.shape[:2]
        dest_w = int(round(orig_w * cfg.scale))
        dest_h = int(round(orig_h * cfg.scale))

        max_allowed_pixels = int(cfg.max_megapixels * 1_000_000)
        orig_pixels = orig_w * orig_h
        dest_pixels = dest_w * dest_h
        if orig_pixels > max_allowed_pixels:
            raise ValueError(
                f"Input image resolution ({orig_w}x{orig_h} = {orig_pixels / 1e6:.2f} MP) "
                f"exceeds safety threshold ({cfg.max_megapixels:.1f} MP). "
                f"Increase max_megapixels in config if intentional to prevent out-of-memory."
            )
        if dest_pixels > max_allowed_pixels:
            raise ValueError(
                f"Output target resolution ({dest_w}x{dest_h} = {dest_pixels / 1e6:.2f} MP at {cfg.scale}x scale) "
                f"exceeds safety threshold ({cfg.max_megapixels:.1f} MP). "
                f"Increase max_megapixels or reduce scale to prevent out-of-memory."
            )

        mode = cfg.mode if isinstance(cfg.mode, ProcessingMode) else ProcessingMode(cfg.mode)

        def report(stage_name: str, ratio: float) -> None:
            if progress_callback:
                progress_callback(stage_name, ratio)

        # -------------------------------------------------------------
        # STAGE -1: AUTONOMOUS DIAGNOSTICS & AUTO-TUNING
        # -------------------------------------------------------------
        diag_result: Optional[DiagnosticsResult] = None
        detail_map: Optional[np.ndarray] = None
        denoise_map: Optional[np.ndarray] = None
        semantic_breakdown = {}

        if cfg.enable_diagnostics or cfg.auto_tune:
            report("Autonomous Signal Diagnostics", 0.02)
            t0 = time.perf_counter()
            diag_result = diagnose_image(cur)
            stage_latencies["diagnostics"] = (time.perf_counter() - t0) * 1000.0

            if cfg.auto_tune and diag_result.recommended_parameters:
                for k, v in diag_result.recommended_parameters.items():
                    if hasattr(cfg, k):
                        setattr(cfg, k, v)

        # -------------------------------------------------------------
        # STAGE -0.5: MULTI-CUE SEMANTIC REGION PARSING
        # -------------------------------------------------------------
        # Gated: the maps are only consumed by SWF blending (denoise_map)
        # and pyramid filtering (detail_map). Neural mode ignores both, so
        # parsing there is pure overhead (~17ms VGA, ~128ms 1080p).
        _needs_detail = cfg.enable_pyramid and (
            abs(cfg.pyramid_micro_texture - 1.0) >= 0.02
            or abs(cfg.pyramid_structure_boost - 1.0) >= 0.02
        )
        _needs_denoise_map = cfg.enable_denoise and cfg.denoise_intensity > 0
        _semantic_consumed = (_needs_detail or _needs_denoise_map) and mode != ProcessingMode.NEURAL_AI
        if cfg.enable_semantic_guidance and _semantic_consumed:
            report("Multi-Cue Semantic Parsing", 0.04)
            t0 = time.perf_counter()
            sem_masks = extract_semantic_masks(cur)
            semantic_breakdown = sem_masks.breakdown
            if diag_result is not None:
                diag_result.semantic_breakdown = semantic_breakdown
            full_guide = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
            detail_map, denoise_map = compute_spatial_guidance_maps(sem_masks, guide=full_guide)
            stage_latencies["semantic_parsing"] = (time.perf_counter() - t0) * 1000.0

        # -------------------------------------------------------------
        # STAGE 0: PRE-RESTORATION CONDITIONING (De-pixelate & Deblur)
        # -------------------------------------------------------------
        if cfg.depixel_strength > 0:
            report("Subpixel Anti-Aliasing", 0.06)
            t0 = time.perf_counter()
            cur = directional_subpixel_depixelate(cur, strength=cfg.depixel_strength)
            stage_latencies["pre_depixel"] = (time.perf_counter() - t0) * 1000.0

        if cfg.deblur_strength > 0:
            report("Structure Tensor Shock Deblur", 0.08)
            t0 = time.perf_counter()
            cur = tensor_steered_shock_filter(cur, strength=cfg.deblur_strength)
            stage_latencies["pre_deblur"] = (time.perf_counter() - t0) * 1000.0

        # -------------------------------------------------------------
        # STAGE 1: ATMOSPHERIC DEHAZING (Dark Channel Prior)
        # -------------------------------------------------------------
        if cfg.enable_dehaze and cfg.dehaze_strength > 0.0:
            report("Atmospheric Dehazing (DCP)", 0.12)
            t0 = time.perf_counter()
            cur, _ = atmospheric_dehaze(cur, strength=cfg.dehaze_strength)
            stage_latencies["dcp_dehaze"] = (time.perf_counter() - t0) * 1000.0

        # -------------------------------------------------------------
        # MODE 1: PURE DSP (Analytical, Zero Neural, 100% Deterministic)
        # -------------------------------------------------------------
        if mode == ProcessingMode.PURE_DSP:
            # 1. Structural Denoise (SWF)
            # Clean-image skip: diagnosed sigma < 2.0 with default strength
            # bypasses ~75ms (VGA) to ~1000ms (1080p). Matches auto_tune
            # Clean -> enable_denoise=False when --auto is used.
            _swf_clean_skip = (
                diag_result is not None
                and diag_result.noise_sigma < 2.0
                and cfg.denoise_intensity <= 35
            )
            if cfg.enable_denoise and cfg.denoise_intensity > 0 and not _swf_clean_skip:
                report("Structural Denoising (SWF)", 0.18)
                t0 = time.perf_counter()
                r = max(1, int(round(cfg.denoise_intensity / 25.0)))
                iters = 1 if cfg.denoise_intensity <= 60 else 2
                denoised = side_window_filter(
                    cur, radius=r, iterations=iters,
                    noise_sigma=diag_result.noise_sigma if diag_result else None,
                )
                if denoise_map is not None:
                    alpha_den = np.clip(denoise_map / 1.5, 0.20, 1.0)[:, :, np.newaxis]
                    cur = np.clip(
                        np.rint(alpha_den * denoised.astype(np.float32) + (1.0 - alpha_den) * cur.astype(np.float32)),
                        0.0,
                        255.0,
                    ).astype(np.uint8)
                else:
                    cur = denoised
                stage_latencies["swf_denoise"] = (time.perf_counter() - t0) * 1000.0

            # 2. Spatial Scaling (EASU)
            if cfg.scale != 1.0:
                report("Edge-Adaptive Spatial Upsampling (EASU)", 0.32)
                t0 = time.perf_counter()
                cur = edge_adaptive_upsample(cur, scale=cfg.scale)
                if alpha is not None:
                    alpha = cv2.resize(alpha, (dest_w, dest_h), interpolation=cv2.INTER_LANCZOS4)
                stage_latencies["easu_upsample"] = (time.perf_counter() - t0) * 1000.0

            # Scale detail guidance map to match new resolution if needed
            scaled_detail_map = None
            if detail_map is not None:
                cur_h, cur_w = cur.shape[:2]
                scaled_detail_map = cv2.resize(detail_map, (cur_w, cur_h), interpolation=cv2.INTER_LINEAR)

            # 3. Multiscale Local Laplacian Pyramid Filtering
            if cfg.enable_pyramid:
                report("Multiscale Local Laplacian Pyramid", 0.48)
                t0 = time.perf_counter()
                cur = multiscale_laplacian_filter(
                    cur,
                    micro_texture_gain=cfg.pyramid_micro_texture,
                    structure_boost=cfg.pyramid_structure_boost,
                    noise_damping=0.15 if (diag_result and diag_result.noise_sigma > 4.0) else 0.0,
                    dynamic_range_compression=cfg.pyramid_dynamic_range,
                    spatial_detail_mask=scaled_detail_map,
                )
                stage_latencies["multiscale_laplacian"] = (time.perf_counter() - t0) * 1000.0

            # 4. Dynamic Range Fusion (BIMEF with Black-Point Pinning)
            if cfg.enable_contrast and cfg.contrast_boost > 1.0:
                report("Dynamic Range Fusion (BIMEF)", 0.60)
                t0 = time.perf_counter()
                cur = bimef_exposure_fusion(cur, contrast_boost=cfg.contrast_boost)
                stage_latencies["bimef_contrast"] = (time.perf_counter() - t0) * 1000.0

            # 5. Radiometric Exposure Offset
            if cfg.brightness_shift != 0:
                report("Exposure Offset", 0.68)
                t0 = time.perf_counter()
                cur = np.clip(cur.astype(np.int16) + cfg.brightness_shift, 0, 255).astype(np.uint8)
                stage_latencies["exposure_offset"] = (time.perf_counter() - t0) * 1000.0

            # 6. Detail Clarity (CAS)
            if cfg.sharpen_strength > 0.0:
                report("Contrast-Adaptive Sharpening (CAS)", 0.76)
                t0 = time.perf_counter()
                cur = contrast_adaptive_sharpen(cur, strength=cfg.sharpen_strength)
                stage_latencies["cas_sharpen"] = (time.perf_counter() - t0) * 1000.0

            # 7. Portrait Retouching (YuNet + FGF)
            if cfg.portrait_smooth > 0 or cfg.eye_clarity > 1.0:
                report("Portrait Retouching (YuNet + FGF)", 0.84)
                t0 = time.perf_counter()
                retoucher = self._get_portrait_retoucher()
                cur, faces_detected = retoucher.enhance(
                    cur,
                    portrait_smooth=cfg.portrait_smooth,
                    eye_clarity=cfg.eye_clarity,
                )
                stage_latencies["portrait_retouch"] = (time.perf_counter() - t0) * 1000.0

            # 8. Perceptual Vibrance (Oklab)
            if cfg.vibrance_boost != 1.0:
                report("Perceptual Vibrance (Oklab)", 0.92)
                t0 = time.perf_counter()
                cur = oklab_vibrance(cur, boost=cfg.vibrance_boost)
                stage_latencies["oklab_vibrance"] = (time.perf_counter() - t0) * 1000.0

            # 9. White Balance (Bradford CAT16)
            if cfg.color_temperature != 0:
                report("White Balance (CAT16)", 0.96)
                t0 = time.perf_counter()
                cur = bradford_cat16_white_balance(cur, temperature_offset=cfg.color_temperature)
                stage_latencies["cat16_wb"] = (time.perf_counter() - t0) * 1000.0

        # -------------------------------------------------------------
        # MODE 2: NEURAL AI (Compact Edge AI Super-Resolution)
        # -------------------------------------------------------------
        elif mode == ProcessingMode.NEURAL_AI:
            report("Neural AI Super-Resolution (Real-ESRGAN Compact)", 0.25)
            t0 = time.perf_counter()
            engine = self._get_neural_engine(cfg.device, cfg.scale)
            active_backend = engine.backend_name

            cur = engine.upscale(
                cur,
                target_scale=cfg.scale,
                tile_size=cfg.tile_size,
                tile_overlap=cfg.tile_overlap,
            )
            if alpha is not None:
                alpha = cv2.resize(alpha, (dest_w, dest_h), interpolation=cv2.INTER_LANCZOS4)
            stage_latencies["neural_superres"] = (time.perf_counter() - t0) * 1000.0

            # Multiscale Laplacian refinement
            if cfg.enable_pyramid:
                report("Multiscale Laplacian Refinement", 0.70)
                t0 = time.perf_counter()
                cur = multiscale_laplacian_filter(
                    cur,
                    micro_texture_gain=cfg.pyramid_micro_texture,
                    structure_boost=cfg.pyramid_structure_boost,
                )
                stage_latencies["multiscale_laplacian"] = (time.perf_counter() - t0) * 1000.0

            # Optional Portrait Retouching
            if cfg.portrait_smooth > 0 or cfg.eye_clarity > 1.0:
                report("Portrait Retouching (YuNet + FGF)", 0.85)
                t0 = time.perf_counter()
                retoucher = self._get_portrait_retoucher()
                cur, faces_detected = retoucher.enhance(
                    cur,
                    portrait_smooth=cfg.portrait_smooth,
                    eye_clarity=cfg.eye_clarity,
                )
                stage_latencies["portrait_retouch"] = (time.perf_counter() - t0) * 1000.0

        # -------------------------------------------------------------
        # MODE 3: HYBRID (Neural AI Edge Synthesis + Multiscale DSP)
        # -------------------------------------------------------------
        elif mode == ProcessingMode.HYBRID:
            # 1. Subtle Structural Denoise before neural pass (SWF)
            _swf_clean_skip = (
                diag_result is not None
                and diag_result.noise_sigma < 2.0
                and cfg.denoise_intensity <= 35
            )
            if cfg.enable_denoise and cfg.denoise_intensity > 0 and not _swf_clean_skip:
                report("Pre-Denoising (SWF)", 0.15)
                t0 = time.perf_counter()
                r = max(1, int(round(cfg.denoise_intensity / 40.0)))
                denoised = side_window_filter(
                    cur, radius=r, iterations=1,
                    noise_sigma=diag_result.noise_sigma if diag_result else None,
                )
                if denoise_map is not None:
                    alpha_den = np.clip(denoise_map / 1.5, 0.20, 1.0)[:, :, np.newaxis]
                    cur = np.clip(
                        np.rint(alpha_den * denoised.astype(np.float32) + (1.0 - alpha_den) * cur.astype(np.float32)),
                        0.0,
                        255.0,
                    ).astype(np.uint8)
                else:
                    cur = denoised
                stage_latencies["swf_pre_denoise"] = (time.perf_counter() - t0) * 1000.0

            # 2. Neural Edge Synthesis
            report("Neural Edge Synthesis (Real-ESRGAN Compact)", 0.35)
            t0 = time.perf_counter()
            engine = self._get_neural_engine(cfg.device, cfg.scale)
            active_backend = engine.backend_name

            cur = engine.upscale(
                cur,
                target_scale=cfg.scale,
                tile_size=cfg.tile_size,
                tile_overlap=cfg.tile_overlap,
            )
            if alpha is not None:
                alpha = cv2.resize(alpha, (dest_w, dest_h), interpolation=cv2.INTER_LANCZOS4)
            stage_latencies["neural_superres"] = (time.perf_counter() - t0) * 1000.0

            scaled_detail_map = None
            if detail_map is not None:
                cur_h, cur_w = cur.shape[:2]
                scaled_detail_map = cv2.resize(detail_map, (cur_w, cur_h), interpolation=cv2.INTER_LINEAR)

            # 3. Multiscale Local Laplacian Pyramid Filtering
            if cfg.enable_pyramid:
                report("Multiscale Local Laplacian Pyramid", 0.55)
                t0 = time.perf_counter()
                cur = multiscale_laplacian_filter(
                    cur,
                    micro_texture_gain=cfg.pyramid_micro_texture,
                    structure_boost=cfg.pyramid_structure_boost,
                    noise_damping=0.10 if (diag_result and diag_result.noise_sigma > 4.0) else 0.0,
                    spatial_detail_mask=scaled_detail_map,
                )
                stage_latencies["multiscale_laplacian"] = (time.perf_counter() - t0) * 1000.0

            # 4. Dynamic Range Fusion (BIMEF with Black-Point Pinning)
            if cfg.enable_contrast and cfg.contrast_boost > 1.0:
                report("Dynamic Range Fusion (BIMEF)", 0.65)
                t0 = time.perf_counter()
                cur = bimef_exposure_fusion(cur, contrast_boost=cfg.contrast_boost)
                stage_latencies["bimef_contrast"] = (time.perf_counter() - t0) * 1000.0

            # 5. Micro-Texture Clarity (CAS)
            if cfg.sharpen_strength > 0.0:
                report("Micro-Texture Sharpening (CAS)", 0.75)
                t0 = time.perf_counter()
                cur = contrast_adaptive_sharpen(cur, strength=cfg.sharpen_strength * 0.7)
                stage_latencies["cas_sharpen"] = (time.perf_counter() - t0) * 1000.0

            # 6. Portrait Retouching (YuNet + FGF)
            if cfg.portrait_smooth > 0 or cfg.eye_clarity > 1.0:
                report("Portrait Retouching (YuNet + FGF)", 0.83)
                t0 = time.perf_counter()
                retoucher = self._get_portrait_retoucher()
                cur, faces_detected = retoucher.enhance(
                    cur,
                    portrait_smooth=cfg.portrait_smooth,
                    eye_clarity=cfg.eye_clarity,
                )
                stage_latencies["portrait_retouch"] = (time.perf_counter() - t0) * 1000.0

            # 7. Perceptual Color Vibrance (Oklab)
            if cfg.vibrance_boost != 1.0:
                report("Perceptual Color Vibrance (Oklab)", 0.90)
                t0 = time.perf_counter()
                cur = oklab_vibrance(cur, boost=cfg.vibrance_boost)
                stage_latencies["oklab_vibrance"] = (time.perf_counter() - t0) * 1000.0

            # 8. White Balance (Bradford CAT16)
            if cfg.color_temperature != 0:
                report("White Balance (CAT16)", 0.96)
                t0 = time.perf_counter()
                cur = bradford_cat16_white_balance(cur, temperature_offset=cfg.color_temperature)
                stage_latencies["cat16_wb"] = (time.perf_counter() - t0) * 1000.0

        if is_grayscale:
            cur = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)

        if alpha is not None and alpha.shape[:2] != cur.shape[:2]:
            alpha = cv2.resize(alpha, (cur.shape[1], cur.shape[0]), interpolation=cv2.INTER_LANCZOS4)

        report("Complete", 1.0)
        total_latency = (time.perf_counter() - t_start) * 1000.0

        return ProcessingResult(
            image=cur,
            alpha=alpha,
            latency_ms=total_latency,
            backend_name=active_backend,
            mode_name=mode.value,
            faces_detected=faces_detected,
            diagnostics=diag_result,
            semantic_breakdown=semantic_breakdown,
            stage_latencies=stage_latencies,
        )
