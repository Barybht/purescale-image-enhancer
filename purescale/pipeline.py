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
    AUTO_STYLE_CONFIDENCE,
    DeviceTarget,
    DiagnosticsResult,
    PipelineConfig,
    ProcessingMode,
    ProcessingResult,
    apply_style_profile,
)
from purescale.dsp.color import bradford_cat16_white_balance, oklab_vibrance
from purescale.dsp.contrast import bimef_exposure_fusion, contrast_adaptive_sharpen
from purescale.dsp.tone import local_tone_mapping
from purescale.dsp.dehaze import atmospheric_dehaze
from purescale.dsp.diagnostics import diagnose_image
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
        style: str = "photo",
    ) -> NeuralSuperResEngine:
        # Scale-aware and style-aware routing: a native x2 model serves target_scale <= 2
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
        model_key = resolve_sr_model(target_scale, style=style)
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

    # -------------------------------------------------------------
    # Shared stage helpers (single implementation used by PureDSP,
    # Neural AI, and Hybrid modes).
    # -------------------------------------------------------------
    @staticmethod
    def _swf_clean_skip(diag_result: Optional[DiagnosticsResult], cfg: "PipelineConfig") -> bool:
        """Clean-image skip: diagnosed sigma < 2.0 with default strength
        bypasses SWF. Matches auto_tune Clean -> enable_denoise=False."""
        return (
            diag_result is not None
            and diag_result.noise_sigma < 2.0
            and cfg.denoise_intensity <= 35
        )

    @staticmethod
    def _blend_with_denoise_map(
        cur: np.ndarray,
        denoised: np.ndarray,
        denoise_map: Optional[np.ndarray],
    ) -> np.ndarray:
        if denoise_map is None:
            return denoised
        alpha_den = np.clip(denoise_map / 1.5, 0.20, 1.0)[:, :, np.newaxis]
        return np.clip(
            np.rint(alpha_den * denoised.astype(np.float32) + (1.0 - alpha_den) * cur.astype(np.float32)),
            0.0,
            255.0,
        ).astype(np.uint8)

    @staticmethod
    def _scaled_detail_map(
        detail_map: Optional[np.ndarray],
        cur: np.ndarray,
    ) -> Optional[np.ndarray]:
        if detail_map is None:
            return None
        cur_h, cur_w = cur.shape[:2]
        return cv2.resize(detail_map, (cur_w, cur_h), interpolation=cv2.INTER_LINEAR)

    def _run_swf(
        self,
        cur: np.ndarray,
        cfg: "PipelineConfig",
        diag_result: Optional[DiagnosticsResult],
        denoise_map: Optional[np.ndarray],
        radius: int,
        iterations: int,
        stage_key: str,
        progress_label: str,
        progress_ratio: float,
        report: Callable[[str, float], None],
        stage_latencies: Dict[str, float],
    ) -> np.ndarray:
        report(progress_label, progress_ratio)
        t0 = time.perf_counter()
        denoised = side_window_filter(
            cur, radius=radius, iterations=iterations,
            noise_sigma=diag_result.noise_sigma if diag_result else None,
        )
        cur = self._blend_with_denoise_map(cur, denoised, denoise_map)
        stage_latencies[stage_key] = (time.perf_counter() - t0) * 1000.0
        return cur

    def _run_pyramid(
        self,
        cur: np.ndarray,
        cfg: "PipelineConfig",
        noise_damping: float,
        spatial_detail_mask: Optional[np.ndarray],
        dynamic_range_compression: float,
        progress_label: str,
        progress_ratio: float,
        report: Callable[[str, float], None],
        stage_latencies: Dict[str, float],
    ) -> np.ndarray:
        report(progress_label, progress_ratio)
        t0 = time.perf_counter()
        cur = multiscale_laplacian_filter(
            cur,
            micro_texture_gain=cfg.pyramid_micro_texture,
            structure_boost=cfg.pyramid_structure_boost,
            noise_damping=noise_damping,
            dynamic_range_compression=dynamic_range_compression,
            spatial_detail_mask=spatial_detail_mask,
        )
        stage_latencies["multiscale_laplacian"] = (time.perf_counter() - t0) * 1000.0
        return cur

    def _run_bimef(
        self,
        cur: np.ndarray,
        cfg: "PipelineConfig",
        progress_label: str,
        progress_ratio: float,
        report: Callable[[str, float], None],
        stage_latencies: Dict[str, float],
    ) -> np.ndarray:
        if not (cfg.enable_contrast and cfg.contrast_boost > 1.0):
            return cur
        report(progress_label, progress_ratio)
        t0 = time.perf_counter()
        cur = bimef_exposure_fusion(cur, contrast_boost=cfg.contrast_boost)
        stage_latencies["bimef_contrast"] = (time.perf_counter() - t0) * 1000.0
        return cur

    def _run_local_tone(
        self,
        cur: np.ndarray,
        cfg: "PipelineConfig",
        progress_label: str,
        progress_ratio: float,
        report: Callable[[str, float], None],
        stage_latencies: Dict[str, float],
    ) -> np.ndarray:
        if not cfg.enable_local_tone or (
            cfg.local_tone_strength <= 0.001
            and cfg.highlight_recovery <= 0.001
            and cfg.shadow_boost <= 0.001
        ):
            return cur
        report(progress_label, progress_ratio)
        t0 = time.perf_counter()
        cur = local_tone_mapping(
            cur,
            strength=cfg.local_tone_strength,
            highlight_recovery=cfg.highlight_recovery,
            shadow_boost=cfg.shadow_boost,
        )
        stage_latencies["local_tone"] = (time.perf_counter() - t0) * 1000.0
        return cur

    def _run_cas(
        self,
        cur: np.ndarray,
        strength: float,
        progress_label: str,
        progress_ratio: float,
        report: Callable[[str, float], None],
        stage_latencies: Dict[str, float],
    ) -> np.ndarray:
        if not strength > 0.0:
            return cur
        report(progress_label, progress_ratio)
        t0 = time.perf_counter()
        cur = contrast_adaptive_sharpen(cur, strength=strength)
        stage_latencies["cas_sharpen"] = (time.perf_counter() - t0) * 1000.0
        return cur

    def _run_portrait(
        self,
        cur: np.ndarray,
        cfg: "PipelineConfig",
        progress_label: str,
        progress_ratio: float,
        report: Callable[[str, float], None],
        stage_latencies: Dict[str, float],
    ) -> Tuple[np.ndarray, int]:
        if not (cfg.portrait_smooth > 0 or cfg.eye_clarity > 1.0):
            return cur, 0
        report(progress_label, progress_ratio)
        t0 = time.perf_counter()
        retoucher = self._get_portrait_retoucher()
        cur, faces_detected = retoucher.enhance(
            cur,
            portrait_smooth=cfg.portrait_smooth,
            eye_clarity=cfg.eye_clarity,
        )
        stage_latencies["portrait_retouch"] = (time.perf_counter() - t0) * 1000.0
        return cur, faces_detected

    def _run_oklab(
        self,
        cur: np.ndarray,
        cfg: "PipelineConfig",
        progress_label: str,
        progress_ratio: float,
        report: Callable[[str, float], None],
        stage_latencies: Dict[str, float],
    ) -> np.ndarray:
        if cfg.vibrance_boost == 1.0:
            return cur
        report(progress_label, progress_ratio)
        t0 = time.perf_counter()
        cur = oklab_vibrance(cur, boost=cfg.vibrance_boost)
        stage_latencies["oklab_vibrance"] = (time.perf_counter() - t0) * 1000.0
        return cur

    def _run_cat16(
        self,
        cur: np.ndarray,
        cfg: "PipelineConfig",
        progress_label: str,
        progress_ratio: float,
        report: Callable[[str, float], None],
        stage_latencies: Dict[str, float],
    ) -> np.ndarray:
        if cfg.color_temperature == 0:
            return cur
        report(progress_label, progress_ratio)
        t0 = time.perf_counter()
        cur = bradford_cat16_white_balance(cur, temperature_offset=cfg.color_temperature)
        stage_latencies["cat16_wb"] = (time.perf_counter() - t0) * 1000.0
        return cur

    def _run_neural_upscale(
        self,
        cur: np.ndarray,
        alpha: Optional[np.ndarray],
        cfg: "PipelineConfig",
        dest_w: int,
        dest_h: int,
        progress_label: str,
        progress_ratio: float,
        report: Callable[[str, float], None],
        stage_latencies: Dict[str, float],
    ) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
        report(progress_label, progress_ratio)
        t0 = time.perf_counter()
        engine = self._get_neural_engine(cfg.device, cfg.scale, style=cfg.sr_style)
        cur = engine.upscale(
            cur,
            target_scale=cfg.scale,
            tile_size=cfg.tile_size,
            tile_overlap=cfg.tile_overlap,
            fast_2x=cfg.enable_fast_2x,
        )
        if alpha is not None:
            alpha = cv2.resize(alpha, (dest_w, dest_h), interpolation=cv2.INTER_LANCZOS4)
        stage_latencies["neural_superres"] = (time.perf_counter() - t0) * 1000.0
        return cur, alpha, engine.backend_name

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
        is_grayscale = (cur.ndim == 2) or (cur.ndim == 3 and cur.shape[2] == 1)
        if is_grayscale:
            if cur.ndim == 3:
                cur = cur[:, :, 0]
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

        if cfg.enable_diagnostics or cfg.auto_tune or cfg.auto_style:
            report("Autonomous Signal Diagnostics", 0.02)
            t0 = time.perf_counter()
            diag_result = diagnose_image(cur)
            stage_latencies["diagnostics"] = (time.perf_counter() - t0) * 1000.0

            if cfg.auto_tune and diag_result.recommended_parameters:
                for k, v in diag_result.recommended_parameters.items():
                    if hasattr(cfg, k):
                        setattr(cfg, k, v)

        # Style DSP profile: fills fields still at library defaults from the
        # resolved style profile (explicit style_profile > sr_style). Runs
        # after auto-tune and auto-style so diagnosed/routed values win.
        if cfg.auto_style and diag_result is not None:
            if diag_result.style_confidence >= AUTO_STYLE_CONFIDENCE:
                suggested = diag_result.suggested_style
                if suggested in ("photo", "anime"):
                    # Weights exist for these: re-route only the default pin.
                    if cfg.sr_style == "photo":
                        cfg.sr_style = suggested
                elif suggested == "manga" and cfg.style_profile is None:
                    # No manga weights exist: DSP profile only, weights stay.
                    cfg.style_profile = "manga"
        cfg = apply_style_profile(cfg)

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
            if cfg.enable_denoise and cfg.denoise_intensity > 0 and not self._swf_clean_skip(diag_result, cfg):
                r = max(1, int(round(cfg.denoise_intensity / 25.0)))
                iters = 1 if cfg.denoise_intensity <= 60 else 2
                cur = self._run_swf(
                    cur, cfg, diag_result, denoise_map, radius=r, iterations=iters,
                    stage_key="swf_denoise", progress_label="Structural Denoising (SWF)",
                    progress_ratio=0.18, report=report, stage_latencies=stage_latencies,
                )

            # 2. Spatial Scaling (EASU)
            if cfg.scale != 1.0:
                report("Edge-Adaptive Spatial Upsampling (EASU)", 0.32)
                t0 = time.perf_counter()
                cur = edge_adaptive_upsample(cur, scale=cfg.scale)
                if alpha is not None:
                    alpha = cv2.resize(alpha, (dest_w, dest_h), interpolation=cv2.INTER_LANCZOS4)
                stage_latencies["easu_upsample"] = (time.perf_counter() - t0) * 1000.0

            # Scale detail guidance map to match new resolution if needed
            scaled_detail_map = self._scaled_detail_map(detail_map, cur)

            # 3. Multiscale Local Laplacian Pyramid Filtering
            if cfg.enable_pyramid:
                cur = self._run_pyramid(
                    cur, cfg,
                    noise_damping=0.15 if (diag_result and diag_result.noise_sigma > 4.0) else 0.0,
                    spatial_detail_mask=scaled_detail_map,
                    dynamic_range_compression=cfg.pyramid_dynamic_range,
                    progress_label="Multiscale Local Laplacian Pyramid",
                    progress_ratio=0.48, report=report, stage_latencies=stage_latencies,
                )

            # 4. Dynamic Range Fusion (BIMEF with Black-Point Pinning)
            cur = self._run_bimef(
                cur, cfg, progress_label="Dynamic Range Fusion (BIMEF)",
                progress_ratio=0.60, report=report, stage_latencies=stage_latencies,
            )

            # 4.5 Local Tone-Mapping & Highlight Reconstruction
            cur = self._run_local_tone(
                cur, cfg, progress_label="Local Tone-Mapping & Highlight Reconstruction",
                progress_ratio=0.64, report=report, stage_latencies=stage_latencies,
            )

            # 5. Radiometric Exposure Offset
            if cfg.brightness_shift != 0:
                report("Exposure Offset", 0.68)
                t0 = time.perf_counter()
                cur = np.clip(cur.astype(np.int16) + cfg.brightness_shift, 0, 255).astype(np.uint8)
                stage_latencies["exposure_offset"] = (time.perf_counter() - t0) * 1000.0

            # 6. Detail Clarity (CAS)
            cur = self._run_cas(
                cur, cfg.sharpen_strength,
                progress_label="Contrast-Adaptive Sharpening (CAS)",
                progress_ratio=0.76, report=report, stage_latencies=stage_latencies,
            )

            # 7. Portrait Retouching (YuNet + FGF)
            cur, faces_detected = self._run_portrait(
                cur, cfg, progress_label="Portrait Retouching (YuNet + FGF)",
                progress_ratio=0.84, report=report, stage_latencies=stage_latencies,
            )

            # 8. Perceptual Vibrance (Oklab)
            cur = self._run_oklab(
                cur, cfg, progress_label="Perceptual Vibrance (Oklab)",
                progress_ratio=0.92, report=report, stage_latencies=stage_latencies,
            )

            # 9. White Balance (Bradford CAT16)
            cur = self._run_cat16(
                cur, cfg, progress_label="White Balance (CAT16)",
                progress_ratio=0.96, report=report, stage_latencies=stage_latencies,
            )

        # -------------------------------------------------------------
        # MODE 2: NEURAL AI (Compact Edge AI Super-Resolution)
        # -------------------------------------------------------------
        elif mode == ProcessingMode.NEURAL_AI:
            cur, alpha, active_backend = self._run_neural_upscale(
                cur, alpha, cfg, dest_w, dest_h,
                progress_label="Neural AI Super-Resolution (Real-ESRGAN Compact)",
                progress_ratio=0.25, report=report, stage_latencies=stage_latencies,
            )

            # Multiscale Laplacian refinement
            if cfg.enable_pyramid:
                cur = self._run_pyramid(
                    cur, cfg, noise_damping=0.0, spatial_detail_mask=None,
                    dynamic_range_compression=0.0,
                    progress_label="Multiscale Laplacian Refinement",
                    progress_ratio=0.70, report=report, stage_latencies=stage_latencies,
                )

            # Optional Portrait Retouching
            cur, faces_detected = self._run_portrait(
                cur, cfg, progress_label="Portrait Retouching (YuNet + FGF)",
                progress_ratio=0.85, report=report, stage_latencies=stage_latencies,
            )

        # -------------------------------------------------------------
        # MODE 3: HYBRID (Neural AI Edge Synthesis + Multiscale DSP)
        # -------------------------------------------------------------
        elif mode == ProcessingMode.HYBRID:
            # 1. Subtle Structural Denoise before neural pass (SWF)
            if cfg.enable_denoise and cfg.denoise_intensity > 0 and not self._swf_clean_skip(diag_result, cfg):
                r = max(1, int(round(cfg.denoise_intensity / 40.0)))
                cur = self._run_swf(
                    cur, cfg, diag_result, denoise_map, radius=r, iterations=1,
                    stage_key="swf_pre_denoise", progress_label="Pre-Denoising (SWF)",
                    progress_ratio=0.15, report=report, stage_latencies=stage_latencies,
                )

            # 2. Neural Edge Synthesis
            cur, alpha, active_backend = self._run_neural_upscale(
                cur, alpha, cfg, dest_w, dest_h,
                progress_label="Neural Edge Synthesis (Real-ESRGAN Compact)",
                progress_ratio=0.35, report=report, stage_latencies=stage_latencies,
            )

            scaled_detail_map = self._scaled_detail_map(detail_map, cur)

            # 3. Multiscale Local Laplacian Pyramid Filtering
            if cfg.enable_pyramid:
                cur = self._run_pyramid(
                    cur, cfg,
                    noise_damping=0.10 if (diag_result and diag_result.noise_sigma > 4.0) else 0.0,
                    spatial_detail_mask=scaled_detail_map,
                    dynamic_range_compression=0.0,
                    progress_label="Multiscale Local Laplacian Pyramid",
                    progress_ratio=0.55, report=report, stage_latencies=stage_latencies,
                )

            # 4. Dynamic Range Fusion (BIMEF with Black-Point Pinning)
            cur = self._run_bimef(
                cur, cfg, progress_label="Dynamic Range Fusion (BIMEF)",
                progress_ratio=0.65, report=report, stage_latencies=stage_latencies,
            )

            # 4.5 Local Tone-Mapping & Highlight Reconstruction
            cur = self._run_local_tone(
                cur, cfg, progress_label="Local Tone-Mapping & Highlight Reconstruction",
                progress_ratio=0.70, report=report, stage_latencies=stage_latencies,
            )

            # 5. Micro-Texture Clarity (CAS)
            cur = self._run_cas(
                cur, cfg.sharpen_strength * 0.7,
                progress_label="Micro-Texture Sharpening (CAS)",
                progress_ratio=0.75, report=report, stage_latencies=stage_latencies,
            )

            # 6. Portrait Retouching (YuNet + FGF)
            cur, faces_detected = self._run_portrait(
                cur, cfg, progress_label="Portrait Retouching (YuNet + FGF)",
                progress_ratio=0.83, report=report, stage_latencies=stage_latencies,
            )

            # 7. Perceptual Color Vibrance (Oklab)
            cur = self._run_oklab(
                cur, cfg, progress_label="Perceptual Color Vibrance (Oklab)",
                progress_ratio=0.90, report=report, stage_latencies=stage_latencies,
            )

            # 8. White Balance (Bradford CAT16)
            cur = self._run_cat16(
                cur, cfg, progress_label="White Balance (CAT16)",
                progress_ratio=0.96, report=report, stage_latencies=stage_latencies,
            )

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
