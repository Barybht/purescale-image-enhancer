"""Command Line Interface (CLI) for PureScale 4.0 with Autonomous Diagnostics."""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image, ImageOps

from purescale import __version__
from purescale.config import (
    DeviceTarget,
    PipelineConfig,
    ProcessingMode,
    get_preset_config,
)
from purescale.dsp.diagnostics import diagnose_image
from purescale.pipeline import PureScalePipeline
from purescale.quality import compare_images

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


def load_image_with_alpha(path: str, return_meta: bool = False):
    """Loads image, handles EXIF orientation tag 274, and separates alpha channel."""
    pil_img = Image.open(path)
    exif_bytes = pil_img.info.get("exif")
    icc_profile = pil_img.info.get("icc_profile")

    pil_img = ImageOps.exif_transpose(pil_img)
    arr = np.array(pil_img)

    alpha = None
    if arr.ndim == 2:
        bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.shape[2] == 4:
        alpha = arr[:, :, 3]
        bgr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)
    else:
        bgr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)

    if return_meta:
        return bgr, alpha, exif_bytes, icc_profile
    return bgr, alpha


def save_image_with_alpha(
    bgr: np.ndarray,
    alpha: Optional[np.ndarray],
    out_path: str,
    format_name: str,
    exif: Optional[bytes] = None,
    icc_profile: Optional[bytes] = None,
) -> None:
    """Saves enhanced image in specified format with alpha, EXIF, and ICC profile if present."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fmt = format_name.upper()

    if alpha is not None and fmt in ("PNG", "WEBP"):
        rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
        rgba[:, :, 3] = alpha
        pil_out = Image.fromarray(rgba)
    else:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_out = Image.fromarray(rgb)

    save_kwargs = {}
    if exif:
        save_kwargs["exif"] = exif
    if icc_profile:
        save_kwargs["icc_profile"] = icc_profile

    if fmt == "PNG":
        pil_out.save(out_path, format="PNG", compress_level=6, **save_kwargs)
    elif fmt in ("JPEG", "JPG"):
        if pil_out.mode != "RGB":
            pil_out = pil_out.convert("RGB")
        pil_out.save(out_path, format="JPEG", quality=95, subsampling=0, **save_kwargs)
    elif fmt == "WEBP":
        pil_out.save(out_path, format="WEBP", quality=95, method=6, **save_kwargs)
    else:
        pil_out.save(out_path, **save_kwargs)


def process_file(pipeline: PureScalePipeline, cfg: PipelineConfig, in_path: str, out_path: str,
                 quiet: bool = False, reference: Optional[np.ndarray] = None) -> Optional[Dict[str, Any]]:
    """Processes a single file. Returns result dict on success, None on failure.

    When ``reference`` (uint8 BGR) is supplied, PSNR/SSIM vs the enhanced
    output are added under ``"quality"`` if shapes match, otherwise a
    ``"quality_error"`` string is recorded instead of failing the run.
    """
    try:
        bgr, alpha, exif, icc = load_image_with_alpha(in_path, return_meta=True)
        h, w = bgr.shape[:2]

        res = pipeline.enhance(bgr, config=cfg, alpha=alpha)

        out_h, out_w = res.image.shape[:2]
        save_image_with_alpha(res.image, res.alpha, out_path, cfg.output_format, exif=exif, icc_profile=icc)

        info: Dict[str, Any] = {
            "input": in_path,
            "output": out_path,
            "input_resolution": [w, h],
            "output_resolution": [out_w, out_h],
            "scale": cfg.scale,
            "mode": res.mode_name,
            "backend": res.backend_name,
            "latency_ms": round(res.latency_ms, 2),
            "faces_detected": res.faces_detected,
            "stage_latencies_ms": {k: round(v, 2) for k, v in res.stage_latencies.items()},
        }
        if res.diagnostics:
            info["diagnostics"] = {
                "noise_sigma": res.diagnostics.noise_sigma,
                "noise_category": res.diagnostics.noise_category,
                "blur_score": res.diagnostics.blur_score,
                "blur_category": res.diagnostics.blur_category,
                "haze_index": res.diagnostics.haze_index,
                "haze_detected": res.diagnostics.haze_detected,
            }
        if reference is not None:
            if reference.shape == res.image.shape:
                info["quality"] = compare_images(reference, res.image)
            else:
                info["quality_error"] = (
                    f"Reference shape {reference.shape} != output shape {res.image.shape}"
                )
        if not quiet:
            print(f"[SUCCESS] {os.path.basename(in_path)} -> {os.path.basename(out_path)}")
            print(f"  Resolution: {w}x{h} -> {out_w}x{out_h} ({cfg.scale}x)")
            print(f"  Mode:       {res.mode_name.upper()}")
            print(f"  Backend:    {res.backend_name}")
            print(f"  Latency:    {res.latency_ms:.2f} ms")
            if res.diagnostics:
                print(f"  Diagnostics: Noise Sigma {res.diagnostics.noise_sigma} [{res.diagnostics.noise_category}], Blur {res.diagnostics.blur_score:.2f} [{res.diagnostics.blur_category}], Haze {res.diagnostics.haze_index:.2f}")
            if res.faces_detected > 0:
                print(f"  Faces:      {res.faces_detected} detected & retouched")
            if "quality" in info:
                q = info["quality"]
                print(f"  Quality:    PSNR {q['psnr_db']:.2f} dB, SSIM {q['ssim']:.4f}")
            elif "quality_error" in info:
                print(f"  Quality:    skipped ({info['quality_error']})")
        return info
    except (OSError, IOError, ValueError, cv2.error, RuntimeError, MemoryError) as err:
        print(f"[ERROR] Failed to process {in_path}: {err}", file=sys.stderr)
        return None


def main_compare(argv: List[str]) -> int:
    """Compares two images with PSNR/SSIM. Usage: compare BEFORE AFTER [--json]."""
    parser = argparse.ArgumentParser(
        prog="enhance_image.py compare",
        description="Compare two images with PSNR and SSIM (alpha channels ignored)",
    )
    parser.add_argument("before", help="Path to reference image")
    parser.add_argument("after", help="Path to image to compare against reference")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parsed = parser.parse_args(argv)

    for label, path in (("before", parsed.before), ("after", parsed.after)):
        if not os.path.isfile(path):
            print(f"[ERROR] {label} image does not exist: {path}", file=sys.stderr)
            return 1
    try:
        ref, _ = load_image_with_alpha(parsed.before)
        out, _ = load_image_with_alpha(parsed.after)
        metrics = compare_images(ref, out)
    except (OSError, ValueError, cv2.error) as err:
        print(f"[ERROR] Comparison failed: {err}", file=sys.stderr)
        return 1

    if parsed.json:
        print(json.dumps({"before": parsed.before, "after": parsed.after, **metrics}))
    else:
        print(f"PSNR: {metrics['psnr_db']:.2f} dB")
        print(f"SSIM: {metrics['ssim']:.4f}")
    return 0


def main(args: List[str] = None) -> int:
    """CLI entrypoint."""
    if args and len(args) > 0 and args[0] == "compare":
        return main_compare(args[1:])
    parser = argparse.ArgumentParser(
        description="PureScale 4.0: Autonomous Multiscale Computational Vision & Edge AI Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("input", help="Path to input image file or directory of images")
    parser.add_argument("-o", "--output", help="Path to output file or destination directory", default=None)

    # Autonomous Diagnostics
    parser.add_argument("--auto", action="store_true", help="Enable autonomous diagnostic auto-tuning")
    parser.add_argument("--diagnostics", action="store_true", help="Print signal diagnostics table without processing")

    # Mode & Hardware Target
    parser.add_argument(
        "--mode",
        choices=["puredsp", "neural", "hybrid"],
        default="puredsp",
        help="Execution mode: puredsp (deterministic DSP), neural (AI super-res), hybrid (AI + Multiscale DSP)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "directml", "cpu", "opencv"],
        default="auto",
        help="Hardware acceleration backend for neural model",
    )

    # Preset Selection
    parser.add_argument(
        "--preset",
        "-p",
        choices=["balanced", "portrait", "landscape", "low-light", "art", "fast"],
        default=None,
        help="Load empirically tuned parameter profile (fast: skips diagnostics/semantic/pyramid/SWF for speed)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Shortcut for --preset fast (disables diagnostics, semantic guidance, pyramid, and SWF denoising)",
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress per-file progress output (errors still go to stderr)")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of human-readable text")
    parser.add_argument("-V", "--version", action="version", version=f"PureScale {__version__}")

    # Advanced CV Parameters (default None = keep preset value; explicit flag wins)
    parser.add_argument("--dehaze", type=float, default=None, help="Dark Channel Prior atmospheric dehazing strength (0.0 to 1.0)")
    parser.add_argument("--no-pyramid", action="store_true", help="Disable Multiscale Local Laplacian Pyramid filtering")
    parser.add_argument("--pyramid-detail", type=float, default=None, help="Multiscale micro-texture detail gain (L1 band: 0.5 to 2.0)")
    parser.add_argument("--pyramid-structure", type=float, default=None, help="Multiscale structural contour gain (L2 band: 0.8 to 1.8)")
    parser.add_argument("--pyramid-dynamic-range", type=float, default=None, help="Base-illumination dynamic range compression G3 (0.0 to 1.0)")
    parser.add_argument("--no-semantic", action="store_true", help="Disable semantic region parsing and guidance")

    # Core Tuning Parameters (default None = keep preset value; explicit flag wins)
    parser.add_argument("--scale", type=float, default=None, help="Resolution scaling factor")
    parser.add_argument("--sharpen", type=float, default=None, help="Contrast-Adaptive Sharpening strength (CAS)")
    parser.add_argument("--no-denoise", action="store_true", help="Disable Side Window Filter denoising")
    parser.add_argument("--denoise-intensity", type=int, default=None, help="Side Window Filter cleaning power (10-100)")
    parser.add_argument("--no-contrast", action="store_true", help="Disable BIMEF dynamic range fusion")
    parser.add_argument("--contrast-boost", type=float, default=None, help="BIMEF exposure curve factor (1.0-4.0)")
    parser.add_argument("--brightness", type=int, default=None, help="Radiometric brightness shift (-50 to +50)")
    parser.add_argument("--vibrance", type=float, default=None, help="Oklab perceptual vibrance boost (1.0-1.5)")
    parser.add_argument("--temperature", type=int, default=None, help="Bradford CAT16 color temperature shift (-30 to +30)")
    parser.add_argument("--depixel", type=int, default=None, help="Directional subpixel de-pixelation & deblocking strength (0-100)")
    parser.add_argument("--deblur", type=int, default=None, help="Structure tensor shock deblur strength (0-100)")
    parser.add_argument("--portrait-smooth", type=int, default=None, help="Fast Guided Filter skin smoothing (0-100)")
    parser.add_argument("--eye-clarity", type=float, default=None, help="Corneal catchlight & iris sharpness (1.0-2.0)")
    parser.add_argument("--tile-size", type=int, default=None, help="Tile dimension for neural super-resolution (>= 32)")
    parser.add_argument("--tile-overlap", type=int, default=None, help="Tile overlap border margin (>= 0 and < tile-size)")
    parser.add_argument("--max-megapixels", type=float, default=None, help="Maximum image megapixels allowed before OOM guard aborts (default: 40.0)")
    parser.add_argument("--reference", default=None, help="Path to ground-truth image for PSNR/SSIM quality scoring (shapes must match output)")
    parser.add_argument("-r", "--recursive", action="store_true", help="Recursively process subdirectories in input directory")
    parser.add_argument("--format", choices=["PNG", "JPEG", "WebP"], default="PNG", help="Output container format")

    parsed = parser.parse_args(args)

    in_path = os.path.abspath(parsed.input)
    if not os.path.exists(in_path):
        print(f"[ERROR] Input path does not exist: {in_path}", file=sys.stderr)
        return 1

    # Diagnostics inspection mode
    if parsed.diagnostics and os.path.isfile(in_path):
        bgr, _ = load_image_with_alpha(in_path)
        diag = diagnose_image(bgr)
        if parsed.json:
            print(json.dumps({
                "input": in_path,
                "noise_sigma": diag.noise_sigma,
                "noise_category": diag.noise_category,
                "blur_score": diag.blur_score,
                "blur_category": diag.blur_category,
                "entropy": diag.entropy,
                "dynamic_range": diag.dynamic_range,
                "shadow_clipping": diag.shadow_clipping,
                "highlight_clipping": diag.highlight_clipping,
                "mean_luminance": diag.mean_luminance,
                "color_cast_kelvin": diag.color_cast_kelvin,
                "color_cast_name": diag.color_cast_name,
                "haze_index": diag.haze_index,
                "haze_detected": diag.haze_detected,
                "semantic_breakdown": diag.semantic_breakdown,
            }))
        else:
            print(diag.summary_table())
        return 0

    # Build Configuration: preset first, explicit flags win (None = keep preset).
    preset_name = "fast" if parsed.fast else parsed.preset
    cfg = PipelineConfig()
    if preset_name:
        cfg = get_preset_config(preset_name, cfg)

    # Apply Explicit Flags (only when the user passed them)
    cfg.mode = ProcessingMode(parsed.mode)
    cfg.device = DeviceTarget(parsed.device)
    cfg.auto_tune = parsed.auto
    if parsed.auto:
        # Auto-tuning needs diagnostics to compute recommendations.
        cfg.enable_diagnostics = True
    if parsed.dehaze is not None:
        cfg.enable_dehaze = parsed.dehaze > 0.0
        cfg.dehaze_strength = parsed.dehaze if parsed.dehaze > 0.0 else 0.50
    if parsed.no_pyramid:
        cfg.enable_pyramid = False
    if parsed.pyramid_detail is not None:
        cfg.pyramid_micro_texture = parsed.pyramid_detail
    if parsed.pyramid_structure is not None:
        cfg.pyramid_structure_boost = parsed.pyramid_structure
    if parsed.pyramid_dynamic_range is not None:
        cfg.pyramid_dynamic_range = parsed.pyramid_dynamic_range
    if parsed.no_semantic:
        cfg.enable_semantic_guidance = False
    if parsed.scale is not None:
        cfg.scale = parsed.scale
    if parsed.sharpen is not None:
        cfg.sharpen_strength = parsed.sharpen
    if parsed.no_denoise:
        cfg.enable_denoise = False
    if parsed.denoise_intensity is not None:
        cfg.denoise_intensity = parsed.denoise_intensity
    if parsed.no_contrast:
        cfg.enable_contrast = False
    if parsed.contrast_boost is not None:
        cfg.contrast_boost = parsed.contrast_boost
    if parsed.brightness is not None:
        cfg.brightness_shift = parsed.brightness
    if parsed.vibrance is not None:
        cfg.vibrance_boost = parsed.vibrance
    if parsed.temperature is not None:
        cfg.color_temperature = parsed.temperature
    if parsed.depixel is not None:
        cfg.depixel_strength = parsed.depixel
    if parsed.deblur is not None:
        cfg.deblur_strength = parsed.deblur
    if parsed.portrait_smooth is not None:
        cfg.portrait_smooth = parsed.portrait_smooth
    if parsed.eye_clarity is not None:
        cfg.eye_clarity = parsed.eye_clarity
    if parsed.tile_size is not None:
        cfg.tile_size = parsed.tile_size
    if parsed.tile_overlap is not None:
        cfg.tile_overlap = parsed.tile_overlap
    if parsed.max_megapixels is not None:
        cfg.max_megapixels = parsed.max_megapixels
    cfg.output_format = parsed.format

    quiet = parsed.quiet or parsed.json

    reference_bgr: Optional[np.ndarray] = None
    if parsed.reference is not None:
        if not os.path.isfile(parsed.reference):
            print(f"[ERROR] Reference image does not exist: {parsed.reference}", file=sys.stderr)
            return 1
        try:
            reference_bgr, _ = load_image_with_alpha(parsed.reference)
        except (OSError, ValueError, cv2.error) as err:
            print(f"[ERROR] Could not load reference image: {err}", file=sys.stderr)
            return 1

    pipeline = PureScalePipeline(config=cfg)

    ext_map = {"PNG": ".png", "JPEG": ".jpg", "WebP": ".webp"}
    target_ext = ext_map.get(cfg.output_format, ".png")

    if os.path.isfile(in_path):
        if parsed.output:
            out_path = os.path.abspath(parsed.output)
            if os.path.isdir(out_path):
                base_name = os.path.splitext(os.path.basename(in_path))[0]
                out_path = os.path.join(out_path, f"{base_name}_enhanced{target_ext}")
        else:
            base_name = os.path.splitext(os.path.basename(in_path))[0]
            out_path = os.path.join(os.path.dirname(in_path), f"{base_name}_enhanced{target_ext}")

        result = process_file(pipeline, cfg, in_path, out_path, quiet=quiet, reference=reference_bgr)
        if parsed.json:
            print(json.dumps(result if result is not None else {"input": in_path, "error": "failed"}))
        return 0 if result is not None else 1

    elif os.path.isdir(in_path):
        out_dir = os.path.abspath(parsed.output) if parsed.output else os.path.join(in_path, "enhanced")
        os.makedirs(out_dir, exist_ok=True)

        rel_files = []
        if parsed.recursive:
            for root, _, filenames in os.walk(in_path):
                for f in sorted(filenames):
                    if os.path.splitext(f.lower())[1] in SUPPORTED_EXTENSIONS:
                        rel_path = os.path.relpath(os.path.join(root, f), in_path)
                        rel_files.append(rel_path)
        else:
            rel_files = sorted([
                f for f in os.listdir(in_path)
                if os.path.splitext(f.lower())[1] in SUPPORTED_EXTENSIONS
            ])

        if not rel_files:
            print(f"[WARNING] No supported image files found in directory: {in_path}")
            return 0

        if not quiet:
            print(f"Batch processing {len(rel_files)} images -> {out_dir}")
        t_batch_start = time.perf_counter()
        successes = 0
        failures = 0
        results = []

        for idx, rel_file in enumerate(rel_files, 1):
            src_file = os.path.join(in_path, rel_file)
            base_rel = os.path.splitext(rel_file)[0]
            dst_file = os.path.join(out_dir, f"{base_rel}_enhanced{target_ext}")
            os.makedirs(os.path.dirname(dst_file), exist_ok=True)
            if not quiet:
                print(f"\n[{idx}/{len(rel_files)}] Processing {rel_file}...")
            info = process_file(pipeline, cfg, src_file, dst_file, quiet=quiet, reference=reference_bgr)
            if info is not None:
                successes += 1
                results.append(info)
            else:
                failures += 1
                results.append({"input": src_file, "error": "failed"})

        total_batch_time = time.perf_counter() - t_batch_start
        if parsed.json:
            print(json.dumps({
                "output_dir": out_dir,
                "succeeded": successes,
                "failed": failures,
                "total_time_s": round(total_batch_time, 2),
                "results": results,
            }))
        else:
            print(f"\n[COMPLETE] Batch finished in {total_batch_time:.2f} s: {successes} succeeded, {failures} failed ({total_batch_time/len(rel_files)*1000:.1f} ms/image average)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
