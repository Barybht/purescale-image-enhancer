"""
PureScale: Simple Non-AI Image Enhancer (Pure Computer Vision).
Uses OpenCV and NumPy - no PyTorch, no AI models, zero GPU required.
Runs in milliseconds on any CPU.
"""

import argparse
import os
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image, ImageOps


def enhance_image(
    image_input: Union[str, Path, np.ndarray],
    scale: float = 2.0,
    sharpen_strength: float = 1.2,
    sharpen_radius: float = 2.5,
    enable_denoise: bool = True,
    denoise_intensity: int = 50,
    enable_clahe: bool = True,
    contrast_boost: float = 2.0,
    brightness_shift: int = 0,
    vibrance_boost: float = 1.1,
    color_temperature: int = 0,
) -> np.ndarray:
    """
    Enhances image quality using classical computer vision techniques.

    Args:
        image_input: Filepath or BGR numpy array.
        scale: Resolution multiplier (1.0 = same size, 2.0 = 2x bigger, 3.0 = 3x bigger).
        sharpen_strength: Makes blurry edges crisp (0.0 = off, 1.0-1.4 = natural, 2.0+ = strong).
        sharpen_radius: Detail size to sharpen (1.0-2.0 = fine hairs/eyelashes, 3.0-5.0 = broad outlines).
        enable_denoise: Whether to clean grainy noise and compression artifacts.
        denoise_intensity: Cleaning power (20-40 = light grain preservation, 50-60 = balanced, 80+ = heavy smoothing).
        enable_clahe: Balances shadows and highlights locally (HDR effect).
        contrast_boost: How strongly shadows and highlights get lifted (1.0 = subtle, 2.0 = balanced, 3.0+ = dramatic).
        brightness_shift: Exposure adjustment (-30 to -10 = dimmer, 0 = untouched, +10 to +30 = brighter).
        vibrance_boost: Color saturation boost (1.0 = original, 1.1 = 10% fresher colors).
        color_temperature: Color warmth (-15 to -5 = cool blue tone, 0 = neutral, +5 to +15 = warm golden sun).

    Returns:
        Enhanced BGR image as a numpy array.
    """
    # 1. Load image and correct smartphone rotation
    if isinstance(image_input, (str, Path)):
        pil_img = Image.open(image_input)
        pil_img = ImageOps.exif_transpose(pil_img)
        img_np = np.array(pil_img)
        if img_np.ndim == 2:
            img = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        elif img_np.shape[2] == 4:
            img = cv2.cvtColor(img_np[:, :, :3], cv2.COLOR_RGB2BGR)
        else:
            img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    else:
        img = image_input.copy()

    h, w = img.shape[:2]

    # 2. Edge-Preserving Denoising (Bilateral Filter)
    if enable_denoise and denoise_intensity > 0:
        sigma = float(denoise_intensity)
        img = cv2.bilateralFilter(img, d=7, sigmaColor=sigma, sigmaSpace=sigma)

    # 3. High-Quality Lanczos-4 Upscaling
    if scale != 1.0:
        target_w = int(round(w * scale))
        target_h = int(round(h * scale))
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    # 4. Adaptive Local Contrast (CLAHE on L-channel in LAB space)
    if enable_clahe and contrast_boost > 0:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=float(contrast_boost), tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)
        img = cv2.cvtColor(cv2.merge((l_enhanced, a_channel, b_channel)), cv2.COLOR_LAB2BGR)

    # 5. Brightness Shift
    if brightness_shift != 0:
        img = np.clip(img.astype(np.int16) + brightness_shift, 0, 255).astype(np.uint8)

    # 6. Unsharp Masking (Frequency Sharpening)
    if sharpen_strength > 0:
        blur = cv2.GaussianBlur(img, (0, 0), sigmaX=float(sharpen_radius))
        unsharp = cv2.addWeighted(img, 1.0 + sharpen_strength, blur, -sharpen_strength, 0)
        img = np.clip(unsharp, 0, 255).astype(np.uint8)

    # 7. Subtle Vibrance / Color Saturation Boost
    if vibrance_boost != 1.0:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * vibrance_boost, 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 8. Color Temperature (Cool Blue vs Warm Golden)
    if color_temperature != 0:
        b_ch, g_ch, r_ch = cv2.split(img.astype(np.float32))
        if color_temperature > 0:
            r_ch = np.clip(r_ch + color_temperature, 0, 255)
            b_ch = np.clip(b_ch - color_temperature * 0.5, 0, 255)
        else:
            b_ch = np.clip(b_ch - color_temperature, 0, 255)
            r_ch = np.clip(r_ch + color_temperature * 0.5, 0, 255)
        img = cv2.merge((b_ch, g_ch, r_ch)).astype(np.uint8)

    return img


def main():
    parser = argparse.ArgumentParser(description="PureScale: Classical Image Enhancer (Pure OpenCV)")
    parser.add_argument("input", type=str, help="Path to input image or directory")
    parser.add_argument("--output", "-o", type=str, default="enhanced_output", help="Output file or directory")
    parser.add_argument("--scale", "-s", type=float, default=2.0, help="Upscale factor (1.0, 1.5, 2.0, 3.0, 4.0)")
    parser.add_argument("--sharpen", type=float, default=1.2, help="Sharpening strength (0.0 to 3.0, default 1.2)")
    parser.add_argument("--radius", type=float, default=2.5, help="Sharpen detail radius (1.0 to 6.0, default 2.5)")
    parser.add_argument("--no-denoise", action="store_true", help="Disable bilateral noise reduction")
    parser.add_argument("--denoise-intensity", type=int, default=50, help="Noise reduction power (10 to 100, default 50)")
    parser.add_argument("--no-contrast", action="store_true", help="Disable adaptive contrast (CLAHE)")
    parser.add_argument("--contrast-boost", type=float, default=2.0, help="Contrast boost level (1.0 to 4.0, default 2.0)")
    parser.add_argument("--brightness", type=int, default=0, help="Brightness shift (-50 to +50, default 0)")
    parser.add_argument("--vibrance", type=float, default=1.1, help="Color saturation multiplier (default 1.1)")
    parser.add_argument("--temperature", type=int, default=0, help="Warmth shift (-30 cool to +30 warm, default 0)")
    parser.add_argument("--format", type=str, default="PNG", choices=["PNG", "JPEG", "WebP"], help="Output format")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    ext = args.format.lower()

    if in_path.is_file():
        if out_path.is_dir() or not out_path.suffix:
            out_path.mkdir(parents=True, exist_ok=True)
            out_file = out_path / f"{in_path.stem}_enhanced_{args.scale}x.{ext}"
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_file = out_path

        enhanced = enhance_image(
            in_path,
            scale=args.scale,
            sharpen_strength=args.sharpen,
            sharpen_radius=args.radius,
            enable_denoise=not args.no_denoise,
            denoise_intensity=args.denoise_intensity,
            enable_clahe=not args.no_contrast,
            contrast_boost=args.contrast_boost,
            brightness_shift=args.brightness,
            vibrance_boost=args.vibrance,
            color_temperature=args.temperature,
        )
        
        # Save according to selected format
        pil_out = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
        if args.format == "JPEG":
            pil_out.save(out_file, format="JPEG", quality=95)
        elif args.format == "WebP":
            pil_out.save(out_file, format="WEBP", quality=95)
        else:
            pil_out.save(out_file, format="PNG", compress_level=3)

        print(f"Enhanced image saved to: {out_file.resolve()}")

    elif in_path.is_dir():
        out_path.mkdir(parents=True, exist_ok=True)
        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        files = [p for p in in_path.iterdir() if p.suffix.lower() in valid_exts]

        print(f"Processing {len(files)} images...")
        for p in files:
            enhanced = enhance_image(
                p,
                scale=args.scale,
                sharpen_strength=args.sharpen,
                sharpen_radius=args.radius,
                enable_denoise=not args.no_denoise,
                denoise_intensity=args.denoise_intensity,
                enable_clahe=not args.no_contrast,
                contrast_boost=args.contrast_boost,
                brightness_shift=args.brightness,
                vibrance_boost=args.vibrance,
                color_temperature=args.temperature,
            )
            out_file = out_path / f"{p.stem}_enhanced_{args.scale}x.{ext}"
            pil_out = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
            if args.format == "JPEG":
                pil_out.save(out_file, format="JPEG", quality=95)
            elif args.format == "WebP":
                pil_out.save(out_file, format="WEBP", quality=95)
            else:
                pil_out.save(out_file, format="PNG", compress_level=3)
            print(f"  {p.name} -> {out_file.name}")

        print(f"\nAll images enhanced and saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
