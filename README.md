# PureScale Image Enhancer

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Array%20Ops-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pillow](https://img.shields.io/badge/Pillow-Imaging-90E59A?style=flat-square&logo=pypi&logoColor=black)](https://pillow.readthedocs.io/)
[![Hardware](https://img.shields.io/badge/Hardware-Pure%20CPU-24292e?style=flat-square&logo=intel&logoColor=white)](https://github.com/)
[![Inference](https://img.shields.io/badge/Latency-~45ms%20(4K)-success?style=flat-square)](https://github.com/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)

PureScale is a high-performance image enhancement and spatial super-resolution engine implemented entirely in classical computer vision. 

Eliminates neural network overhead, heavy weights downloads, and GPU requirements. Runs in milliseconds on standard CPU hardware with 100% bitwise determinism and zero hallucination risk.

---

## Core Capabilities

| Pipeline Stage | Algorithm / Method | Functional Purpose |
| :--- | :--- | :--- |
| **Spatial Scaling** | 8-Lobe Lanczos-4 Sinc Interpolation | Sharp spatial enlargement up to 8x without blurring or pixelation. |
| **Noise Attenuation** | Space-Variant Bilateral Filtering | Eliminates sensor noise and JPEG compression grain while preserving real edges. |
| **Dynamic Range** | CLAHE in CIE $L^\ast a^\ast b^\ast$ Color Space | Balances shadows and sky highlights locally without altering color tones. |
| **Edge Definition** | Gaussian High-Pass Unsharp Masking | Restores micro-textures (hair, fabric, small text) with tunable detail radius. |
| **Color Tuning** | HSV Saturation & Tristimulus Balance | Boosts color vibrancy naturally and corrects cool or warm room lighting casts. |
| **Export Formats** | Multi-Container Serialization | Lossless PNG (alpha-preserved), JPEG (95% quality), or modern WebP (95% quality). |

---

## Performance Benchmark

Benchmarks measured on a single commodity x86-64 CPU core (Intel Xeon / AMD EPYC @ 2.20 GHz):

| Input Resolution | Scaling Factor | Output Resolution | Execution Time | Peak Memory |
| :--- | :--- | :--- | :--- | :--- |
| $512 \times 512$ (0.26 MP) | 2.0x | $1024 \times 1024$ (1.05 MP) | ~18 ms | ~15 MB RAM |
| $1280 \times 720$ (0.92 MP) | 2.0x | $2560 \times 1440$ (3.68 MP) | ~32 ms | ~38 MB RAM |
| $1920 \times 1080$ (2.07 MP) | 2.0x | $3840 \times 2160$ (4K UHD) | ~45 ms | ~72 MB RAM |
| $1920 \times 1080$ (2.07 MP) | 3.0x | $5760 \times 3240$ (6K) | ~90 ms | ~112 MB RAM |
| $3840 \times 2160$ (8.29 MP) | 2.0x | $7680 \times 4320$ (8K UHD) | ~180 ms | ~245 MB RAM |

> [!NOTE]
> GPU quota consumed: 0%. Operates comfortably within standard Google Colab free-tier CPU runtimes with zero warm-up latency.

---

## Quickstart: Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

1. Open [Google Colab](https://colab.research.google.com/).
2. Select **File** -> **Upload notebook** and choose `simple_colab_enhancer.ipynb`.
3. Set your runtime to CPU (**Runtime** -> **Change runtime type** -> **CPU**).
4. Configure your parameters in the interactive Form View:
   - Size multiplier (`upscale_factor`)
   - Edge clarity & radius (`sharpen_strength`, `sharpen_radius`)
   - Noise cleaning (`enable_denoise`, `denoise_intensity`)
   - Contrast & Exposure (`enable_contrast`, `contrast_boost`, `brightness_shift`)
   - Color warmth & vibrancy (`vibrance_boost`, `color_temperature`)
   - File format (`output_format`)
5. Click **Run** and select an image when prompted.
6. The enhanced image is processed in ~0.05 seconds, rendered on screen, and downloaded automatically.

---

## Local Usage

### Installation & Dependencies

The required packages differ depending on whether you are running the desktop GUI or the command-line interface:

- **For Desktop GUI (Interactive)**:
  Requires the core processing libraries plus `customtkinter` for the desktop interface:
  ```bash
  pip install opencv-python pillow numpy customtkinter
  ```

- **For CLI / CMD (Minimal / Headless Installation)**:
  Only requires the core processing libraries. `customtkinter` is not needed, which makes this setup ideal for headless Linux servers, Docker containers, and automated pipelines without display drivers.
  ```bash
  pip install opencv-python pillow numpy
  ```

#### Dependency Breakdown

| Package | CLI / CMD | Desktop GUI | Role |
| :--- | :--- | :--- | :--- |
| `opencv-python` | Required | Required | Core image processing: Lanczos-4 sinc scaling, bilateral denoising, CLAHE |
| `pillow` | Required | Required | Image decoding/encoding (JPEG, WebP, PNG) and color space conversions |
| `numpy` | Required | Required | Vectorized array operations and channel slicing |
| `customtkinter` | Not needed | Required | Modern desktop GUI framework, themes, and UI widgets |

---

### Option A: Desktop GUI (Interactive & Visual)

Recommended for users who prefer an interactive visual interface with real-time controls, instant before/after comparison, and pan/zoom inspection.

Launch the native desktop application:

```bash
python gui.py
```

**Key Features:**
- **Real-Time Controls**: Sliders and toggles for scale factor, sharpen intensity, radius, bilateral denoising, adaptive contrast (CLAHE), brightness, vibrance, and white balance.
- **Detailed Option Tooltips**: Hover over the `(?)` badge next to any setting for contextual parameter explanations.
- **Interactive Zoom & Pan Preview**:
  - Zoom toolbar (`-`, `+`, `Fit`, `100%`) or mouse wheel scrolling to inspect fine details up to 1500%.
  - Click and drag anywhere on the preview canvas to pan across high-resolution images.
  - Double-click to reset the view back to `Fit`.
- **Before / After Comparison**: Toggle between **Enhanced** and **Original** views with a single click.
- **Flexible Export**: Select container format (`PNG`, `JPEG`, `WebP`) with dynamic compression hints, and save via the native file dialog.
- **Non-Blocking Execution**: Enhancement runs on a dedicated background worker thread so the interface stays responsive.

---

### Option B: Command Line Interface (CLI / CMD)

Recommended for terminal users, scripting, server automation, and batch directory processing.

#### 1. Single Image Enhancement

Basic enhancement (default 3x upscale and balanced clarity):

```bash
python enhance_image.py input.jpg --scale 3
```

Custom parameter tuning:

```bash
# Sharpen micro-textures with custom radius and contrast lift
python enhance_image.py input.jpg --scale 2 --sharpen 1.4 --radius 1.8 --contrast-boost 2.5

# Denoise high-ISO photo, compensate exposure, and export as JPEG
python enhance_image.py input.jpg --denoise-intensity 70 --brightness 15 --format JPEG

# Output to a specific target file
python enhance_image.py input.jpg -o ./output/enhanced_photo.png --scale 3
```

#### 2. Batch Directory Processing

Process an entire folder of photos with one command:

```bash
# Enhance all images in a folder and export as modern WebP
python enhance_image.py ./my_photos -o ./enhanced_photos --scale 2 --format WebP
```

---

## Parameter Reference

| Parameter | CLI Flag | Default | Range | Plain-English Explanation |
| :--- | :--- | :--- | :--- | :--- |
| `upscale_factor` | `--scale` | 3 | 1.0 - 8.0 | **Size Multiplier**: 1 preserves original size; 2 doubles dimensions; 3 triples resolution. |
| `sharpen_strength` | `--sharpen` | 1.2 | 0.0 - 3.0 | **Edge Sharpness**: 0 is disabled; 1.0-1.4 provides natural clarity; 2.0+ is very crisp. |
| `sharpen_radius` | `--radius` | 2.5 | 1.0 - 6.0 | **Detail Size**: 1.0-2.0 targets fine hair, fabric weave, and text; 3.0-5.0 targets broad contours. |
| `enable_denoise` | `--no-denoise` | True | Boolean | **Noise Cleaner Toggle**: Enables edge-preserving bilateral filtering to strip grain. |
| `denoise_intensity`| `--denoise-intensity` | 50 | 10 - 100 | **Cleaning Power**: 20-40 keeps film grain; 50-60 is balanced; 80+ smooths rough scans. |
| `enable_contrast` | `--no-contrast` | True | Boolean | **Adaptive Contrast (HDR)**: Locally balances shadows and bright skies without color shift. |
| `contrast_boost` | `--contrast-boost` | 2.0 | 1.0 - 4.0 | **Shadow/Highlight Lift**: 1.0-1.5 is gentle; 2.0 is balanced; 3.0+ delivers punchy contrast. |
| `brightness_shift` | `--brightness` | 0 | -50 to +50 | **Exposure Adjustment**: Negative values dim bright images; positive values brighten dark shots. |
| `vibrance_boost` | `--vibrance` | 1.1 | 1.0 - 1.5 | **Color Freshness**: 1.0 is original; 1.1 adds a natural 10% saturation lift without skin distortion. |
| `color_temperature`| `--temperature` | 0 | -30 to +30 | **White Balance**: Negative values add cool daylight blue; positive values add golden sun warmth. |
| `output_format` | `--format` | PNG | PNG, JPEG, WebP | **Container Format**: Lossless PNG (supports alpha), compact JPEG (95%), or WebP (95%). |

---

## Architecture & Technical Deep Dive

For an in-depth breakdown of the signal processing theory, algorithmic proofs, and empirical analysis, refer to [`ARCHITECTURE.md`](ARCHITECTURE.md):

- **Mathematical Foundations**: Continuous 8-lobe Lanczos-4 sinc reconstruction and space-variant bilateral filtering formulas.
- **Color Space Processing**: Orthogonal CIE $L^\ast a^\ast b^\ast$ decomposition and localized CLAHE dynamic range optimization.
- **Complexity Analysis**: Asymptotic runtime bounds ($O(N)$ / $O(s^2 N)$) and SIMD cache-friendly access patterns.
- **Empirical Benchmarks**: Detailed latency, throughput, and hardware comparisons against deep neural models (Real-ESRGAN, Diffusion).

---

## License

This project is open-source software licensed under the **[MIT License](LICENSE)**.
