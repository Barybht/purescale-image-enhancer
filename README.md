# PureScale 4.0: Autonomous Multiscale Vision & Edge AI Engine

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)
[![DirectML](https://img.shields.io/badge/DirectML-GPU%20Accelerated-0078D4?style=flat-square&logo=windows&logoColor=white)](https://github.com/microsoft/DirectML)
[![NumPy](https://img.shields.io/badge/NumPy-Vectorized-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![Hardware](https://img.shields.io/badge/Hardware-DirectML%20GPU%20%2B%20AVX512%20CPU-24292e?style=flat-square&logo=amd&logoColor=white)](https://github.com/)
[![Latency](https://img.shields.io/badge/Latency-~2s%20(1080p%20PureDSP)-blue?style=flat-square)](BENCH.md)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)

PureScale 4.0 is an autonomous, multiscale computational vision and edge neural restoration engine. It unifies physical signal diagnostics, Dark Channel Prior atmospheric dehazing, 4-octave Local Laplacian pyramid filtering, soft semantic region guidance, and lightweight edge neural super-resolution (Real-ESRGAN Compact) with hardware acceleration across AMD Radeon GPUs and Zen CPUs.

---

## What's New in PureScale 4.0

1. **Autonomous Signal Quality Diagnostics**:
   - **Wavelet Donoho-Johnstone MAD Noise Estimator**: Measures sensor noise variance $\hat{\sigma}_{\text{noise}} = \text{median}(|HH_1|)/0.6745$ directly from 2D Haar wavelet high-frequency subbands.
   - **Laplacian Spectral Blur Index**: Quantifies optical defocus and motion blur without reference images.
   - **Shannon Dynamic Range Entropy**: Evaluates histogram distribution, shadow crushing, and highlight clipping.
   - **Shades-of-Gray ($L_p$-norm) Illuminant**: Estimates illuminant cast bias (Kelvin offset).
   - **Atmospheric Veiling Index**: Analyzes dark channel density to detect fog, haze, and smoke.
2. **One-Click Autonomous Auto-Tuner**:
   - Analyzes the input image signal and automatically synthesizes mathematically optimal parameters tailored specifically to its degradation characteristics.
3. **Multiscale Local Laplacian Pyramids**:
   - Decomposes imagery into 4 octave frequency bands ($L_0$ noise, $L_1$ micro-textures, $L_2$ structural contours, $G_3$ base illumination).
   - Non-linear transfer functions enhance fine textures without ringing or boundary halos.
4. **Dark Channel Prior (DCP) Atmospheric Dehazing**:
   - Recovers true scene radiance, atmospheric depth, and color contrast using Fast Guided Filter boundary refinement.
5. **Multi-Cue Semantic Region Guidance**:
   - Generates soft continuous masks for Sky, Foliage, Skin, Shadow, and Architecture to spatially modulate filtering (suppressing grain in sky/shadow while boosting micro-textures in foliage).
6. **Live Signal Diagnostics HUD in Obsidian Studio**:
   - Real-time telemetry card displaying noise floor, blur index, entropy, and haze index with one-click Auto-Enhance.

---

## Three Processing Engines

| Mode | Technology Stack | Hardware Target | Latency | Primary Advantage |
| :--- | :--- | :--- | :--- | :--- |
| **PureDSP** | EASU + Laplacian Pyramids + Dehaze + CAS + SWF + BIMEF + Oklab + CAT16 | AMD Zen 4 CPU (AVX-512) | **~0.7–5.0 s (resolution-dependent)** | 100% bitwise deterministic, zero neural weights, zero hallucination. |
| **Neural AI** | Real-ESRGAN General x4v3 (~4.87 MB ONNX) + Laplacian Pyramids | AMD Radeon 760M (DirectML GPU) | ~350-500 ms | Deep perceptual edge synthesis and compression artifact removal. |
| **Hybrid** | Neural Super-Resolution + Multiscale Pyramids + Pinned BIMEF + Oklab + CAS | DirectML GPU + Zen CPU | ~400-600 ms | Neural edge reconstruction with pure mathematical color science and halo-free micro-clarity. |

---

## Core Pipeline Stages

| Stage | Algorithm / Method | Mathematical Formulation | Technical Advantage |
| :--- | :--- | :--- | :--- |
| **Stage -1: Diagnostics** | Wavelet MAD & Spectral Analysis | $\hat{\sigma} = \text{median}(\|HH_1\|)/0.6745$, LoG variance | Physical degradation estimation; autonomous parameter auto-tuning. |
| **Stage -0.5: Semantics** | Multi-Cue Region Parsing | Chromatic opponent signatures + Fast Guided Filter | Soft continuous masks for Sky, Foliage, Skin, Shadow, and Structure. |
| **Stage 0: Conditioning** | Structure Tensor Shock Deblur & Anti-Aliasing | Second directional derivative $I_{\eta\eta}$ along isophotes | Reverses optical lens diffusion and removes pixel jaggies prior to scaling. |
| **Stage 1: Dehazing** | Dark Channel Prior (DCP) | $J(x) = (I(x) - A) / \max(t(x), t_0) + A$ with Guided Filter | Removes atmospheric fog and smoke; restores deep landscape contrast. |
| **Stage 2: Super-Res** | EASU (DSP) or Real-ESRGAN (AI) | Directional sinc reconstruction or 6-block CNN | Vector-sharp edges up to 4x scaling with overlap patch tiling (<300 MB RAM). |
| **Stage 3: Pyramids** | Multiscale Local Laplacian | 4-octave Gaussian/Laplacian band decomposition | Halo-free micro-texture ($L_1$) and structural contour ($L_2$) synthesis. |
| **Stage 4: Dynamic Range** | BIMEF with Black-Point Pinning | Anchored S-curve + detail clarity gating | Seamless midtone recovery with inky blacks and zero haloing. |
| **Stage 5: Edge Clarity** | Contrast-Adaptive Sharpening (CAS) | 3x3 local contrast min/max bound clamping | Halo-free micro-texture enhancement with zero edge overshoot. |
| **Stage 6: Perceptual Color**| Oklab Vibrance & Bradford CAT16 | Human cone LMS cubic-root space + Von Kries transform | Straight hue lines; eliminates blue-to-purple shifts and shadow noise. |
| **Stage 7: Portrait Retouch**| Fast Guided Filter (FGF) + YuNet | $O(N)$ subsampled local linear model + facial gating | Organic skin blemish softening preserving pores; targeted eye catchlights. |

---

## Performance Benchmark

PureDSP figures re-measured with `bench/bench.py` (balanced preset, deterministic
fixtures) on AMD64 CPU; historical neural figures below are DirectML-GPU
estimates pending re-measurement on reference hardware:

| Input Resolution | Scaling Factor | Output Resolution | PureDSP Mode | Hybrid Mode | Neural AI Mode |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 512 x 512 (0.26 MP) | 2.0x | 1024 x 1024 (1.05 MP) | **~0.7 s** | ~140 ms | ~120 ms |
| 1280 x 720 (0.92 MP) | 2.0x | 2560 x 1440 (3.68 MP) | **~2.3 s** | ~280 ms | ~240 ms |
| 1920 x 1080 (2.07 MP) | 2.0x | 3840 x 2160 (4K UHD) | **~5.0 s** | ~420 ms | ~360 ms |
| 3840 x 2160 (8.29 MP) | 2.0x | 7680 x 4320 (8K UHD) | **~20 s** | ~1,450 ms | ~1,250 ms |

Peak RAM footprint remains strictly bounded under **65 MB** for PureDSP and under **280 MB** for Neural/Hybrid modes via overlap tiling.
Comprehensive stage latency benchmarks are documented in [`BENCH.md`](BENCH.md).

---

## Local Usage

### Installation

```bash
# Clone the repository
git clone https://github.com/Barybht/purescale-image-enhancer.git
cd purescale-image-enhancer

# Minimal CLI dependencies (OpenCV, NumPy, Pillow only)
pip install -r requirements.txt

# Full desktop GUI dependencies (CustomTkinter + DirectML GPU)
pip install -r requirements-gui.txt

# Or install locally as a package with optional GUI support
pip install -e .[gui]
```

### Desktop GUI (Obsidian Studio)

Launch the modern desktop application:
```bash
python gui.py
```
- Click **Open Image** to load an image.
- Check the **Live Signal Diagnostics HUD** for physical signal metrics.
- Click **Auto-Enhance (CV Engine)** for autonomous zero-click optimization.
- Use the **Split View** slider to compare original and enhanced viewports side by side.

### Command Line Interface (CLI)

#### 1. Autonomous Auto-Tuning
Analyze signal degradation and automatically enhance with optimal settings:
```bash
python enhance_image.py input.jpg --auto
```

#### 2. Print Physical Signal Diagnostics
Inspect noise sigma, blur index, dynamic entropy, and haze without modifying the image:
```bash
python enhance_image.py input.jpg --diagnostics
```

#### 3. Atmospheric Dehazing & Multiscale Laplacian Detail
```bash
python enhance_image.py landscape.jpg --mode puredsp --dehaze 0.65 --pyramid-detail 1.35 --scale 2.0
```

#### 4. Recursive Batch Directory Processing
```bash
python enhance_image.py path/to/photos/ -o path/to/output/ --mode hybrid --preset landscape -r
```

### CLI Parameters Reference

| Parameter | CLI Flag | Default | Valid Range | Description |
| :--- | :--- | :--- | :--- | :--- |
| `auto_tune` | `--auto` | False | Boolean | **Autonomous Auto-Tuning**: Automatically configures parameters based on diagnostics. |
| `diagnostics` | `--diagnostics` | False | Boolean | **Signal Diagnostics**: Prints formatted ASCII telemetry table without processing. |
| `mode` | `--mode` | `puredsp` | `puredsp`, `neural`, `hybrid` | **Execution Engine**: Analytical DSP, Neural AI, or Hybrid mode. |
| `device` | `--device` | `auto` | `auto`, `directml`, `cpu`, `opencv` | **Hardware Accelerator**: DirectML GPU, CPU SIMD, or OpenCV DNN fallback. |
| `preset` | `-p`, `--preset` | None | `balanced`, `portrait`, `landscape`, `low-light`, `art` | **Parameter Profile**: Loads pre-tuned empirical parameter profiles. |
| `dehaze` | `--dehaze` | 0.0 | 0.0 - 1.0 | **Atmospheric Dehaze**: Dark Channel Prior (DCP) fog/haze removal strength. |
| `pyramid_detail` | `--pyramid-detail` | 1.20 | 0.5 - 2.0 | **Micro-Texture Gain**: Multiscale Local Laplacian octave band L1 detail boost. |
| `pyramid_structure`| `--pyramid-structure`| 1.10 | 0.8 - 1.8 | **Structural Gain**: Multiscale Local Laplacian octave band L2 contour boost. |
| `scale` | `--scale` | 2.0 | 0.5 - 4.0 | **Spatial Magnification**: Resolution scaling factor (0.5x to 4.0x). |
| `sharpen_strength` | `--sharpen` | 1.1 | 0.0 - 3.0 | **Detail Clarity**: Contrast-Adaptive Sharpening (CAS) halo-free gain. |
| `denoise_intensity`| `--denoise-intensity` | 40 | 10 - 100 | **Cleaning Power**: Side Window Filter (SWF) noise attenuation strength. |
| `contrast_boost` | `--contrast-boost` | 1.8 | 1.0 - 4.0 | **HDR Dynamic Range**: BIMEF exposure fusion tone curve factor. |
| `brightness` | `--brightness` | 0 | -50 to +50 | **Radiometric Exposure**: Additive brightness shift offset. |
| `vibrance` | `--vibrance` | 1.10 | 1.0 - 1.5 | **Perceptual Vibrance**: Color saturation multiplier in Oklab LMS cone space. |
| `temperature` | `--temperature` | 0 | -30 to +30 | **White Balance**: Bradford CAT16 chromatic adaptation offset. |
| `depixel` | `--depixel` | 0 | 0 - 100 | **Anti-Aliasing**: Directional subpixel de-pixelation & deblocking strength. |
| `deblur` | `--deblur` | 0 | 0 - 100 | **Shock Deblur**: Structure tensor morphological shock deblur strength. |
| `portrait_smooth` | `--portrait-smooth` | 35 | 0 - 100 | **Skin Softening**: Fast Guided Filter (FGF) skin smoothing intensity. |
| `eye_clarity` | `--eye-clarity` | 1.30 | 1.0 - 2.0 | **Eye Clarity**: Corneal catchlight and iris sharpness. |
| `tile_size` | `--tile-size` | 256 | >= 32 | **Neural Patch Dimension**: Spatial tile dimension for neural inference. |
| `tile_overlap` | `--tile-overlap` | 32 | 0 <= overlap < tile | **Patch Overlap**: Border overlap margin with raised-cosine feathering. |
| `max_megapixels` | `--max-megapixels` | 40.0 | > 0.0 | **Memory OOM Guard**: Maximum allowed megapixels (input/target) before aborting. |
| `recursive` | `-r`, `--recursive` | False | Boolean | **Recursive Processing**: Recursively process subdirectories in batch mode. |
| `no_pyramid` | `--no-pyramid` | False | Boolean | **Bypass Flag**: Disables Multiscale Local Laplacian Pyramid filtering. |
| `no_semantic` | `--no-semantic` | False | Boolean | **Bypass Flag**: Disables multi-cue semantic region parsing. |
| `no_denoise` | `--no-denoise` | False | Boolean | **Bypass Flag**: Disables Side Window Filter (SWF) denoising. |
| `no_contrast` | `--no-contrast` | False | Boolean | **Bypass Flag**: Disables BIMEF dynamic range fusion. |
| `format` | `--format` | PNG | PNG, JPEG, WebP | **Container Format**: Lossless PNG (alpha supported), JPEG (95%), or WebP (95%). |


---

## Technical Documentation

For complete mathematical derivations, signal processing proofs, and algorithmic formulations, consult [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## License

PureScale 4.0 is open-source software licensed under the **[MIT License](LICENSE)**.
