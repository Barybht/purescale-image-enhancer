# PureScale 4.0 Performance Benchmarks

## Environment Specifications
- **Processor**: AMD Ryzen 5 PRO 7640HS (Zen 4, 6 Cores / 12 Threads, AVX-512 / AVX2)
- **Memory**: DDR5-5600 Dual-Channel
- **Operating System**: Windows 11 Pro 64-bit
- **Python**: 3.11.9
- **OpenCV**: 5.0.0 (Pre-release Build with Zen 4 Vector SIMD)
- **NumPy**: 1.26.4

---

## PureDSP Stage Latency Telemetry

Benchmark methodology: 5 consecutive iterations per resolution after warm-up pass.
All advanced computer vision algorithms enabled simultaneously:
- Autonomous Signal Diagnostics
- Multi-Cue Analytical Semantic Parsing
- Dark Channel Prior Atmospheric Dehazing
- Side Window Filter (SWF) Corner-Preserving Denoising
- Edge-Adaptive Spatial Upsampling (EASU)
- Multiscale Local Laplacian 4-Octave Frequency Decomposition
- Bio-Inspired Multi-Exposure Fusion (BIMEF) HDR Dynamic Range Recovery
- Contrast-Adaptive Sharpening (CAS)
- Synchronized Facial Retouching (YuNet + Fast Guided Filter)
- Perceptual LMS Cone Vibrance (Oklab)

---

### Low Resolution: 120p (160x120 -> 320x240, 2.0x Scale)
- **Total End-to-End Pipeline Latency**: `46.22 +/- 4.38 ms`
- **Throughput**: ~21.6 FPS

| Pipeline Stage | Algorithm Formulation | Latency (ms) | Percentage |
| :--- | :--- | :--- | :--- |
| **Diagnostics** | Wavelet MAD, Blur, Shannon Entropy | 2.52 ms | 5.5% |
| **Semantic Parsing** | Multi-cue opponent soft probability masks | 1.90 ms | 4.1% |
| **Dehazing** | Dark Channel Prior & Guided Filter transmission | 2.76 ms | 6.0% |
| **SWF Denoise** | 8-Directional variance-minimizing box filter | 6.24 ms | 13.5% |
| **Upsampling** | Directional Edge-Adaptive Spatial Upsampling (EASU) | 4.50 ms | 9.7% |
| **Laplacian Pyramid** | 4-Octave Local Laplacian frequency decomposition | 2.67 ms | 5.8% |
| **BIMEF Fusion** | Black-point pinned exposure fusion | 3.96 ms | 8.6% |
| **CAS Sharpening** | Contrast-Adaptive Sharpening kernel | 6.80 ms | 14.7% |
| **Portrait Retouch** | YuNet aspect-ratio detection + Fast Guided Filter | 3.04 ms | 6.6% |
| **Oklab Vibrance** | Perceptual LMS cone chroma modulation | 11.55 ms | 25.0% |

---

### High Definition: 1080p (1920x1080 Native Resolution)
- **Total End-to-End Pipeline Latency**: `2294.66 +/- 89.56 ms`
- **Total Pixels Processed**: 2,073,600 pixels

| Pipeline Stage | Algorithm Formulation | Latency (ms) | Percentage |
| :--- | :--- | :--- | :--- |
| **Diagnostics** | Wavelet MAD, Blur, Shannon Entropy | 77.41 ms | 3.4% |
| **Semantic Parsing** | Multi-cue opponent soft probability masks | 127.59 ms | 5.6% |
| **Dehazing** | Dark Channel Prior & Guided Filter transmission | 318.22 ms | 13.9% |
| **SWF Denoise** | 8-Directional variance-minimizing box filter | 1004.65 ms (pre-proxy; see Phase 2c) | 43.8% |
| **Laplacian Pyramid** | 4-Octave Local Laplacian frequency decomposition | 68.91 ms | 3.0% |
| **BIMEF Fusion** | Black-point pinned exposure fusion | 132.44 ms | 5.8% |
| **CAS Sharpening** | Contrast-Adaptive Sharpening kernel | 192.79 ms | 8.4% |
| **Portrait Retouch** | Fast Guided Filter skin smoothing | 12.46 ms | 0.5% |
| **Oklab Vibrance** | Perceptual LMS cone chroma modulation | 356.29 ms | 15.5% |

### Post-Optimization: 1080p Balanced, Scale 1.0x (Phase 2b)

Re-measured after the luminance-selection SWF, Oklab transfer LUTs, and
dehaze proxy (same machine class, `bench/bench.py` deterministic fixture,
dehazing off per balanced defaults):

- **Total**: `~1958 ms` (was `~2294 ms`, ~15% faster)

| Pipeline Stage | Latency (ms) |
| :--- | :--- |
| **Diagnostics** | 90.1 ms |
| **Semantic Parsing** | 174.4 ms |
| **SWF Denoise** | 844.1 ms (pre-proxy; see Phase 2c) |
| **Laplacian Pyramid** | 96.1 ms |
| **BIMEF Fusion** | 186.0 ms |
| **CAS Sharpening** | 228.3 ms |
| **Portrait Retouch** | 20.4 ms |
| **Oklab Vibrance** | 306.4 ms |

Standalone wins folded into the total, each measured in isolation:
SWF stage ~2.0x (244 ms → 122 ms on a 640x480 noisy fixture),
Oklab vibrance ~1.5x (155 ms → 106 ms on a 960x640 fixture),
dehaze ~1.9x (408 ms → 218 ms at 1080p, strength 0.65).

### Post-Optimization: SWF proxy denoising (Phase 2c)

Large images (long edge > 480 px) are denoised on an INTER_AREA
downsampled proxy and upsampled with INTER_LINEAR
(`side_window_filter(..., proxy_max_dim=480)`; `0` restores the
full-resolution path). Small fixtures (all golden/unit inputs) take the
identical full path, so golden outputs are bitwise unchanged.

- **Speedup**: ~1.7x on 640x480 (74.7 ms → 43.1 ms), ~7x at 720p
  (234 ms → 33 ms), ~12x at 1080p (453 ms → 37 ms), radius 2.
- **Quality gates** (`test_swf_proxy_consistency`, mirroring the dehaze
  proxy test): PSNR > 35 dB vs full path, SSIM > 0.90 (lower than
  dehaze's 0.98 because two valid denoises differ in residual grain,
  not structure), flat-region noise still halved, step-edge delta
  preserved (> 60), deterministic across runs.

### Post-Optimization: Neural 2x Fast Path (Task 1)

In Neural AI and Hybrid modes at `--scale 2.0`, the reference path runs full 4x
inference followed by Lanczos-4 downsampling ($O(N)$ input pixels). The
experimental fast path (`--fast-2x` / `enable_fast_2x=True`) downscales the
input 0.5x with `INTER_AREA`, executes 4x neural inference on $N/4$ pixels, and
produces the 2.0x target directly.

Measured on AMD Ryzen 5 PRO 7640HS (DirectML GPU and Zen 4 CPU AVX-512):

| Fixture / Resolution | DirectML Ref (ms) | DirectML Fast (ms) | Speedup (GPU) | CPU Ref (ms) | CPU Fast (ms) | Speedup (CPU) | PSNR (dB) | SSIM |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1080p (1920x1080)** | 9019.0 ms | 2373.1 ms | **3.80x** | 32406.9 ms | 7771.3 ms | **4.17x** | 47.59 dB | 0.9918 |
| **720p (1280x720)** | 4167.5 ms | 948.8 ms | **4.39x** | 13909.8 ms | 2858.2 ms | **4.87x** | 46.82 dB | 0.9915 |
| **VGA (640x480)** | 1427.1 ms | 353.2 ms | **4.04x** | — | — | — | 45.42 dB | 0.9909 |
| **QVGA (320x240)** | 363.4 ms | 60.8 ms | **5.98x** | 741.6 ms | 88.0 ms | **8.42x** | 43.00 dB | 0.9893 |
| **160x120** | 70.2 ms | 25.2 ms | **2.79x** | 92.5 ms | 23.4 ms | **3.96x** | 40.04 dB | 0.9854 |
| **golden_photo (96x96)** | 37.5 ms | 17.4 ms | **2.16x** | 50.0 ms | 12.7 ms | **3.94x** | 39.20 dB | 0.9851 |
| **golden_edge (64x64)** | 13.4 ms | 12.4 ms | **1.08x** | 26.5 ms | 9.0 ms | **2.96x** | 35.99 dB | 0.9802 |
| **golden_texture (80x80)**| 27.0 ms | 13.8 ms | **1.96x** | 32.1 ms | 10.8 ms | **2.99x** | 17.02 dB | 0.6956 |

- **Quality analysis**: On natural imagery, photos, and structural contours, the
  fast path achieves PSNR > 39-47 dB and SSIM > 0.98-0.99 with an empirical ~4x
  latency reduction across both GPU and CPU backends. However, on fine stochastic
  micro-textures (`golden_texture`), 0.5x pre-decimation removes high frequencies
  before neural feature extraction, resulting in PSNR dropping to 17.02 dB and
  SSIM to 0.6956.
- **Outcome**: The fast path is shipped as an optional feature (`enable_fast_2x`,
  CLI `--fast-2x`, GUI toggle, enabled in preset `fast`) rather than the default.
  Reference 4x-then-downsample remains default for fidelity.

### Post-Optimization: Local Tone-Mapping & Highlight Reconstruction (Task 3)

Local Tone-Mapping decomposes luminance via a proxy-accelerated Fast Guided Filter
into large-scale illumination base and high-frequency reflectance detail layers.
Asymmetric toe expansion lifts crushed shadows below 0.40, while shoulder
compression and specular desaturation reconstruct clipped channels (> 0.88).

Measured on AMD Ryzen 5 PRO 7640HS (Zen 4 CPU, AVX-512 / AVX2):

| Fixture / Resolution | Full-Res Latency (ms) | Proxy (480px) Latency (ms) | Speedup | Proxy Consistency PSNR | Proxy Consistency SSIM |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1080p (1920x1080)** | 227.0 ms | 139.3 ms | **1.63x** | 54.28 dB | 0.9986 |
| **720p (1280x720)** | 105.8 ms | 48.2 ms | **2.19x** | 54.35 dB | 0.9986 |
| **VGA (640x480)** | 28.5 ms | 16.7 ms | **1.71x** | 54.50 dB | 0.9986 |
| **QVGA (320x240)** | 3.3 ms | 3.3 ms (direct) | **1.00x** | inf (identical) | 1.0000 |

- **Quality gates** (`test_local_tone_proxy_consistency`): Exceeds gating
  thresholds (PSNR > 35 dB, SSIM > 0.98), achieving > 54 dB PSNR and > 0.998 SSIM
  with 100% bitwise determinism ($L_\infty = 0$).

---

## Architectural Insights & Optimizations

1. **Analytical Semantic Parsing**:
   Combining the five individual semantic guidance passes into two guided filter passes for the detail and denoise guidance maps achieved a 65% latency reduction at 1080p (from ~360 ms to ~128 ms) with exact mathematical equivalence.

2. **Multiscale Local Laplacian Filtering**:
   Vectorized Burt-Adelson Gaussian/Laplacian pyramid construction with border reflection achieves sub-70ms execution on full 1080p frames, providing halo-free micro-texture and structural contour synthesis.

3. **Bitwise Determinism Invariance**:
   PureDSP executes with bitwise determinism ($L_\infty = 0$) across independent runs, guaranteed through `np.clip(np.rint(...), 0.0, 255.0).astype(np.uint8)` round-half-up integer quantization.

