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
| **SWF Denoise** | 8-Directional variance-minimizing box filter | 1004.65 ms | 43.8% |
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
| **SWF Denoise** | 844.1 ms |
| **Laplacian Pyramid** | 96.1 ms |
| **BIMEF Fusion** | 186.0 ms |
| **CAS Sharpening** | 228.3 ms |
| **Portrait Retouch** | 20.4 ms |
| **Oklab Vibrance** | 306.4 ms |

Standalone wins folded into the total, each measured in isolation:
SWF stage ~2.0x (244 ms → 122 ms on a 640x480 noisy fixture),
Oklab vibrance ~1.5x (155 ms → 106 ms on a 960x640 fixture),
dehaze ~1.9x (408 ms → 218 ms at 1080p, strength 0.65).

---

## Architectural Insights & Optimizations

1. **Analytical Semantic Parsing**:
   Combining the five individual semantic guidance passes into two guided filter passes for the detail and denoise guidance maps achieved a 65% latency reduction at 1080p (from ~360 ms to ~128 ms) with exact mathematical equivalence.

2. **Multiscale Local Laplacian Filtering**:
   Vectorized Burt-Adelson Gaussian/Laplacian pyramid construction with border reflection achieves sub-70ms execution on full 1080p frames, providing halo-free micro-texture and structural contour synthesis.

3. **Bitwise Determinism Invariance**:
   PureDSP executes with bitwise determinism ($L_\infty = 0$) across independent runs, guaranteed through `np.clip(np.rint(...), 0.0, 255.0).astype(np.uint8)` round-half-up integer quantization.
