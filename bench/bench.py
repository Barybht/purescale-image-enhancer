"""Benchmark harness for PureScale stage latencies.

Runs fixed deterministic fixtures through selected presets/scales and
records per-stage latencies. Used to measure optimizations (Phase 2) and
as a CI regression gate in --quick mode.

Examples:
    python bench/bench.py --quick                # CI gate (~seconds)
    python bench/bench.py --full --csv out.csv   # full matrix for BENCH.md
    python bench/bench.py --quick --check        # fail if >15% over baseline
    python bench/bench.py --quick --update-baseline
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from purescale.config import PipelineConfig, ProcessingMode, get_preset_config
from purescale.pipeline import PureScalePipeline

BASELINE_PATH = os.path.join(os.path.dirname(__file__), "baseline.json")
REGRESSION_TOLERANCE = 0.15

# (name, width, height) fixtures. Content is deterministic (fixed seeds).
FIXTURES_QUICK = [("320x240", 320, 240)]
FIXTURES_FULL = [("320x240", 320, 240), ("640x480", 640, 480), ("1280x720", 1280, 720)]


def make_fixture(w: int, h: int, seed: int = 7) -> np.ndarray:
    """Builds a deterministic photo-like test image (gradient + texture + shapes)."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, w, dtype=np.float32)[None, :].repeat(h, axis=0)
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None].repeat(w, axis=1)
    base = np.stack([x * 180 + 30, y * 150 + 40, (1 - x) * y * 160 + 30], axis=-1)
    grain = rng.normal(0, 6.0, (h, w, 1)).astype(np.float32)
    img = base + grain
    import cv2

    cv2.circle(img, (w // 3, h // 3), min(w, h) // 8, (200, 150, 100), -1)
    cv2.rectangle(img, (w // 2, h // 2), (w // 2 + w // 5, h // 2 + h // 5), (40, 40, 40), -1)
    return np.clip(img, 0, 255).astype(np.uint8)


def run_case(pipeline: PureScalePipeline, img: np.ndarray, preset: str, scale: float,
             warmup: bool = True) -> dict:
    """Runs one preset/scale case, returns timings dict."""
    cfg = get_preset_config(preset)
    cfg.mode = ProcessingMode.PURE_DSP
    cfg.scale = scale
    if warmup:
        pipeline.enhance(img, config=cfg)
    t0 = time.perf_counter()
    res = pipeline.enhance(img, config=cfg)
    total = (time.perf_counter() - t0) * 1000.0
    return {
        "preset": preset,
        "scale": scale,
        "input": f"{img.shape[1]}x{img.shape[0]}",
        "total_ms": round(total, 1),
        "stages_ms": {k: round(v, 1) for k, v in res.stage_latencies.items()},
    }


def markdown_table(results: list) -> str:
    lines = ["| Fixture | Preset | Scale | Total (ms) | Slowest stage |", "|---|---|---|---|---|"]
    for r in results:
        stages = r["stages_ms"]
        slowest = max(stages.items(), key=lambda kv: kv[1]) if stages else ("-", 0)
        lines.append(f"| {r['input']} | {r['preset']} | {r['scale']}x | {r['total_ms']} | {slowest[0]} ({slowest[1]} ms) |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="PureScale benchmark harness")
    parser.add_argument("--quick", action="store_true", help="Small fixture only (CI gate)")
    parser.add_argument("--full", action="store_true", help="Full fixture matrix")
    parser.add_argument("--csv", default=None, help="Write per-stage CSV to path")
    parser.add_argument("--check", action="store_true", help="Fail if totals regress vs baseline")
    parser.add_argument("--tolerance", type=float, default=REGRESSION_TOLERANCE,
                        help="Allowed fractional regression vs baseline "
                             "(default: 0.15). Use a looser value on heterogeneous CI runners.")
    parser.add_argument("--update-baseline", action="store_true", help="Rewrite baseline.json from this run")
    args = parser.parse_args()

    fixtures = FIXTURES_QUICK if (args.quick or not args.full) else FIXTURES_FULL
    presets = ["balanced", "fast"]
    scales = [1.0, 2.0]

    pipeline = PureScalePipeline()
    results = []
    for name, w, h in fixtures:
        img = make_fixture(w, h)
        for preset in presets:
            for scale in scales:
                results.append(run_case(pipeline, img, preset, scale))

    print(markdown_table(results))

    if args.csv:
        with open(args.csv, "w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(["fixture", "preset", "scale", "total_ms", "stage", "stage_ms"])
            for r in results:
                for stage, ms in r["stages_ms"].items():
                    writer.writerow([r["input"], r["preset"], r["scale"], r["total_ms"], stage, ms])
        print(f"\nWrote {args.csv}")

    if args.update_baseline:
        baseline = {(r["input"], r["preset"], r["scale"]): r["total_ms"] for r in results}
        with open(BASELINE_PATH, "w") as fp:
            json.dump({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in baseline.items()}, fp, indent=2)
        print(f"Updated {BASELINE_PATH}")

    if args.check:
        if not os.path.exists(BASELINE_PATH):
            print(f"No baseline at {BASELINE_PATH}; run --update-baseline first.", file=sys.stderr)
            return 1
        with open(BASELINE_PATH) as fp:
            baseline = json.load(fp)
        failed = False
        for r in results:
            key = f"{r['input']}|{r['preset']}|{r['scale']}"
            if key not in baseline:
                print(f"WARN: no baseline for {key}, skipping")
                continue
            base = baseline[key]
            if r["total_ms"] > base * (1 + args.tolerance):
                print(f"REGRESSION: {key} {base} ms -> {r['total_ms']} ms (> {args.tolerance:.0%})")
                failed = True
        if failed:
            return 1
        print("No regressions vs baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
