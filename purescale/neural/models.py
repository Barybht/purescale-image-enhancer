"""Model weight registry and cache management."""

import hashlib
import os
import sys
import urllib.request
import urllib.error
import http.client
from typing import Dict, List, Optional, Callable

DEFAULT_MODELS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "models")
)

# Registry schema per entry:
#   filename   - local cache file name under models/
#   url        - download source (only add URLs verified to exist)
#   size_bytes - expected download size (sanity check)
#   sha256     - hex digest verified against the reference file, or None
#   scale      - native super-resolution factor (1 = no scaling, e.g. detectors)
#   task       - "super-resolution" | "face-detection" | "face-restoration"
#   license    - upstream license of the weights
#   description - human-readable summary
MODEL_REGISTRY = {
    "realesr-general-x4v3": {
        "filename": "realesr-general-x4v3.onnx",
        "url": "https://huggingface.co/Heliosoph/realesrgan-onnx/resolve/main/realesr-general-x4v3.onnx",
        "size_bytes": 4871181,
        "sha256": "09b757accd747d7e423c1d352b3e8f23e77cc5742d04bae958d4eb8082b76fa4",
        "scale": 4,
        "task": "super-resolution",
        "license": "BSD-3-Clause",
        "description": "Compact Real-ESRGAN General x4v3 (6-block edge super-resolution, ~4.87 MB)",
    },
    "yunet-face": {
        "filename": "face_detection_yunet_2023mar.onnx",
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "size_bytes": 232589,
        "sha256": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        "scale": 1,
        "task": "face-detection",
        "license": "Apache-2.0",
        "description": "OpenCV Zoo YuNet face and landmark detector (~232 KB)",
    }
}


class ModelManager:
    """Manages downloading, caching, and path resolution for neural models."""

    def __init__(self, models_dir: Optional[str] = None):
        self.models_dir = os.path.abspath(models_dir or DEFAULT_MODELS_DIR)
        os.makedirs(self.models_dir, exist_ok=True)

    def get_model_path(self, model_key: str, auto_download: bool = True, progress_callback: Optional[Callable[[int, int], None]] = None) -> Optional[str]:
        """Resolves local model path, downloading if necessary."""
        if model_key not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model key: {model_key}. Available: {list(MODEL_REGISTRY.keys())}")

        info = MODEL_REGISTRY[model_key]
        dest_path = os.path.join(self.models_dir, info["filename"])

        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 10000:
            # Corrupt/stale cache (hash mismatch) is treated as missing so it
            # gets re-downloaded below instead of crashing inference later.
            if info.get("sha256") and not self.verify_model(model_key):
                try:
                    os.remove(dest_path)
                except OSError:
                    pass
            else:
                return dest_path

        if not auto_download:
            return None

        url = info["url"]
        temp_path = dest_path + ".download"

        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "PureScale/4.0"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp, open(temp_path, "wb") as out_file:
                total_size = int(resp.headers.get("Content-Length", info["size_bytes"]))
                downloaded = 0
                chunk_size = 65536
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback:
                        progress_callback(downloaded, total_size)

            if os.path.exists(dest_path):
                os.remove(dest_path)
            os.rename(temp_path, dest_path)

            if info.get("sha256") and not self.verify_model(model_key):
                try:
                    os.remove(dest_path)
                except OSError:
                    pass
                raise RuntimeError(
                    f"Downloaded model '{model_key}' failed SHA256 integrity check; "
                    f"expected {info['sha256']}"
                )
            return dest_path

        except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError, TimeoutError) as err:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            raise RuntimeError(f"Failed to download model '{model_key}' from {url}: {err}") from err

    def verify_model(self, model_key: str) -> bool:
        """Verifies a cached model file against its registry SHA256.

        Returns False when the file is missing or has no recorded digest.
        """
        if model_key not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model key: {model_key}. Available: {list(MODEL_REGISTRY.keys())}")
        info = MODEL_REGISTRY[model_key]
        expected = info.get("sha256")
        if not expected:
            return False
        dest_path = os.path.join(self.models_dir, info["filename"])
        if not os.path.exists(dest_path):
            return False
        digest = hashlib.sha256()
        with open(dest_path, "rb") as fp:
            while True:
                chunk = fp.read(1 << 20)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest() == expected

    def list_models(self) -> List[Dict[str, object]]:
        """Returns registry status: key, task, scale, cached and verified flags."""
        status = []
        for key, info in MODEL_REGISTRY.items():
            dest_path = os.path.join(self.models_dir, info["filename"])
            cached = os.path.exists(dest_path) and os.path.getsize(dest_path) > 10000
            status.append({
                "key": key,
                "task": info.get("task", "unknown"),
                "scale": info.get("scale", 1),
                "license": info.get("license", "unknown"),
                "description": info.get("description", ""),
                "cached": cached,
                "verified": self.verify_model(key) if cached else False,
            })
        return status


def resolve_sr_model(target_scale: float, registry: Optional[Dict[str, Dict[str, object]]] = None) -> str:
    """Selects the super-resolution model key for a target scale.

    Picks the smallest native scale >= ``target_scale`` (avoids wasteful
    4x inference followed by downsampling); falls back to the largest
    available scale when none covers the target.

    Args:
        target_scale: Requested magnification factor.
        registry: Model registry to search (defaults to ``MODEL_REGISTRY``;
            injectable for tests).
    """
    reg = registry if registry is not None else MODEL_REGISTRY
    candidates = [(k, v) for k, v in reg.items() if v.get("task") == "super-resolution"]
    if not candidates:
        raise ValueError("No super-resolution models in registry")
    covering = sorted(
        ((k, v) for k, v in candidates if float(v.get("scale", 1)) >= float(target_scale)),
        key=lambda kv: float(kv[1].get("scale", 1)),
    )
    if covering:
        return covering[0][0]
    return max(candidates, key=lambda kv: float(kv[1].get("scale", 1)))[0]


def get_default_model_path(auto_download: bool = True) -> Optional[str]:
    """Convenience helper returning path to the default super-resolution model."""
    manager = ModelManager()
    return manager.get_model_path("realesr-general-x4v3", auto_download=auto_download)
