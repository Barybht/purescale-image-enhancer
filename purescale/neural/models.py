"""Model weight registry and cache management."""

import os
import sys
import urllib.request
import urllib.error
import http.client
from typing import Optional, Callable

DEFAULT_MODELS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "models")
)

MODEL_REGISTRY = {
    "realesr-general-x4v3": {
        "filename": "realesr-general-x4v3.onnx",
        "url": "https://huggingface.co/Heliosoph/realesrgan-onnx/resolve/main/realesr-general-x4v3.onnx",
        "size_bytes": 4871181,
        "scale": 4,
        "description": "Compact Real-ESRGAN General x4v3 (6-block edge super-resolution, ~4.87 MB)",
    },
    "yunet-face": {
        "filename": "face_detection_yunet_2023mar.onnx",
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "size_bytes": 232589,
        "scale": 1,
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
            return dest_path

        except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError, TimeoutError) as err:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            raise RuntimeError(f"Failed to download model '{model_key}' from {url}: {err}") from err


def get_default_model_path(auto_download: bool = True) -> Optional[str]:
    """Convenience helper returning path to the default super-resolution model."""
    manager = ModelManager()
    return manager.get_model_path("realesr-general-x4v3", auto_download=auto_download)
