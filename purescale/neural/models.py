"""Model weight registry and cache management."""

import hashlib
import os
import time
import urllib.request
import urllib.error
import http.client
import uuid
from typing import Dict, List, Optional, Callable

DEFAULT_MODELS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "models")
)

# Download-hardening knobs.
_DOWNLOAD_RETRIES = 3
_LOCK_POLL_INTERVAL_S = 0.05
_LOCK_TIMEOUT_S = 120.0      # wait this long for a concurrent downloader
_LOCK_STALE_S = 600.0        # steal locks left by dead processes after 10 min

# Registry schema per entry:
#   filename   - local cache file name under models/
#   url        - download source, pinned to an immutable upstream revision
#                (a mutable /main/ URL can silently change bytes and leave the
#                pinned SHA256 permanently unverifiable)
#   size_bytes - expected download size (early corruption detection)
#   sha256     - hex digest verified against the reference file, or None
#   scale      - native super-resolution factor (1 = no scaling, e.g. detectors)
#   task       - "super-resolution" | "face-detection" | "face-restoration"
#   style      - "photo" | "anime" (for super-resolution models)
#   license    - upstream license of the weights
#   description - human-readable summary
MODEL_REGISTRY = {
    "realesr-general-x4v3": {
        "filename": "realesr-general-x4v3.onnx",
        "url": "https://huggingface.co/Heliosoph/realesrgan-onnx/resolve/488e5dda07333179f229a6205d92135eea4c25e9/realesr-general-x4v3.onnx",
        "size_bytes": 4871181,
        "sha256": "09b757accd747d7e423c1d352b3e8f23e77cc5742d04bae958d4eb8082b76fa4",
        "scale": 4,
        "task": "super-resolution",
        "style": "photo",
        "license": "BSD-3-Clause",
        "description": "Compact Real-ESRGAN General x4v3 (6-block edge super-resolution, ~4.87 MB)",
    },
    "realesr-animevideov3-x4": {
        "filename": "realesr-animevideov3-x4.onnx",
        "url": "https://huggingface.co/xiaojiaenen/lingtuan-sr-models/resolve/29adb237e42934ce2cc87a1a4887f4950cb8ab55/realesr-animevideov3-x4.onnx",
        "size_bytes": 2496535,
        "sha256": "14114520123405712c92c2b266a49e906f03b595335e53e3604cdc03c5a408e4",
        "scale": 4,
        "task": "super-resolution",
        "style": "anime",
        "license": "BSD-3-Clause",
        "description": "Compact Real-ESRGAN AnimeVideo-v3 x4 (edge anime/illustration super-resolution, ~2.49 MB)",
    },
    "yunet-face": {
        "filename": "face_detection_yunet_2023mar.onnx",
        "url": "https://github.com/opencv/opencv_zoo/raw/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
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
        self._sweep_stale_temp(self.models_dir)

    @staticmethod
    def _sweep_stale_temp(models_dir: str) -> None:
        """Removes litter from hard-killed downloads: ``.download`` temp files
        and ``.lock`` files older than the stale-lock window. Fresh artifacts
        from another live process are never touched."""
        now = time.time()
        try:
            names = os.listdir(models_dir)
        except OSError:
            return
        for name in names:
            if name.endswith(".download") or name.endswith(".lock"):
                path = os.path.join(models_dir, name)
                try:
                    if now - os.path.getmtime(path) > _LOCK_STALE_S:
                        os.remove(path)
                except OSError:
                    pass

    @staticmethod
    def _lock_is_stale(lock_path: str) -> bool:
        try:
            return (time.time() - os.path.getmtime(lock_path)) > _LOCK_STALE_S
        except OSError:
            return True

    def _acquire_lock(self, lock_path: str) -> bool:
        """Advisory cross-process lock via O_EXCL creation.

        Losers poll until the winner finishes (or the timeout elapses). A lock
        left by a dead process is stolen after ``_LOCK_STALE_S``.
        """
        deadline = time.monotonic() + _LOCK_TIMEOUT_S
        while True:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                return True
            except FileExistsError:
                if self._lock_is_stale(lock_path):
                    try:
                        os.remove(lock_path)
                        continue  # stolen: retry acquisition immediately
                    except OSError:
                        pass
                if time.monotonic() >= deadline:
                    return False
                time.sleep(_LOCK_POLL_INTERVAL_S)
            except OSError:
                # FS forbids O_EXCL here: degrade to the old unlocked behavior.
                return True

    def get_model_path(self, model_key: str, auto_download: bool = True, progress_callback: Optional[Callable[[int, int], None]] = None) -> Optional[str]:
        """Resolves local model path, downloading if necessary.

        Download hardening: unique per-attempt temp file (no cross-process
        ``.download`` collisions), an advisory lock so concurrent downloaders
        cannot corrupt each other, SHA256 verified BEFORE the atomic rename
        into the cache, and ``_DOWNLOAD_RETRIES`` attempts with backoff.
        """
        if model_key not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model key: {model_key}. Available: {list(MODEL_REGISTRY.keys())}")

        info = MODEL_REGISTRY[model_key]
        dest_path = os.path.join(self.models_dir, info["filename"])
        lock_path = dest_path + ".lock"

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

        if not self._acquire_lock(lock_path):
            # Another process downloaded (or is downloading) the model.
            if os.path.exists(dest_path) and os.path.getsize(dest_path) > 10000:
                return dest_path
            raise RuntimeError(
                f"Timed out waiting for a concurrent download of '{model_key}' "
                f"({lock_path}); delete the lock file if no other process is running."
            )
        try:
            return self._download_and_verify(model_key, info, dest_path, progress_callback)
        finally:
            try:
                os.remove(lock_path)
            except OSError:
                pass

    def _download_and_verify(
        self,
        model_key: str,
        info: Dict[str, object],
        dest_path: str,
        progress_callback: Optional[Callable[[int, int], None]],
    ) -> str:
        """Downloads to a unique temp file, verifies size + SHA256, then atomically moves it into the cache."""
        url = str(info["url"])
        expected_size = int(info["size_bytes"])
        last_err: Exception = None  # type: ignore[assignment]

        for attempt in range(1, _DOWNLOAD_RETRIES + 1):
            temp_path = f"{dest_path}.{os.getpid()}.{uuid.uuid4().hex[:12]}.download"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "PureScale/4.0"})
                with urllib.request.urlopen(req, timeout=30) as resp, open(temp_path, "wb") as out_file:
                    try:
                        total_size = int(resp.headers.get("Content-Length", expected_size))
                    except (TypeError, ValueError):
                        total_size = expected_size
                    downloaded = 0
                    chunk_size = 65536
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        out_file.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback:
                            # Callback bugs must surface as themselves, not as
                            # misleading "download failed" errors.
                            progress_callback(downloaded, total_size)

                if downloaded != expected_size:
                    raise RuntimeError(
                        f"size mismatch: got {downloaded} bytes, expected {expected_size}"
                    )

                # Verify BEFORE the cache sees the file: a corrupt payload must
                # never briefly become the canonical entry (the old code renamed
                # first and verified after).
                if info.get("sha256") and not self._verify_file(temp_path, str(info["sha256"])):
                    raise RuntimeError(f"SHA256 mismatch; expected {info['sha256']}")

                os.replace(temp_path, dest_path)  # atomic on the same volume
                return dest_path

            except (urllib.error.URLError, http.client.HTTPException, OSError, TimeoutError, RuntimeError) as err:
                last_err = err
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass
                if attempt < _DOWNLOAD_RETRIES:
                    time.sleep(min(30.0, 2.0 ** attempt))  # 2 s, 4 s backoff

        raise RuntimeError(
            f"Failed to download model '{model_key}' from {url} "
            f"after {_DOWNLOAD_RETRIES} attempts: {last_err}"
        ) from last_err

    @staticmethod
    def _verify_file(path: str, expected: str) -> bool:
        digest = hashlib.sha256()
        try:
            with open(path, "rb") as fp:
                while True:
                    chunk = fp.read(1 << 20)
                    if not chunk:
                        break
                    digest.update(chunk)
        except OSError:
            return False
        return digest.hexdigest() == expected

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
        return self._verify_file(dest_path, str(expected))

    def list_models(self) -> List[Dict[str, object]]:
        """Returns registry status: key, task, scale, cached and verified flags."""
        status = []
        for key, info in MODEL_REGISTRY.items():
            dest_path = os.path.join(self.models_dir, info["filename"])
            cached = os.path.exists(dest_path) and os.path.getsize(dest_path) > 10000
            status.append({
                "key": key,
                "task": info.get("task", "unknown"),
                "style": info.get("style"),
                "scale": info.get("scale", 1),
                "license": info.get("license", "unknown"),
                "description": info.get("description", ""),
                "cached": cached,
                "verified": self.verify_model(key) if cached else False,
            })
        return status


def resolve_sr_model(
    target_scale: float,
    style: str = "photo",
    registry: Optional[Dict[str, Dict[str, object]]] = None,
) -> str:
    """Selects the super-resolution model key for a target scale and style.

    Picks the smallest native scale >= ``target_scale`` for the requested style
    (avoids wasteful 4x inference followed by downsampling); falls back to the
    largest available scale for that style when none covers the target.

    Args:
        target_scale: Requested magnification factor.
        style: Content style filter ("photo" or "anime"). Default "photo".
        registry: Model registry to search (defaults to ``MODEL_REGISTRY``;
            injectable for tests).

    Raises:
        ValueError: If ``style`` is unknown or no super-resolution models exist
            for the requested style.
    """
    reg = registry if registry is not None else MODEL_REGISTRY
    sr_models = [(k, v) for k, v in reg.items() if v.get("task") == "super-resolution"]
    if not sr_models:
        raise ValueError("No super-resolution models in registry")

    available_styles = {str(v.get("style", "photo")).lower() for _, v in sr_models}
    normalized_style = str(style).lower().strip()
    if normalized_style not in available_styles:
        raise ValueError(
            f"Unknown super-resolution style: {style!r}. Available: {sorted(available_styles)}"
        )

    candidates = [
        (k, v) for k, v in sr_models
        if str(v.get("style", "photo")).lower() == normalized_style
    ]
    if not candidates:
        raise ValueError(f"No super-resolution models found for style {style!r}")

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
