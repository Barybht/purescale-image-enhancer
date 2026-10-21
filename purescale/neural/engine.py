"""Hardware-accelerated neural inference engine for compact super-resolution models."""

import logging
import os
from typing import Optional, Callable
import cv2
import numpy as np

from purescale.config import DeviceTarget
from purescale.device import cpu_label, select_providers
from purescale.neural.models import MODEL_REGISTRY, ModelManager, get_default_model_path
from purescale.neural.tiling import tile_process

logger = logging.getLogger(__name__)


class NeuralSuperResEngine:
    """
    Executes compact ONNX super-resolution models across DirectML GPU, CPU AVX-512,
    and OpenCV DNN backends with overlap patch tiling.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        target_device: DeviceTarget = DeviceTarget.AUTO,
        model_key: Optional[str] = None,
        native_scale: Optional[int] = None,
    ):
        if model_key is not None:
            if model_key not in MODEL_REGISTRY:
                raise ValueError(f"Unknown model key: {model_key}. Available: {list(MODEL_REGISTRY.keys())}")
            self.model_key = model_key
            self.native_scale = int(MODEL_REGISTRY[model_key].get("scale", 4))
            manager = ModelManager() if model_path is None else None
            self.model_path = model_path or manager.get_model_path(model_key, auto_download=True)
        else:
            self.model_key = None
            self.model_path = model_path or get_default_model_path(auto_download=True)
            # Registry-less paths historically assumed 4x weights; make the
            # assumption overridable so real x2/x3 weights can be supplied.
            self.native_scale = int(native_scale) if native_scale else 4
        self.target_device = target_device
        self.session = None
        self.opencv_net = None
        self.input_name = ""
        self.output_name = ""
        self.backend_name = "Uninitialized"

        self._init_backend()

    def _init_backend(self) -> None:
        """Initializes the optimal execution provider.

        Fallback chain: requested accelerator (or AUTO pick) -> plain CPU ORT
        (when an accelerator session fails to construct) -> OpenCV DNN. The
        reported backend name reflects the providers the session ACTUALLY
        loaded, not the ones requested — ORT can silently degrade an
        accelerator to its CPU execution provider.
        """
        if not self.model_path or not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        try:
            import onnxruntime as ort
        except (ImportError, OSError):
            self._init_opencv_dnn()
            return

        try:
            chosen_providers, requested_name = select_providers(self.target_device)
        except ImportError:
            self._init_opencv_dnn()
            return

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = max(1, os.cpu_count() or 4)

        try:
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_opts,
                providers=chosen_providers,
            )
        except Exception as err:
            if chosen_providers == ["CPUExecutionProvider"]:
                logger.debug("ONNX Runtime CPU session failed: %s", err)
            else:
                # Accelerator/driver fault: retry plain CPU ORT (10-100x faster
                # than the OpenCV DNN fallback) before leaving ONNX Runtime.
                logger.warning(
                    "Session with %s failed (%s); retrying CPU-only ONNX Runtime.",
                    chosen_providers, err,
                )
                try:
                    self.session = ort.InferenceSession(
                        self.model_path,
                        sess_options=sess_opts,
                        providers=["CPUExecutionProvider"],
                    )
                except Exception as cpu_err:
                    logger.warning("CPU-only ORT also failed (%s); falling back to OpenCV DNN.", cpu_err)
                    self._init_opencv_dnn()
                    return

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        try:
            active = [str(p) for p in self.session.get_providers()]
        except Exception:
            active = []
        if active and active == ["CPUExecutionProvider"] and chosen_providers != ["CPUExecutionProvider"]:
            self.backend_name = f"{cpu_label()} (accelerator unavailable)"
        else:
            self.backend_name = requested_name

    def _init_opencv_dnn(self) -> None:
        """Last-resort backend: OpenCV's built-in DNN module on CPU."""
        try:
            self.opencv_net = cv2.dnn.readNetFromONNX(self.model_path)
            # OpenCV >= 5 uses the new graph engine where setPreferableBackend/
            # Target are no-ops that spam:
            #   "Targets are not supported by the new graph engine for now".
            # Default is already CPU, so only configure the classic engine.
            try:
                major = int(str(cv2.__version__).split(".")[0])
            except (ValueError, AttributeError):
                major = 4
            if major < 5:
                self.opencv_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.opencv_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self.backend_name = "OpenCV DNN (CPU Fallback)"
        except (cv2.error, RuntimeError, ValueError, OSError) as err:
            raise RuntimeError(f"Failed to load ONNX model via ONNX Runtime or OpenCV DNN: {err}") from err

    def _infer_patch(self, patch_bgr: np.ndarray) -> np.ndarray:
        """Runs single-patch forward pass through the active neural backend."""
        # Convert BGR [0..255] uint8 -> RGB [0..1] float32 NCHW
        rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        nchw = np.transpose(rgb, (2, 0, 1))[np.newaxis, :, :, :]

        if self.session is not None:
            raw_out = self.session.run([self.output_name], {self.input_name: nchw})[0]
        else:
            self.opencv_net.setInput(nchw)
            raw_out = self.opencv_net.forward()

        # NCHW -> HWC RGB -> BGR [0..255] uint8
        out_hwc = np.transpose(raw_out[0], (1, 2, 0))
        out_bgr = cv2.cvtColor(np.clip(out_hwc, 0.0, 1.0), cv2.COLOR_RGB2BGR)
        return (out_bgr * 255.0 + 0.5).astype(np.uint8)

    def upscale(
        self,
        img: np.ndarray,
        target_scale: float = 2.0,
        tile_size: int = 256,
        tile_overlap: int = 32,
        fast_2x: bool = False,
        cancel_check: Optional["Callable[[], bool]"] = None,
    ) -> np.ndarray:
        """
        Upscales input image using compact neural super-resolution with overlap tiling.
        Adapts output to arbitrary target_scale cleanly.

        NOTE (2x penalty): only a native 4x model is registered, so any
        ``target_scale < 4`` without ``fast_2x`` runs full 4x inference followed
        by a Lanczos-4 downsample (~4x the inference work a native model would
        need; measured 2.0x == 4.0x wall time). When ``fast_2x=True`` (and target_scale <= 2),
        the input is pre-downscaled 0.5x before 4x inference, yielding 2.0x directly
        with ~4x less inference work (~3.8x-4.9x faster) at the cost of sub-pixel
        texture detail on high-frequency stochastic noise.
        """
        # Input contract: silent garbage (float inputs, RGB order, 2-channel
        # arrays) is worse than a clear error — validate before inference.
        if not isinstance(img, np.ndarray):
            raise ValueError(f"img must be a numpy array, got {type(img).__name__}.")
        if img.size == 0:
            raise ValueError("img is empty (0 pixels).")
        if img.dtype != np.uint8:
            raise ValueError(
                f"engine.upscale requires uint8 BGR input, got {img.dtype}. "
                "Convert first, e.g. np.clip(img, 0, 255).astype(np.uint8)."
            )
        if img.ndim not in (2, 3) or (img.ndim == 3 and img.shape[2] not in (1, 3, 4)):
            raise ValueError(f"img must be gray [H, W] or BGR(A) [H, W, 3/4], got shape {img.shape}.")

        orig_h, orig_w = img.shape[:2]
        dest_w = int(round(orig_w * target_scale))
        dest_h = int(round(orig_h * target_scale))

        # Bypass neural inference when scaling is 1.0 or downscaling
        if target_scale <= 1.0:
            if target_scale == 1.0:
                return img.copy()
            return cv2.resize(img, (dest_w, dest_h), interpolation=cv2.INTER_AREA)

        # Experimental 2x fast path without new weights
        if fast_2x and self.native_scale == 4 and 1.0 < target_scale <= 2.0:
            pre_w = max(1, int(round(dest_w / self.native_scale)))
            pre_h = max(1, int(round(dest_h / self.native_scale)))
            logger.debug(
                "2x fast path: downscaling input (%dx%d -> %dx%d) before 4x inference to target %dx%d",
                orig_w, orig_h, pre_w, pre_h, dest_w, dest_h,
            )
            downscaled = cv2.resize(img, (pre_w, pre_h), interpolation=cv2.INTER_AREA)
            fast_raw = tile_process(
                downscaled,
                process_fn=self._infer_patch,
                scale=self.native_scale,
                tile_size=tile_size,
                overlap=tile_overlap,
                cancel_check=cancel_check,
            )
            if (fast_raw.shape[1], fast_raw.shape[0]) != (dest_w, dest_h):
                return cv2.resize(fast_raw, (dest_w, dest_h), interpolation=cv2.INTER_LANCZOS4)
            return fast_raw

        # Execute tiled inference at the model's native 4x scaling
        upscaled_4x = tile_process(
            img,
            process_fn=self._infer_patch,
            scale=self.native_scale,
            tile_size=tile_size,
            overlap=tile_overlap,
            cancel_check=cancel_check,
        )

        # If target scale equals native 4x, return directly
        if target_scale == float(self.native_scale):
            return upscaled_4x

        # If target scale is different, downscale / upscale to exact requested dimensions
        # using high-precision Lanczos-4 resampling
        logger.debug(
            "Target %.2fx != native %dx inference; resampling %dx output to %dx%d",
            target_scale, self.native_scale, self.native_scale, dest_w, dest_h,
        )
        resampled = cv2.resize(upscaled_4x, (dest_w, dest_h), interpolation=cv2.INTER_LANCZOS4)
        return resampled

