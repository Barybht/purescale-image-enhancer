"""Hardware-accelerated neural inference engine for compact super-resolution models."""

import logging
import os
from typing import Optional, Tuple
import cv2
import numpy as np

from purescale.config import DeviceTarget
from purescale.device import select_providers
from purescale.neural.models import get_default_model_path
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
    ):
        self.model_path = model_path or get_default_model_path(auto_download=True)
        self.target_device = target_device
        self.session = None
        self.opencv_net = None
        self.input_name = ""
        self.output_name = ""
        self.backend_name = "Uninitialized"
        self.native_scale = 4

        self._init_backend()

    def _init_backend(self) -> None:
        """Initializes the optimal execution provider."""
        if not self.model_path or not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Attempt to use ONNX Runtime with DirectML or CPU.
        # Provider selection and display names live in purescale.device.
        try:
            import onnxruntime as ort

            chosen_providers, self.backend_name = select_providers(self.target_device)

            # Configure session options for optimal concurrency
            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_opts.intra_op_num_threads = max(1, os.cpu_count() or 4)

            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_opts,
                providers=chosen_providers,
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            return

        except (ImportError, AttributeError, RuntimeError, ValueError, OSError) as err:
            logger.debug("ONNX Runtime initialization failed, falling back to OpenCV DNN: %s", err)

        # Fallback: Built-in OpenCV DNN module
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
    ) -> np.ndarray:
        """
        Upscales input image using compact neural super-resolution with overlap tiling.
        Adapts output to arbitrary target_scale cleanly.
        """
        orig_h, orig_w = img.shape[:2]
        dest_w = int(round(orig_w * target_scale))
        dest_h = int(round(orig_h * target_scale))

        # Bypass neural inference when scaling is 1.0 or downscaling
        if target_scale <= 1.0:
            if target_scale == 1.0:
                return img.copy()
            return cv2.resize(img, (dest_w, dest_h), interpolation=cv2.INTER_AREA)

        # Execute tiled inference at the model's native 4x scaling
        upscaled_4x = tile_process(
            img,
            process_fn=self._infer_patch,
            scale=self.native_scale,
            tile_size=tile_size,
            overlap=tile_overlap,
        )

        # If target scale equals native 4x, return directly
        if target_scale == float(self.native_scale):
            return upscaled_4x

        # If target scale is different, downscale / upscale to exact requested dimensions
        # using high-precision Lanczos-4 resampling
        resampled = cv2.resize(upscaled_4x, (dest_w, dest_h), interpolation=cv2.INTER_LANCZOS4)
        return resampled
