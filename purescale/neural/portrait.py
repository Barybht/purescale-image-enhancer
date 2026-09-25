"""Synchronized portrait retouching with aspect-ratio preserving YuNet and Fast Guided Filter."""

import logging
import os
from typing import List, Optional, Tuple
import cv2
import numpy as np

from purescale.dsp.filters import fast_guided_filter
from purescale.neural.models import ModelManager

logger = logging.getLogger(__name__)


class PortraitRetoucher:
    """Detects faces preserving aspect ratio and applies Fast Guided Filter skin retouching."""

    def __init__(self, model_path: Optional[str] = None):
        if not model_path:
            manager = ModelManager()
            model_path = manager.get_model_path("yunet-face", auto_download=True)

        self.model_path = model_path
        self.detector = None
        self._init_detector()

    def _init_detector(self) -> None:
        if self.model_path and os.path.exists(self.model_path):
            # FaceDetectorYN builds an internal DNN net that emits the same
            # OpenCV 5.x WARN ("Targets are not supported..."). Mute WARN
            # for the duration of construction, then restore prior level.
            prev_level = None
            try:
                if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
                    prev_level = cv2.utils.logging.getLogLevel()
                    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
            except (AttributeError, RuntimeError):
                prev_level = None
            try:
                self.detector = cv2.FaceDetectorYN.create(
                    model=self.model_path,
                    config="",
                    input_size=(320, 320),
                    score_threshold=0.6,
                    nms_threshold=0.3,
                    top_k=10,
                )
            except (cv2.error, RuntimeError, ValueError, TypeError) as e:
                logger.warning("Failed to initialize FaceDetectorYN from %s: %s", self.model_path, e)
                self.detector = None
            finally:
                try:
                    if prev_level is not None:
                        cv2.utils.logging.setLogLevel(prev_level)
                except (AttributeError, RuntimeError):
                    pass

    def enhance(
        self,
        img: np.ndarray,
        portrait_smooth: int = 35,
        eye_clarity: float = 1.3,
    ) -> Tuple[np.ndarray, int]:
        """
        Applies aspect-ratio preserving face detection, selective FGF skin smoothing,
        and iris catchlight clarity.
        """
        if portrait_smooth <= 0 and eye_clarity <= 1.0:
            return img.copy(), 0

        cur_h, cur_w = img.shape[:2]
        face_mask = np.zeros((cur_h, cur_w), dtype=np.uint8)
        eye_regions: List[Tuple[Tuple[int, int], int]] = []
        faces_detected = 0

        if self.detector is not None:
            # Preserve aspect ratio up to 640px maximum dimension
            max_dim = 640
            scale = min(1.0, max_dim / float(max(cur_h, cur_w)))
            proxy_w = int(round(cur_w * scale))
            proxy_h = int(round(cur_h * scale))
            proxy_w = proxy_w if proxy_w % 2 == 0 else proxy_w + 1
            proxy_h = proxy_h if proxy_h % 2 == 0 else proxy_h + 1

            small = cv2.resize(img, (proxy_w, proxy_h), interpolation=cv2.INTER_AREA)
            self.detector.setInputSize((proxy_w, proxy_h))
            _, faces = self.detector.detect(small)

            if faces is not None and len(faces) > 0:
                faces_detected = len(faces)
                scale_x = cur_w / float(proxy_w)
                scale_y = cur_h / float(proxy_h)

                for face in faces:
                    fx = int(face[0] * scale_x)
                    fy = int(face[1] * scale_y)
                    fw = int(face[2] * scale_x)
                    fh = int(face[3] * scale_y)

                    cx, cy = fx + fw // 2, fy + fh // 2
                    axes = (int(fw * 0.46), int(fh * 0.56))
                    cv2.ellipse(face_mask, (cx, cy), axes, 0, 0, 360, 255, -1)

                    # Extract eye landmarks: face[4:6] is right eye, face[6:8] is left eye
                    lex, ley = int(face[4] * scale_x), int(face[5] * scale_y)
                    rex, rey = int(face[6] * scale_x), int(face[7] * scale_y)
                    eye_radius = max(8, int(fw * 0.08))

                    eye_regions.append(((lex, ley), eye_radius))
                    eye_regions.append(((rex, rey), eye_radius))

        out = img.copy()

        if np.any(face_mask):
            # Skin color segmentation within facial envelope
            if portrait_smooth > 0:
                ycrcb = cv2.cvtColor(out, cv2.COLOR_BGR2YCrCb)
                skin_ycrcb = cv2.inRange(ycrcb, (15, 120, 65), (255, 185, 145))

                hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
                skin_hsv = cv2.inRange(hsv, (0, 15, 20), (28, 240, 255))

                skin_roi = cv2.bitwise_and(cv2.bitwise_and(skin_ycrcb, skin_hsv), face_mask)

                # Protect eyes from skin smoothing
                for center, radius in eye_regions:
                    cv2.circle(skin_roi, center, int(radius * 1.1), 0, -1)

                final_skin_mask = cv2.GaussianBlur(skin_roi, (15, 15), 0)
                alpha_skin = (final_skin_mask.astype(np.float32) / 255.0)[:, :, np.newaxis]

                # Blend weight mapped from portrait_smooth [0..100] -> [0.0..0.85]
                blend_factor = (portrait_smooth / 100.0) * 0.85

                gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
                smoothed_bgr = fast_guided_filter(gray, out, radius=8, eps=0.04, subsample=4)

                blended = (alpha_skin * blend_factor * smoothed_bgr.astype(np.float32) +
                           (1.0 - alpha_skin * blend_factor) * out.astype(np.float32))
                out = np.clip(np.rint(blended), 0.0, 255.0).astype(np.uint8)

            # Targeted eye clarity / catchlight boost
            if eye_clarity > 1.0 and eye_regions:
                for center, radius in eye_regions:
                    cx, cy = center
                    patch_r = int(radius * 1.25)
                    x1 = max(0, cx - patch_r)
                    y1 = max(0, cy - patch_r)
                    x2 = min(cur_w, cx + patch_r)
                    y2 = min(cur_h, cy + patch_r)

                    if x2 > x1 and y2 > y1:
                        patch = out[y1:y2, x1:x2].astype(np.float32)
                        blur_patch = cv2.GaussianBlur(patch, (5, 5), 1.0)
                        sharp_patch = np.clip(patch * eye_clarity - blur_patch * (eye_clarity - 1.0), 0.0, 255.0)

                        patch_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.float32)
                        cv2.circle(patch_mask, (cx - x1, cy - y1), patch_r, 1.0, -1)
                        patch_mask = cv2.GaussianBlur(patch_mask, (9, 9), 0)[:, :, np.newaxis]

                        out[y1:y2, x1:x2] = (patch_mask * sharp_patch + (1.0 - patch_mask) * patch).astype(np.uint8)

        return out, faces_detected


def detect_and_enhance_portrait(
    img: np.ndarray,
    portrait_smooth: int = 35,
    eye_clarity: float = 1.3,
    model_path: Optional[str] = None,
) -> Tuple[np.ndarray, int]:
    """Functional convenience wrapper for portrait enhancement."""
    retoucher = PortraitRetoucher(model_path=model_path)
    return retoucher.enhance(img, portrait_smooth=portrait_smooth, eye_clarity=eye_clarity)
