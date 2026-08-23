import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class IlluminanceNormalizer:
    """
    Adaptive Illuminance & Environmental Normalizer.
    Normalizes low-light, nighttime, backlit, or heavy-glare surveillance footage.
    """

    def __init__(self, target_brightness: float = 120.0):
        self.target_brightness = target_brightness

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Main entry point. Evaluates image luminance and applies appropriate
        enhancement (Zero-DCE style dynamic curve or Retinex CLAHE).
        """
        if image is None or image.size == 0:
            return image

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mean_lum = float(np.mean(gray))

            # Severe low-light (Night conditions: mean < 45)
            if mean_lum < 45.0:
                return self._apply_zero_dce_curve(image, iterations=4)
            
            # Moderate underexposure / heavy shadows (mean between 45 and 85)
            elif mean_lum < 85.0:
                return self._apply_retinex_clahe(image, clip_limit=3.0)
            
            # Overexposed / harsh daytime glare (mean > 190)
            elif mean_lum > 190.0:
                return self._suppress_glare(image)

            return image

        except Exception as e:
            logger.warning(f"IlluminanceNormalizer error: {e}")
            return image

    def _apply_zero_dce_curve(self, image: np.ndarray, iterations: int = 4) -> np.ndarray:
        """
        Zero-DCE style iterative quadratic illumination enhancement:
        LE_n(x) = LE_{n-1}(x) + A * LE_{n-1}(x) * (1 - LE_{n-1}(x))
        """
        try:
            norm_img = image.astype(np.float32) / 255.0
            # Higher alpha in darker regions
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            alpha = np.clip(1.0 - gray, 0.2, 0.8)[..., np.newaxis]

            x = norm_img
            for _ in range(iterations):
                x = x + alpha * x * (1.0 - x)

            enhanced = np.clip(x * 255.0, 0, 255).astype(np.uint8)
            return enhanced
        except Exception:
            return image

    def _apply_retinex_clahe(self, image: np.ndarray, clip_limit: float = 2.5) -> np.ndarray:
        """
        Applies Contrast Limited Adaptive Histogram Equalization in LAB space.
        """
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            enhanced_l = clahe.apply(l)
            merged = cv2.merge([enhanced_l, a, b])
            return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        except Exception:
            return image

    def _suppress_glare(self, image: np.ndarray) -> np.ndarray:
        """
        Compresses highlight extremes while preserving mid-tone contrast.
        """
        try:
            gamma = 1.25
            inv_gamma = 1.0 / gamma
            lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
            return cv2.LUT(image, lut)
        except Exception:
            return image
