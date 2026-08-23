import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PlateSuperResolver:
    """
    Real-Time License Plate Super-Resolution & Gradient Sharpening Engine.
    Reconstructs edge gradients and character contours on low-resolution / blurry plate crops.
    """

    def __init__(self, target_min_height: int = 64, target_min_width: int = 140):
        self.target_min_height = target_min_height
        self.target_min_width = target_min_width

    def resolve(self, plate_crop: np.ndarray) -> np.ndarray:
        """
        Main entry point for plate super-resolution.
        """
        if plate_crop is None or plate_crop.size == 0:
            return plate_crop

        try:
            h, w = plate_crop.shape[:2]
            scale_factor = 1.0

            # Calculate required upscale multiplier if plate is small
            if h < self.target_min_height or w < self.target_min_width:
                scale_h = self.target_min_height / max(h, 1)
                scale_w = self.target_min_width / max(w, 1)
                scale_factor = max(scale_h, scale_w, 2.0)
                scale_factor = min(scale_factor, 4.0)

            # Step 1: High-Order Bicubic / Lanczos Upsampling
            if scale_factor > 1.0:
                new_w = int(w * scale_factor)
                new_h = int(h * scale_factor)
                upscaled = cv2.resize(plate_crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            else:
                upscaled = plate_crop.copy()

            # Step 2: Unsharp Masking with Multi-Scale Gaussian Gradient
            blurred = cv2.GaussianBlur(upscaled, (0, 0), sigmaX=2.0)
            sharpened = cv2.addWeighted(upscaled, 1.6, blurred, -0.6, 0)

            # Step 3: Morphological Character Contrast Enhancement
            enhanced = self._enhance_character_edges(sharpened)

            return enhanced

        except Exception as e:
            logger.warning(f"PlateSuperResolver error: {e}")
            return plate_crop

    def _enhance_character_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Enhances stroke contrast of embossed and printed characters.
        """
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            # CLAHE on luminance
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            enhanced_l = clahe.apply(l)

            # Morphological Top-Hat / Black-Hat filtering for character separation
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            top_hat = cv2.morphologyEx(enhanced_l, cv2.MORPH_TOPHAT, kernel)
            black_hat = cv2.morphologyEx(enhanced_l, cv2.MORPH_BLACKHAT, kernel)
            
            enhanced_l = cv2.add(enhanced_l, top_hat)
            enhanced_l = cv2.subtract(enhanced_l, black_hat)

            merged = cv2.merge([enhanced_l, a, b])
            return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        except Exception:
            return image
