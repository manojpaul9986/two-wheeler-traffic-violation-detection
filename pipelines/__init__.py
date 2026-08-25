"""
Pipeline Registry & Factory.
Provides access to:
- V1: Baseline High-Throughput Detector (Single 640px Pass + PANet)
- V2: Multi-Scale Feature Pyramid Detector (Production SOTA with 320/640/960px Pyramid + Stacked OCR)
"""

from .v1_baseline import V1BaselineDetector
from .v2_multiscale import V2MultiScaleDetector

__all__ = [
    "V1BaselineDetector",
    "V2MultiScaleDetector",
    "get_pipeline",
]


def get_pipeline(variant: str = "v2", model_dir: str = "./models"):
    """
    Factory function to retrieve the desired pipeline instance.

    Args:
        variant: 'v1' (Baseline) or 'v2' (Multi-Scale Production)
        model_dir: Directory containing trained YOLOv8 & PaddleOCR weights

    Returns:
        Instantiated Detector Pipeline
    """
    variant = variant.lower().strip()
    if variant == "v1":
        return V1BaselineDetector(model_dir=model_dir)
    elif variant == "v2":
        return V2MultiScaleDetector(model_dir=model_dir)
    else:
        raise ValueError(
            f"Unknown pipeline variant '{variant}'. Available variants: ['v1', 'v2']"
        )
