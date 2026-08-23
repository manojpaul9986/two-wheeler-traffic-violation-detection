"""
Pipeline Factory and Registry for Traffic Violation Detectors.
"""

from .base_pipeline import BaseTrafficViolationDetector
from .v1_baseline import V1BaselineDetector
from .v2_multiscale import V2MultiScaleDetector
from .v3_transformer import V3TransformerDetector
from .v4_sota import V4SOTADetector

PIPELINE_REGISTRY = {
    "v1": V1BaselineDetector,
    "v1_baseline": V1BaselineDetector,
    "v2": V2MultiScaleDetector,
    "v2_multiscale": V2MultiScaleDetector,
    "v3": V3TransformerDetector,
    "v3_transformer": V3TransformerDetector,
    "v4": V4SOTADetector,
    "v4_sota": V4SOTADetector,
}


def get_pipeline(variant: str = "v1", model_dir: str = "./models") -> BaseTrafficViolationDetector:
    """
    Instantiates and returns the requested pipeline variant.
    
    Args:
        variant (str): One of 'v1' (baseline), 'v2' (multi-scale), 'v3' (transformer), 'v4' (sota).
        model_dir (str): Path to directory containing deep learning weights.

    Returns:
        BaseTrafficViolationDetector
    """
    key = variant.lower().strip()
    if key not in PIPELINE_REGISTRY:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(PIPELINE_REGISTRY.keys())}")
    
    return PIPELINE_REGISTRY[key](model_dir=model_dir)


__all__ = [
    "get_pipeline",
    "PIPELINE_REGISTRY",
    "BaseTrafficViolationDetector",
    "V1BaselineDetector",
    "V2MultiScaleDetector",
    "V3TransformerDetector",
    "V4SOTADetector",
]
