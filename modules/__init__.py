"""
Specialized Computer Vision Modules for Traffic Violation Detection.
"""

from .illuminance_enhancer import IlluminanceNormalizer
from .super_resolution import PlateSuperResolver

__all__ = ["IlluminanceNormalizer", "PlateSuperResolver"]
