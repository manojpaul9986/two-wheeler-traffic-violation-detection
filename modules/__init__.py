"""
Specialized Computer Vision Modules for Traffic Violation Detection.
"""

from .illuminance_enhancer import IlluminanceNormalizer
from .super_resolution import PlateSuperResolver
from .visualizer import annotate_traffic_violations

__all__ = ["IlluminanceNormalizer", "PlateSuperResolver", "annotate_traffic_violations"]
