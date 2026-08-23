"""
V1: Baseline Pipeline (YOLOv8s + Standard PANet + Offline PaddleOCR 3.x).
"""

from solution import TrafficViolationDetector
from .base_pipeline import BaseTrafficViolationDetector


class V1BaselineDetector(BaseTrafficViolationDetector):
    """
    V1 Baseline Implementation wrapping the course submission solution.
    """

    def __init__(self, model_dir: str = "./models"):
        super().__init__(model_dir=model_dir, variant_name="V1_Baseline")
        self.detector = TrafficViolationDetector(model_dir=model_dir)

    def predict(self, image_path: str) -> dict:
        return self.detector.predict(image_path)
