"""
Base Pipeline Interface for Two-Wheeler Traffic Violation Detectors.
"""

import os
import sys
import re
import cv2
import numpy as np
import logging
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)


class BaseTrafficViolationDetector(ABC):
    """
    Abstract Base Class for all traffic violation detection pipelines (V1, V2, V3, V4).
    Guarantees standard JSON output and Indian plate post-processing.
    """

    INDIAN_PLATE_REGEX = r"^[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{1,4}$"

    def __init__(self, model_dir: str = "./models", variant_name: str = "Base"):
        self.variant_name = variant_name
        self.model_dir = os.path.abspath(model_dir)

    @abstractmethod
    def predict(self, image_path: str) -> dict:
        """
        Runs the full detection pipeline on the given image.
        Returns {"violations": [{"num_riders": int, "helmet_violations": int, "license_plate": str}]}
        """
        pass

    def clean_plate_text(self, raw_text: str) -> str:
        """
        Universal License Plate cleaner supporting both Indian (RTO) and International plates.
        Filters out long news captions / watermarks (>11 chars).
        """
        try:
            if not raw_text:
                return ""

            # Remove special characters and spaces
            text = re.sub(r"[^A-Z0-9]", "", raw_text.upper().strip())
            
            # Reject long sentences/captions (>12 characters) or very short noise (<4 chars)
            if len(text) < 4:
                return text if len(text) >= 3 else ""
            if len(text) > 12:
                # Try to extract an embedded plate substring (4-10 alphanumeric chars)
                m = re.search(r"[A-Z0-9]{4,10}", text)
                if m:
                    text = m.group(0)
                else:
                    return ""

            # Check if an Indian RTO pattern is present (e.g., extracts 'UP65EB1464' from 'FUP65EB1464')
            m_ind = re.search(r"[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{1,4}", text)
            if m_ind and len(m_ind.group(0)) >= 7:
                text = m_ind.group(0)
                is_indian_format = True
            else:
                is_indian_format = bool(re.match(r"^[A-Z]{2}\d{1,2}", text) or re.match(r"^\d{2}[A-Z]{2}", text))

            if is_indian_format:
                digit_to_letter = {"0": "O", "1": "I", "8": "B", "5": "S", "6": "G", "2": "Z", "4": "A"}
                letter_to_digit = {"O": "0", "I": "1", "B": "8", "S": "5", "G": "6", "Z": "2", "D": "0", "A": "4", "T": "7"}
                
                # Check for standard Indian registration pattern inside string (e.g. strips leading frame noise 'F' in 'FUP65EB1464')
                m_ind = re.search(r"[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{1,4}", text)
                if m_ind:
                    text = m_ind.group(0)

                corrected = list(text)

                for i in range(min(2, len(corrected))):
                    c = corrected[i]
                    if c.isdigit() and c in digit_to_letter:
                        corrected[i] = digit_to_letter[c]

                for i in range(2, min(4, len(corrected))):
                    c = corrected[i]
                    if c.isalpha() and c in letter_to_digit:
                        corrected[i] = letter_to_digit[c]

                return "".join(corrected)

            # International / Generic plate format: return clean alphanumeric string
            return text

        except Exception:
            return re.sub(r"[^A-Z0-9]", "", raw_text.upper().strip()) if raw_text else ""

    def safe_crop(self, image: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
        """Safely crops a bounding box without out-of-bounds errors."""
        try:
            h, w = image.shape[:2]
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            if x2 <= x1 or y2 <= y1:
                return None
            return image[y1:y2, x1:x2]
        except Exception:
            return None

    def compute_iou(self, box1: list, box2: list) -> float:
        """Computes Intersection over Union between two xyxy boxes."""
        try:
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            if x2 <= x1 or y2 <= y1:
                return 0.0
            inter = (x2 - x1) * (y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            return inter / max(area1 + area2 - inter, 1e-6)
        except Exception:
            return 0.0
