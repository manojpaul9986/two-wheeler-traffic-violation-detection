"""
Traffic Rule Violation Detection System
AID 728 - IIIT Bangalore Computer Vision Course Project

Three-model architecture:
  Model 1 (rider_group_best.pt) — YOLOv8s, 2 classes: rider_group / triple_riding
  Model 2 (helmet_best.pt)      — YOLOv8s, 2 classes: helmet / no_helmet
  Model 3 (plate_best.pt)       — YOLOv8s, 1 class:  license_plate
  OCR    (PaddleOCR 3.x)        — PaddleOCR English, offline cache in ~/.paddleocr

Pipeline per image:
  rider_group_model → crop each group → helmet_model on crop
                                      → plate_model on expanded region → OCR
"""

import os
import sys
import time
import logging
import re

os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

# Point PaddleX to bundled models — must be set before any paddle import (cache.py reads it at module init)
_PADDLE_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "paddle_ocr")
if os.path.isdir(_PADDLE_CACHE):
    os.environ["PADDLE_PDX_CACHE_HOME"] = _PADDLE_CACHE

import cv2
import numpy as np
import torch
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s",
                    stream=sys.stderr)
logger = logging.getLogger(__name__)


class TrafficViolationDetector:

    # rider_group model class IDs
    RIDER_GROUP_CLS   = 0
    TRIPLE_RIDING_CLS = 1

    # helmet model class IDs
    HELMET_CLS    = 0
    NO_HELMET_CLS = 1

    # Confidence thresholds (calibrated for both high-res and distant/low-res CCTV footage)
    CONF_RIDER_GROUP = 0.20
    CONF_HELMET      = 0.12
    CONF_NO_HELMET   = 0.10
    CONF_PLATE       = 0.15

    # Indian license plate regex
    INDIAN_PLATE_REGEX = r"^[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{1,4}$"

    def __init__(self, model_dir="./models"):
        """Load all models. Called once."""
        try:
            self.device = 0 if torch.cuda.is_available() else "cpu"

            # Resolve model_dir relative to this file, not the cwd
            if not os.path.isabs(model_dir):
                model_dir = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), model_dir
                )

            self.rider_group_model = YOLO(
                os.path.join(model_dir, "rider_group_best.pt")
            )
            self.helmet_model = YOLO(
                os.path.join(model_dir, "helmet_best.pt")
            )
            self.plate_model = YOLO(
                os.path.join(model_dir, "plate_best.pt")
            )

            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(
                use_textline_orientation=True,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                enable_mkldnn=False,
                lang="en",
            )
            logger.info("All models loaded. Device: %s", self.device)

        except Exception as e:
            logger.error("Model loading failed: %s", e)
            raise

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict(self, image_path: str) -> dict:
        """
        Main pipeline. Returns {"violations": [...]} always, never crashes.
        Each violation: {"num_riders": int, "helmet_violations": int, "license_plate": str}
        """
        start_time = time.time()
        MAX_TIME   = 30.0

        try:
            if not isinstance(image_path, str) or not os.path.exists(image_path):
                return {"violations": []}

            image = cv2.imread(image_path)
            if image is None or image.size == 0 or len(image.shape) != 3:
                return {"violations": []}

            image    = self._maybe_resize(image, max_dim=2500)
            original = image.copy()
            image    = self._enhance_if_dark(image)

            # Step 1: detect all rider groups
            rg_results = self.rider_group_model.predict(
                image, imgsz=640, conf=self.CONF_RIDER_GROUP,
                verbose=False, device=self.device,
            )

            if (not rg_results
                    or rg_results[0].boxes is None
                    or len(rg_results[0].boxes) == 0):
                return {"violations": []}

            # Deduplicate overlapping detections
            raw_boxes = []
            for i in range(len(rg_results[0].boxes)):
                raw_boxes.append({
                    "bbox":   rg_results[0].boxes.xyxy[i].cpu().numpy().tolist(),
                    "cls":    int(rg_results[0].boxes.cls[i].item()),
                    "conf":   float(rg_results[0].boxes.conf[i].item()),
                })
            raw_boxes = self._deduplicate(raw_boxes, iou_threshold=0.5)

            violations = []
            for det in raw_boxes:
                if time.time() - start_time > MAX_TIME:
                    logger.warning("Approaching time limit, returning partial results")
                    break

                try:
                    bbox      = det["bbox"]
                    is_triple = (det["cls"] == self.TRIPLE_RIDING_CLS)

                    # Step 2: helmet detection on rider_group crop
                    num_riders, helmet_violations = self._check_helmets(
                        image, bbox, is_triple
                    )

                    # Only record if there is an actual violation
                    if num_riders <= 2 and helmet_violations == 0:
                        continue

                    # Step 3: plate detection on expanded region
                    plate_text = self._detect_and_read_plate(original, bbox)

                    violations.append({
                        "num_riders":        num_riders,
                        "helmet_violations": helmet_violations,
                        "license_plate":     plate_text,
                    })

                except Exception as e:
                    logger.warning("Error on detection %s: %s", det, e)
                    continue

            elapsed = time.time() - start_time
            logger.info(
                "%s | Groups:%d Violations:%d | %.2fs",
                os.path.basename(image_path),
                len(raw_boxes), len(violations), elapsed,
            )
            return {"violations": violations}

        except Exception as e:
            logger.error("Pipeline error for %s: %s", image_path, e)
            return {"violations": []}

    # ------------------------------------------------------------------
    # Helmet detection
    # ------------------------------------------------------------------

    def _check_helmets(self, image, bbox, is_triple: bool):
        """
        Run helmet model on the rider_group crop.
        Returns (num_riders, helmet_violations).
        """
        try:
            x1, y1, x2, y2 = bbox
            crop = self._safe_crop(image, x1, y1, x2, y2)

            if crop is None or crop.shape[0] < 20 or crop.shape[1] < 20:
                # Can't analyse — assume violation
                return (3 if is_triple else 1), 1

            results = self.helmet_model.predict(
                crop, imgsz=960, conf=self.CONF_NO_HELMET,
                verbose=False, device=self.device,
            )

            helmet_boxes    = []
            no_helmet_boxes = []

            if results and results[0].boxes is not None:
                for j in range(len(results[0].boxes)):
                    h_cls  = int(results[0].boxes.cls[j].item())
                    h_conf = float(results[0].boxes.conf[j].item())
                    h_bbox = results[0].boxes.xyxy[j].cpu().numpy().tolist()
                    if h_cls == self.HELMET_CLS and h_conf >= self.CONF_HELMET:
                        helmet_boxes.append({"bbox": h_bbox, "conf": h_conf})
                    elif h_cls == self.NO_HELMET_CLS and h_conf >= self.CONF_NO_HELMET:
                        no_helmet_boxes.append({"bbox": h_bbox, "conf": h_conf})

            # Cross-class dedup: if helmet and no_helmet overlap, keep higher conf
            filtered_no_helmet = []
            for nh in no_helmet_boxes:
                dominated = False
                for h in helmet_boxes:
                    if self._compute_iou(nh["bbox"], h["bbox"]) > 0.5 and h["conf"] > nh["conf"]:
                        dominated = True
                        break
                if not dominated:
                    filtered_no_helmet.append(nh)

            filtered_helmet = []
            for h in helmet_boxes:
                dominated = False
                for nh in no_helmet_boxes:
                    if self._compute_iou(h["bbox"], nh["bbox"]) > 0.5 and nh["conf"] > h["conf"]:
                        dominated = True
                        break
                if not dominated:
                    filtered_helmet.append(h)

            helmet_count    = len(filtered_helmet)
            no_helmet_count = len(filtered_no_helmet)
            num_riders      = helmet_count + no_helmet_count

            # TTA fallback: zero detections → try CLAHE-enhanced crop
            if num_riders == 0:
                try:
                    enhanced_results = self.helmet_model.predict(
                        self._enhance_crop(crop), imgsz=960, conf=self.CONF_NO_HELMET,
                        verbose=False, device=self.device,
                    )
                    if enhanced_results and enhanced_results[0].boxes is not None:
                        for j in range(len(enhanced_results[0].boxes)):
                            h_cls  = int(enhanced_results[0].boxes.cls[j].item())
                            h_conf = float(enhanced_results[0].boxes.conf[j].item())
                            if h_cls == self.HELMET_CLS and h_conf >= self.CONF_HELMET:
                                helmet_count += 1
                            elif h_cls == self.NO_HELMET_CLS and h_conf >= self.CONF_NO_HELMET:
                                no_helmet_count += 1
                        num_riders = helmet_count + no_helmet_count
                except Exception:
                    pass

            # Reconcile rider count: Trust resolved head detections over bulky luggage/backpacks.
            # If 2 or more heads are clearly detected, use the actual head count.
            # If heads are occluded (<=1 head) and is_triple is True, fall back to 3.
            if num_riders >= 2:
                num_riders = num_riders
            elif is_triple:
                num_riders = 3
            elif num_riders == 0:
                num_riders = 1

            num_riders = min(num_riders, 5)

            return num_riders, no_helmet_count

        except Exception as e:
            logger.warning("Helmet check error: %s", e)
            return (3 if is_triple else 1), 1

    # ------------------------------------------------------------------
    # Plate detection + OCR
    # ------------------------------------------------------------------

    def _detect_and_read_plate(self, original_image, bbox):
        """Detect plate in/below rider group bbox, then OCR."""
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox]
            h = y2 - y1
            w = x2 - x1

            exp_x1 = x1 - int(w * 0.20)
            exp_x2 = x2 + int(w * 0.20)
            exp_y1 = y1 - int(h * 0.10)
            exp_y2 = y2 + int(h * 0.50)

            search_crop = self._safe_crop(original_image, exp_x1, exp_y1, exp_x2, exp_y2)
            if search_crop is None:
                return ""

            results = self.plate_model.predict(
                search_crop, imgsz=640, conf=self.CONF_PLATE,
                verbose=False, device=self.device,
            )

            best_text = ""
            best_conf = 0.0

            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes          = results[0].boxes
                sorted_indices = boxes.conf.argsort(descending=True)

                for idx in sorted_indices[:3]:
                    try:
                        px1, py1, px2, py2 = boxes.xyxy[idx].cpu().numpy().astype(int)
                        abs_x1 = max(0, int(exp_x1)) + px1
                        abs_y1 = max(0, int(exp_y1)) + py1
                        abs_x2 = max(0, int(exp_x1)) + px2
                        abs_y2 = max(0, int(exp_y1)) + py2

                        plate_crop = self._safe_crop(original_image,
                                                     abs_x1, abs_y1, abs_x2, abs_y2)
                        if plate_crop is None:
                            continue
                        ph, pw = plate_crop.shape[:2]
                        if ph < 8 or pw < 15:
                            continue

                        text, conf = self._ocr_plate(plate_crop)
                        if conf > best_conf and len(text) >= 4:
                            best_conf = conf
                            best_text = text
                    except Exception:
                        continue

            # Fallback: OCR on lower 40% of search region
            if not best_text:
                try:
                    sh         = search_crop.shape[0]
                    lower_crop = search_crop[int(sh * 0.6):, :]
                    if lower_crop.shape[0] > 10 and lower_crop.shape[1] > 20:
                        text, conf = self._ocr_plate(lower_crop)
                        if len(text) >= 4:
                            best_text = text
                except Exception:
                    pass

            return best_text

        except Exception as e:
            logger.warning("Plate detection error: %s", e)
            return ""

    def _ocr_plate(self, plate_crop):
        """Run PaddleOCR on preprocessed variants, return (text, confidence)."""
        try:
            if plate_crop is None or plate_crop.size == 0:
                return ("", 0.0)

            variants  = self._generate_plate_variants(plate_crop)
            best_text = ""
            best_conf = 0.0

            for variant in variants:
                if variant is None or variant.size == 0:
                    continue
                try:
                    if len(variant.shape) == 2:
                        variant = cv2.cvtColor(variant, cv2.COLOR_GRAY2BGR)

                    result = self.ocr.predict(variant)
                    if not result:
                        continue

                    texts = []
                    confs = []
                    for item in result:
                        if not item:
                            continue
                        for text, score in zip(
                            item.get("rec_texts", []),
                            item.get("rec_scores", []),
                        ):
                            if text:
                                texts.append(text)
                                confs.append(float(score))

                    if not texts:
                        continue

                    avg_conf  = sum(confs) / len(confs)
                    full_text = " ".join(texts)

                    if avg_conf > best_conf and len(full_text.strip()) >= 3:
                        best_conf = avg_conf
                        best_text = full_text

                except Exception:
                    continue

            if best_text:
                best_text = self._clean_plate_text(best_text)

            return (best_text, best_conf)

        except Exception as e:
            logger.warning("OCR error: %s", e)
            return ("", 0.0)

    def _generate_plate_variants(self, plate_crop):
        """3 preprocessed versions targeting different plate degradation types."""
        variants = []
        try:
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        except Exception:
            return [plate_crop]

        # 1: original
        variants.append(plate_crop)

        # 2: CLAHE — faded/low-contrast plates
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            variants.append(cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR))
        except Exception:
            pass

        # 3: 2x upscale + sharpen — small/distant plates
        try:
            h, w = plate_crop.shape[:2]
            if h < 60 or w < 120:
                up = cv2.resize(plate_crop, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
                blur = cv2.GaussianBlur(up, (0, 0), 3)
                sharpened = cv2.addWeighted(up, 1.5, blur, -0.5, 0)
                variants.append(sharpened)
        except Exception:
            pass

        return variants if variants else [plate_crop]

    def _clean_plate_text(self, raw_text):
        """Post-process OCR output for Indian license plates."""
        try:
            if not raw_text:
                return ""

            text = re.sub(r"[^A-Z0-9]", "", raw_text.upper().strip())

            if len(text) < 4:
                return text

            digit_to_letter = {"0": "O", "1": "I", "8": "B", "5": "S",
                                "6": "G", "2": "Z", "4": "A"}
            letter_to_digit = {"O": "0", "I": "1", "B": "8", "S": "5",
                                "G": "6", "Z": "2", "D": "0", "A": "4", "T": "7"}

            corrected = list(text)

            # pos 0-1: state code — must be letters
            for i in range(min(2, len(corrected))):
                c = corrected[i]
                if c.isdigit() and c in digit_to_letter:
                    corrected[i] = digit_to_letter[c]

            # pos 2-3: district code — must be digits
            for i in range(2, min(4, len(corrected))):
                c = corrected[i]
                if c.isalpha() and c in letter_to_digit:
                    corrected[i] = letter_to_digit[c]

            # remaining: series letters then final digits
            if len(corrected) > 4:
                rest        = corrected[4:]
                digit_start = len(rest)
                for j in range(len(rest) - 1, -1, -1):
                    if rest[j].isdigit() or rest[j] in letter_to_digit:
                        digit_start = j
                    else:
                        break
                for j in range(digit_start):
                    if rest[j].isdigit() and rest[j] in digit_to_letter:
                        rest[j] = digit_to_letter[rest[j]]
                for j in range(digit_start, len(rest)):
                    if rest[j].isalpha() and rest[j] in letter_to_digit:
                        rest[j] = letter_to_digit[rest[j]]
                corrected = corrected[:4] + rest

            result = "".join(corrected)

            if re.match(self.INDIAN_PLATE_REGEX, result):
                return result

            return result

        except Exception:
            return re.sub(r"[^A-Z0-9]", "", raw_text.upper().strip()) if raw_text else ""

    # ------------------------------------------------------------------
    # Pre-processing helpers
    # ------------------------------------------------------------------

    def _maybe_resize(self, image, max_dim=2500):
        try:
            h, w = image.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                return cv2.resize(image, (int(w * scale), int(h * scale)),
                                  interpolation=cv2.INTER_AREA)
            return image
        except Exception:
            return image

    def _enhance_if_dark(self, image):
        try:
            mean_intensity = float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean())
            if mean_intensity < 50:
                inv_gamma = 1.0 / 0.4
                lut = np.array(
                    [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
                    dtype=np.uint8,
                )
                return cv2.LUT(image, lut)
            if mean_intensity < 80:
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)
            return image
        except Exception:
            return image

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _enhance_crop(self, crop):
        """CLAHE enhancement on a BGR crop. TTA fallback for dark/low-contrast images."""
        try:
            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)
        except Exception:
            return crop

    def _deduplicate(self, boxes, iou_threshold):
        try:
            if not boxes:
                return boxes
            sorted_boxes = sorted(boxes, key=lambda x: x["conf"], reverse=True)
            kept = []
            for box in sorted_boxes:
                if all(
                    self._compute_iou(box["bbox"], k["bbox"]) < iou_threshold
                    for k in kept
                ):
                    kept.append(box)
            return kept
        except Exception:
            return boxes

    def _safe_crop(self, image, x1, y1, x2, y2):
        try:
            h, w = image.shape[:2]
            x1 = max(0, int(x1));  y1 = max(0, int(y1))
            x2 = min(w, int(x2));  y2 = min(h, int(y2))
            if x2 <= x1 or y2 <= y1:
                return None
            return image[y1:y2, x1:x2]
        except Exception:
            return None

    def _compute_iou(self, box1, box2):
        try:
            x1 = max(box1[0], box2[0]);  y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2]);  y2 = min(box1[3], box2[3])
            if x2 <= x1 or y2 <= y1:
                return 0.0
            inter  = (x2 - x1) * (y2 - y1)
            area1  = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2  = (box2[2] - box2[0]) * (box2[3] - box2[1])
            return inter / max(area1 + area2 - inter, 1e-6)
        except Exception:
            return 0.0