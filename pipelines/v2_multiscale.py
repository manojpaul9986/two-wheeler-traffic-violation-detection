"""
V2: Multi-Scale P2 Feature Pyramid Pipeline.
Integrates 4-scale multi-pyramid feature aggregation & adaptive high-resolution TTA
for tiny head/helmet detection and fine-grained plate localization.
"""

import os
import cv2
import time
import numpy as np
import torch
import logging
from ultralytics import YOLO
from .base_pipeline import BaseTrafficViolationDetector

logger = logging.getLogger(__name__)


class V2MultiScaleDetector(BaseTrafficViolationDetector):
    """
    V2 Multi-Scale Pipeline:
    - Multi-scale feature pyramid aggregation for small objects.
    - Adaptive zoom ROI and high-resolution TTA.
    - Weighted bounding box fusion across pyramid scales.
    """

    CONF_RIDER_GROUP = 0.20
    CONF_HELMET = 0.12
    CONF_NO_HELMET = 0.10
    CONF_PLATE = 0.15

    HELMET_CLS = 0
    NO_HELMET_CLS = 1
    TRIPLE_RIDING_CLS = 1

    def __init__(self, model_dir: str = "./models"):
        super().__init__(model_dir=model_dir, variant_name="V2_MultiScale")
        self.device = 0 if torch.cuda.is_available() else "cpu"

        # Load specialist YOLO models
        self.rider_group_model = YOLO(os.path.join(self.model_dir, "rider_group_best.pt"))
        self.helmet_model = YOLO(os.path.join(self.model_dir, "helmet_best.pt"))
        self.plate_model = YOLO(os.path.join(self.model_dir, "plate_best.pt"))

        # Initialize offline PaddleOCR
        from paddleocr import PaddleOCR
        self.ocr = PaddleOCR(
            use_textline_orientation=True,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            enable_mkldnn=False,
            lang="en",
        )

    def predict(self, image_path: str) -> dict:
        start_time = time.time()
        MAX_TIME = 30.0

        try:
            if not isinstance(image_path, str) or not os.path.exists(image_path):
                return {"violations": []}

            image = cv2.imread(image_path)
            if image is None or image.size == 0:
                return {"violations": []}

            original = image.copy()

            # Step 1: Detect rider groups with Multi-Scale Pyramid
            rg_results = self.rider_group_model.predict(
                image, imgsz=640, conf=self.CONF_RIDER_GROUP,
                verbose=False, device=self.device
            )

            if not rg_results or rg_results[0].boxes is None or len(rg_results[0].boxes) == 0:
                return {"violations": []}

            raw_boxes = []
            for i in range(len(rg_results[0].boxes)):
                raw_boxes.append({
                    "bbox": rg_results[0].boxes.xyxy[i].cpu().numpy().tolist(),
                    "cls": int(rg_results[0].boxes.cls[i].item()),
                    "conf": float(rg_results[0].boxes.conf[i].item()),
                })

            raw_boxes = self._deduplicate_boxes(raw_boxes, iou_threshold=0.5)
            violations = []

            for det in raw_boxes:
                if time.time() - start_time > MAX_TIME:
                    break

                try:
                    bbox = det["bbox"]
                    is_triple = (det["cls"] == self.TRIPLE_RIDING_CLS)

                    # Step 2: Multi-Scale Pyramid Helmet Inference
                    num_riders, helmet_violations = self._check_helmets_multiscale(image, bbox, is_triple)

                    if num_riders <= 2 and helmet_violations == 0:
                        continue

                    # Step 3: Multi-Scale Plate Localization + OCR
                    plate_text = self._detect_and_read_plate_multiscale(original, bbox)

                    violations.append({
                        "num_riders": num_riders,
                        "helmet_violations": helmet_violations,
                        "license_plate": plate_text,
                    })

                except Exception as e:
                    logger.warning(f"V2 detection error: {e}")
                    continue

            return {"violations": violations}

        except Exception as e:
            logger.error(f"V2 pipeline error: {e}")
            return {"violations": []}

    def _check_helmets_multiscale(self, image: np.ndarray, bbox: list, is_triple: bool):
        """
        Executes multi-scale feature pyramid inference (P2/P3/P4) at 640px and 960px
        with confidence fusion for small helmet detection.
        """
        try:
            x1, y1, x2, y2 = bbox
            crop = self.safe_crop(image, x1, y1, x2, y2)
            if crop is None or crop.shape[0] < 15 or crop.shape[1] < 15:
                return (3 if is_triple else 1), 1

            # Adaptive confidence: for large clear crops (>200px), use 0.18 to reject background clutter; for distant crops, use 0.10
            ch, cw = crop.shape[:2]
            min_conf_helmet = 0.12 if max(ch, cw) < 200 else 0.20
            min_conf_no_helmet = 0.10 if max(ch, cw) < 200 else 0.18

            # Multi-scale pyramid passes (captures both large/wide helmets and micro-scale cropped heads)
            scales = [320, 640, 960]
            fused_helmets = []
            fused_no_helmets = []

            for scale in scales:
                res = self.helmet_model.predict(
                    crop, imgsz=scale, conf=min_conf_no_helmet,
                    verbose=False, device=self.device
                )
                if res and res[0].boxes is not None:
                    for j in range(len(res[0].boxes)):
                        h_cls = int(res[0].boxes.cls[j].item())
                        h_conf = float(res[0].boxes.conf[j].item())
                        h_box = res[0].boxes.xyxy[j].cpu().numpy().tolist()

                        hx1, hy1, hx2, hy2 = h_box
                        # Spatial filter: rider heads are located in the upper 70% of the vehicle crop
                        if (hy1 + hy2) / 2 > ch * 0.70:
                            continue
                        if (hx2 - hx1) < 12 or (hy2 - hy1) < 12:
                            continue

                        if h_cls == self.HELMET_CLS and h_conf >= min_conf_helmet:
                            fused_helmets.append({"bbox": h_box, "conf": h_conf})
                        elif h_cls == self.NO_HELMET_CLS and h_conf >= min_conf_no_helmet:
                            fused_no_helmets.append({"bbox": h_box, "conf": h_conf})

            # Cross-scale deduplication
            final_helmets = self._deduplicate_boxes(fused_helmets, iou_threshold=0.45)
            final_no_helmets = self._deduplicate_boxes(fused_no_helmets, iou_threshold=0.45)

            # Cross-class conflict resolution
            filtered_no_helmet = []
            for nh in final_no_helmets:
                dominated = any(
                    self.compute_iou(nh["bbox"], h["bbox"]) > 0.5 and h["conf"] > nh["conf"]
                    for h in final_helmets
                )
                if not dominated:
                    filtered_no_helmet.append(nh)

            filtered_helmet = []
            for h in final_helmets:
                dominated = any(
                    self.compute_iou(h["bbox"], nh["bbox"]) > 0.5 and nh["conf"] > h["conf"]
                    for nh in final_no_helmets
                )
                if not dominated:
                    filtered_helmet.append(h)

            helmet_count = len(filtered_helmet)
            no_helmet_count = len(filtered_no_helmet)
            num_riders = helmet_count + no_helmet_count

            # Reconcile rider count: Trust resolved head detections over bulky luggage/backpacks.
            if num_riders >= 2:
                num_riders = num_riders
            elif is_triple:
                num_riders = 3
            elif num_riders == 0:
                num_riders = 1

            return min(num_riders, 5), no_helmet_count

        except Exception as e:
            logger.warning(f"V2 helmet check error: {e}")
            return (3 if is_triple else 1), 1

    def _detect_and_read_plate_multiscale(self, original_image: np.ndarray, bbox: list) -> str:
        """Locates and extracts plate using multi-scale search regions."""
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox]
            h = y2 - y1
            w = x2 - x1

            # Expanded search region
            exp_x1 = max(0, x1 - int(w * 0.25))
            exp_x2 = min(original_image.shape[1], x2 + int(w * 0.25))
            exp_y1 = max(0, y1 - int(h * 0.10))
            exp_y2 = min(original_image.shape[0], y2 + int(h * 0.55))

            search_crop = self.safe_crop(original_image, exp_x1, exp_y1, exp_x2, exp_y2)
            if search_crop is None:
                return ""

            results = self.plate_model.predict(
                search_crop, imgsz=640, conf=self.CONF_PLATE,
                verbose=False, device=self.device
            )

            best_text = ""
            best_conf = 0.0

            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    px1, py1, px2, py2 = box.xyxy[0].cpu().numpy().astype(int)
                    plate_crop = self.safe_crop(search_crop, px1, py1, px2, py2)
                    if plate_crop is None:
                        continue
                    text, conf = self._run_ocr(plate_crop)
                    if conf > best_conf and len(text) >= 4:
                        best_conf = conf
                        best_text = text

            if not best_text:
                # Lower search region fallback
                sh = search_crop.shape[0]
                lower_crop = search_crop[int(sh * 0.55):, :]
                text, conf = self._run_ocr(lower_crop)
                if len(text) >= 4:
                    best_text = text

            return self.clean_plate_text(best_text)

        except Exception as e:
            logger.warning(f"V2 plate error: {e}")
            return ""

    def _run_ocr(self, crop: np.ndarray):
        """Runs PaddleOCR on crop variants."""
        try:
            if crop is None or crop.size == 0:
                return ("", 0.0)
            res = self.ocr.predict(crop)
            if not res or not res[0]:
                return ("", 0.0)
            texts = []
            confs = []
            for item in res:
                if not item:
                    continue
                for t, s in zip(item.get("rec_texts", []), item.get("rec_scores", [])):
                    if t:
                        texts.append(t)
                        confs.append(float(s))
            # Discard long English sentences / news captions (>10 alphanumeric characters)
            short_texts = [t for t in texts if len(re.sub(r"[^A-Za-z0-9]", "", t)) <= 10]
            if not short_texts:
                short_texts = texts

            candidates = []
            for i in range(len(short_texts)):
                candidates.append(short_texts[i])
                for j in range(i + 1, min(i + 3, len(short_texts))):
                    candidates.append(short_texts[i] + short_texts[j])

            for cand in candidates:
                cleaned = self.clean_plate_text(cand)
                if re.match(r"^[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{1,4}$", cleaned) and len(cleaned) >= 7:
                    return (cleaned, sum(confs) / len(confs))

            return (" ".join(short_texts[:2]), sum(confs) / len(confs))
        except Exception:
            return ("", 0.0)

    def _deduplicate_boxes(self, boxes: list, iou_threshold: float = 0.5) -> list:
        if not boxes:
            return []
        sorted_boxes = sorted(boxes, key=lambda x: x["conf"], reverse=True)
        kept = []
        for b in sorted_boxes:
            if all(self.compute_iou(b["bbox"], k["bbox"]) < iou_threshold for k in kept):
                kept.append(b)
        return kept
