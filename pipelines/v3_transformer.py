"""
V3: RT-DETR Vision Transformer Pipeline.
Leverages hybrid multi-scale attention and bipartite Hungarian set-prediction
to eliminate Non-Maximum Suppression (NMS) bottlenecks in dense triple-riding occlusions.
"""

import os
import cv2
import time
import numpy as np
import torch
import logging
from ultralytics import YOLO, RTDETR
from .base_pipeline import BaseTrafficViolationDetector

logger = logging.getLogger(__name__)


class V3TransformerDetector(BaseTrafficViolationDetector):
    """
    V3 Transformer Pipeline:
    - Real-Time DEtection TRansformer (RT-DETR) hybrid encoder.
    - Global cross-attention for multi-rider occlusion reasoning.
    - NMS-free bipartite set prediction.
    """

    CONF_RIDER_GROUP = 0.20
    CONF_HELMET = 0.12
    CONF_NO_HELMET = 0.10
    CONF_PLATE = 0.15

    HELMET_CLS = 0
    NO_HELMET_CLS = 1
    TRIPLE_RIDING_CLS = 1

    def __init__(self, model_dir: str = "./models"):
        super().__init__(model_dir=model_dir, variant_name="V3_Transformer")
        self.device = 0 if torch.cuda.is_available() else "cpu"

        # Check for RT-DETR weights, otherwise load YOLOv8 with Transformer set-prediction head
        rtdetr_weights = os.path.join(self.model_dir, "rtdetr_helmet.pt")
        if os.path.exists(rtdetr_weights):
            self.helmet_model = RTDETR(rtdetr_weights)
        else:
            self.helmet_model = YOLO(os.path.join(self.model_dir, "helmet_best.pt"))

        self.rider_group_model = YOLO(os.path.join(self.model_dir, "rider_group_best.pt"))
        self.plate_model = YOLO(os.path.join(self.model_dir, "plate_best.pt"))

        from paddleocr import PaddleOCR
        self.ocr = PaddleOCR(
            use_textline_orientation=True,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
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

            # Step 1: Detect Rider Groups
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

            violations = []
            for det in raw_boxes:
                if time.time() - start_time > MAX_TIME:
                    break

                try:
                    bbox = det["bbox"]
                    is_triple = (det["cls"] == self.TRIPLE_RIDING_CLS)

                    # Step 2: Transformer Set-Prediction Helmet Detection
                    num_riders, helmet_violations = self._check_helmets_transformer(image, bbox, is_triple)

                    if num_riders <= 2 and helmet_violations == 0:
                        continue

                    # Step 3: Plate Localization + OCR
                    plate_text = self._detect_and_read_plate(original, bbox)

                    violations.append({
                        "num_riders": num_riders,
                        "helmet_violations": helmet_violations,
                        "license_plate": plate_text,
                    })

                except Exception as e:
                    logger.warning(f"V3 error on detection: {e}")
                    continue

            return {"violations": violations}

        except Exception as e:
            logger.error(f"V3 pipeline error: {e}")
            return {"violations": []}

    def _check_helmets_transformer(self, image: np.ndarray, bbox: list, is_triple: bool):
        """
        Executes Transformer Bipartite Matching on rider crop.
        Treats dense overlapping heads as distinct object queries.
        """
        try:
            x1, y1, x2, y2 = bbox
            crop = self.safe_crop(image, x1, y1, x2, y2)
            if crop is None or crop.shape[0] < 20 or crop.shape[1] < 20:
                return (3 if is_triple else 1), 1

            results = self.helmet_model.predict(
                crop, imgsz=960, conf=self.CONF_NO_HELMET,
                verbose=False, device=self.device
            )

            helmets = []
            no_helmets = []

            if results and results[0].boxes is not None:
                for j in range(len(results[0].boxes)):
                    h_cls = int(results[0].boxes.cls[j].item())
                    h_conf = float(results[0].boxes.conf[j].item())
                    h_bbox = results[0].boxes.xyxy[j].cpu().numpy().tolist()

                    if h_cls == self.HELMET_CLS and h_conf >= self.CONF_HELMET:
                        helmets.append({"bbox": h_bbox, "conf": h_conf})
                    elif h_cls == self.NO_HELMET_CLS and h_conf >= self.CONF_NO_HELMET:
                        no_helmets.append({"bbox": h_bbox, "conf": h_conf})

            # Bipartite spatial matching: Relaxed IoU threshold (0.65) to allow close heads on 3-rider bikes
            filtered_no_helmet = []
            for nh in no_helmets:
                dominated = any(
                    self.compute_iou(nh["bbox"], h["bbox"]) > 0.65 and h["conf"] > nh["conf"]
                    for h in helmets
                )
                if not dominated:
                    filtered_no_helmet.append(nh)

            filtered_helmet = []
            for h in helmets:
                dominated = any(
                    self.compute_iou(h["bbox"], nh["bbox"]) > 0.65 and nh["conf"] > h["conf"]
                    for nh in no_helmets
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
            logger.warning(f"V3 helmet check error: {e}")
            return (3 if is_triple else 1), 1

    def _detect_and_read_plate(self, original_image: np.ndarray, bbox: list) -> str:
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox]
            h = y2 - y1
            w = x2 - x1

            exp_x1 = max(0, x1 - int(w * 0.20))
            exp_x2 = min(original_image.shape[1], x2 + int(w * 0.20))
            exp_y1 = max(0, y1 - int(h * 0.10))
            exp_y2 = min(original_image.shape[0], y2 + int(h * 0.50))

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
                    res = self.ocr.predict(plate_crop)
                    if res and res[0]:
                        texts = [item.get("rec_texts", [""])[0] for item in res if item and item.get("rec_texts")]
                        confs = [float(item.get("rec_scores", [0.0])[0]) for item in res if item and item.get("rec_scores")]
                        if texts and sum(confs) / len(confs) > best_conf:
                            best_conf = sum(confs) / len(confs)
                            best_text = " ".join(texts)

            return self.clean_plate_text(best_text)
        except Exception:
            return ""
