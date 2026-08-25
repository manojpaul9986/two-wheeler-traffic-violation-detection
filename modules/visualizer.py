"""
Visual Annotation Module for Traffic Rule Violations.
Draws color-coded bounding boxes for:
- Two-Wheelers / Rider Groups (Cyan / Orange)
- Helmets (Green)
- Helmet Violations (Red)
- License Plates with Decoded Text (Blue / Yellow)
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


def annotate_traffic_violations(image: np.ndarray, detector, output_path: str = None) -> np.ndarray:
    """
    Runs detection and annotates color-coded bounding boxes on the image.
    """
    if image is None or image.size == 0:
        return image

    vis = image.copy()
    img_h, img_w = image.shape[:2]

    try:
        # 1. Detect Rider Groups
        rg_res = detector.rider_group_model.predict(image, imgsz=640, conf=0.18, verbose=False, device=detector.device)
        if not rg_res or rg_res[0].boxes is None:
            if output_path:
                cv2.imwrite(output_path, vis)
            return vis

        for box in rg_res[0].boxes:
            bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy().astype(int)
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            is_triple = (cls_id == 1)

            rg_label = f"{'TRIPLE RIDING' if is_triple else 'Rider Group'}: {conf:.2f}"
            rg_color = (0, 140, 255) if is_triple else (255, 200, 0) # Orange if triple, Cyan if normal
            
            # Draw motorcycle box
            cv2.rectangle(vis, (bx1, by1), (bx2, by2), rg_color, 2)
            cv2.putText(vis, rg_label, (bx1, max(18, by1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, rg_color, 2)

            # 2. Detect Helmets in Crop
            crop = image[max(0, by1):min(img_h, by2), max(0, bx1):min(img_w, bx2)]
            if crop.size > 0:
                for scale in [320, 640, 960]:
                    h_res = detector.helmet_model.predict(crop, imgsz=scale, conf=0.10, verbose=False, device=detector.device)
                    if h_res and h_res[0].boxes is not None:
                        for h_box in h_res[0].boxes:
                            hx1, hy1, hx2, hy2 = h_box.xyxy[0].cpu().numpy().astype(int)
                            abs_hx1, abs_hy1 = bx1 + hx1, by1 + hy1
                            abs_hx2, abs_hy2 = bx1 + hx2, by1 + hy2
                            h_cls = int(h_box.cls[0].item())
                            h_conf = float(h_box.conf[0].item())

                            if h_cls == 0 and h_conf >= 0.12:  # Helmet
                                h_color = (0, 255, 0) # Green
                                h_text = f"Helmet: {h_conf:.2f}"
                                cv2.rectangle(vis, (abs_hx1, abs_hy1), (abs_hx2, abs_hy2), h_color, 2)
                                cv2.putText(vis, h_text, (abs_hx1, max(12, abs_hy1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, h_color, 1)
                            elif h_cls == 1 and h_conf >= 0.10: # No Helmet
                                h_color = (0, 0, 255) # Red
                                h_text = f"NO HELMET: {h_conf:.2f}"
                                cv2.rectangle(vis, (abs_hx1, abs_hy1), (abs_hx2, abs_hy2), h_color, 2)
                                cv2.putText(vis, h_text, (abs_hx1, max(12, abs_hy1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, h_color, 1)

            # 3. Detect License Plate in Expanded Area
            h_crop, w_crop = by2 - by1, bx2 - bx1
            exp_x1 = max(0, int(bx1 - w_crop * 0.20))
            exp_x2 = min(img_w, int(bx2 + w_crop * 0.20))
            exp_y1 = max(0, int(by1 - h_crop * 0.10))
            exp_y2 = min(img_h, int(by2 + h_crop * 0.50))
            search_crop = image[exp_y1:exp_y2, exp_x1:exp_x2]

            if search_crop.size > 0:
                plate_res = detector.plate_model.predict(search_crop, imgsz=640, conf=0.10, verbose=False, device=detector.device)
                if plate_res and plate_res[0].boxes is not None:
                    for p_box in plate_res[0].boxes:
                        px1, py1, px2, py2 = p_box.xyxy[0].cpu().numpy().astype(int)
                        abs_px1, abs_py1 = exp_x1 + px1, exp_y1 + py1
                        abs_px2, abs_py2 = exp_x1 + px2, exp_y1 + py2
                        p_conf = float(p_box.conf[0].item())

                        plate_crop = search_crop[py1:py2, px1:px2]
                        if plate_crop.size > 0:
                            plate_text = ""
                            try:
                                if hasattr(detector, '_ocr_plate'):
                                    plate_text, _ = detector._ocr_plate(plate_crop)
                                elif hasattr(detector, '_run_ocr'):
                                    plate_text, _ = detector._run_ocr(plate_crop)
                            except Exception:
                                pass

                            p_label = f"Plate [{plate_text}]" if plate_text else f"Plate: {p_conf:.2f}"
                            cv2.rectangle(vis, (abs_px1, abs_py1), (abs_px2, abs_py2), (255, 255, 0), 2)
                            cv2.putText(vis, p_label, (abs_px1, max(16, abs_py1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 2)

        if output_path:
            cv2.imwrite(output_path, vis)

        return vis

    except Exception as e:
        logger.warning(f"Visualization error: {e}")
        if output_path:
            cv2.imwrite(output_path, vis)
        return vis
