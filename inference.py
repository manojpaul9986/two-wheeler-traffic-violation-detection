"""
Two-Wheeler Traffic Rule Violation Detection — Inference CLI
Group 22 | MT2025709 & MT2025714 | IIIT Bangalore

Usage:
    # Run Production Multi-Scale Detector (Default: V2)
    python inference.py --image test_image_1.jpg

    # Run with Visual Bounding Box Annotations saved to file
    python inference.py --image test_image_3.png --save_vis

    # Run Fast Baseline Detector (V1)
    python inference.py --variant v1 --image test_image_8.png --save_vis "output_8.jpg"
"""

import sys
import os
import json
import cv2
import argparse
from pipelines import get_pipeline
from modules.visualizer import annotate_traffic_violations


def main():
    parser = argparse.ArgumentParser(
        description="Two-Wheeler Traffic Rule Violation Inference (Group 22 - IIIT Bangalore)"
    )
    parser.add_argument("pos_image", nargs="?", default=None, help="Positional image path")
    parser.add_argument("--image", "-i", type=str, default=None, help="Path to input traffic frame")
    parser.add_argument(
        "--variant", "-v", type=str, default="v2", choices=["v1", "v2"],
        help="Pipeline variant: 'v1' (Fast Baseline) or 'v2' (Multi-Scale Production SOTA, default: v2)"
    )
    parser.add_argument("--model_dir", "-m", type=str, default="./models", help="Models directory path")
    parser.add_argument(
        "--save_vis", "-s", nargs="?", const="default", default=None,
        help="Save visual color-coded bounding box annotations to image file"
    )
    args = parser.parse_args()

    # Determine image path from either positional argument or --image flag
    image_path = args.image or args.pos_image

    if not image_path:
        image_path = "test_image_1.jpg"
        print(f"[INFO] No image path provided. Running default test path: '{image_path}'\n")

    print(f"[*] Initializing Pipeline Variant: {args.variant.upper()}...")
    detector = get_pipeline(variant=args.variant, model_dir=args.model_dir)

    print(f"[*] Running inference on: {image_path}")
    output = detector.predict(image_path)

    print("\n--- Detection Result ---")
    print(json.dumps(output, indent=2))

    # Visual Annotation Generation
    if args.save_vis is not None and os.path.exists(image_path):
        if args.save_vis == "default":
            base, ext = os.path.splitext(image_path)
            vis_path = f"{base}_annotated{ext or '.jpg'}"
        else:
            vis_path = args.save_vis

        img = cv2.imread(image_path)
        if img is not None:
            annotate_traffic_violations(img, detector, output_path=vis_path)
            print(f"\n[+] Visual bounding boxes saved to: {vis_path}")


if __name__ == "__main__":
    main()