"""
Inference script supporting Multi-Pipeline Variants (V1, V2, V3, V4).
Usage:
    python inference.py --image path/to/image.jpg
    python inference.py --variant v2 --image path/to/image.jpg
    python inference.py --variant v3 --image path/to/image.jpg
    python inference.py --variant v4 --image path/to/image.jpg
"""

import sys
import os
import json
import argparse
from pipelines import get_pipeline


def main():
    parser = argparse.ArgumentParser(description="Two-Wheeler Traffic Rule Violation Inference")
    parser.add_argument("pos_image", nargs="?", default=None, help="Positional image path")
    parser.add_argument("--image", "-i", type=str, default=None, help="Path to input traffic frame")
    parser.add_argument("--variant", "-v", type=str, default="v1", choices=["v1", "v2", "v3", "v4"],
                        help="Pipeline variant to run (default: v1)")
    parser.add_argument("--model_dir", "-m", type=str, default="./models", help="Models directory")
    args = parser.parse_args()

    # Determine image path from either positional argument or --image flag
    image_path = args.image or args.pos_image

    if not image_path:
        image_path = "test_image.jpg"
        print(f"[INFO] No image path provided. Usage: python inference.py --variant {args.variant} --image <path_to_image>")
        print(f"[INFO] Running default test path: '{image_path}' (safe non-crashing fallback)\n")

    print(f"[*] Initializing Pipeline Variant: {args.variant.upper()}...")
    detector = get_pipeline(variant=args.variant, model_dir=args.model_dir)

    print(f"[*] Running inference on: {image_path}")
    output = detector.predict(image_path)

    print("\n--- Detection Result ---")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()