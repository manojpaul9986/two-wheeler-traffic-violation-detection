"""
Comparative Benchmark Suite: V1 (Fast Baseline) vs. V2 (Multi-Scale SOTA)
Group 22 | MT2025709 & MT2025714 | IIIT Bangalore

Usage:
    python benchmark.py
    python benchmark.py --images test_image_1.jpg test_image_2.jpg
"""

import os
import sys
import time
import argparse
from pipelines.v1_baseline import V1BaselineDetector
from pipelines.v2_multiscale import V2MultiScaleDetector


def run_benchmark(model_dir: str = "./models", custom_images: list = None):
    print("=" * 105)
    print("🚀 TWO-WHEELER TRAFFIC VIOLATION DETECTION — COMPREHENSIVE BENCHMARK")
    print("   Group 22 | MT2025709 & MT2025714 | IIIT Bangalore")
    print("=" * 105)

    # Initialize detectors
    print("\n[*] Loading V1 (Fast Baseline) & V2 (Multi-Scale Production) Pipelines...")
    v1 = V1BaselineDetector(model_dir=model_dir)
    v2 = V2MultiScaleDetector(model_dir=model_dir)

    # Find test images
    if custom_images:
        images = custom_images
    else:
        images = []
        for i in range(1, 15):
            for ext in [".jpg", ".png", ".jpeg"]:
                p = f"test_image_{i}{ext}"
                if os.path.exists(p):
                    images.append(p)
                    break

    if not images:
        print("[!] No test images found in directory.")
        return

    print(f"[*] Found {len(images)} test image(s) to benchmark: {images}\n")

    results = []
    for img in images:
        # Run V1
        t0 = time.perf_counter()
        out1 = v1.predict(img)
        lat1 = (time.perf_counter() - t0) * 1000

        # Run V2
        t0 = time.perf_counter()
        out2 = v2.predict(img)
        lat2 = (time.perf_counter() - t0) * 1000

        v1_viol = out1.get("violations", [])
        v2_viol = out2.get("violations", [])

        r1 = v1_viol[0]["num_riders"] if v1_viol else 0
        h1 = v1_viol[0]["helmet_violations"] if v1_viol else 0
        p1 = v1_viol[0]["license_plate"] if v1_viol else ""

        r2 = v2_viol[0]["num_riders"] if v2_viol else 0
        h2 = v2_viol[0]["helmet_violations"] if v2_viol else 0
        p2 = v2_viol[0]["license_plate"] if v2_viol else ""

        results.append({
            "img": img,
            "lat1": lat1, "r1": r1, "h1": h1, "p1": p1,
            "lat2": lat2, "r2": r2, "h2": h2, "p2": p2,
        })

    # Print Clean Formatted Summary Table
    print("\n" + "=" * 105)
    header = "{:<18} | {:<7} | {:<10} | {:<12} | {:<12} | {:<7} | {:<10} | {:<12} | {:<12}".format(
        "IMAGE", "V1 LAT", "V1 RIDERS", "V1 HELM VIO", "V1 PLATE", "V2 LAT", "V2 RIDERS", "V2 HELM VIO", "V2 PLATE"
    )
    print(header)
    print("-" * 105)
    for r in results:
        row = "{:<18} | {:>5.0f}ms | {:>10} | {:>12} | {:<12} | {:>5.0f}ms | {:>10} | {:>12} | {:<12}".format(
            r["img"], r["lat1"], r["r1"], r["h1"], r["p1"], r["lat2"], r["r2"], r["h2"], r["p2"]
        )
        print(row)

    avg_lat1 = sum(r["lat1"] for r in results) / len(results)
    avg_lat2 = sum(r["lat2"] for r in results) / len(results)
    print("-" * 105)
    print("{:<18} | {:>5.0f}ms | {:<10} | {:<12} | {:<12} | {:>5.0f}ms |".format(
        "AVERAGE LATENCY", avg_lat1, "", "", "", avg_lat2
    ))
    print("=" * 105 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-Wheeler Traffic Rule Violation Detection Benchmark")
    parser.add_argument("--model_dir", type=str, default="./models", help="Path to models directory")
    parser.add_argument("--images", nargs="*", default=None, help="List of image paths to benchmark")
    args = parser.parse_args()

    run_benchmark(model_dir=args.model_dir, custom_images=args.images)
