"""
Automated Benchmarking & Comparative Evaluation Suite
Compares V1 (Baseline), V2 (Multi-Scale), V3 (Transformer), and V4 (SOTA) pipelines.
"""

import os
import sys
import time
import argparse
import numpy as np
import logging
from pipelines import get_pipeline, PIPELINE_REGISTRY

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("Benchmark")


def format_table(rows: list, headers: list) -> str:
    """Formats rows and headers as a clean GitHub-style Markdown table without external dependencies."""
    if not rows:
        return "No data."
    col_widths = [len(str(h)) for h in headers]
    for r in rows:
        for i, val in enumerate(r):
            col_widths[i] = max(col_widths[i], len(str(val)))
    
    header_line = "| " + " | ".join(f"{str(h):<{col_widths[i]}}" for i, h in enumerate(headers)) + " |"
    sep_line = "|-" + "-|-".join("-" * col_widths[i] for i in range(len(headers))) + "-|"
    
    data_lines = []
    for r in rows:
        data_lines.append("| " + " | ".join(f"{str(val):<{col_widths[i]}}" for i, val in enumerate(r)) + " |")
        
    return "\n".join([header_line, sep_line] + data_lines)


def run_benchmark(image_paths: list, variants: list = None, model_dir: str = "./models"):
    if variants is None:
        variants = ["v1", "v2", "v3", "v4"]

    print("=" * 85)
    print("🚦 TRAFFIC VIOLATION DETECTION — MULTI-PIPELINE BENCHMARK")
    print("=" * 85)
    print(f"Total Test Images: {len(image_paths)}")
    print(f"Variants to Evaluate: {', '.join(variants)}\n")

    results = []

    for var in variants:
        print(f"[*] Initializing Pipeline Variant: {var.upper()}...")
        try:
            detector = get_pipeline(variant=var, model_dir=model_dir)
        except Exception as e:
            print(f"[!] Failed to initialize {var}: {e}")
            continue

        latencies = []
        total_violations = 0
        total_triples = 0
        total_helmet_violations = 0
        plates_detected = 0

        # Warmup pass
        if image_paths and os.path.exists(image_paths[0]):
            try:
                detector.predict(image_paths[0])
            except Exception:
                pass

        print(f"[*] Running inference across {len(image_paths)} images...")
        for img_path in image_paths:
            t0 = time.time()
            res = detector.predict(img_path)
            dt = (time.time() - t0) * 1000.0  # ms
            latencies.append(dt)

            violations = res.get("violations", [])
            total_violations += len(violations)
            for v in violations:
                if v.get("num_riders", 0) > 2:
                    total_triples += 1
                total_helmet_violations += v.get("helmet_violations", 0)
                if v.get("license_plate"):
                    plates_detected += 1

        avg_lat = float(np.mean(latencies)) if latencies else 0.0
        fps = 1000.0 / avg_lat if avg_lat > 0 else 0.0

        results.append({
            "Variant": var.upper(),
            "Avg Latency (ms)": f"{avg_lat:.2f} ms",
            "Throughput (FPS)": f"{fps:.1f} FPS",
            "Total Violations": total_violations,
            "Triple Riding": total_triples,
            "Helmet Violations": total_helmet_violations,
            "Plates Extracted": plates_detected,
        })

    print("\n" + "=" * 85)
    print("📊 BENCHMARK COMPARATIVE RESULTS SUMMARY")
    print("=" * 85)
    headers = list(results[0].keys()) if results else []
    rows = [list(r.values()) for r in results]
    print(format_table(rows, headers))
    print("=" * 85 + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Multi-Pipeline Traffic Violation Benchmark")
    parser.add_argument("--image_dir", type=str, default="./", help="Path to directory containing test images")
    parser.add_argument("--model_dir", type=str, default="./models", help="Path to model weights directory")
    parser.add_argument("--variants", nargs="+", default=["v1", "v2", "v3", "v4"], help="Variants to benchmark")
    args = parser.parse_args()

    # Collect valid images
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = []
    if os.path.isdir(args.image_dir):
        for root, _, files in os.walk(args.image_dir):
            for f in files:
                if os.path.splitext(f.lower())[1] in valid_exts:
                    images.append(os.path.join(root, f))
    elif os.path.isfile(args.image_dir):
        images.append(args.image_dir)

    if not images:
        print("[INFO] No external images provided. Running non-crashing test harness demonstration:")
        images = ["sample_traffic_frame.jpg"]

    run_benchmark(images, variants=args.variants, model_dir=args.model_dir)


if __name__ == "__main__":
    main()
