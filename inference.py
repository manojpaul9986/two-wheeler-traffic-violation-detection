"""
Two-Wheeler Traffic Rule Violation Detection — Inference CLI
Group 22 | MT2025709 & MT2025714 | IIIT Bangalore

Supports both Static Images (.jpg, .png) and Video Streams (.mp4, .avi, .mov, .mkv).

Usage:
    # 1. Run inference on Image
    python inference.py --image test_image_8.png --save_vis "my_result.jpg"

    # 2. Run inference on MP4 Video & save annotated video
    python inference.py --video "traffic_feed.mp4" --save_vis "output_annotated.mp4"

    # 3. Quick run (auto-detects image or video file extension)
    python inference.py traffic_feed.mp4 --save_vis
"""

import sys
import os
import json
import cv2
import argparse
from pipelines import get_pipeline
from modules.visualizer import annotate_traffic_violations


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm"}


def process_video_file(video_path: str, detector, output_path: str = None, frame_skip: int = 1):
    """
    Processes an MP4/video file frame-by-frame, annotates traffic violations,
    and saves the annotated output video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not output_path:
        base, _ = os.path.splitext(video_path)
        output_path = f"{base}_annotated.mp4"

    # Use mp4v codec for standard universal playback
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps / frame_skip, (width, height))

    print(f"\n[*] Processing Video: {video_path}")
    print(f"    - Resolution: {width}x{height} | FPS: {fps:.1f} | Total Frames: {total_frames}")
    print(f"    - Saving Annotated Output to: {output_path}\n")

    frame_idx = 0
    processed_count = 0
    all_violations = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            # Annotate bounding boxes directly onto frame
            annotated_frame = annotate_traffic_violations(frame, detector)
            out.write(annotated_frame)
            processed_count += 1

            # Progress bar
            progress = (frame_idx + 1) / max(total_frames, 1) * 100
            sys.stdout.write(f"\r[Progress: {progress:5.1f}%] Frame {frame_idx + 1}/{total_frames} processed")
            sys.stdout.flush()

        frame_idx += 1

    cap.release()
    out.release()
    print(f"\n\n[+] Video annotation complete! Saved to: {output_path}")
    print(f"[+] You can view the annotated video using VLC, Windows Media Player, or your browser.")


def main():
    parser = argparse.ArgumentParser(
        description="Two-Wheeler Traffic Violation Inference (Images & MP4 Videos)"
    )
    parser.add_argument("pos_input", nargs="?", default=None, help="Input image or video path")
    parser.add_argument("--image", "-i", type=str, default=None, help="Path to input image")
    parser.add_argument("--video", type=str, default=None, help="Path to input MP4 video file")
    parser.add_argument(
        "--variant", "-v", type=str, default="v2", choices=["v1", "v2"],
        help="Pipeline variant: 'v1' (Fast Baseline) or 'v2' (Multi-Scale Production SOTA, default: v2)"
    )
    parser.add_argument("--model_dir", "-m", type=str, default="./models", help="Models directory path")
    parser.add_argument(
        "--save_vis", "-s", nargs="?", const="default", default=None,
        help="Save visual color-coded bounding box annotations to image or MP4 file"
    )
    parser.add_argument("--frame_skip", type=int, default=1, help="Process every N-th video frame (default: 1)")
    args = parser.parse_args()

    input_path = args.video or args.image or args.pos_input

    if not input_path:
        input_path = "test_image_1.jpg"
        print(f"[INFO] No input provided. Defaulting to: '{input_path}'\n")

    if not os.path.exists(input_path):
        print(f"[ERROR] Input file '{input_path}' does not exist.")
        return

    ext = os.path.splitext(input_path)[1].lower()
    is_video = ext in VIDEO_EXTENSIONS or args.video is not None

    print(f"[*] Initializing Pipeline Variant: {args.variant.upper()}...")
    detector = get_pipeline(variant=args.variant, model_dir=args.model_dir)

    if is_video:
        vis_out = None
        if args.save_vis and args.save_vis != "default":
            vis_out = args.save_vis
        process_video_file(input_path, detector, output_path=vis_out, frame_skip=args.frame_skip)
    else:
        # Static Image Processing
        print(f"[*] Running inference on: {input_path}")
        output = detector.predict(input_path)

        print("\n--- Detection Result ---")
        print(json.dumps(output, indent=2))

        if args.save_vis is not None:
            if args.save_vis == "default":
                base, e = os.path.splitext(input_path)
                vis_path = f"{base}_annotated{e or '.jpg'}"
            else:
                vis_path = args.save_vis

            img = cv2.imread(input_path)
            if img is not None:
                annotate_traffic_violations(img, detector, output_path=vis_path)
                print(f"\n[+] Visual bounding boxes saved to: {vis_path}")


if __name__ == "__main__":
    main()