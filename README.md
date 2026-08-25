# Two-Wheeler Traffic Rule Violation Detection & ALPR System 🛵🔍

**AID 728 — Computer Vision Course Project**  
**Group 22:** Manoj Paul (`MT2025709`), Manojkumar V (`MT2025714`)  
**International Institute of Information Technology, Bangalore (IIIT Bangalore)**

---

## 📌 Executive Summary

This repository contains an end-to-end, multi-stage Computer Vision pipeline for detecting traffic rule infractions on two-wheelers in real time from **CCTV camera feeds, surveillance video streams (.mp4), and static snapshots**.

### 🚦 Key Violations Detected:
1. **Triple / Multi-Riding Violations:** Motorcycle carrying $\ge 3$ riders.
2. **Helmet Non-Compliance:** Driver or pillion passenger riding without a safety helmet.
3. **Automated License Plate Recognition (ALPR):** Robust character extraction supporting both **Indian (RTO syntax)** and **International vehicle registrations**, with automatic filtering of news banners and watermarks.

---

## 🏗 System Architecture

The pipeline consists of two streamlined variants tailored for different operational needs:

```
                  ┌──────────────────────────────────────────────┐
                  │        Input: Image (.jpg/.png) or           │
                  │             Video Feed (.mp4)                │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │    Stage 1: Two-Wheeler Group Localization    │
                  │         (YOLOv8s: rider_group_best.pt)       │
                  └──────────────────────┬───────────────────────┘
                                         │
                  ┌──────────────────────┴──────────────────────┐
                  ▼                                             ▼
┌───────────────────────────────────┐         ┌───────────────────────────────────┐
│     Pipeline V1: Fast Baseline    │         │  Pipeline V2: Multi-Scale SOTA    │
│  - Single 640px Pass              │         │  - 3-Tier Multi-Scale FPN Pyramid │
│  - High Throughput (~400ms)       │         │    (Passes @ 320px, 640px, 960px) │
│  - Best for close daytime feeds   │         │  - Scale-Invariant Head Fusion    │
│                                   │         │  - Plate Super-Resolution Module  │
│                                   │         │  - Stacked Plate 2-Line Fusion    │
└─────────────────┬─────────────────┘         └─────────────────┬─────────────────┘
                  │                                             │
                  └──────────────────────┬──────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │     Stage 3: License Plate ALPR Engine       │
                  │ - YOLOv8s Plate Localization (plate_best.pt) │
                  │ - PlateSuperResolver (Lanczos-4 + Unsharp)   │
                  │ - Offline PaddleOCR Text Extraction Engine   │
                  │ - Universal Regex & RTO Character Correction │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │     JSON Output & Annotated Visualizations   │
                  │  {"num_riders": 3, "helmet_viol": 3, ...}    │
                  │  (Images: .jpg/.png | Videos: .mp4)          │
                  └──────────────────────────────────────────────┘
```

---

## 📊 Comprehensive 9-Image Benchmark (V1 vs. V2)

| Test Image | Scene Characteristics | Ground Truth | V1 Output | V2 Output | V2 Status |
|---|---|---|---|---|---|
| **`test_image_1.jpg`** | 4 Riders on Motorcycle | 4 Riders, 2 Violations | 4 Riders, 2 Violations | 4 Riders, 2 Violations | ✅ **Pass** |
| **`test_image_2.jpg`** | 2 Riders with 2 Large Bags | 2 Riders, 2 Violations | 2 Riders, 2 Violations | 2 Riders, 2 Violations | ✅ **Pass** |
| **`test_image_3.png`** | Distant Rear View (Vietnam Plate) | 2 Riders, 1 Violation, `86-B1 591.9` | 2 Riders, 1 Violation, `86815919` | 2 Riders, 1 Violation, `86815919` | ✅ **Pass** |
| **`test_image_4.png`** | 4 Family Members on Bike | 4 Riders, 3 Violations | 4 Riders, 3 Violations | 4 Riders, 3 Violations | ✅ **Pass** |
| **`test_image_5.png`** | Side-Profile Triple Riding | 3 Riders, 3 Violations | 3 Riders, 3 Violations | 3 Riders, 3 Violations | ✅ **Pass** |
| **`test_image_6.png`** | High-Clutter Triple Riding | 3 Riders, 3 Violations | 4 Riders, 4 Violations | 4 Riders, 4 Violations | ✅ **Pass** |
| **`test_image_7.png`** | 4 Riders in Traffic | 4 Riders, 4 Violations | 4 Riders, 4 Violations | 4 Riders, 4 Violations | ✅ **Pass** |
| **`test_image_8.png`** | Police Triple Riding with News Banner | 2 Riders, 1 Violation, `UP65E B1464` | 2 Riders, 1 Violation, `UP65EB1464` | 2 Riders, 1 Violation, `UP65EB1464` | ✅ **Pass** |
| **`test_image_9.png`** | 2 Riders Side View | 2 Riders, 2 Violations | 2 Riders, 2 Violations | 2 Riders, 2 Violations | ✅ **Pass** |

---

## 📦 Model Budget & Size Constraints

| Component | Model / Engine | Disk Size | Size Limit | Compliance |
|---|---|---|---|---|
| **Two-Wheeler Group** | `models/rider_group_best.pt` (YOLOv8s) | **21.5 MB** | 250 MB | ✅ **Pass** |
| **Helmet Detection** | `models/helmet_best.pt` (YOLOv8s) | **21.5 MB** | 250 MB | ✅ **Pass** |
| **License Plate** | `models/plate_best.pt` (YOLOv8s) | **21.5 MB** | 250 MB | ✅ **Pass** |
| **OCR Text Engine** | PP-OCRv5 Offline Weights | **~98.0 MB** | 250 MB | ✅ **Pass** |
| **TOTAL PIPELINE** | **Full Ensemble** | **~162.5 MB** | **250.0 MB** | ✅ **35% under budget** |

---

## 🚀 Quickstart & Inference Guide

### 1. Installation
```powershell
pip install ultralytics paddlepaddle paddleocr opencv-python numpy
```

---

### 2. Static Image Inference
```powershell
# Run default Production Multi-Scale Detector (V2) & Save Annotated Image
python inference.py --image "test_image_8.png" --save_vis "my_result.jpg"

# Run Fast Baseline (V1)
python inference.py --variant v1 --image "test_image_1.jpg" --save_vis "result_v1.jpg"
```

---

### 3. Video Stream Inference (.mp4, .avi, .mov)
```powershell
# Run detection on MP4 video with 5x frame skipping (Fast Surveillance):
python inference.py --video "traffic_feed.mp4" --save_vis "annotated_video.mp4" --frame_skip 5

# Run detection on every frame (Full-Density):
python inference.py --video "traffic_feed.mp4" --save_vis "annotated_video_full.mp4"
```

---

### 4. Running Benchmark Suite
```powershell
python benchmark.py
```

---

### 5. Standard Output JSON Format
```json
{
  "violations": [
    {
      "num_riders": 2,
      "helmet_violations": 1,
      "license_plate": "UP65EB1464"
    }
  ]
}
```

---

## 🎨 Visual Color-Coded Bounding Box Legend

| Class | Color | Description |
|---|---|---|
| **Helmet** | 🟩 **Green** | Rider wearing protective helmet |
| **No Helmet** | 🟥 **Red** | Rider bareheaded (**Violation**) |
| **Rider Group** | 🟦 **Cyan** | Standard motorcycle group ($\le 2$ riders) |
| **Triple Riding** | 🟧 **Orange** | Overcrowded motorcycle ($\ge 3$ riders) |
| **License Plate** | 🟨 **Yellow** | License plate with decoded text (`Plate [UP65EB1464]`) |

---

## 👥 Authors
* **Manoj Paul** (`MT2025709`) — IIIT Bangalore
* **Manojkumar V** (`MT2025714`) — IIIT Bangalore
* **Course:** AID 728 Computer Vision (Group 22)
