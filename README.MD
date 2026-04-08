# LiDAR Data Processing for Car and Pedestrian Detection/Tracking

Verification and validation prototype for a solid-state LiDAR case study in autonomous driving.

## Project Scope

This project implements an offline pipeline to:

- review and sanity-check LiDAR point cloud data,
- detect surrounding objects (car and pedestrian focused),
- track objects across frames,
- evaluate detection/tracking performance against a labeled subset.

The implementation is designed for algorithm validation, not embedded real-time deployment.

## Task Context

Based on the assignment constraints:

- LiDAR range: 5 m to 250 m
- Range resolution: < 0.01 m
- Horizontal FOV: 70 degrees
- Vertical FOV: 30 degrees
- Scanlines per second: > 500
- Update rate: 1 to 30 Hz (speed dependent)
- Wavelength: 900 nm

Target performance from task definition:

- Correct classification rate: 0.99
- False alarm rate: 0.01 per hour

## Method Overview

Pipeline stages:

1. Load sequential CSV point clouds.
2. Apply preprocessing (ROI filtering and ground suppression).
3. Run DBSCAN-based clustering for object proposals.
4. Classify clusters with geometric and motion-based heuristics.
5. Track objects using Kalman filter + Hungarian assignment.
6. Evaluate with frame-wise matching and MOT-style metrics.

## Repository Structure

```text
.
├── data/                     # Local input data (not committed)
├── outputs/                  # Exported analysis artifacts
├── load_data.py
├── data_review.py
├── detection.py
├── tracking.py
├── evaluation.py
├── visualize.py
├── main.py
├── ground_truth_subset.csv
├── requirements.txt
└── README.MD
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

Full pipeline:

```bash
python main.py
```

Evaluation:

```bash
python evaluation.py
```

## Outputs Included in Repository

To keep the repository clean, only selected tabular outputs are tracked.
Large/raw data and all plot images are excluded by `.gitignore`.

Tracked output examples:

- `outputs/data_summary.csv`
- `outputs/frame_timing.csv`
- `outputs/noise_stats.csv`
- `outputs/range_stats.csv`
- `outputs/density_by_part.csv`

## Notes
Blackfield cube 1 data sheet used.
