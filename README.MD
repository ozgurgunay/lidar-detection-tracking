# LiDAR Object Detection and Tracking

An academic project developed during my MSc in Artificial Intelligence for the course Localization, Motion Planning, and Sensor Fusion.

The project explores vehicle and pedestrian detection and tracking in LiDAR point clouds using Python. It is an offline prototype for studying and evaluating the methods, rather than a system intended for use in a vehicle.

## Approach

The pipeline includes:

1. Loading sequential point-cloud frames from CSV files.
2. Filtering the region of interest and removing ground points using a height threshold.
3. Grouping points with DBSCAN.
4. Classifying clusters using geometric rules, with motion information used during tracking.
5. Tracking objects with a Kalman filter and Hungarian assignment.
6. Comparing predictions with a labeled subset.

The implementation uses NumPy, pandas, SciPy, scikit-learn and Matplotlib.

## Findings and Limitations

Vehicle detection performed better than pedestrian detection in the evaluated subset. Small static objects were often classified as pedestrians, and nearby objects could merge into a single cluster.

The evaluation uses a limited labeled subset and centroid-distance matching. The results should therefore be treated as exploratory, rather than evidence of performance in other scenes or driving conditions.

Possible improvements include reducing pedestrian false positives, separating merged clusters and strengthening the evaluation procedure.

## Repository Structure

```text
.
├── data/                     # Local input data; not included
├── outputs/                  # Selected analysis outputs
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

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Running the Project

The original LiDAR dataset is not included. Running the pipeline requires compatible input data and the appropriate local paths.

Check the input folder paths in `main.py` and `data_review.py` before running:

```bash
python data_review.py
python main.py
```

To run the evaluation, check the prediction and annotation paths in `evaluation.py`, then run:

```bash
python evaluation.py
```

The evaluation currently assumes 10 frames per second. This setting should match the data being evaluated.

## Outputs

The scripts produce data summaries, detection and tracking results, and point-cloud visualizations.

Selected analysis tables are included:

- `outputs/data_summary.csv`
- `outputs/frame_timing.csv`
- `outputs/noise_stats.csv`
- `outputs/range_stats.csv`
- `outputs/density_by_part.csv`

Raw data and generated plot images are excluded by `.gitignore`.

## Background

The sensor review in the accompanying academic report used the Blickfeld Cube 1 datasheet.
