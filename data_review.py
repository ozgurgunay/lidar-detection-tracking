import pandas as pd
from pathlib import Path
import numpy as np
from load_data import load_lidar_frames_from_folders

folders = [
    "data/192.168.26.26_2020-11-25_20-01-45_frame-1899_part_1",
    "data/192.168.26.26_2020-11-25_20-01-45_frame-2155_part_2",
    "data/192.168.26.26_2020-11-25_20-01-45_frame-2414_part_3",
    "data/192.168.26.26_2020-11-25_20-01-45_frame-2566_part_4",
]

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

frames = load_lidar_frames_from_folders(folders)

rows = []
for df in frames:
    rows.append({
        "part_id": df["part_id"].iloc[0],
        "frame_id": df["frame_id"].iloc[0],
        "num_points": len(df),
        "timestamp": df["TIMESTAMP"].iloc[0],
    })

summary = pd.DataFrame(rows)
summary = summary.sort_values("timestamp")

summary.to_csv(OUTPUT_DIR / "data_summary.csv", index=False)
print("Saved data_summary.csv")

summary["dt_ms"] = summary["timestamp"].diff() / 1e6
summary["frame_gap"] = summary["frame_id"].diff()

timing = summary[["part_id", "frame_id", "dt_ms", "frame_gap"]]
timing.to_csv(OUTPUT_DIR / "frame_timing.csv", index=False)

print("Saved frame_timing.csv")
print("Sampling rate (Hz):", 1000 / timing["dt_ms"].mean())


all_points = pd.concat(frames, ignore_index=True)

noise_stats = all_points[["X", "Y", "Z"]].describe()
noise_stats.to_csv(OUTPUT_DIR / "noise_stats.csv")

# Derived range statistics
ranges = np.sqrt(all_points["X"]**2 + all_points["Y"]**2)
range_stats = ranges.describe()
range_stats.to_csv(OUTPUT_DIR / "range_stats.csv")

print("Saved noise_stats.csv and range_stats.csv")

density = summary.groupby("part_id")["num_points"].agg(
    frames="count",
    avg_points="mean",
    min_points="min",
    max_points="max"
)

density.to_csv(OUTPUT_DIR / "density_by_part.csv")
print("Saved density_by_part.csv")
