from load_data import load_lidar_frames_from_folders
from pathlib import Path
import pandas as pd
from detection import preprocess, detect_objects
from tracking import update_tracks
from visualize import (
    plot_3d,
    plot_3d_pcshow_like
)

folders = [
    "data/192.168.26.26_2020-11-25_20-01-45_frame-1899_part_1",
    "data/192.168.26.26_2020-11-25_20-01-45_frame-2155_part_2",
    "data/192.168.26.26_2020-11-25_20-01-45_frame-2414_part_3",
    "data/192.168.26.26_2020-11-25_20-01-45_frame-2566_part_4",
]

frames = load_lidar_frames_from_folders(folders)

tracks = []
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

all_results = []

for i, df in enumerate(frames):
    df = preprocess(df)
    detections = detect_objects(df)
    tracks = update_tracks(tracks, detections)

    tracked_objects = []

    for track in tracks:
        if track.confirmed and track.missed == 0:
            tracked_objects.append({
                "bbox": track.bbox,
                "object_type": track.object_type
            })

    frame_id = df["frame_id"].iloc[0]
    part_id = df["part_id"].iloc[0]

    print(
        f"Frame {df['frame_id'].iloc[0]} | "
        f"Detections: {len(detections)} | "
        f"Active tracks: {len(tracks)}"
    )

    # Save tracking results
    for track in tracks:
        # Only export confirmed tracks that were updated this frame
        if track.confirmed and track.missed == 0:
            x_min, y_min, z_min, x_max, y_max, z_max = track.bbox

            all_results.append({
                "frame_id": frame_id,
                "part_id": part_id,
                "track_id": track.id,
                "object_type": track.object_type,
                "cx": track.position[0],
                "cy": track.position[1],
                "cz": track.cz,
                "width": track.width,
                "length": track.length,
                "height": track.height,
                "volume": track.volume,
                "x_min": x_min,
                "y_min": y_min,
                "z_min": z_min,
                "x_max": x_max,
                "y_max": y_max,
                "z_max": z_max,
                "num_points": track.num_points,
            })

    if i % 2 == 0:
        plot_3d(df, title=f"Frame {df['frame_id'].iloc[0]}")
        plot_3d_pcshow_like(
            df,
            detections=tracked_objects,
            title=f"Frame {df['frame_id'].iloc[0]}"
        )


results_df = pd.DataFrame(all_results)
results_df.to_csv(OUTPUT_DIR / "detections_tracking.csv", index=False)

print("Saved detections_tracking.csv")