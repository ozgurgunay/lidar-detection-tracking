import re
from pathlib import Path
import pandas as pd

def get_frame_id(filename):
    m = re.search(r"frame-(\d+)", filename)
    return int(m.group(1)) if m else -1

def load_lidar_frames_from_folders(folders):
    frames = []

    for part_id, folder in enumerate(folders, start=1):
        files = sorted(
            Path(folder).glob("*.csv"),
            key=lambda f: get_frame_id(f.name)
        )

        for f in files:
            df = pd.read_csv(f, sep=";")
            df["frame_id"] = get_frame_id(f.name)
            df["part_id"] = part_id
            df["source_file"] = f.name
            frames.append(df)

    return frames
