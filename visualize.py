import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("outputs/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)





def plot_3d(df, title=None, max_points=50000, save=True):
    # Optional subsampling for speed
    if len(df) > max_points:
        df = df.sample(max_points, random_state=0)

    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        df["X"],
        df["Y"],
        df["Z"],
        s=1,
        c=df["Z"],
        cmap="viridis"
    )

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    fig.colorbar(sc, ax=ax, label="Z [m]")
    frame_id = df["frame_id"].iloc[0]
    if title:
        ax.set_title(title)
    if save:
        fname = OUTPUT_DIR / f"3d_frame_{frame_id}.png"
        plt.savefig(fname, dpi=300, bbox_inches="tight")

    plt.close(fig)


def plot_3d_pcshow_like(df, detections=None, title=None, max_points=50000, save=True):
    # Optional subsampling for speed
    if len(df) > max_points:
        df = df.sample(max_points, random_state=0)

    # Distance-based coloring (MATLAB-like)
    r = np.sqrt(df["X"]**2 + df["Y"]**2 + df["Z"]**2)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        df["X"],
        df["Y"],
        df["Z"],
        c=r,
        s=1,
        cmap="jet",
        linewidths=0
    )

    # MATLAB pcshow-style axis limits
    ax.set_xlim(-25, 25)
    ax.set_ylim(0, 60)
    ax.set_zlim(0, 30)

    #ax.view_init(elev=35, azim=-32)
    ax.view_init(elev=60, azim=-90)
    ax.set_box_aspect([50, 60, 20])

    ax.set_xlabel("x in m")
    ax.set_ylabel("y in m")
    ax.set_zlabel("z in m")

    fig.colorbar(sc, ax=ax, label="Range [m]")

    frame_id = df["frame_id"].iloc[0]
    part_id = df["part_id"].iloc[0]

    if title:
        ax.set_title(title)

    # Draw bounding boxes if provided
    if detections is not None:

        color_map = {
            "vehicle": "blue",
            "pedestrian": "green",
            "static": "gray",
        }

        for det in detections:
            x_min, y_min, z_min, x_max, y_max, z_max = det["bbox"]

            # 8 corners of the box
            corners = np.array([
                [x_min, y_min, z_min],
                [x_min, y_max, z_min],
                [x_max, y_max, z_min],
                [x_max, y_min, z_min],
                [x_min, y_min, z_max],
                [x_min, y_max, z_max],
                [x_max, y_max, z_max],
                [x_max, y_min, z_max],
            ])

            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7)
            ]

            box_color = color_map.get(det.get("object_type", "static"), "red")

            for edge in edges:
                ax.plot(
                    [corners[edge[0], 0], corners[edge[1], 0]],
                    [corners[edge[0], 1], corners[edge[1], 1]],
                    [corners[edge[0], 2], corners[edge[1], 2]],
                    color=box_color,
                    linewidth=1
                )

    if save:
        fname = OUTPUT_DIR / f"part{part_id}_pcshow_frame_{frame_id}.png"
        plt.savefig(fname, dpi=300, bbox_inches="tight")

    plt.close(fig)