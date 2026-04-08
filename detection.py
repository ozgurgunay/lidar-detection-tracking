import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from sklearn.neighbors import NearestNeighbors
import numpy as np

def estimate_eps(points, k=10):
    """
    Estimate DBSCAN eps using k-nearest neighbor distances.
    """
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(points)
    distances, _ = neigh.kneighbors(points)

    # Take distance to kth neighbor
    k_distances = np.sort(distances[:, -1])

    # Simple heuristic: take 90th percentile
    eps_estimated = np.percentile(k_distances, 90)

    return eps_estimated


def classify_object(length, width, height, volume, num_points, z_std):
    horizontal_area = width * length
    #range_distance = np.sqrt(cx ** 2 + cy ** 2)
    aspect_ratio = length / (width + 1e-6)
    if aspect_ratio > 2.2:
        return "static"



    # Reject very large blobs (trees, buildings)
    if width > 8 or length > 10:
        return "static"

    # Reject flat ground patches
    if height < 0.05:
        return "static"

    # Reject irregular vertical structures (tree canopy)
    if z_std > 0.15:
        return "static"

    # --- Pedestrian ---
    if (
        0.1 <= height <= 1.0 and
        width <= 1.0 and
        length <= 1.0 and
        horizontal_area <= 2.0 and
        num_points >= 30
    ):
        return "pedestrian"

    # --- Vehicle ---
    if (
        0.1 <= height <= 1.305 and
        1.8 <= width <= 3.6 and
        1.0 <= length <= 6.5 and
        horizontal_area >= 2.0 and
        num_points >= 150
    ):
        return "vehicle"

    return "static"



def preprocess(df):
    # ROI limits (tunable)
    df = df[
        (df["X"] > -30) & (df["X"] < 30) &
        (df["Y"] > 0) & (df["Y"] < 60) &
        (df["Z"] > -2) & (df["Z"] < 5)
    ]

    # Ground removal
    df = df[df["Z"] > -1.2]

    return df


def detect_objects(df, eps=None, min_samples=15):
    points = df[["X", "Y", "Z"]].values

    if eps is None:
        eps = estimate_eps(points, k=min_samples)

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_

    df = df.copy()
    df["cluster_id"] = labels

    detections = []

    unique_labels = set(labels)
    unique_labels.discard(-1)  # remove noise

    for cluster_id in unique_labels:
        cluster_points = df[df["cluster_id"] == cluster_id]

        x_min, x_max = cluster_points["X"].min(), cluster_points["X"].max()
        y_min, y_max = cluster_points["Y"].min(), cluster_points["Y"].max()
        z_min, z_max = cluster_points["Z"].min(), cluster_points["Z"].max()

        # Centroid
        cx = cluster_points["X"].mean()
        cy = cluster_points["Y"].mean()
        cz = cluster_points["Z"].mean()

        # Derived dimensions
        width = x_max - x_min
        length = y_max - y_min
        height = z_max - z_min
        volume = width * length * height
        z_std = cluster_points["Z"].std()

        num_points = len(cluster_points)

        object_type = classify_object(
            length,
            width,
            height,
            volume,
            num_points,
            z_std
        )

        print(f"Cluster {cluster_id}: "
              f"L={length:.2f}, W={width:.2f}, H={height:.2f}, "
              f"Pts={num_points}")

        detections.append({
            "cluster_id": cluster_id,
            "cx": cx,
            "cy": cy,
            "cz": cz,
            "bbox": (x_min, y_min, z_min, x_max, y_max, z_max),
            "width": width,
            "length": length,
            "height": height,
            "volume": volume,
            "num_points": len(cluster_points),
            "object_type": object_type
        })

    return detections
