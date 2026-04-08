import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def evaluate_tracking(
    pred_csv,
    gt_csv,
    distance_thresh=1.5,
    fps=10
):
    """
    Evaluate multi-object detection & tracking using 3D center distance
    and Hungarian matching.

    Returns:
        metrics (dict)
    """

    pred_df = pd.read_csv(pred_csv)
    gt_df = pd.read_csv(gt_csv)

    frames = sorted(gt_df["frame_id"].unique())

    # Global counters
    TP = 0
    FP = 0
    FN = 0
    ID_switches = 0
    total_distance = 0
    match_count = 0

    # Per-class counters
    class_stats = {}

    # Track GT → track_id assignment across frames
    prev_assignments = {}

    for frame in frames:

        gt_frame = gt_df[gt_df["frame_id"] == frame]
        pred_frame = pred_df[pred_df["frame_id"] == frame]

        # Process per class
        for obj_class in gt_frame["object_type"].unique():

            gt_objs = gt_frame[gt_frame["object_type"] == obj_class]
            pred_objs = pred_frame[pred_frame["object_type"] == obj_class]

            if obj_class not in class_stats:
                class_stats[obj_class] = {"TP": 0, "FP": 0, "FN": 0}

            if len(gt_objs) == 0 and len(pred_objs) == 0:
                continue

            if len(gt_objs) > 0 and len(pred_objs) > 0:

                gt_centers = gt_objs[["cx", "cy", "cz"]].values
                pred_centers = pred_objs[["cx", "cy", "cz"]].values

                distances = cdist(gt_centers, pred_centers)

                row_ind, col_ind = linear_sum_assignment(distances)

                matched_gt = set()
                matched_pred = set()

                for r, c in zip(row_ind, col_ind):
                    if distances[r, c] <= distance_thresh:

                        gt_id = gt_objs.iloc[r]["GT_id"]
                        track_id = pred_objs.iloc[c]["track_id"]

                        TP += 1
                        class_stats[obj_class]["TP"] += 1

                        total_distance += distances[r, c]
                        match_count += 1

                        matched_gt.add(r)
                        matched_pred.add(c)

                        # ID switch check
                        if gt_id in prev_assignments:
                            if prev_assignments[gt_id] != track_id:
                                ID_switches += 1

                        prev_assignments[gt_id] = track_id

                FN += len(gt_objs) - len(matched_gt)
                FP += len(pred_objs) - len(matched_pred)

                class_stats[obj_class]["FN"] += len(gt_objs) - len(matched_gt)
                class_stats[obj_class]["FP"] += len(pred_objs) - len(matched_pred)

            elif len(gt_objs) > 0:
                FN += len(gt_objs)
                class_stats[obj_class]["FN"] += len(gt_objs)

            elif len(pred_objs) > 0:
                FP += len(pred_objs)
                class_stats[obj_class]["FP"] += len(pred_objs)

    # =========================
    # GLOBAL METRICS
    # =========================
    GT_total = TP + FN

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / GT_total if GT_total > 0 else 0

    MOTA = 1 - (FN + FP + ID_switches) / GT_total if GT_total > 0 else 0
    MOTP = total_distance / match_count if match_count > 0 else 0

    # False alarms per hour
    total_frames = len(frames)
    hours = total_frames / fps / 3600
    false_alarms_per_hour = FP / hours if hours > 0 else 0

    # =========================
    # PER-CLASS METRICS
    # =========================
    for cls in class_stats:
        cls_TP = class_stats[cls]["TP"]
        cls_FP = class_stats[cls]["FP"]
        cls_FN = class_stats[cls]["FN"]

        cls_precision = cls_TP / (cls_TP + cls_FP) if (cls_TP + cls_FP) > 0 else 0
        cls_recall = cls_TP / (cls_TP + cls_FN) if (cls_TP + cls_FN) > 0 else 0

        class_stats[cls]["Precision"] = cls_precision
        class_stats[cls]["Recall"] = cls_recall

    metrics = {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "ID Switches": ID_switches,
        "Precision": precision,
        "Recall": recall,
        "MOTA": MOTA,
        "MOTP (m)": MOTP,
        "False Alarms / Hour": false_alarms_per_hour,
        "Per Class": class_stats
    }

    return metrics


if __name__ == "__main__":

    metrics = evaluate_tracking(
        pred_csv="outputs/detections_tracking.csv",
        gt_csv="ground_truth_subset.csv",
        distance_thresh=1.5,
        fps=10  # adjust if needed
    )

    print("\n==== GLOBAL RESULTS ====")
    for k, v in metrics.items():
        if k != "Per Class":
            print(f"{k}: {v}")

    print("\n==== PER-CLASS RESULTS ====")
    for cls, stats in metrics["Per Class"].items():
        print(f"\nClass: {cls}")
        for k, v in stats.items():
            print(f"  {k}: {v}")