import numpy as np
from scipy.optimize import linear_sum_assignment


class Track:
    _next_id = 0

    def __init__(self, detection, dt=0.1):
        self.id = Track._next_id
        Track._next_id += 1

        # State: [x, y, vx, vy]
        self.x = np.array([
            detection["cx"],
            detection["cy"],
            0.0,
            0.0
        ])

        self.P = np.eye(4) * 10.0

        self.dt = dt

        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])


        self.Q = np.eye(4) * 0.01

        # Measurement model
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        self.R = np.eye(2) * 1.0

        self.hits = 1
        self.missed = 0
        self.confirmed = False

        # --- Motion history ---
        self.speed_history = []


        self.update_detection_info(detection)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, detection):
        z = np.array([detection["cx"], detection["cy"]])

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

        # --- Update motion history ---
        current_speed = self.speed
        self.speed_history.append(current_speed)

        if len(self.speed_history) > 5:
            self.speed_history.pop(0)
        print(f"Track {self.id} mean_speed: {self.mean_speed:.4f}")

        self.hits += 1
        self.missed = 0

        if self.hits >= 3:
            self.confirmed = True

        self.update_detection_info(detection)

    def update_detection_info(self, detection):

        self.cz = detection["cz"]
        self.width = detection["width"]
        self.length = detection["length"]
        self.height = detection["height"]
        self.volume = detection["volume"]
        self.bbox = detection["bbox"]
        self.num_points = detection["num_points"]

        # --- Fusion of geometry + motion ---
        geometry_class = detection["object_type"]
        speed = self.mean_speed
        horizontal_area = self.width * self.length

        geometry_class = detection["object_type"]
        speed = self.mean_speed
        horizontal_area = self.width * self.length

        # Strong geometry vehicle
        if geometry_class == "vehicle":
            self.object_type = "vehicle"

        # Strong geometry pedestrian
        elif geometry_class == "pedestrian":
            self.object_type = "pedestrian"

        # Uncertain → use motion
        else:
            if 0.01 < speed < 3.0 and horizontal_area < 2.0:
                self.object_type = "pedestrian"
            else:
                self.object_type = "static"



    @property
    def position(self):
        return self.x[0], self.x[1]

    @property
    def speed(self):
        vx = self.x[2]
        vy = self.x[3]
        return np.sqrt(vx ** 2 + vy ** 2)

    @property
    def mean_speed(self):
        if len(self.speed_history) == 0:
            return 0.0
        return np.mean(self.speed_history)


# ======================================================
# Main Tracking Update
# ======================================================

def update_tracks(tracks, detections,
                  max_age=5,
                  distance_threshold=3.0):

    # 1 Predict all tracks
    for track in tracks:
        track.predict()

    # 2 If no tracks yet
    if len(tracks) == 0:
        return [Track(det) for det in detections]

    # 3 If no detections
    if len(detections) == 0:
        for track in tracks:
            track.missed += 1
        return [t for t in tracks if t.missed <= max_age]

    # 4 Build cost matrix
    cost = np.zeros((len(tracks), len(detections)))

    for i, t in enumerate(tracks):
        tx, ty = t.position
        for j, d in enumerate(detections):
            dx = d["cx"]
            dy = d["cy"]
            cost[i, j] = np.sqrt((tx - dx)**2 + (ty - dy)**2)

    # 5 Hungarian matching
    row_ind, col_ind = linear_sum_assignment(cost)

    matched_tracks = set()
    matched_dets = set()

    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < distance_threshold:
            tracks[r].update(detections[c])
            matched_tracks.add(r)
            matched_dets.add(c)

    # 6 Handle unmatched tracks
    for i, t in enumerate(tracks):
        if i not in matched_tracks:
            t.missed += 1

    tracks = [t for t in tracks if t.missed <= max_age]

    # 8 Create new tracks
    for j, d in enumerate(detections):
        if j not in matched_dets:
            tracks.append(Track(d))

    return tracks
