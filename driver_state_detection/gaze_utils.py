import numpy as np
from typing import Optional, Tuple, Dict, Any
import matplotlib.pyplot as plt


def angle_from_vector(vec_xy: np.ndarray) -> float:
    """
    Return angle in degrees [0,360) for a 2D vector (x,y) in image coordinates.
    x positive = right, y positive = down.
    """
    dx, dy = float(vec_xy[0]), float(vec_xy[1])
    ang_rad = np.arctan2(-dy, dx)
    return (np.degrees(ang_rad) + 360.0) % 360.0

def circular_angle_diff(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)

def classify_by_proximity(gaze_points, gaze_magnitude, angle_close_thresh= 5.0) -> Tuple[str, float]:
    """
    Classify gaze to one of four corners using proximity.
    Returns: (corner_name, min_dist)
    """
    if gaze_points is None:
        return "none", 0.0

    gaze_angle = np.asarray(gaze_points, dtype=np.float32)
    gaze_magnitude = np.asarray(gaze_magnitude, dtype=np.float32)

    ROIs= {
    "left_mirror":  {"angle": 6.0,   "left_mag": 1.2, "right_mag": 1.2},
    "right_mirror": {"angle": 188.0, "left_mag": 1.2, "right_mag": 1.2},
    "radio":        {"angle": 85.0,  "left_mag": 0.8, "right_mag": 0.8},
    "road":         {"angle": 0.0,   "left_mag": 1.0, "right_mag": 1.0},
    "lap":          {"angle": 113.0, "left_mag": 0.6, "right_mag": 0.6},
    "rearmirror":   {"angle": 34.0,  "left_mag": 1.1, "right_mag": 1.1},
    "left_window":  {"angle": 23.0,  "left_mag": 1.3, "right_mag": 1.3},
    "right_window": {"angle": 200.0, "left_mag": 1.3, "right_mag": 1.3},
    }

    gaze_mag = np.mean(gaze_magnitude)
    left_mag = gaze_magnitude[0]
    right_mag = gaze_magnitude[1]
    
        
    angle_dists = {roi: circular_angle_diff(gaze_angle, info["angle"]) for roi, info in ROIs.items()}
    top3 = sorted(angle_dists.items(), key=lambda kv: kv[1])[:3]

    _, best_angle = top3[0]

    # makes sure each of the three angles individually are within the threshold angle_close_thresh, remove them if they are not 
    # the final candidate may be determined at this stage if only one remains within the threshold (please ensure at least one remains, the smallest) 
    # then otherwise proceed to use magnitude to break the tie among the remaining candidates
    candidates = [roi for roi, d in top3 if d <= (best_angle + angle_close_thresh)]


    if len(candidates) == 1:
        roi = candidates[0]
        return roi, float(angle_dists[roi])

    roi = min(candidates, key=lambda r: (abs(left_mag- ROIs[r]["left_mag"]) + abs(right_mag - ROIs[r]["right_mag"])))
    return roi, float(angle_dists[roi])


def classify_by_angle(gaze_points: np.ndarray, iris_points: np.ndarray) -> Tuple[str, float]:
    """
    Classify gaze to one of four corners using midpoint angle.
    Returns: (corner_name, mid_angle_deg)
    """
    if gaze_points is None or iris_points is None:
        return "none", 0.0

    gaze_points = np.asarray(gaze_points, dtype=np.float32)
    iris_points = np.asarray(iris_points, dtype=np.float32)

    corner_names = ["top_left", "top_right", "bottom_right", "bottom_left"]

    def angle_to_corner_idx(angle_deg: float) -> int:
        a = angle_deg % 360.0
        if 0 <= a < 90:
            return corner_names.index("top_right")
        if 90 <= a < 180:
            return corner_names.index("top_left")
        if 180 <= a < 270:
            return corner_names.index("bottom_left")
        return corner_names.index("bottom_right")

    return corner_names[angle_to_corner_idx(gaze_points)], gaze_points


def apply_headpose_adjustment(
    gaze_points: np.ndarray,
    iris_points: np.ndarray,
    pitch,
    yaw,
    yaw_gain: float = 0.005,
    pitch_gain: float = 0.005) -> Optional[np.ndarray]:
    """
    Adjust gaze endpoints using head pose. Inputs are in normalized coords (0..1).
    Returns adjusted endpoints or None if inputs invalid.
    """
    if gaze_points is None or iris_points is None:
        return None

    # handle yaw/pitch being list/np arrays or scalars
    yaw_val = float(np.ravel(yaw)[0])
    pitch_val = float(np.ravel(pitch)[0])

    adjusted_endpoints = []
    for i in range(len(gaze_points)):
        gaze_vec = gaze_points[i] - iris_points[i]
        gaze_vec_adj = gaze_vec #+ np.array([0, 0], dtype=np.float32)#yaw_val * yaw_gain#pitch_val * pitch_gain
        end_pt = iris_points[i] + gaze_vec_adj
        adjusted_endpoints.append(end_pt)

    return np.asarray(adjusted_endpoints, dtype=np.float32)


class GazeLogger:
    """Simple logger for per-frame gaze angles and timestamps."""

    def __init__(self, enabled: bool = False):
        self.enabled = bool(enabled)
        self._t = []
        self._left = []
        self._right = []
        self._mid = []
        self._mag_right = []
        self._mag_left = []

    def log(self, left_ang: float, right_ang: float, mid_ang: float, left_mag: float, right_mag: float, wall_time: float) -> None:
        if not self.enabled:
            return
        self._t.append(wall_time)
        self._left.append(left_ang)
        self._right.append(right_ang)
        self._mid.append(mid_ang)
        self._mag_left.append(left_mag)
        self._mag_right.append(right_mag)

    def has_samples(self) -> bool:
        return len(self._t) > 0

    def to_numpy(self):
        t= np.asarray(self._t, dtype=float)
        L= np.asarray(self._left, dtype=float)
        R= np.asarray(self._right, dtype=float)
        M= np.asarray(self._mid, dtype=float)

        def unwrap_deg(d):
                return np.rad2deg(np.unwrap(np.deg2rad(d)))
        L = unwrap_deg(L)
        R = unwrap_deg(R)
        M = unwrap_deg(M)
        return (t, L, R, M)
    
    def print_summary(self, cam_index):
        if not self.has_samples():
            print("No gaze samples logged.")
            return
        t, L, R, M = self.to_numpy()
        print(f"\n=== Gaze Angle Summary (Camera {cam_index}) ===")
        print(f"Samples: {len(t)}")
        print(f"Time range: {t[0]:.2f}s -> {t[-1]:.2f}s")
        print(f"Left Eye: mean:{np.mean(L):.2f}°, std:{np.std(L):.2f}°")
        print(f"Right Eye: mean:{np.mean(R):.2f}°, std:{np.std(R):.2f}°")
        print(f"Midpoint: mean:{np.mean(M):.2f}°, std:{np.std(M):.2f}°")
        print(f"Gaze Magnitude Left Eye: mean:{np.mean(self._mag_left):.4f}, std:{np.std(self._mag_left):.4f}")
        print(f"Gaze Magnitude Right Eye: mean:{np.mean(self._mag_right):.4f}, std:{np.std(self._mag_right):.4f}")

    def plot(self, cam_index):
        if not self.has_samples():
            return
        t, L, R, M = self.to_numpy()
        plt.figure(figsize=(8, 3))
        plt.plot(t, L, label="Left eye")
        plt.plot(t, R, label="Right eye")
        plt.plot(t, M, label="Midpoint", linewidth=1)
        plt.xlabel("Time (seconds)")
        plt.ylabel("Angle (degrees)")
        plt.title(f"Gaze Angles (Camera {cam_index})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


class GazeProcessor:
    """adjust by headpose -> determine midpoint -> compute vectors/angles -> classify"""

    def __init__(
        self,
        yaw_gain: float = 0.005,
        pitch_gain: float = 0.005,
        history_points: int = 5,
        std_threshold: float = 15.0,
        ratio_threshold: float = 2.0,
):
        self.yaw_gain = yaw_gain
        self.pitch_gain = pitch_gain
        self.history_points = history_points
        self.left_history = []
        self.right_history = []
        self.std_threshold = std_threshold
        self.ratio_threshold = ratio_threshold

    def process(self, gaze_points, iris_points, gaze_magnitude, pitch, yaw) -> Optional[Dict[str, Any]]:
        if gaze_points is None or iris_points is None:
            return None

        gp_adj = apply_headpose_adjustment(
            gaze_points, iris_points, pitch, yaw, yaw_gain=self.yaw_gain, pitch_gain=self.pitch_gain)
        left_vec = gp_adj[0] - iris_points[0]
        right_vec = gp_adj[1] - iris_points[1]
        left_ang = angle_from_vector(left_vec)
        right_ang = angle_from_vector(right_vec)

        # Maintain a short history og angles for each eye 
        self.left_history.append(left_ang)
        self.right_history.append(right_ang)
        if len(self.left_history) > self.history_points:
            self.left_history.pop(0)
        if len(self.right_history) > self.history_points:
            self.right_history.pop(0)

        # compute standard deviations for recent history
        left_std = float(np.std(np.asarray(self.left_history))) if len(self.left_history) > 0 else 0.0
        right_std = float(np.std(np.asarray(self.right_history))) if len(self.right_history) > 0 else 0.0

        # must have larger std in one eye than the other and must exceed threshold
        if (left_std > right_std * self.ratio_threshold and left_std > self.std_threshold):
            mid_ang = right_ang
        elif (right_std > left_std * self.ratio_threshold and right_std > self.std_threshold):
            mid_ang = left_ang
        else:
            # for regular case, use average of both eyes
            mid_vec = np.mean(gp_adj, axis=0) - np.mean(iris_points, axis=0)
            mid_ang = angle_from_vector(mid_vec)

        corner_name, _ = classify_by_angle(mid_ang, iris_points)
        roi, _ = classify_by_proximity(mid_ang, gaze_magnitude)

        return {
            "gaze_points_adj": gp_adj,
            "left_vec": left_vec,
            "right_vec": right_vec,
            "left_ang": left_ang,
            "right_ang": right_ang,
            "mid_ang": mid_ang,
            "corner_name": corner_name,
            "roi": roi,
        }
