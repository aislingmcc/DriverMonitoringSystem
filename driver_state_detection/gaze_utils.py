import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import matplotlib.pyplot as plt


def angle_from_vector(vec_xy: np.ndarray):
    dx, dy = float(vec_xy[0]), float(vec_xy[1])
    ang_rad = np.arctan2(-dy, dx)
    return (np.degrees(ang_rad) + 360.0) % 360.0

def circular_angle_diff(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)

def vec_jitter_std(vecs):
    if len(vecs) < 2:
        return 0.0
    angles = [angle_from_vector(v) for v in vecs]
    diffs = [
        circular_angle_diff(angles[i], angles[i-1])
        for i in range(1, len(angles))
    ]
    return float(np.std(diffs))


def select_reliable_eye(left_history, right_history, left_vec, right_vec, 
                       iris_points, gp_adj,std_threshold=5.0, ratio_threshold=2.0):

    left_std = vec_jitter_std(left_history)
    right_std = vec_jitter_std(right_history)

    if left_std > right_std * ratio_threshold and left_std > std_threshold:
        eye_used = "right"
        gaze_vec = right_vec
        gaze_point = gp_adj[1]

    elif right_std > left_std * ratio_threshold and right_std > std_threshold:
        eye_used = "left"
        gaze_vec = left_vec
        gaze_point = gp_adj[0]

    else:
        eye_used = "blend"
        # ensure float dtype before normalization to avoid casting errors
        gaze_vec = (left_vec + right_vec).astype(np.float32, copy=False)
        norm = np.linalg.norm(gaze_vec) + 1e-8
        gaze_vec = gaze_vec / norm
        gaze_point = 0.5 * (gp_adj[0] + gp_adj[1])

    angle = angle_from_vector(gaze_vec)

    return eye_used, gaze_point, angle

def _get_default_rois():
    return {
        "left_mirror":  {"angle": 6.0,   "left_mag": 1.2, "right_mag": 1.2},
        "right_mirror": {"angle": 188.0, "left_mag": 1.2, "right_mag": 1.2},
        "radio":        {"angle": 85.0,  "left_mag": 0.8, "right_mag": 0.8},
        "road":         {"angle": 0.0,   "left_mag": 1.0, "right_mag": 1.0},
        "lap":          {"angle": 113.0, "left_mag": 0.6, "right_mag": 0.6},
        "rearmirror":   {"angle": 34.0,  "left_mag": 1.1, "right_mag": 1.1},
        "left_window":  {"angle": 23.0,  "left_mag": 1.3, "right_mag": 1.3},
        "right_window": {"angle": 200.0, "left_mag": 1.3, "right_mag": 1.3},
    }

def classify_by_proximity(gaze_points, gaze_magnitude, angle_close_thresh=5.0, calibrated_rois = None):
    """
    Classify gaze to ROI using angle and magnitude proximity.       
    Returns: (roi_name, angle_distance)
    """
    if gaze_points is None:
        return "none", 0.0

    gaze_angle = np.asarray(gaze_points, dtype=np.float32)
    gaze_magnitude = np.asarray(gaze_magnitude, dtype=np.float32)

    # Use calibrated ROIs if provided, otherwise use defaults
    if calibrated_rois is None:
        ROIs = _get_default_rois()
    else:
        ROIs = calibrated_rois

    left_mag = gaze_magnitude[0]
    right_mag = gaze_magnitude[1]
    
    angle_dists = {roi: circular_angle_diff(gaze_angle, info["angle"]) for roi, info in ROIs.items()}
    top3 = sorted(angle_dists.items(), key=lambda kv: kv[1])[:3]

    _, best_angle = top3[0]

    # Filter candidates within angle_close_thresh of best angle
    candidates = [roi for roi, d in top3 if d <= (best_angle + angle_close_thresh)]

    if len(candidates) == 1:
        roi = candidates[0]
        return roi, float(angle_dists[roi])

    # Break tie using magnitude distance
    roi = min(candidates, key=lambda r: (abs(left_mag - ROIs[r]["left_mag"]) + abs(right_mag - ROIs[r]["right_mag"])))
    return roi, float(angle_dists[roi])


def classify_by_point_proximity(gaze_points, calibrated_rois= None):
    if gaze_points is None:
        return "none", float('inf')

    gaze_point = np.asarray(gaze_points, dtype=np.float32)
    if len(gaze_point) < 2:
        return "none", float('inf')
    
    gaze_x, gaze_y = float(gaze_point[0]), float(gaze_point[1])

    # Use calibrated ROIs if provided, otherwise cannot classify
    if calibrated_rois is None:
        return "none", float('inf')

    # Calculate Euclidean distance to each ROI centroid
    distances = {}
    for roi_name, roi_data in calibrated_rois.items():
        if "centroid_x" not in roi_data or "centroid_y" not in roi_data:
            # Skip ROIs without centroid data
            continue
        
        centroid_x = float(roi_data["centroid_x"])
        centroid_y = float(roi_data["centroid_y"])
        
        # Euclidean distance formula: sqrt((x2-x1)^2 + (y2-y1)^2)
        distance = np.sqrt((gaze_x - centroid_x)**2 + (gaze_y - centroid_y)**2)
        distances[roi_name] = distance

    if not distances:
        return "none", float('inf')

    # Find ROI with minimum distance
    roi = min(distances, key=distances.get)
    return roi, float(distances[roi])


def classify_by_proximity_multicam(gaze_data_list, calibrated_rois_list = None, angle_close_thresh = 5.0):
    """
    Algorithm:
    1. For each camera, compute circular angle difference for all ROIs
    2. Sum angle differences across cameras for each ROI
    3. Select top 3 ROIs by summed angle difference
    4. Among top 3, sum magnitude differences across cameras
    5. Select ROI with minimum summed magnitude difference
    """
    if not gaze_data_list:
        return "none", 0.0

    # Handle case of single camera
    if len(gaze_data_list) == 1:
        gaze_data = gaze_data_list[0]
        rois = calibrated_rois_list[0] if calibrated_rois_list else None
        return classify_by_proximity(gaze_data["mid_ang"], gaze_data["gaze_magnitude"], angle_close_thresh, rois)

    # Get ROI definitions per camera
    if calibrated_rois_list is None:
        calibrated_rois_list = [None] * len(gaze_data_list)

    rois_per_camera = []
    for rois in calibrated_rois_list:
        if rois is None:
            rois_per_camera.append(_get_default_rois())
        else:
            rois_per_camera.append(rois)

    # Get all ROI names from first camera (assuming all cameras have same ROI names)
    roi_names = list(rois_per_camera[0].keys())

    # Sum angle differences across cameras for each ROI
    summed_angle_dists = {roi: 0.0 for roi in roi_names}
    
    for cam_idx, gaze_data in enumerate(gaze_data_list):
        if gaze_data is None or gaze_data.get("mid_ang") is None:
            continue
        
        mid_ang = gaze_data["mid_ang"]
        rois = rois_per_camera[cam_idx]
        
        for roi_name in roi_names:
            angle_diff = circular_angle_diff(mid_ang, rois[roi_name]["angle"])
            summed_angle_dists[roi_name] += angle_diff

    # Select top 3 ROIs by summed angle difference
    top3 = sorted(summed_angle_dists.items(), key=lambda kv: kv[1])[:3]
    
    if not top3:
        return "none", 0.0

    _, best_summed_angle = top3[0]

    # Filter candidates within angle_close_thresh of best summed angle
    candidates = [roi for roi, d in top3 if d <= (best_summed_angle + angle_close_thresh)]

    if len(candidates) == 1:
        roi = candidates[0]
        return roi, float(summed_angle_dists[roi])

    # Among candidates, sum magnitude differences across cameras
    summed_mag_dists = {}
    
    for candidate_roi in candidates:
        total_mag_dist = 0.0
        
        for cam_idx, gaze_data in enumerate(gaze_data_list):
            if gaze_data is None or gaze_data.get("gaze_magnitude") is None:
                continue
            
            gaze_magnitude = gaze_data["gaze_magnitude"]
            rois = rois_per_camera[cam_idx]
            
            left_mag = gaze_magnitude[0]
            right_mag = gaze_magnitude[1]
            roi_config = rois[candidate_roi]
            
            # Sum absolute magnitude differences for this camera
            mag_dist = abs(left_mag - roi_config["left_mag"]) + abs(right_mag - roi_config["right_mag"])
            total_mag_dist += mag_dist
        
        summed_mag_dists[candidate_roi] = total_mag_dist

    # Select ROI with minimum summed magnitude distance
    roi = min(summed_mag_dists, key=summed_mag_dists.get)
    return roi, float(summed_angle_dists[roi])


def classify_by_angle(gaze_points, iris_points):
    """
    Classify gaze to one of four corners using midpoint angle for camera test 
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


def apply_headpose_adjustment(gaze_points, iris_points, pitch, yaw, yaw_gain=0.005, pitch_gain=0.005):
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

    def __init__(self, angles: bool = False, scatter: bool = False):
        self.angles = bool(angles)
        self.scatter = bool(scatter)
        self._t = []
        self._left = []
        self._right = []
        self._mid = []
        self._mag_right = []
        self._mag_left = []
        self._gaze_points_x = []
        self._gaze_points_y = []

    def log(self, left_ang, right_ang, mid_ang, left_mag, right_mag, time, gaze_points_adj = None):
        if not self.angles and not self.scatter:
            return
        if self.angles:
            self._t.append(time)
            self._left.append(left_ang)
            self._right.append(right_ang)
            self._mid.append(mid_ang)
            self._mag_left.append(left_mag)
            self._mag_right.append(right_mag)
        if self.scatter and gaze_points_adj is not None:
            # Extract x, y coordinates from gaze points
            gaze_points_adj = np.asarray(gaze_points_adj, dtype=np.float32)
            self._gaze_points_x.append(gaze_points_adj[0])
            self._gaze_points_y.append(gaze_points_adj[1])

    # def has_samples(self) -> bool:
    #     return len(self._t) > 0

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
        # if not self.has_samples():
        #     print("No gaze samples logged.")
        #     return
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
        # if not self.has_samples():
        #     return
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

    def scatter_plot(self, cam_index):
        if len(self._gaze_points_x) == 0 or len(self._gaze_points_y) == 0:
            print("No gaze point samples available for scatter plot.")
            return
        
        plt.figure(figsize=(8, 6))
        plt.scatter(self._gaze_points_x, self._gaze_points_y, alpha=0.5, s=20, c=range(len(self._gaze_points_x)), cmap='viridis')
        plt.colorbar(label="Frame sequence")
        plt.xlabel("X coordinate (pixels)")
        plt.ylabel("Y coordinate (pixels)")
        plt.title(f"Gaze Points Scatter Plot (Camera {cam_index})")
        plt.grid(True, alpha=0.3)
        plt.gca().invert_yaxis()  # Invert Y-axis to match image coordinates
        plt.tight_layout()
        plt.show()

class GazeProcessor:
    """adjust by headpose -> determine 
    t -> compute vectors/angles -> classify"""

    def __init__(
        self,
        yaw_gain = 0.005,
        pitch_gain = 0.005,
        history_points= 20,
        std_threshold= 15.0,
        ratio_threshold = 2.0,
        calibrated_rois= None,
        roi_classifier = "proximity"):

        self.yaw_gain = yaw_gain
        self.pitch_gain = pitch_gain
        self.history_points = history_points
        self.left_history = []
        self.right_history = []
        self.std_threshold = std_threshold
        self.ratio_threshold = ratio_threshold
        self.calibrated_rois = calibrated_rois
        self.roi_classifier = roi_classifier

    def process(self, gaze_points, iris_points, gaze_magnitude, pitch, yaw) -> Optional[Dict[str, Any]]:
        if gaze_points is None or iris_points is None:
            return None

        gp_adj = apply_headpose_adjustment(
            gaze_points, iris_points, pitch, yaw, yaw_gain=self.yaw_gain, pitch_gain=self.pitch_gain)
        left_vec = gp_adj[0] - iris_points[0]
        right_vec = gp_adj[1] - iris_points[1]
        left_ang = angle_from_vector(left_vec)
        right_ang = angle_from_vector(right_vec)

        # Maintain a short history for each eye 
        self.left_history.append(left_vec)
        self.right_history.append(right_vec)
        if len(self.left_history) > self.history_points:
            self.left_history.pop(0)
        if len(self.right_history) > self.history_points:
            self.right_history.pop(0)

        # Use select_reliable_eye to determine which eye data to use
        eye_used, mid_pt, mid_ang = select_reliable_eye(
            self.left_history, self.right_history,
            left_vec, right_vec,
            iris_points, gp_adj,
            std_threshold=self.std_threshold,
            ratio_threshold=self.ratio_threshold
        )

        corner_name, _ = classify_by_angle(mid_ang, iris_points)

        # Use selected classifier method with the reliable eye data
        roi, _ = classify_by_point_proximity(mid_pt, calibrated_rois=self.calibrated_rois)
        roi_cluster, _ = classify_by_proximity(mid_ang, gaze_magnitude, calibrated_rois=self.calibrated_rois)

        return {
            "gaze_points_adj": mid_pt,
            "left_vec": left_vec,
            "right_vec": right_vec,
            "left_ang": left_ang,
            "right_ang": right_ang,
            "mid_ang": mid_ang,
            "corner_name": corner_name,
            "roi": roi,
            "roi_cluster": roi_cluster, 
            "gaze_magnitude": gaze_magnitude,
        }


class CameraPrioritySelector:
    """
    Prioritizes cameras with better frontal views 
    """

    def __init__(self, priority_threshold=0.2):
        self.priority_threshold = priority_threshold

    def select_cameras(self, gaze_results_list, head_poses_list):
        if len(gaze_results_list) <= 1:
            return [i for i, r in enumerate(gaze_results_list) if r is not None]

        scores = []
        for i, (gaze_result, head_pose) in enumerate(zip(gaze_results_list, head_poses_list)):
            if gaze_result is None or head_pose is None:
                scores.append(0.0)
                continue
            
            # Head pose: lower values better (frontal view) 
            roll, pitch, yaw = head_pose
            pose_score = 1.0 / (1.0 + abs(roll) + abs(pitch) + abs(yaw))  # Higher is better
            # may need to be adjusted for car setup

            pose_score = 0.7 * pose_score #+ 0.3 
            scores.append(pose_score)

        # Find valid cameras (those with results)
        valid_cameras = [i for i, score in enumerate(scores) if score > 0]

        # Sort by score descending
        sorted_valid = sorted(valid_cameras, key=lambda i: scores[i], reverse=True)
        best_score = scores[sorted_valid[0]]
        second_score = scores[sorted_valid[1]] if len(sorted_valid) > 1 else 0

        # Check if significant difference between best and second
        if best_score > second_score * (1 + self.priority_threshold):
            print("Camera ", sorted_valid[0], " prioritized for fusion")
            return [sorted_valid[0]]
        else:
            return sorted_valid


class MultiCameraROIClassifier:
    def __init__(self, calibrated_rois_list= None, angle_close_thresh = 5.0):
        self.calibrated_rois_list = calibrated_rois_list
        self.angle_close_thresh = angle_close_thresh

    def classify(self, gaze_results_list, selected_indices=None):
        # Filter to selected cameras if provided
        if selected_indices is not None:
            filtered_results = [gaze_results_list[i] if i in selected_indices else None for i in range(len(gaze_results_list))]
        else:
            filtered_results = gaze_results_list

        # Filter out None results (cameras without landmarks)
        valid_results = [r for r in filtered_results if r is not None]
        
        if not valid_results:
            return "none", 0.0

        # Build gaze data list for classification
        gaze_data_list = []
        valid_indices = []
        for idx, result in enumerate(filtered_results):
            if result is not None:
                gaze_data_list.append({
                    "mid_ang": result["mid_ang"],
                    "gaze_magnitude": result.get("gaze_magnitude", None)
                })
                valid_indices.append(idx)

        # Build calibrated ROIs list for valid cameras
        if self.calibrated_rois_list is None:
            calibrated_rois_subset = None
        else:
            calibrated_rois_subset = [self.calibrated_rois_list[i] for i in valid_indices]

        # Classify using combined method
        roi_name, score = classify_by_proximity_multicam(
            gaze_data_list,
            calibrated_rois_list=calibrated_rois_subset,
            angle_close_thresh=self.angle_close_thresh,
        )

        return roi_name, score
