from typing import List, Optional, Dict
import time
from calibration import AudioPlayer


def build_confusion_matrix(labels, y_true, y_pred):
    idx = {lab: i for i, lab in enumerate(labels)}
    nlab = len(labels)
    cm = [[0] * nlab for _ in range(nlab)]
    for t, p in zip(y_true, y_pred):
        ti = idx.get(t, idx.get("none", nlab - 1))
        pi = idx.get(p, idx.get("none", nlab - 1))
        cm[ti][pi] += 1
    return cm


def print_confusion_matrix(labels, cm):
    print("\nConfusion matrix (rows=expected, cols=predicted):")
    print("\t" + "\t".join(labels))
    for i, row in enumerate(cm):
        print(f"{labels[i]}\t" + "\t".join(str(x) for x in row))


def print_metrics(labels, cm):
    nlab = len(labels)
    print("\nPrecision/Recall/F1/Support:")
    for i, lab in enumerate(labels):
        tp = cm[i][i]
        pred_sum = sum(cm[r][i] for r in range(nlab))
        true_sum = sum(cm[i][c] for c in range(nlab))
        prec = tp / pred_sum if pred_sum > 0 else 0.0
        rec = tp / true_sum if true_sum > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        print(f"{lab}: precision={prec:.3f}, recall={rec:.3f}, f1={f1:.3f}, support={true_sum}")


class CornerEvaluator:

    def __init__(self, args, total_frames):
        self.args = args
        self.corners = ["top_right", "top_left", "bottom_left", "bottom_right"]
        self.n = len(self.corners)
        self.total_frames = int(total_frames)
        self.fps = 20

        self.eval_frames = max(1, int(round(float(self.args.eval_segment_secs) * self.fps)))

        self.exclude_frames = int(round(float(self.args.eval_exclude_secs) * self.fps))
        if (self.eval_frames - 2 * self.exclude_frames) <= 0:
            self.exclude_frames = max(0, (self.eval_frames - 1) // 2)

        self.last_eval_end = self.n * self.eval_frames

        # metrics
        self.counts = [0] * self.n
        self.total_counted = [0] * self.n
        self.y_true= []
        self.y_pred= []

    def process_frame(self, frame_idx, detected):

        current_frame = max(0, int(frame_idx))
        corner_idx = min(current_frame // self.eval_frames, self.n - 1)

        corner_start = corner_idx * self.eval_frames
        corner_end = corner_start + self.eval_frames
        mid_start = corner_start + self.exclude_frames
        mid_end = corner_end - self.exclude_frames

        # Record only if in mid window
        if mid_start <= current_frame < mid_end:
            expected = self.corners[corner_idx]
            pred = detected if (detected in self.corners) else "none"
            self.y_true.append(expected)
            self.y_pred.append(pred)
            self.total_counted[corner_idx] += 1
            if pred == expected:
                self.counts[corner_idx] += 1

        return self.corners[corner_idx]

    def print_results(self):
        print("\n=============== CORNER EVALUATION RESULTS ===============")

        print("\nPer-corner accuracy:")
        for i, c in enumerate(self.corners):
            total = self.total_counted[i]
            correct = self.counts[i]
            accuracy = (correct/total * 100.0)
            print(f"  {c}: {accuracy:.1f}% ({correct}/{total})")

        labels = list(self.corners) + ["none"]
        cm = build_confusion_matrix(labels, self.y_true, self.y_pred)
        print_confusion_matrix(labels, cm)
        print_metrics(labels, cm)


class CarEvaluator:
    def __init__(self, roi_duration=4.0, transition_duration=2.0, roi_names=None, audio_file=None, ear_thresh=0.2):
        self.roi_duration = float(roi_duration)
        self.transition_duration = float(transition_duration)
        self.roi_names = roi_names or ["left_mirror", "right_mirror", "radio", "road", "lap", "rearmirror", "left_window", "right_window"]
        self.roi_index = 0
        self.in_transition = False
        self.segment_start_time = None
        self.audio_player = None
        self.audio_file = audio_file
        self.ear_thresh = ear_thresh

        # stores metric information 
        self.metrics = {}
        for name in self.roi_names:
            self.metrics[name] = {}
        
        self.classifations = {
            "proximity": ([], []),
            "point_proximity": ([], [])
        }

    def start(self):
        self.roi_index = 0
        self.in_transition = True
        self.segment_start_time = time.time()
        # Start audio playback if provided
        if self.audio_file:
            self.audio_player = AudioPlayer(self.audio_file)
            self.audio_player.play()

    def stop_audio(self):
        if self.audio_player:
            self.audio_player.stop()
        self.audio_player = None

    def add_classifier(self, classifier_name, true_label, pred_label, roi_name):
        # # Add to classifier lists
        if classifier_name not in self.classifations:
            self.classifations[classifier_name] = ([], [])
        
        y_true, y_pred = self.classifations[classifier_name]
        y_true.append(true_label)
        y_pred.append(pred_label)
        
        # # add new space for classifier for first entry
        if classifier_name not in self.metrics[roi_name]:
            self.metrics[roi_name][classifier_name] = {"correct": 0, "total": 0}
        
        self.metrics[roi_name][classifier_name]["total"] += 1
        if pred_label == true_label:
            self.metrics[roi_name][classifier_name]["correct"] += 1

    def update(self, true_roi, gaze_result, calibrated_rois= None, ear_score=None):
        elapsed = time.time() - self.segment_start_time

        # Determine if eyes are closed (blink detected)
        eye_closed = (ear_score is not None) and (ear_score <= self.ear_thresh)

        # Check if evaluation is complete
        if self.roi_index >= len(self.roi_names):
            return True

        # Only process if true_roi is a valid ROI name
        if true_roi not in self.roi_names:
            return False

        if self.in_transition:
            if elapsed >= self.transition_duration:
                self.in_transition = False
                self.segment_start_time = time.time()
            return False

        # Collect classifier outputs during recording window, but only if eyes are open (not blinking)
        if gaze_result is not None and true_roi is not None and not eye_closed:
            roi_angle = gaze_result.get("roi", None)
            roi_point = gaze_result.get("roi_cluster", None)
            self.add_classifier("proximity", true_roi, roi_angle, true_roi)
            self.add_classifier("point_proximity", true_roi, roi_point, true_roi)

    ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ##
        if elapsed >= self.roi_duration:
            self.roi_index += 1
            if self.roi_index >= len(self.roi_names):
                self.stop_audio()
                return True
            # next transition
            self.in_transition = True
            self.segment_start_time = time.time()

        return False

    def get_current_roi(self):
        if self.roi_index < len(self.roi_names):
            return self.roi_names[self.roi_index]
        return "complete"

    def print_results(self):
        labels = self.roi_names + ["none"]
        
        for classifier_name in sorted(self.classifations.keys()):
            y_true, y_pred = self.classifations[classifier_name]
            
            if classifier_name == "proximity":
                print("\n=============== PROXIMITY CLASSIFIER (angle + magnitude) ===============")
            elif classifier_name == "point_proximity":
                print("\n=============== POINT PROXIMITY CLASSIFIER (gaze point centroid distance) ===============")

            # Per-ROI accuracy
            print("\nROI accuracy:")
            for roi in self.roi_names:
                if classifier_name in self.metrics[roi]:
                    total = self.metrics[roi][classifier_name]["total"]
                    correct = self.metrics[roi][classifier_name]["correct"]
                    accuracy = (correct / total * 100.0) 
                    print(f"  {roi}: {accuracy:.1f}% ({correct}/{total})")

            # Confusion matrix and metrics
            cm = build_confusion_matrix(labels, y_true, y_pred)
            print_confusion_matrix(labels, cm)
            print_metrics(labels, cm)

