from typing import List, Optional

class CornerEvaluator:
    """
    Frame-based corner evaluator.
    Splits a video (or stream) into N equal frame segments (default 4).
    Uses exclude_frames at start/end of each segment to form a middle window for scoring.
    Collects y_true / y_pred for the middle windows and prints a confusion matrix
    and precision/recall/f1 at the end.
    """

    def __init__(self, args, total_frames: int, fps: float, corners: Optional[List[str]] = None):
        self.args = args
        self.corners = corners or ["top_right", "top_left", "bottom_left", "bottom_right"]
        self.n = len(self.corners)
        self.total_frames = int(total_frames or 0)
        self.fps = float(fps or 0.0)

        # compute frames per segment
        if self.total_frames > 0 and self.fps > 0:
            self.eval_frames = max(1, self.total_frames // self.n)
        else:
            # fallback using seconds -> guess fps (25)
            fps_guess = max(self.fps, 25.0)
            self.eval_frames = max(1, int(round(float(self.args.eval_segment_secs) * fps_guess)))

        self.exclude_frames = int(round(float(self.args.eval_exclude_secs) * max(self.fps, 25.0)))
        if (self.eval_frames - 2 * self.exclude_frames) <= 0:
            self.exclude_frames = max(0, (self.eval_frames - 1) // 2)

        self.last_eval_end = self.n * self.eval_frames

        # metrics
        self.counts = [0] * self.n
        self.total_counted = [0] * self.n
        self.y_true: List[str] = []
        self.y_pred: List[str] = []
        self.printed = False

    def process_frame(self, frame_idx: int, detected: Optional[str]):
        """Process one frame. Returns overlay info dict.

        keys: instr, time_left, finished, accuracies, corner_idx
        """
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

        instr = f"Look at: {self.corners[corner_idx]} ({corner_idx+1}/{self.n})"
        rem_frames = max(0, corner_end - current_frame)
        if self.fps and self.fps > 0:
            time_left_str = f"{rem_frames / self.fps:.1f}s"
        else:
            time_left_str = f"{rem_frames} frames"

        finished = current_frame >= self.last_eval_end

        accuracies = []
        for i in range(self.n):
            tot = self.total_counted[i]
            ok = self.counts[i]
            accuracies.append((ok / tot) if tot > 0 else 0.0)

        return {
            "instr": instr,
            "time_left": time_left_str,
            "finished": finished,
            "accuracies": accuracies,
            "corner_idx": corner_idx,
            "mid_window": (mid_start, mid_end),
        }

    def print_results(self):
        if self.printed:
            return
        self.printed = True

        print("\nCorner evaluation finished. Per-segment accuracy:")
        for i, c in enumerate(self.corners):
            tot = self.total_counted[i]
            ok = self.counts[i]
            pct = (ok / tot * 100.0) if tot > 0 else 0.0
            print(f"  {c}: {pct:.1f}% ({ok}/{tot})")

        labels = list(self.corners) + ["none"]
        idx = {lab: i for i, lab in enumerate(labels)}
        nlab = len(labels)
        cm = [[0] * nlab for _ in range(nlab)]
        for t, p in zip(self.y_true, self.y_pred):
            ti = idx.get(t, idx["none"])
            pi = idx.get(p, idx["none"])
            cm[ti][pi] += 1

        print("\nConfusion matrix (rows=expected, cols=predicted):")
        print("\t" + "\t".join(labels))
        for i, row in enumerate(cm):
            print(f"{labels[i]}\t" + "\t".join(str(x) for x in row))

        print("\nprecision/recall/f1/support:")
        for i, lab in enumerate(labels):
            tp = cm[i][i]
            pred_sum = sum(cm[r][i] for r in range(nlab))
            true_sum = sum(cm[i][c] for c in range(nlab))
            prec = tp / pred_sum if pred_sum > 0 else 0.0
            rec = tp / true_sum if true_sum > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            print(f"{lab}: precision={prec:.3f}, recall={rec:.3f}, f1={f1:.3f}, support={true_sum}")
