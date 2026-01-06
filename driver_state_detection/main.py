import time
import pprint

import cv2
import platform
import os
import mediapipe as mp
import numpy as np

from attention_scorer import AttentionScorer as AttScorer
from eye_detector import EyeDetector as EyeDet
from args_parser import get_args
from pose_estimation import HeadPoseEstimator as HeadPoseEst
from utils import get_landmarks, load_camera_parameters
import matplotlib.pyplot as plt

def angle_from_vector(vec_xy):
    """
    Return angle in degrees [0,360) for a 2D vector (x,y) in image coordinates.
    x positive = right, y positive = down.
    """
    dx, dy = float(vec_xy[0]), float(vec_xy[1])
    ang_rad = np.arctan2(-dy, dx)
    return (np.degrees(ang_rad) + 360.0) % 360.0

def classify_by_angle(gaze_points, iris_points):
                # use the midpoint of the two gaze points
                mid_px = np.mean(gaze_points, axis=0)
                mid_iris = np.mean(iris_points, axis=0)

                corner_names = ["top_left", "top_right", "bottom_right", "bottom_left"]

                def angle_to_corner_idx(angle_deg):
                    a = angle_deg % 360.0
                    if 0 <= a < 90:
                        return corner_names.index("top_right")
                    if 90 <= a < 180:
                        return corner_names.index("top_left")
                    if 180 <= a < 270:
                        return corner_names.index("bottom_left")
                    return corner_names.index("bottom_right")

                # midpoint-based angle (what you already had)
                mid_vec = mid_px - mid_iris
                mid_ang = angle_from_vector(mid_vec)

                return corner_names[angle_to_corner_idx(mid_ang)], mid_ang

# def eyeGaze_using_headPose(gaze_points, iris_points, pitch, yaw):
#     """
#     Combine eye gaze and head pose to determine overall gaze direction.
#     This is a placeholder function; actual implementation would depend on
#     the specific method of combining these inputs.
#     """
#     # Example: Adjust gaze points based on head pose angles
#     adjusted_gaze_points = []
#     for i in range(len(gaze_points)):
#         gaze_vec = gaze_points[i] - iris_points[i]
#         # Simple adjustment based on yaw and pitch (this is illustrative)
#         adjusted_x = gaze_vec[0] + (yaw[0] * .01)  # scale factor for yaw
#         adjusted_y = gaze_vec[1] + (pitch[0] *.01)  # scale factor for pitch
#         adjusted_gaze_points.append(np.array([adjusted_x, adjusted_y]))#iris_points[i] + 
    
#     return np.array(adjusted_gaze_points)

import numpy as np

def eyeGaze_using_headPose(gaze_points, iris_points, pitch, yaw,
                           yaw_gain=0.005, pitch_gain=0.005, clamp=True):
    """
    Adjust gaze endpoint using head pose, all in NORMALISED coordinates.

    gaze_points: (2,2) normalised endpoints (0..1) from get_Gaze_Vector
    iris_points: (2,2) normalised iris centres (0..1)
    pitch, yaw: arrays or scalars; uses first element if array
    returns: (2,2) normalised endpoints (0..1)
    """

    if gaze_points is None or iris_points is None:
        return None

    gaze_points = np.asarray(gaze_points, dtype=np.float32)
    iris_points = np.asarray(iris_points, dtype=np.float32)

    # handle yaw/pitch being list/np arrays or scalars
    yaw_val = float(np.ravel(yaw)[0])
    pitch_val = float(np.ravel(pitch)[0])

    adjusted_endpoints = []
    for i in range(len(gaze_points)):
        # vector from iris to current gaze endpoint (still normalised space)
        gaze_vec = gaze_points[i] - iris_points[i]

        # apply head pose offsets to the VECTOR (small bias)
        gaze_vec_adj = gaze_vec + np.array([yaw_val * yaw_gain, pitch_val * pitch_gain], dtype=np.float32)

        # convert back to an ENDPOINT so your drawing code works
        end_pt = iris_points[i] + gaze_vec_adj

        # keep inside image bounds if desired
        if clamp:
            end_pt = np.clip(end_pt, 0.0, 1.0)

        adjusted_endpoints.append(end_pt)

    return np.asarray(adjusted_endpoints, dtype=np.float32)



def main():
    args = get_args()

    if not cv2.useOptimized():
        try:
            cv2.setUseOptimized(True)  # set OpenCV optimization to True
        except Exception as e:
            print(
                f"OpenCV optimization could not be set to True, the script may be slower than expected.\nError: {e}"
            )

    if args.camera_params:
        camera_matrix, dist_coeffs = load_camera_parameters(args.camera_params)
    else:
        camera_matrix, dist_coeffs = None, None

    if args.verbose:
        print("Arguments and Parameters used:\n")
        pprint.pp(vars(args), indent=4)
        print("\nCamera Matrix:")
        pprint.pp(camera_matrix, indent=4)
        print("\nDistortion Coefficients:")
        pprint.pp(dist_coeffs, indent=4)
        print("\n")

    """instantiation of mediapipe face mesh model. This model give back 478 landmarks
    if the rifine_landmarks parameter is set to True. 468 landmarks for the face and
    the last 10 landmarks for the irises
    """
    Detector = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_landmarks=True,
    )

    # instantiation of the Eye Detector and Head Pose estimator objects
    Eye_det = EyeDet(show_processing=args.show_eye_proc)

    Head_pose = HeadPoseEst(
        show_axis=args.show_axis, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs
    )

    # timing variables
    prev_time = time.perf_counter()
    fps = 0.0  # Initial FPS value

    t_now = time.perf_counter()

    # instantiation of the attention scorer object, with the various thresholds
    # NOTE: set verbose to True for additional printed information about the scores
    Scorer = AttScorer(
        t_now=t_now,
        ear_thresh=args.ear_thresh,
        gaze_time_thresh=args.gaze_time_thresh,
        roll_thresh=args.roll_thresh,
        pitch_thresh=args.pitch_thresh,
        yaw_thresh=args.yaw_thresh,
        ear_time_thresh=args.ear_time_thresh,
        gaze_thresh=args.gaze_thresh,
        pose_time_thresh=args.pose_time_thresh,
        verbose=args.verbose,
    )

    angle_log = None
    if getattr(args, "angles", False):
        angle_log = {
            "t": [],       # wall time (s) since start
            "left": [],    # left-eye angle deg [0,360)
            "right": [],   # right-eye angle deg [0,360)
            "mid": []      # midpoint (your current “combined” angle), optional
        }

    # Helper to print evaluation results (confusion, classification report)
    def _print_eval_results(eval_info, corners):
        
        accuracies = []
        for i in range(len(corners)):
            tot = eval_info.get("total_frames", [0] * len(corners))[i]
            ok = eval_info.get("counts", [0] * len(corners))[i]
            acc = (ok / tot) if tot > 0 else 0.0
            accuracies.append(acc)

        print("accuracy:")
        for i, corner in enumerate(corners):
            print(f"  {corner}: {accuracies[i]*100:.1f}% ({eval_info.get('counts',[0]*len(corners))[i]}/{eval_info.get('total_frames',[0]*len(corners))[i]})")

        labels = corners + ["none"]

        y_true = eval_info.get("y_true", [])
        y_pred = eval_info.get("y_pred", [])


        idx = {lab: i for i, lab in enumerate(labels)}
        cm = [[0 for _ in labels] for _ in labels]
        for t, p in zip(y_true, y_pred):
            ti = idx.get(t, idx["none"]) if t in idx else idx["none"]
            pi = idx.get(p, idx["none"]) if p in idx else idx["none"]
            cm[ti][pi] += 1

        print("\nConfusion matrix (rows=expected, cols=predicted):")
        print("\t" + "\t".join(labels))
        for i, row in enumerate(cm):
            print(f"{labels[i]}\t" + "\t".join(str(x) for x in row))

        # per-class metrics
        print("\nprecision/recall/f1:")
        n = len(labels)
        for i, lab in enumerate(labels):
            tp = cm[i][i]
            pred_sum = sum(cm[r][i] for r in range(n))
            true_sum = sum(cm[i][c] for c in range(n))
            prec = tp / pred_sum if pred_sum > 0 else 0.0
            rec = tp / true_sum if true_sum > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            print(f"{lab}: precision={prec:.3f}, recall={rec:.3f}, f1={f1:.3f}, support={true_sum}")


    # If the user provided an input video file, try to open it first
    if getattr(args, "video", None):
        video_path = args.video
        cap = cv2.VideoCapture(video_path)
    else:
        # capture the input from the default system camera (camera number 0)
        cap = None
        cap = cv2.VideoCapture(args.camera)

    frame_idx = -1
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps_val = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    except Exception:
        total_frames, fps_val = (0, 0.0)
    
    start_time = time.time()
    frame_count = 0


    while True:  # infinite loop for webcam video capture
        # get current time in seconds
        frame_count += 1
        t_now = time.perf_counter()

        # Calculate the time taken to process the previous frame
        elapsed_time = t_now - prev_time
        prev_time = t_now

        # calculate FPS
        if elapsed_time > 0:
            fps = np.round(1 / elapsed_time, 3)

        ret, frame = cap.read()  # read a frame from the webcam

        frame_idx += 1 ####################

        if not ret:  # if a frame can't be read, exit the program
            print("Can't receive frame from camera/stream end")
            break

        if args.corner_eval and not hasattr(main, "_corner_eval_state"):
            main._corner_eval_state = {
                "start_time": t_now,
                "counts": [0, 0, 0, 0],
                "total_frames": [0, 0, 0, 0],
                "y_true": [],
                "y_pred": [],
            }

        detected = None

        # if the frame comes from webcam, flip it so it looks like a mirror.
        if args.camera == 0:
            frame = cv2.flip(frame, 2)

        try:
            scale = float(getattr(args, "video_scale", 1.0))
        except Exception:
            scale = 1.0

        # clamp sensible range
        if scale <= 0:
            scale = 1.0
        if scale < 1.0:
            h, w = frame.shape[:2]
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            if new_w != w or new_h != h:
                # use INTER_AREA for downscaling
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # start the tick counter for computing the processing time for each frame
        e1 = cv2.getTickCount()

        # transform the BGR frame in grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # get the frame size
        frame_size = frame.shape[1], frame.shape[0]

        # apply a bilateral filter to lower noise but keep frame details. create a 3D matrix from gray image to give it to the model
        # gray = cv2.bilateralFilter(gray, 5, 10, 10)
        gray = np.expand_dims(gray, axis=2)
        gray = np.concatenate([gray, gray, gray], axis=2)

        # find the faces using the face mesh model
        lms = Detector.process(gray).multi_face_landmarks

        if lms:  # process the frame only if at least a face is found
            # getting face landmarks and then take only the bounding box of the biggest face
            landmarks = get_landmarks(lms)

            # shows the eye keypoints (can be commented)
            Eye_det.show_eye_keypoints(
                color_frame=frame, landmarks=landmarks, frame_size=frame_size
            )

            # compute the EAR score of the eyes
            ear = Eye_det.get_EAR(landmarks=landmarks)

            # compute the *rolling* PERCLOS score and state of tiredness
            # if you don't want to use the rolling PERCLOS, use the get_PERCLOS method instead
            tired, perclos_score = Scorer.get_rolling_PERCLOS(t_now, ear)
            
            
            
            frame_det, roll, pitch, yaw = Head_pose.get_pose(
                            frame=frame, landmarks=landmarks, frame_size=frame_size
                        )
            # compute the Gaze Score
            gaze = Eye_det.get_Gaze_Score(
                frame=gray, landmarks=landmarks, frame_size=frame_size
            )

            

            # compute and draw the gaze direction on the frame
            # gaze_points, iris_points = Eye_det.get_Gaze_Vector(frame=frame, landmarks=landmarks, frame_size=frame_size)
            # # gaze_points = eyeGaze_using_headPose(gaze_point, iris_points, pitch, yaw)
            # gaze_points = eyeGaze_using_headPose(gaze_points, iris_points, pitch, yaw)

            # h, w = frame.shape[:2]

            # if gaze_points is not None and iris_points is not None:

            #     # Left eye
            #     start1 = (
            #         int(iris_points[0][0] * w),
            #         int(iris_points[0][1] * h)
            #     )
            #     end1 = (
            #         int(gaze_points[0][0] * w),
            #         int(gaze_points[0][1] * h)
            #     )
            #     cv2.arrowedLine(frame, start1, end1, (255, 255, 0), 2, tipLength=0.2)

            #     # Right eye
            #     start2 = (
            #         int(iris_points[1][0] * w),
            #         int(iris_points[1][1] * h)
            #     )
            #     end2 = (
            #         int(gaze_points[1][0] * w),
            #         int(gaze_points[1][1] * h)
            #     )
            #     cv2.arrowedLine(frame, start2, end2, (255, 255, 0), 2, tipLength=0.2)

            gaze_points, iris_points = Eye_det.get_Gaze_Vector(frame=frame, landmarks=landmarks, frame_size=frame_size)
            gaze_points = eyeGaze_using_headPose(gaze_points, iris_points, pitch, yaw)

            h, w = frame.shape[:2]

            if gaze_points is not None and iris_points is not None:
                # Left eye
                start1 = (int(iris_points[0][0] * w), int(iris_points[0][1] * h))
                end1   = (int(gaze_points[0][0] * w), int(gaze_points[0][1] * h))
                cv2.arrowedLine(frame, start1, end1, (255, 255, 0), 2, tipLength=0.2)

                # Right eye
                start2 = (int(iris_points[1][0] * w), int(iris_points[1][1] * h))
                end2   = (int(gaze_points[1][0] * w), int(gaze_points[1][1] * h))
                cv2.arrowedLine(frame, start2, end2, (255, 255, 0), 2, tipLength=0.2)


            # if gaze_points is not None and iris_points is not None:
            #     end1 = (int(gaze_points[0][0]), int(gaze_points[0][1]))
            #     start1 = (int(iris_points[0][0]), int(iris_points[0][1]))
            #     cv2.arrowedLine(frame, start1, end1, (255,255,0), 2, tipLength=0.2)

            #     end2 = (int(gaze_points[1][0]), int(gaze_points[1][1]))
            #     start2 = (int(iris_points[1][0]), int(iris_points[1][1]))
            #     cv2.arrowedLine(frame, start2, end2, (255,255,0), 2, tipLength=0.2)
            #     # pt1 = tuple(map(int, np.ravel(iris_points)[:2]))
                # pt2 = tuple(map(int, np.ravel(gaze_points)[:2]))

                # cv2.arrowedLine(frame, pt1, pt2, (0, 255, 0), 2, tipLength=0.2)

            # def combine_eyeGaze_headPose(gaze_points, iris_points):

            corner_name, mid_ang = classify_by_angle(gaze_points, iris_points)
            cv2.putText(frame, f"CornerAngle: {corner_name}", (10, 320), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255), 2)
            detected = corner_name

            # per-eye angles
            # assume index 0 = left eye, 1 = right eye as returned by Eye_det
            left_vec  = gaze_points[0] - iris_points[0]
            right_vec = gaze_points[1] - iris_points[1]
            left_ang  = angle_from_vector(left_vec)
            right_ang = angle_from_vector(right_vec)


            # Optional HUD if --angles
            if angle_log is not None:
                elapsed = time.time() - start_time
                angle_log["t"].append(elapsed)
                angle_log["left"].append(left_ang)
                angle_log["right"].append(right_ang)
                angle_log["mid"].append(mid_ang)

                # Small on-screen readout (kept compact)
                cv2.putText(frame, f"L:{left_ang:.1f}°  R:{right_ang:.1f}°  M:{mid_ang:.1f}°",
                            (10, 205), cv2.FONT_HERSHEY_PLAIN, 1.3, (180, 255, 180), 2)

            # --- end corner detection block (evaluation moved outside to ensure timeline aligns with video start) ---

            # compute the head pose
            
            
            # evaluate the scores for EAR, GAZE and HEAD POSE
            asleep, looking_away, distracted = Scorer.eval_scores(
                t_now=t_now,
                ear_score=ear,
                gaze_score=gaze,
                head_roll=roll,
                head_pitch=pitch,
                head_yaw=yaw,
            )

            # if the head pose estimation is successful, show the results
            if frame_det is not None:
                frame = frame_det

            # show the real-time EAR score
            if ear is not None:
                cv2.putText(
                    frame,
                    "EAR:" + str(round(ear, 3)),
                    (10, 50),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            # show the real-time Gaze Score
            if gaze is not None:
                cv2.putText(
                    frame,
                    "Gaze Score:" + str(round(gaze, 3)),
                    (10, 80),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            # show the real-time PERCLOS score
            cv2.putText(
                frame,
                "PERCLOS:" + str(round(perclos_score, 3)),
                (10, 110),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            if roll is not None:
                cv2.putText(
                    frame,
                    "roll:" + str(roll.round(1)[0]),
                    (450, 40),
                    cv2.FONT_HERSHEY_PLAIN,
                    1.5,
                    (255, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
            if pitch is not None:
                cv2.putText(
                    frame,
                    "pitch:" + str(pitch.round(1)[0]),
                    (450, 70),
                    cv2.FONT_HERSHEY_PLAIN,
                    1.5,
                    (255, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
            if yaw is not None:
                cv2.putText(
                    frame,
                    "yaw:" + str(yaw.round(1)[0]),
                    (450, 100),
                    cv2.FONT_HERSHEY_PLAIN,
                    1.5,
                    (255, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

            # if the driver is tired, show and alert on screen
            if tired:
                cv2.putText(
                    frame,
                    "TIRED!",
                    (10, 280),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

            # if the state of attention of the driver is not normal, show an alert on screen
            if asleep:
                cv2.putText(
                    frame,
                    "ASLEEP!",
                    (10, 300),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
            if looking_away:
                cv2.putText(
                    frame,
                    "LOOKING AWAY!",
                    (10, 320),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
            if distracted:
                cv2.putText(
                    frame,
                    "DISTRACTED!",
                    (10, 340),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

           # --- Corner evaluation mode (always frame-based) ---
            if args.corner_eval:
                corners = ["top_right", "top_left", "bottom_left", "bottom_right"]

                # ensure eval state exists (defensive)
                if not hasattr(main, "_corner_eval_state") or not isinstance(getattr(main, "_corner_eval_state"), dict):
                    main._corner_eval_state = {}

                eval_info = main._corner_eval_state

                # If any required keys are missing, (re)initialize them defensively
                required_info = ["counts", "total_frames", "y_true", "y_pred", "_init_done", "_eval_frames", "_exclude_frames", "_last_eval_end"]
                if not all(k in eval_info for k in required_info):
                    eval_info.update({
                        "counts": [0] * len(corners),
                        "total_frames": [0] * len(corners),
                        "y_true": [],
                        "y_pred": [],
                        "_init_done": False,
                        "_eval_frames": None,
                        "_exclude_frames": 0,
                        "_last_eval_end": None,
                    })

                # One-time init: compute segment sizes in FRAMES (never in seconds)
                if not eval_info.get("_init_done", False):
                    eval_secs = float(args.eval_segment_secs)
                    exclude_secs = float(args.eval_exclude_secs)

                    if total_frames > 0 and fps_val > 0:
                        eval_frames = max(1, total_frames // len(corners))
                        exclude_frames = int(round(exclude_secs * fps_val))
                        if (eval_frames - 2 * exclude_frames) <= 0:
                            exclude_frames = max(0, (eval_frames - 1) // 2)
                        last_eval_end = (len(corners) - 1) * eval_frames + eval_frames
                    else:
                        # fps unknown: estimate frames from seconds with a safe guess
                        fps_guess = fps_val
                        eval_frames = int(max(1, round(eval_secs * fps_guess)))
                        exclude_frames = int(round(exclude_secs * fps_guess))
                        if (eval_frames - 2 * exclude_frames) <= 0:
                            exclude_frames = max(0, (eval_frames - 1) // 2)
                        last_eval_end = len(corners) * eval_frames

                    eval_info["_eval_frames"] = eval_frames
                    eval_info["_exclude_frames"] = exclude_frames
                    eval_info["_last_eval_end"] = last_eval_end
                    eval_info["_init_done"] = True

                eval_frames = eval_info["_eval_frames"]
                exclude_frames = eval_info["_exclude_frames"]
                last_eval_end = eval_info["_last_eval_end"]

                current_frame = max(0, frame_idx)

                # Determine segment index (clamped)
                corner_eval_idx = int(current_frame // eval_frames) if eval_frames > 0 else 0
                if corner_eval_idx >= len(corners):
                    corner_eval_idx = len(corners) - 1

                # Bounds and mid-window in FRAMES
                corner_start_frame = corner_eval_idx * eval_frames
                corner_end_frame   = corner_start_frame + eval_frames
                mid_start_frame = corner_start_frame + exclude_frames
                mid_end_frame   = corner_end_frame   - exclude_frames

                # Record labels only in the mid-window
                if mid_start_frame <= current_frame < mid_end_frame:
                    eval_info["total_frames"][corner_eval_idx] += 1
                    expected_label = corners[corner_eval_idx]
                    pred_label = detected if (detected in corners) else "none"
                    eval_info["y_true"].append(expected_label)
                    eval_info["y_pred"].append(pred_label)
                    if detected == expected_label:
                        eval_info["counts"][corner_eval_idx] += 1

                # UI: instruction + remaining
                instr = f"Look at: {corners[corner_eval_idx]} ({corner_eval_idx+1}/{len(corners)})"
                cv2.putText(frame, instr, (10, 360), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

                rem_frames = max(0, corner_end_frame - current_frame)
                if fps_val and fps_val > 0:
                    rem = rem_frames / fps_val
                    cv2.putText(frame, f"time left: {rem:.1f}s", (10, 390),
                                cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 0), 2)
                else:
                    cv2.putText(frame, f"frames left: {rem_frames}", (10, 390),
                                cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 0), 2)

                # Finish when past the end of the 4th segment
                finished = current_frame >= last_eval_end

                # Print once per run
                if finished and not hasattr(main, "_printed_final_eval"):
                    _print_eval_results(eval_info, corners)

                    # annotate per-segment accuracy
                    accuracies = []
                    for i in range(len(corners)):
                        tot = eval_info["total_frames"][i]
                        ok  = eval_info["counts"][i]
                        acc = (ok / tot) if tot > 0 else 0.0
                        accuracies.append(acc)
                    y = 420
                    for i, corner in enumerate(corners):
                        cv2.putText(frame, f"{corner}: {accuracies[i]*100:.1f}%", (10, y),
                                    cv2.FONT_HERSHEY_PLAIN, 1.2, (200, 200, 200), 1)
                        y += 20

                    main._printed_final_eval = True  # prevent duplicate prints

        e2 = cv2.getTickCount()
        # processign time in milliseconds
        proc_time_frame_ms = ((e2 - e1) / cv2.getTickFrequency()) * 1000
        # print fps and processing time per frame on screen
        if args.show_fps:
            cv2.putText(
                frame,
                "FPS:" + str(round(fps)),
                (10, 400),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (255, 0, 255),
                1,
            )
        if args.show_proc_time:
            cv2.putText(
                frame,
                "PROC. TIME FRAME:" + str(round(proc_time_frame_ms, 0)) + "ms",
                (10, 430),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (255, 0, 255),
                1,
            )

        # show the frame on screen
        cv2.imshow("Press 'q' to terminate", frame)

        # if the key "q" is pressed on the keyboard, the program is terminated
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

    end_time = time.time()
    # Calculate frames per second
    fps = frame_count / (end_time - start_time)
    print("Frames processed:", frame_count)
    print("Time taken:", end_time - start_time)
    print("FPS:", fps)

    # If the capture ended but evaluation state still exists, print the final results
    # ---- Post-run display/print for --angles ----
    if angle_log is not None and len(angle_log["t"]) > 0:

        t = np.asarray(angle_log["t"])
        L = np.asarray(angle_log["left"])
        R = np.asarray(angle_log["right"])
        M = np.asarray(angle_log["mid"])

        # Unwrap for smoother output (remove 360° jumps)
        def unwrap_deg(d):
            return np.rad2deg(np.unwrap(np.deg2rad(d)))

        L = unwrap_deg(L)
        R = unwrap_deg(R)
        M = unwrap_deg(M)

        # Print summary
        print("\n=== Gaze Angle Summary ===")
        print(f"Samples: {len(t)}")
        print(f"Time range: {t[0]:.2f}s → {t[-1]:.2f}s")
        print(f"Left eye mean angle: {np.mean(L):.2f}°, std: {np.std(L):.2f}")
        print(f"Right eye mean angle: {np.mean(R):.2f}°, std: {np.std(R):.2f}")
        print(f"Midpoint mean angle: {np.mean(M):.2f}°, std: {np.std(M):.2f}")

        plt.figure(figsize=(8, 3))
        plt.plot(t, L, label="Left eye")
        plt.plot(t, R, label="Right eye")
        plt.plot(t, M, label="Midpoint", linewidth=1)
        plt.xlabel("Time (s)")
        plt.ylabel("Angle (deg, unwrapped)")
        plt.title("Gaze Angles (preview)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    else:
        print("No gaze angle data to display.")

    if args.corner_eval and hasattr(main, "_corner_eval_state"):
        try:
            eval_info = main._corner_eval_state
            corners = ["top_right", "top_left", "bottom_left", "bottom_right"]
            _print_eval_results(eval_info, corners)
            delattr(main, "_corner_eval_state")
        except Exception as e:
            print(f"Failed to print final evaluation results after capture end: {e}")

    cap.release()
    cv2.destroyAllWindows()

    return


if __name__ == "__main__":
    main()
