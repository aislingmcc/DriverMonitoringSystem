import time
import pprint

import cv2
# import platform
# import os
import mediapipe as mp
import numpy as np

from attention_scorer import AttentionScorer as AttScorer
from eye_detector import EyeDetector as EyeDet
from evaluation import CornerEvaluator
from args_parser import get_args
from pose_estimation import HeadPoseEstimator as HeadPoseEst
from utils import get_landmarks, load_camera_parameters
import matplotlib.pyplot as plt
from gaze_utils import GazeProcessor, GazeLogger

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
    # Note: Mediapipe FaceMesh, EyeDetector and HeadPoseEstimator are created once
    # and reused for sequential per-camera processing to avoid timer/thread issues.

    # timing variables
    prev_time = time.perf_counter()
    fps = 0.0  # Initial FPS value

    t_now = time.perf_counter()

    # Per-camera AttentionScorer instances are created later (one per opened capture).

    # Gaze processors (one per camera)
    # logging is created per-camera later
    gaze_processors = []

    # Evaluation printing is handled by the CornerEvaluator class now.


    # If the user provided an input video file, open single video capture; otherwise open one or more cameras
    caps = []
    cam_indices = []
    num_cams = 0
    if getattr(args, "video", None):
        video_path = args.video
        caps = [cv2.VideoCapture(video_path)]
        cam_indices = ["video"]
        num_cams = 1
    else:
        # args.camera may be a list of ints (support multi-camera)
        cam_indices = list(getattr(args, "camera", [0]))
        caps = [cv2.VideoCapture(int(ci)) for ci in cam_indices]
        num_cams = len(cam_indices)

    # Gather per-capture metadata and create per-camera evaluators/loggers/scorers
    total_frames_list = []
    fps_list = []
    for cap in caps:
        try:
            total_frames_list.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
            fps_list.append(float(cap.get(cv2.CAP_PROP_FPS) or 0.0))
        except Exception:
            total_frames_list.append(0)
            fps_list.append(0.0)

    # Create per-camera evaluator (or None)
    evaluators = []
    if args.corner_eval:
        for tf, fv in zip(total_frames_list, fps_list):
            evaluators.append(CornerEvaluator(args=args, total_frames=tf, fps=fv))
    else:
        evaluators = [None] * len(caps)

    # Create per-camera gaze loggers
    gaze_loggers = [GazeLogger(enabled=getattr(args, "angles", False)) for _ in caps]

    # Create per-camera AttentionScorer instances
    scorers = []
    for _ in caps:
        scorers.append(
            AttScorer(
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
        )

    # per-camera frame index counters
    frame_idxs = [-1] * len(caps)

    start_time = time.time()
    frame_count = 0
    # per-camera previous timestamp for local FPS calculation
    prev_time_per_cam = [time.perf_counter() for _ in caps]

    # create per-camera Mediapipe and detector instances to avoid shared-state issues
    detectors = []
    eye_dets = []
    head_poses = []
    for _ in caps:
        detectors.append(
            mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                refine_landmarks=True,
            )
        )
        eye_dets.append(EyeDet(show_processing=args.show_eye_proc))
        head_poses.append(HeadPoseEst(show_axis=args.show_axis, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs))
        gaze_processors.append(GazeProcessor())

    # Sequential per-camera processing loop
    while True:
        any_active = False
        t_now = time.perf_counter()
        elapsed_time = t_now - prev_time
        prev_time = t_now
        if elapsed_time > 0:
            fps = np.round(1 / elapsed_time, 3)

        for i, cap in enumerate(caps):
            if cap is None:
                continue
            any_active = True

            frame_idxs[i] += 1
            ret, frame = cap.read()
            if not ret:
                try:
                    cap.release()
                except Exception:
                    pass
                caps[i] = None
                continue

            # mirror only when using a local camera index 0
            try:
                cam_idx_val = int(cam_indices[i])
            except Exception:
                cam_idx_val = None
            if cam_idx_val == 0:
                frame = cv2.flip(frame, 2)

            # handle optional video scaling
            try:
                scale = float(getattr(args, "video_scale", 1.0))
            except Exception:
                scale = 1.0
            if scale <= 0:
                scale = 1.0
            if scale < 1.0:
                h, w = frame.shape[:2]
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                if new_w != w or new_h != h:
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            e1 = cv2.getTickCount()

            # prepare image for Mediapipe (RGB) and for other processors (BGR)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_size = frame.shape[1], frame.shape[0]

            lms = detectors[i].process(rgb).multi_face_landmarks

            detected = None
            evaluator = evaluators[i]
            gaze_logger = gaze_loggers[i]
            scorer = scorers[i]

            # initialize per-frame variables to safe defaults
            landmarks = None
            ear = None
            perclos_score = 0.0
            frame_det = None
            roll = pitch = yaw = None
            gaze = None
            gaze_points = None
            iris_points = None
            gaze_result = None

            if lms:
                landmarks = get_landmarks(lms)
                eye_dets[i].show_eye_keypoints(color_frame=frame, landmarks=landmarks, frame_size=frame_size)
                ear = eye_dets[i].get_EAR(landmarks=landmarks)
                tired, perclos_score = scorer.get_rolling_PERCLOS(t_now, ear)

                frame_det, roll, pitch, yaw = head_poses[i].get_pose(frame=frame, landmarks=landmarks, frame_size=frame_size)
                gaze = eye_dets[i].get_Gaze_Score(frame=frame, landmarks=landmarks, frame_size=frame_size)

                gaze_points, iris_points, gaze_magnitude = eye_dets[i].get_Gaze_Vector(frame=frame, landmarks=landmarks, frame_size=frame_size)
                gaze_result = gaze_processors[i].process(gaze_points, iris_points, gaze_magnitude, pitch, yaw)


                if gaze_result is not None:
                    gp_adj = gaze_result["gaze_points_adj"]
                    start1 = (int(iris_points[0][0]), int(iris_points[0][1]))
                    end1 = (int(gp_adj[0][0]), int(gp_adj[0][1]))
                    cv2.arrowedLine(frame, start1, end1, (255, 255, 0), 2, tipLength=0.2)
                    start2 = (int(iris_points[1][0]), int(iris_points[1][1]))
                    end2 = (int(gp_adj[1][0]), int(gp_adj[1][1]))
                    cv2.arrowedLine(frame, start2, end2, (255, 255, 0), 2, tipLength=0.2)

                    corner_name = gaze_result["corner_name"]
                    mid_ang = gaze_result["mid_ang"]
                    cv2.putText(frame, f"CornerAngle: {corner_name}", (10, 320), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)
                    detected = corner_name

                    roi = gaze_result["roi"]
                    cv2.putText(frame, f"ROI: {roi}", (10, 350), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)

                    left_vec = gaze_result["left_vec"]
                    right_vec = gaze_result["right_vec"]
                    left_ang = gaze_result["left_ang"]
                    right_ang = gaze_result["right_ang"]
                    left_mag = gaze_magnitude[0]# array (2)
                    right_mag = gaze_magnitude[1]# array (2)

                    if gaze_logger.enabled:
                        elapsed = time.time() - start_time
                        gaze_logger.log(left_ang, right_ang, mid_ang, left_mag, right_mag, elapsed)
                        cv2.putText(frame, f"L:{left_ang:.1f}  R:{right_ang:.1f}  M:{mid_ang:.1f}",
                                    (10, 205), cv2.FONT_HERSHEY_PLAIN, 1.3, (180, 255, 180), 2)

            # else:
            #     # optional verbose logging to help debug why landmarks are not found
            #     if getattr(args, 'verbose', False) and (frame_idxs[i] % 30 == 0):
            #         print(f"Camera {cam_indices[i]}: no landmarks at frame {frame_idxs[i]}")

            # evaluate attention scores even when landmarks are missing (scorer handles None values)
            asleep, looking_away, distracted = scorer.eval_scores(
                t_now=t_now,
                ear_score=ear,
                gaze_score=gaze,
                head_roll=roll,
                head_pitch=pitch,
                head_yaw=yaw,
            )

            if frame_det is not None:
                frame = frame_det

            if ear is not None:
                cv2.putText(frame, "EAR:" + str(round(ear, 3)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, cv2.LINE_AA)
            if gaze is not None:
                cv2.putText(frame, "Gaze Score:" + str(round(gaze, 3)), (10, 80), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "PERCLOS:" + str(round(perclos_score, 3)), (10, 110), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, cv2.LINE_AA)

            if roll is not None:
                cv2.putText(frame, "roll:" + str(roll.round(1)[0]), (450, 40), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 1, cv2.LINE_AA)
            if pitch is not None:
                cv2.putText(frame, "pitch:" + str(pitch.round(1)[0]), (450, 70), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 1, cv2.LINE_AA)
            if yaw is not None:
                cv2.putText(frame, "yaw:" + str(yaw.round(1)[0]), (450, 100), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 1, cv2.LINE_AA)

            if tired:
                cv2.putText(frame, "TIRED!", (10, 280), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
            if asleep:
                cv2.putText(frame, "ASLEEP!", (10, 300), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
            if looking_away:
                cv2.putText(frame, "LOOKING AWAY!", (10, 320), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
            if distracted:
                cv2.putText(frame, "DISTRACTED!", (10, 340), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)

            # Use the CornerEvaluator object if available to handle segmenting and metrics
            if evaluator is not None:
                info = evaluator.process_frame(frame_idxs[i], detected)
                cv2.putText(frame, info["instr"], (10, 360), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                cv2.putText(frame, f"time left: {info['time_left']}", (10, 390), cv2.FONT_HERSHEY_PLAIN, 1.2, (255,255,0), 2)
                y = 420
                for ci, corner in enumerate(evaluator.corners):
                    cv2.putText(frame, f"{corner}: {info['accuracies'][ci]*100:.1f}%", (10, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (200,200,200), 1)
                    y += 18
                if info.get("finished"):
                    evaluator.print_results()

            e2 = cv2.getTickCount()
            proc_time_frame_ms = ((e2 - e1) / cv2.getTickFrequency()) * 1000

            # update per-camera local FPS
            now_local = time.perf_counter()
            dt_local = now_local - prev_time_per_cam[i]
            prev_time_per_cam[i] = now_local
            fps_local = (np.round(1 / dt_local, 3) if dt_local > 0 else 0.0)
            if args.show_fps:
                cv2.putText(frame, "FPS:" + str(round(fps_local)), (10, 400), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 1)
            if args.show_proc_time:
                cv2.putText(frame, "PROC. TIME FRAME:" + str(round(proc_time_frame_ms, 0)) + "ms", (10, 430), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 1)

            # increment global frame counter and show the frame on screen (per-camera window)
            frame_count += 1
            win_name = f"Camera {cam_indices[i]} - Press 'q' to terminate"
            cv2.imshow(win_name, frame)

        # one keypress check for loop
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
        if not any_active:
            break

    end_time = time.time()
    # Calculate frames per second
    fps = (frame_count / (end_time - start_time)) / num_cams
    print("Frames processed:", frame_count)
    print("Time taken:", end_time - start_time)
    print("FPS:", fps)

    # If the capture ended but evaluation state still exists, print the final results
    # ---- Post-run display/print for --angles (per-camera) ----
    for i, gl in enumerate(gaze_loggers):
        if gl is not None and gl.has_samples():
            gl.print_summary(cam_index=cam_indices[i])
            gl.plot(cam_index=cam_indices[i])

    # Print final evaluation results per-camera
    for i, ev in enumerate(evaluators):
        if ev is not None:
            try:
                print(f"\n=== Evaluation results for Camera {cam_indices[i]} ===")
                ev.print_results()
            except Exception as e:
                print(f"Failed to print final evaluation results for camera {cam_indices[i]}: {e}")

    # release all captures and destroy windows
    for cap in caps:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
    # close mediapipe detectors
    try:
        for d in detectors:
            try:
                d.close()
            except Exception:
                pass
    except Exception:
        pass
    cv2.destroyAllWindows()
    return


if __name__ == "__main__":
    main()
