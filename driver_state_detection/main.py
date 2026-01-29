import time
import pprint
import json

import cv2
import mediapipe as mp
import numpy as np

from attention_scorer import AttentionScorer as AttScorer
from eye_detector import EyeDetector as EyeDet
from evaluation import CornerEvaluator
from args_parser import get_args
from pose_estimation import HeadPoseEstimator as HeadPoseEst
from utils import get_landmarks, load_camera_parameters
import matplotlib.pyplot as plt
from gaze_utils import GazeProcessor, GazeLogger, MultiCameraROIClassifier
from calibration import CalibrationManager


def run_calibration(args, camera_matrix, dist_coeffs):
    cam_indices = list(getattr(args, "camera", [0]))
    caps = [cv2.VideoCapture(int(ci)) for ci in cam_indices]
    
    detectors,eye_dets, head_poses, gaze_procs, calib_mgrs = [],[],[],[],[]
    
    # Audio file for calibration instructions
    audio_file = getattr(args, "calibration_audio", None)
    
    # Creates Mediapipe and related detection instances per camera
    for i in range(len(caps)):
        detectors.append(mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_landmarks=True,
        ))

        eye_dets.append(EyeDet(show_processing=args.show_eye_proc))
        head_poses.append(HeadPoseEst(show_axis=args.show_axis, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs))
        gaze_procs.append(GazeProcessor())
        
        # Creates calibration manager 
        calib_mgrs.append(CalibrationManager(roi_duration=args.calibration_duration, transition_duration=2.0, audio_file=audio_file))
        calib_mgrs[i].start_calibration()

    all_complete = [False] * len(caps)

    while not all(all_complete):
        for i, cap in enumerate(caps):
            if cap is None or all_complete[i]:
                continue
                
            ret, frame = cap.read()
            if not ret:
                all_complete[i] = True
                continue

            # mirror 
            frame = cv2.flip(frame, 2)

            frame_size = frame.shape[1], frame.shape[0]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            lms = detectors[i].process(rgb).multi_face_landmarks

            is_complete = False
            if lms:
                landmarks = get_landmarks(lms)
                eye_dets[i].show_eye_keypoints(color_frame=frame, landmarks=landmarks, frame_size=frame_size)

                gaze_points, iris_points, gaze_magnitude = eye_dets[i].get_Gaze_Vector(
                    frame=frame, landmarks=landmarks, frame_size=frame_size
                )
                _, roll, pitch, yaw = head_poses[i].get_pose(frame=frame, landmarks=landmarks, frame_size=frame_size)

                if gaze_points is not None and iris_points is not None:
                    gaze_result = gaze_procs[i].process(gaze_points, iris_points, gaze_magnitude, pitch, yaw)
                    if gaze_result is not None:
                        mid_ang = gaze_result["mid_ang"]
                        left_mag = gaze_magnitude[0]
                        right_mag = gaze_magnitude[1]
                        gp_adj = gaze_result["gaze_points_adj"]
                        left_ang = gaze_result["left_ang"]
                        right_ang = gaze_result["right_ang"]
                        left_vec = gaze_result.get("left_vec", None)
                        right_vec = gaze_result.get("right_vec", None)

                        # Update calibration current frame data (pass gaze_points_adj and eye vectors for proper calibration)
                        is_complete = calib_mgrs[i].update(mid_ang, left_mag, right_mag, gaze_points_adj=gp_adj, 
                                                           left_vec=left_vec, right_vec=right_vec)

            else:
                # No face detected still call update for synchronized time with audio
                is_complete = calib_mgrs[i].update(None, None, None, gaze_points_adj=None, left_vec=None, right_vec=None)

            if is_complete:
                all_complete[i] = True

            # Draw calibration overlay 
            calib_mgrs[i].draw_overlay(frame)

            cv2.imshow(f"Calibration - Camera {cam_indices[i]}", frame)

            key = cv2.waitKey(20) & 0xFF
            if key == ord('q'):
                print("Calibration cancelled\n")
                for mgr in calib_mgrs:
                    mgr.stop_audio()
                    mgr.is_active = False
                for cap_item in caps:
                    cap_item.release()
                for det in detectors:
                    det.close()
                cv2.destroyAllWindows()
                return None

    # Release resources
    for cap in caps:
        cap.release()
    for det in detectors:
        det.close()
    cv2.destroyAllWindows()

    # Collect calibration results from all cameras
    calibrated_rois_list = []
    for i, calib_mgr in enumerate(calib_mgrs):
        calibrated_rois = calib_mgr.get_calibration_rois()
        if calibrated_rois:
            print(f"\n=== CALIBRATION COMPLETE - Camera {cam_indices[i]} ===")
            print("Calibrated ROI angles and magnitudes:")
            for roi_name, data in calibrated_rois.items():
                print(
                    f"  {roi_name}: angle={data['angle']:.1f}°, "
                    f"left_mag={data['left_mag']:.4f}, right_mag={data['right_mag']:.4f}"
                )
            calibrated_rois_list.append(calibrated_rois)
        else:
            print(f"No calibration data collected for camera {cam_indices[i]}.")
            calibrated_rois_list.append(None)

    # Save calibration dictionary to JSON file 
    calibration_outputs = getattr(args, "calibration_output", None)
    if calibration_outputs:
        if isinstance(calibration_outputs, str):
            calibration_outputs = [calibration_outputs]
        
        # Warn for mismatch in number of cameras and calibration json
        if len(calibration_outputs) != len(caps):
            print(f"WARNING: {len(calibration_outputs)} output file(s) specified but {len(caps)} camera(s) detected.")
            print(f"Using first {min(len(calibration_outputs), len(caps))} file(s).")
        
        for i, output_file in enumerate(calibration_outputs):
            if i < len(calibrated_rois_list) and calibrated_rois_list[i] is not None:
                try:
                    with open(output_file, "w") as f:
                        json.dump(calibrated_rois_list[i], f, indent=2)
                    print(f"\nCalibration data for camera {cam_indices[i]} saved to {output_file}")
                except Exception as e:
                    print(f"Failed to save calibration data to {output_file}: {e}")

    return calibrated_rois_list if any(calibrated_rois_list) else None


def load_calibration(calibration_files):
    """Load calibration data from JSON file or multiple"""
    # Normalize to list
    if isinstance(calibration_files, str):
        calibration_files = [calibration_files]
    
    calibrated_rois_list = []
    
    for calib_file in calibration_files:
        try:
            with open(calib_file, "r") as f:
                data = json.load(f)
            print(f"Loaded calibration from {calib_file}")
            for roi_name, roi_data in data.items():
                print(
                    f"  {roi_name}: angle={roi_data['angle']:.1f}°, "
                    f"left_mag={roi_data['left_mag']:.4f}, right_mag={roi_data['right_mag']:.4f}"
                )
            calibrated_rois_list.append(data)
        except FileNotFoundError:
            print(f"Calibration file not found: {calib_file}")
            calibrated_rois_list.append(None)
        except Exception as e:
            print(f"Failed to load calibration data from {calib_file}: {e}")
            calibrated_rois_list.append(None)
    
    return calibrated_rois_list if any(calibrated_rois_list) else None

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

    # Check in calibration mode 
    if getattr(args, "calibrate", False):
        calibrated_rois_list = run_calibration(args, camera_matrix, dist_coeffs)
        return

    # if calibration file provided load them 
    calibrated_rois_list = None
    if getattr(args, "calibration_output", None):
        calibrated_rois_list = load_calibration(args.calibration_output)

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

    # Gaze processors per camera
    gaze_processors = []
    
    # Multi-camera ROI classifier
    multicam_roi_classifier = None

    caps = []
    cam_indices = []
    num_cams = 0
    # if video input is provided
    if getattr(args, "video", None):
        video_path = args.video
        caps = [cv2.VideoCapture(video_path)]
        cam_indices = ["video"]
        num_cams = 1
    else:
        cam_indices = list(getattr(args, "camera", [0]))
        caps = [cv2.VideoCapture(int(ci)) for ci in cam_indices]
        num_cams = len(cam_indices)

    total_frames_list = []
    fps_list = []
    for cap in caps:
        try:
            total_frames_list.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
            fps_list.append(float(cap.get(cv2.CAP_PROP_FPS) or 0.0))
        except Exception:
            total_frames_list.append(0)
            fps_list.append(0.0)

    # Create evaluators
    evaluators = []
    if args.corner_eval:
        for tf, fv in zip(total_frames_list, fps_list):
            evaluators.append(CornerEvaluator(args=args, total_frames=tf, fps=fv))
    else:
        evaluators = [None] * len(caps)

    # Create gaze loggers
    gaze_loggers = [GazeLogger(enabled=getattr(args, "angles", False), scatter=getattr(args, "scatter", False)) for _ in caps]

    # Create AttentionScorer instances
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

    # per camera frame index counters
    frame_idxs = [-1] * len(caps)
    start_time = time.time()
    frame_count = 0
    prev_time_per_cam = [time.perf_counter() for _ in caps]

    # create per camera Mediapipe and detector instances 
    detectors = []
    eye_dets = []
    head_poses = []
    for i in range(len(caps)):
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
        
        # assign calibration results to the camera for gaze processing
        cam_calibrated_rois = None
        if calibrated_rois_list is not None:
            cam_calibrated_rois = calibrated_rois_list[i]

        gaze_processors.append(GazeProcessor(calibrated_rois=cam_calibrated_rois, roi_classifier=getattr(args, "roi_classifier", "proximity")))

    # Initialize multi-camera ROI classifier
    multicam_roi_classifier = MultiCameraROIClassifier(calibrated_rois_list=calibrated_rois_list)
    gaze_results_current = [None] * len(caps)
    prev_fusion_roi = None

    # Processing loop (single pass with one frame delayed fusion)
    while True:
        any_active = False
        t_now = time.perf_counter()
        elapsed_time = t_now - prev_time
        prev_time = t_now
        if elapsed_time > 0:
            fps = np.round(1 / elapsed_time, 3)

        for i, cap in enumerate(caps):
            if cap is None:
                gaze_results_current[i] = None
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
                gaze_results_current[i] = None
                continue

            # mirror
            frame = cv2.flip(frame, 2)

            # scale
            scale = float(getattr(args, "video_scale", 1.0))
            if scale != 1.0:
                h, w = frame.shape[:2]
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            e1 = cv2.getTickCount()

            # prepare image for Mediapipe (RGB)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_size = frame.shape[1], frame.shape[0]

            lms = detectors[i].process(rgb).multi_face_landmarks

            # initialize variables
            landmarks = None
            ear = None
            perclos_score = 0.0
            frame_det = None
            roll = pitch = yaw = None
            gaze = None
            gaze_result = None
            iris_points = None
            gaze_magnitude = None

            if lms:
                landmarks = get_landmarks(lms)
                eye_dets[i].show_eye_keypoints(color_frame=frame, landmarks=landmarks, frame_size=frame_size)
                ear = eye_dets[i].get_EAR(landmarks=landmarks)
                tired, perclos_score = scorers[i].get_rolling_PERCLOS(t_now, ear)

                frame_det, roll, pitch, yaw = head_poses[i].get_pose(frame=frame, landmarks=landmarks, frame_size=frame_size)
                gaze = eye_dets[i].get_Gaze_Score(frame=frame, landmarks=landmarks, frame_size=frame_size)

                gaze_points, iris_points, gaze_magnitude = eye_dets[i].get_Gaze_Vector(frame=frame, landmarks=landmarks, frame_size=frame_size)
                
                # Compute gaze result for current frame
                if gaze_points is not None and iris_points is not None:
                    gaze_result = gaze_processors[i].process(gaze_points, iris_points, gaze_magnitude, pitch, yaw)
            
            # Store current gaze result for next iteration fusion
            gaze_results_current[i] = gaze_result

            detected = None
            evaluator = evaluators[i]
            gaze_logger = gaze_loggers[i]
            scorer = scorers[i]

            # Display previous frame info
            if gaze_result is not None and iris_points is not None:
                # Draw gaze vectors
                gp_adj = gaze_result["gaze_points_adj"]
                # start1 = (int(iris_points[0][0]), int(iris_points[0][1]))
                # end1 = (int(gp_adj[0][0]), int(gp_adj[0][1]))
                # cv2.arrowedLine(frame, start1, end1, (255, 255, 0), 2, tipLength=0.2)
                # start2 = (int(iris_points[1][0]), int(iris_points[1][1]))
                # end2 = (int(gp_adj[1][0]), int(gp_adj[1][1]))
                # cv2.arrowedLine(frame, start2, end2, (255, 255, 0), 2, tipLength=0.2)

                corner_name = gaze_result["corner_name"]
                mid_ang = gaze_result["mid_ang"]
                cv2.putText(frame, f"CornerAngle: {corner_name}", (10, 320), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)
                detected = corner_name

                ix = (iris_points[0][0] + iris_points[1][0]) / 2.0
                iy = (iris_points[0][1] + iris_points[1][1]) / 2.0

                start = (int(ix), int(iy))
                end = (int(gp_adj[0]), int(gp_adj[1]))

                cv2.arrowedLine(frame, start, end, (0, 255, 255), 2, tipLength=0.2)


                # Display previous frame info
                if prev_fusion_roi is not None:
                    cv2.putText(frame, f"ROI: {prev_fusion_roi}", (10, 350), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)

                left_vec = gaze_result["left_vec"]
                right_vec = gaze_result["right_vec"]
                left_ang = gaze_result["left_ang"]
                right_ang = gaze_result["right_ang"]
                left_mag = gaze_magnitude[0]
                right_mag = gaze_magnitude[1]
                gp_adj = gaze_result["gaze_points_adj"]

                if gaze_logger.enabled or gaze_logger.scatter:
                    elapsed = time.time() - start_time
                    gaze_logger.log(left_ang, right_ang, mid_ang, left_mag, right_mag, elapsed, gaze_points_adj=gp_adj)
                    if gaze_logger.enabled:
                        cv2.putText(frame, f"L:{left_ang:.1f}  R:{right_ang:.1f}  M:{mid_ang:.1f}",
                                    (10, 205), cv2.FONT_HERSHEY_PLAIN, 1.3, (180, 255, 180), 2)

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

            frame_count += 1
            cv2.imshow(f"Camera {cam_indices[i]} - Press 'q' to terminate", frame)

        # compute fusion result
        fusion_result = multicam_roi_classifier.classify(gaze_results_current)
        if fusion_result is not None:
            prev_fusion_roi, prev_fusion_score = fusion_result

        # break on Q key pressed 
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

    # print the final results
    for i, gl in enumerate(gaze_loggers):
        if gl is not None and gl.has_samples():
            gl.print_summary(cam_index=cam_indices[i])
            gl.plot(cam_index=cam_indices[i])
        if gl is not None and gl.scatter and len(gl._gaze_points_x) > 0:
            gl.scatter_plot(cam_index=cam_indices[i])

    for i, ev in enumerate(evaluators):
        if ev is not None:
            print(f"\n==== Evaluation results for Camera {cam_indices[i]} ====")
            ev.print_results()

    # destroy all windows
    for cap in caps:
        cap.release()

    for d in detectors:
        d.close()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
