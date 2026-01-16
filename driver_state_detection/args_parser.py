import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Driver State Detection")

    # selection the camera number, default is 0 (webcam)
    parser.add_argument(
        "-c",
        "--camera",
        type=int,
        nargs='+',
        default=[0],
        metavar="",
        help="Camera number(s). Provide one or more camera indices (e.g. -c 0 1). Default is 0.",
    )

    parser.add_argument(
        "--camera_params",
        type=str,
        help="Path to the camera parameters file (JSON or YAML).",
    )

    # optional input video file (when provided, video is used instead of camera)
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        metavar="",
        help="Path to an input video file. If provided, the video is used instead of the camera.",
    )

    parser.add_argument(
        "--video_scale",
        type=float,
        default=1.0,
        metavar="",
        help="Scale factor (0.1-1.0) to downscale video frames when using --video; values <1.0 reduce resolution and CPU usage.",
    )

    # visualisation parameters
    parser.add_argument(
        "--show_fps",
        type=bool,
        default=True,
        metavar="",
        help="Show the actual FPS of the capture stream, default is true",
    )
    parser.add_argument(
        "--show_proc_time",
        type=bool,
        default=True,
        metavar="",
        help="Show the processing time for a single frame, default is true",
    )
    parser.add_argument(
        "--show_eye_proc",
        type=bool,
        default=False,
        metavar="",
        help="Show the eyes processing, deafult is false",
    )
    parser.add_argument(
        "--show_axis",
        type=bool,
        default=True,
        metavar="",
        help="Show the head pose axis, default is true",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        metavar="",
        help="Prints additional info, default is false",
    )

    # corner evaluation mode: runs a timed sequence where the user looks at each corner
    parser.add_argument(
        "--corner_eval",
        action="store_true",
        help="Run corner-evaluation routine (top-right, top-left, bottom-left, bottom-right).",
    )
    parser.add_argument(
        "--eval_segment_secs",
        type=float,
        default=5.0,
        metavar="",
        help="Seconds per corner segment in evaluation (default 10).",
    )
    parser.add_argument(
        "--eval_exclude_secs",
        type=float,
        default=1.0,
        metavar="",
        help="Seconds to exclude at start and end of each segment when computing accuracy (default 1 -> middle 8s used for 10s segment).",
    )

    parser.add_argument(
        "--angles",
        # type=bool,
        # default=False,
        action="store_true",
        help="returns the right and left eye gaze angles plot and information for duration of the run"
    )

    # Attention Scorer parameters (EAR, Gaze Score, Pose)
    parser.add_argument(
        "--smooth_factor",
        type=float,
        default=0.5,
        metavar="",
        help="Sets the smooth factor for the head pose estimation keypoint smoothing, default is 0.5",
    )
    parser.add_argument(
        "--ear_thresh",
        type=float,
        default=0.15,
        metavar="",
        help="Sets the EAR threshold for the Attention Scorer, default is 0.15",
    )
    parser.add_argument(
        "--ear_time_thresh",
        type=float,
        default=2,
        metavar="",
        help="Sets the EAR time (seconds) threshold for the Attention Scorer, default is 2 seconds",
    )
    parser.add_argument(
        "--gaze_thresh",
        type=float,
        default=0.015,
        metavar="",
        help="Sets the Gaze Score threshold for the Attention Scorer, default is 0.2",
    )
    parser.add_argument(
        "--gaze_time_thresh",
        type=float,
        default=2,
        metavar="",
        help="Sets the Gaze Score time (seconds) threshold for the Attention Scorer, default is 2. seconds",
    )
    parser.add_argument(
        "--pitch_thresh",
        type=float,
        default=20,
        metavar="",
        help="Sets the PITCH threshold (degrees) for the Attention Scorer, default is 30 degrees",
    )
    parser.add_argument(
        "--yaw_thresh",
        type=float,
        default=20,
        metavar="",
        help="Sets the YAW threshold (degrees) for the Attention Scorer, default is 20 degrees",
    )
    parser.add_argument(
        "--roll_thresh",
        type=float,
        default=20,
        metavar="",
        help="Sets the ROLL threshold (degrees) for the Attention Scorer, default is 30 degrees",
    )
    parser.add_argument(
        "--pose_time_thresh",
        type=float,
        default=2.5,
        metavar="",
        help="Sets the Pose time threshold (seconds) for the Attention Scorer, default is 2.5 seconds",
    )

    # parse the arguments and store them in the args variable dictionary
    args, _ = parser.parse_known_args()

    return args
