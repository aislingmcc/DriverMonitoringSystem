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
        help="Scale factor (0.1-1.0) to downscale video input",
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
        help="Seconds per corner segment in evaluation.",
    )
    parser.add_argument(
        "--eval_exclude_secs",
        type=float,
        default=1.0,
        metavar="",
        help="Seconds to exclude at start and end of each segment",
    )

    parser.add_argument(
        "--angles",
        action="store_true",
        help="returns the right and left eye gaze angles plot",
    )

    parser.add_argument(
        "--scatter",
        action="store_true",
        help="returns a scatter plot of gaze points (x, y coordinates)"
    )

    parser.add_argument(
        "--ellipse",
        action="store_true",
        help="When used with --scatter and --calibration_output display Mahalanobis ellipses for each ROI on scatter plots",
    )

    parser.add_argument(
        "--centroid",
        action="store_true",
        help="When used with --scatter and --calibration_output display ROI centroids on scatter plots",
    )

    parser.add_argument(
        "--roi_classifier",
        type=str,
        default="proximity",
        choices=["proximity", "point_proximity"],
        metavar="",
        help="ROI classification method: 'proximity' uses angle+magnitude (default), 'point_proximity' uses gaze point centroid with Euclidean distance"
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
        default=0.1,
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

    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run calibration mode: look at each of the 8 ROIs for 5 seconds each to collect gaze signatures.",
    )
    parser.add_argument(
        "--calibration_duration",
        type=float,
        default=4,
        metavar="",
        help="Duration (seconds) to look at each ROI during calibration.",
    )
    parser.add_argument(
        "--calibration_output",
        type=str,
        nargs='+',
        default=None,
        metavar="",
        help="Optional file path(s) to save/load calibration data as JSON. Provide one file per camera.",
    )
    parser.add_argument(
        "--calibration_audio",
        type=str,
        default=None,
        metavar="",
        help="Optional audio file path with ROI instructions to play during calibration.",
    )

    parser.add_argument(
        "--car_eval",
        action="store_true",
        help="Run ROI classification evaluation (transition 2s, record 4s per ROI) to measure classifier accuracy.",
    )

    parser.add_argument(
        "--record",
        type=str,
        default=None,
        metavar="",
        help=(
            "Record camera output to the specified file or path."
        ),
    )

    parser.add_argument(
        "--live_graph",
        action="store_true",
        help="Enable visualization of ROI classification over time",
    )

    parser.add_argument(
        "--return_graph",
        action="store_true",
        help="Show a final graph of all collected ROI classification data (with ROI reference lines) after recording, even if --live_graph is not enabled.",
    )

    parser.add_argument(
        "--live_graph_history",
        type=int,
        default=60,
        metavar="",
        help="Duration of data history displayed on the graph",
    )

    parser.add_argument(
        "--live_graph_normalise",
        action="store_true",
        help="Enable normalized ROI metric visualization. Converts distances to confidence-based scores positioned at ROI y-coordinates.",
    )

    # parse the arguments and store them in the args variable dictionary
    args, _ = parser.parse_known_args()

    return args
