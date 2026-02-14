import time
import cv2
import numpy as np
import threading
import subprocess
# from gaze_utils import select_reliable_eye

class Calibration:

    def __init__(self, roi_duration= 4.0, transition_duration= 2.0, audio_file=None, ear_thresh=0.2):
        self.roi_duration = roi_duration
        self.transition_duration = transition_duration
        self.roi_names = ["left_mirror", "right_mirror", "radio", "road", "lap", "rearmirror","left_window", "right_window"]
        self.roi_index = 0
        self.calibration_data = {}
        self.in_transition = False
        self.segment_start_time = None
        self.current_mid_angles = []
        self.current_left_mags = []
        self.current_right_mags = []
        self.current_gaze_points = []
        self.audio_player = None
        self.audio_file = audio_file
        self.ear_thresh = ear_thresh
        self.eye_counter = 0

    def start_calibration(self):
        self.roi_index = 0
        self.calibration_data = {}
        self.in_transition = True
        self.segment_start_time = time.time()
        
        # Start audio playback if provided
        if self.audio_file:
            self.audio_player = AudioPlayer(self.audio_file)
            self.audio_player.play()

    def update(self, mid_ang, left_mag, right_mag, gaze_points_adj= None, 
               left_vec = None, right_vec = None, ear_score=None):
        
        """
        Update calibration with gaze info
        """
        elapsed = time.time() - self.segment_start_time
        
        # Determine if eyes are closed (blink detected)
        eye_closed = (ear_score is not None) and (ear_score <= self.ear_thresh)
        if eye_closed:
            self.eye_counter = self.eye_counter + 1 
            print("eyes closed!!!!!: ", self.eye_counter)

        # transition period
        if self.in_transition:
            if elapsed >= self.transition_duration:
                self.in_transition = False
                # recording period 
                self.segment_start_time = time.time()
                self.current_mid_angles = []
                self.current_left_mags = []
                self.current_right_mags = []
                self.current_gaze_points = []
            return False

        # Only collect data if valid gaze data is available AND eyes are open (not blinking)
        if mid_ang is not None and left_mag is not None and right_mag is not None and not eye_closed:
            # arrays for current ROI 
            self.current_mid_angles.append(mid_ang)
            self.current_left_mags.append(left_mag)
            self.current_right_mags.append(right_mag)
            self.current_gaze_points.append(gaze_points_adj)
            
        # record roi data
        if elapsed >= self.roi_duration:
            roi_name = self.roi_names[self.roi_index]
            
            # if len(self.current_mid_angles) > 0:
            roi_data = {"angle": float(np.median(self.current_mid_angles)) if len(self.current_mid_angles) > 0 else 0.0,
                        "left_mag": float(np.median(self.current_left_mags)) if len(self.current_left_mags) > 0 else 0.0,
                        "right_mag": float(np.median(self.current_right_mags)) if len(self.current_right_mags) > 0 else 0.0,
                        "angle_std": float(np.std(self.current_mid_angles)) if len(self.current_mid_angles) > 0 else 0.0}
            
            # Calculate centroid of gaze points 
            if len(self.current_gaze_points) >= 1:
                gaze_points_array = np.asarray(self.current_gaze_points, dtype=np.float32)
                centroid_x = float(np.median(gaze_points_array[:, 0]))
                centroid_y = float(np.median(gaze_points_array[:, 1]))
                # print(f"Centroid for {roi_name}: ({centroid_x}, {centroid_y})")
                roi_data["centroid_x"] = centroid_x
                roi_data["centroid_y"] = centroid_y

            self.calibration_data[roi_name] = roi_data

            # Move on to next ROI
            self.roi_index += 1

            if self.roi_index >= len(self.roi_names):
                self.stop_audio()
                return True  # calibration complete

            # transition period
            self.in_transition = True
            self.segment_start_time = time.time()
        return False

    def stop_audio(self):
        # Safely stop audio player if it exists
        if self.audio_player:
            try:
                self.audio_player.stop()
            except Exception:
                pass
        self.audio_player = None

    def get_current_roi(self):
        if self.roi_index < len(self.roi_names):
            return self.roi_names[self.roi_index]
        return "complete"

    def get_calibration_rois(self):
        return self.calibration_data

    def draw_overlay(self, frame, cam_idx=0):
        """ calibration instructions and progress """

        roi_name = self.get_current_roi()
        time_remaining = 0.0
        elapsed = time.time() - self.segment_start_time

        if self.in_transition:
            # breather message
            colour = (255, 255, 0)  
            cv2.putText(frame, f"TRANSITION: {roi_name} next", (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, colour, 2)
            total_duration = self.transition_duration
            time_remaining = max(0.0, self.transition_duration - elapsed)
        else:
            # ROI instruction
            colour = (0, 255, 255)  
            cv2.putText(frame, f"CALIBRATING: Look at {roi_name}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, colour, 2)
            total_duration = self.roi_duration 
            time_remaining = max(0.0, self.roi_duration - elapsed)

        progress_decimal = (total_duration - time_remaining) / total_duration
        progress_bar = (10 + int(300 * progress_decimal), 100)

        cv2.rectangle(frame, (10,80), (310, 100), (255, 255, 255), 1)
        cv2.rectangle(frame, (10,80), progress_bar, colour, -1)

class AudioPlayer:
    def __init__(self, audio_file):
        self.audio_file = audio_file
        self.playback_thread = None
        self.is_playing = False

    def play(self):
        """Start audio playback in background thread"""
        if self.is_playing:
            return
        
        self.is_playing = True
        self.playback_thread = threading.Thread(target=self._play_audio, daemon=True)
        self.playback_thread.start()

    def _play_audio(self):
        try:
            subprocess.Popen(["ffplay", "-nodisp", "-autoexit", self.audio_file])
            return
        except FileNotFoundError:
            pass

    def stop(self):
        self.is_playing = False