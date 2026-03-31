import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import time

class ROILiveGraph:
    CAMERA_COLOURS = {
        0: ("#f92a9f", "Camera 0"),
        1: ("#0effab", "Camera 1"),
        None: ("#b0b0b06e", "Unknown"),
    }
    
    def __init__(self, history_duration=60, roi_list=None, normalise=False, plot_mode="scalar"):
        self.history_duration = history_duration
        self.roi_list = roi_list
        self.normalised = normalise
        self.plot_mode = plot_mode
        self.current_camera_idx = None
        self.roi_names = ["left_mirror", "right_mirror", "road", "rearmirror", "left_window","right_window", "over_left_shoulder", "over_right_shoulder", "none"]
        self.roi_colours = {"left_mirror": "pink",
                            "right_mirror": "lime", 
                            "road": "magenta",
                            "rearmirror": "purple",
                            "left_window": "cyan",
                            "right_window": "blue",
                            "over_left_shoulder": "orange",
                            "over_right_shoulder": "red",
                            "none": "gray"}
        self.roi_to_y = {roi: idx for idx, roi in enumerate(self.roi_names, start=1)}
        
        self.roi_ordered_angles = None
        self.roi_line_segments = {roi: [] for roi in self.roi_colours.keys()}
        self.last_roi_change_time = None
        size = int(self.history_duration * 15)
        self.data_buffer = {'times': deque(maxlen=size),
                            'values': deque(maxlen=size),
                            'cameras': deque(maxlen=size)}
        self.full_history = {'times': [],
                             'values': [],
                             'cameras': []}
                                    
        self.fig = None
        self.axes = None
        self.lines = None
        self.start_time = None
        self.animation = None
        self.is_running = False
    
    def roi_ordering(self):
        if not self.roi_list or self.current_camera_idx is None or not (0 <= self.current_camera_idx < len(self.roi_list)):
            return []

        calibrated_rois = self.roi_list[self.current_camera_idx]
        if not calibrated_rois:
            return []

        roi_angles = []
        for roi, data in calibrated_rois.items():
            angle = data.get('angle') if isinstance(data, dict) else None
            if angle is not None:
                roi_angles.append((roi, angle))

        roi_angles.sort(key=lambda x: x[1])
        # returns roi and angle smallest to largest
        return roi_angles

    def normalise(self, value):
        if value is None or not self.roi_ordered_angles:
            return None
        
        # valid angle range
        value = max(0.0, min(360.0, value))

        # normalise angles from 0 to smallest roi
        _, first_angle = self.roi_ordered_angles[0]
        if value < first_angle:
            return value / first_angle

        # normalise between roi angle range
        for i in range(len(self.roi_ordered_angles) - 1):
            _, angle_lower = self.roi_ordered_angles[i]
            _, angle_upper = self.roi_ordered_angles[i + 1]

            if angle_lower <= value <= angle_upper:
                index_lower = i + 1
                diff = angle_upper-angle_lower
                if diff > 0:
                    offset =(value-angle_lower)/diff
                    return index_lower + offset
                return float(index_lower)

        # normalise greater than largest roi
        _, last_angle = self.roi_ordered_angles[-1]
        diff = 360.0 - last_angle
        n_rois = len(self.roi_ordered_angles)
        if diff > 0:
            offset =(value-last_angle)/diff
            if offset > 0.5:
                return offset
            else:
                return n_rois+offset

        return float(n_rois)
    
    def add_data_point(self, current_time, angle_score=None, camera_idx=None, roi_label=None):
        if self.start_time is None:
            self.start_time = current_time
            self.last_roi_change_time = 0

            if camera_idx is not None and self.roi_list and 0 <= camera_idx < len(self.roi_list) and self.roi_list[camera_idx]:
                self.current_camera_idx = camera_idx
            elif self.roi_list:
                # fallback to first valid calibrated camera if available
                for idx, roi in enumerate(self.roi_list):
                    if roi:
                        self.current_camera_idx = idx
                        break
                else:
                    self.current_camera_idx = None
            else:
                self.current_camera_idx = None

            self.roi_ordered_angles = self.roi_ordering()

        relative_time = current_time - self.start_time

        if self.plot_mode == "roi":
            roi_value = self.roi_to_y.get(roi_label, self.roi_to_y["none"])
            self.data_buffer['times'].append(relative_time)
            self.data_buffer['values'].append(roi_value)
            self.data_buffer['cameras'].append(camera_idx)

            self.full_history['times'].append(relative_time)
            self.full_history['values'].append(roi_value)
            self.full_history['cameras'].append(camera_idx)
            return

        # update lines for current camera
        if camera_idx is not None and self.roi_list:
            if camera_idx != self.current_camera_idx:
                relative_time = current_time - self.start_time
                
                # close previous section of lines 
                if self.roi_line_segments[list(self.roi_colours.keys())[0]]:
                    for roi in self.roi_colours.keys():
                        if self.roi_line_segments[roi] and self.roi_line_segments[roi][-1][1] is None:
                            self.roi_line_segments[roi][-1]=(self.roi_line_segments[roi][-1][0], relative_time, self.roi_line_segments[roi][-1][2])
                
                # switch to calibration for new camera
                if 0 <= camera_idx < len(self.roi_list) and self.roi_list[camera_idx]:
                    self.calibrated_rois = self.roi_list[camera_idx]
                    self.current_camera_idx = camera_idx
                    self.roi_ordered_angles = self.roi_ordering()
                    for roi in self.roi_colours.keys():
                        if roi in self.calibrated_rois:
                            if self.normalised:
                                for idx, (r, _) in enumerate(self.roi_ordered_angles, start=1):
                                    if r == roi:
                                        y_value = idx
                                        break
                                else:
                                    y_value = None
                            else:
                                roi_data=self.calibrated_rois[roi]
                                y_value=(roi_data.get('angle',0.0)*.7+(roi_data.get('left_mag', 0.0)+roi_data.get('right_mag', 0.0))*.3*360/40)
                            
                            if y_value is not None:
                                self.roi_line_segments[roi].append((relative_time, None, y_value))
        
        relative_time = current_time - self.start_time
        
        # initialise roi lines
        if not any(self.roi_line_segments.values()):
            if self.roi_list and self.current_camera_idx is not None and 0 <= self.current_camera_idx < len(self.roi_list):
                for roi in self.roi_colours.keys():
                    if self.roi_list[self.current_camera_idx] and roi in self.roi_list[self.current_camera_idx]:
                        if self.normalised:
                            for idx, (r, _) in enumerate(self.roi_ordered_angles, start=1):
                                if r == roi:
                                    y_value = idx
                                    break
                            else:
                                y_value = None
                        else:
                            roi_data=self.roi_list[self.current_camera_idx][roi]
                            y_value=(roi_data.get('angle', 0.0)*.7+(roi_data.get('left_mag',0.0)+roi_data.get('right_mag', 0.0))*.3*360/40)
                        if y_value is not None:
                            self.roi_line_segments[roi].append((0, None, y_value))
        
        if self.normalised:
            angle_value = self.normalise(angle_score)
        else:
            angle_value = angle_score

        if angle_value is None:
            angle_value = float('nan')

        self.data_buffer['times'].append(relative_time)
        self.data_buffer['values'].append(angle_value)
        self.data_buffer['cameras'].append(camera_idx)
        
        # full history 
        self.full_history['times'].append(relative_time)
        self.full_history['values'].append(angle_value)
        self.full_history['cameras'].append(camera_idx)
    
    def _get_roi_marker(self):
        if self.calibrated_rois is None:
            return {}
        
        roi_names = ["left_mirror", "right_mirror", "road", "rearmirror", "left_window", "right_window"]
        calibration_representative = {}
        for roi_name in roi_names:
            roi_data = self.calibrated_rois[roi_name]
            calibration_representative[roi_name] = (roi_data.get('angle', 0.0) * .7 +
                (roi_data.get('left_mag', 0.0) + roi_data.get('right_mag', 0.0))*.3*360/40)
        return calibration_representative
    
    def start(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 4))
        self.line, = self.ax.plot([], [], "o", markersize=4)
        self.is_running = True
        self.animation = FuncAnimation(self.fig,self.update_plot,interval=100,blit=False ,repeat=True,cache_frame_data=False)

        plt.show(block=False)
    
    def _update_plot_with_buffer(self, frame, buffer, show_all=False):
        times = list(buffer['times'])
        values = list(buffer['values'])
        cameras = list(buffer['cameras'])
        if not times or self.ax is None:
            return
        
        self.ax.clear()

        if self.plot_mode == "roi":
            x_max = max(times) if times else 0
            labeled_rois = set()
            for roi in self.roi_names:
                y_value = self.roi_to_y[roi]
                linestyle = ':' if roi == 'none' else '--'
                label = roi.replace('_', ' ') if roi not in labeled_rois else None
                self.ax.hlines(y_value, 0, x_max, colors=self.roi_colours.get(roi, 'gray'), linestyles=linestyle, linewidth=1.5, alpha=0.8,label=label)
                labeled_rois.add(roi)

            for camera_idx in sorted(set(cameras), key=lambda value: -1 if value is None else value): 
                mask = [i for i, c in enumerate(cameras) if c == camera_idx]
                cam_times = [times[i] for i in mask]
                cam_values = [values[i] for i in mask]
                colour, label = self.CAMERA_COLOURS.get(camera_idx, self.CAMERA_COLOURS[None])
                self.ax.scatter(cam_times, cam_values, color=colour, label=f"{label} classification", s=18, alpha=0.8)

            if show_all or self.history_duration is None or self.history_duration <= 0:
                self.ax.set_xlim(0, x_max + 1)
            else:
                self.ax.set_xlim(max(0, x_max - self.history_duration), x_max + 1)

            self.ax.set_ylim(0.5, len(self.roi_names) + 0.5)
            self.ax.set_yticks([self.roi_to_y[roi] for roi in self.roi_names])
            self.ax.set_yticklabels([roi.replace('_', ' ') for roi in self.roi_names])
            self.ax.set_xlabel("Time (seconds)", fontsize=10)
            self.ax.set_ylabel("ROI", fontsize=10)
            self.ax.legend(loc="upper left", fontsize=8, ncol=2)
            return

        # plot ROI reference lines 
        labeled_rois = set()
        for roi, segments in self.roi_line_segments.items():
            colour = self.roi_colours[roi]
            for start_t, end_t, y_value in segments:
                if end_t is None:
                    end_t = max(times) if times else start_t
                if roi not in labeled_rois:
                    self.ax.hlines(y_value, start_t, end_t, colors=colour, linestyles='--', linewidth=2, label=roi)
                    labeled_rois.add(roi)
                else:
                    self.ax.hlines(y_value, start_t, end_t, colors=colour, linestyles='--', linewidth=2)

        for camera_idx in set(cameras):
            mask = [i for i, c in enumerate(cameras) if c == camera_idx]
            cam_times = [times[i] for i in mask]
            cam_values = [values[i] for i in mask]
            colour, label = self.CAMERA_COLOURS.get(camera_idx, self.CAMERA_COLOURS[None])

            self.ax.plot(cam_times,cam_values,"o",color=colour,label=f"{label} data",markersize=4)

        # Determine x-axis limits
        if show_all or self.history_duration is None or self.history_duration <= 0:
            self.ax.set_xlim(0, max(times) + 1)
        else:
            self.ax.set_xlim(max(0, max(times) - self.history_duration), max(times) + 1)

        self.ax.set_ylim((0, 7) if self.normalised else (-10, 400))
        self.ax.set_xlabel("Time (seconds)", fontsize=10)
        self.ax.set_ylabel("Angle", fontsize=10)
        self.ax.legend(loc="upper left", fontsize=8)
    
    def update_plot(self, frame):
        self._update_plot_with_buffer(frame, self.data_buffer, show_all=False)

    def show_full(self):
        # initialised 
        self.fig, self.ax = plt.subplots(figsize=(12, 4))
        self.fig.suptitle('Driver Gaze Scatter over Time', fontsize=14, fontweight='bold')

        if self.animation:
            self.animation.event_source.stop()
            self.animation = None

        # Use full_history buffer to show all data from time 0
        self._update_plot_with_buffer(None, self.full_history, show_all=True)
        plt.show(block=True)

    def stop(self):
        if self.start_time is not None:
            end_time = time.time() - self.start_time
            for roi in self.roi_colours.keys():
                if self.roi_line_segments[roi] and self.roi_line_segments[roi][-1][1] is None:
                    self.roi_line_segments[roi][-1] = (self.roi_line_segments[roi][-1][0], end_time, self.roi_line_segments[roi][-1][2])
        if self.animation:
            self.animation.event_source.stop()
        if self.fig:
            plt.close(self.fig)
        self.is_running = False