import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

class ROILiveGraph:
    CAMERA_COLOURS = {
        0: ("#f92a9f", "Camera 0"),
        1: ("#0effab", "Camera 1"),
        None: ("#b0b0b06e", "Unknown"),
    }
    
    def __init__(self, history_duration,
                 classification_types= None,
                 calibrated_rois= None):
        self.history_duration = history_duration
        self.calibrated_rois = calibrated_rois
        
        if classification_types is None:
            classification_types = ["angle", "centroid"]
        self.classification_types = [m.lower() for m in classification_types]
        
        self.data_buffers = {}
        size = history_duration * 15
        for metric in self.classification_types: 
            self.data_buffers[metric] = {
                'times': deque(maxlen=size),  
                'values': deque(maxlen=size),
                'cameras': deque(maxlen=size)     
            }
        
        # plot setup
        self.fig = None
        self.axes = None
        self.lines = {}
        self.start_time = None
        self.animation = None
        self.is_running = False
    
    def add_data_point(self,current_time,angle_score = None,centroid_score = None, camera_idx= None):
        if self.start_time is None:
            self.start_time = current_time
        
        relative_time = current_time - self.start_time
        
        # Add data points for each requested metric
        if 'angle' in self.classification_types and angle_score is not None:
            self.data_buffers['angle']['times'].append(relative_time)
            self.data_buffers['angle']['values'].append(angle_score)
            self.data_buffers['angle']['cameras'].append(camera_idx)
        
        if 'centroid' in self.classification_types and centroid_score is not None:
            self.data_buffers['centroid']['times'].append(relative_time)
            self.data_buffers['centroid']['values'].append(centroid_score)
            self.data_buffers['centroid']['cameras'].append(camera_idx)

    def _get_roi_marker(self, classification_type):
        if self.calibrated_rois is None:
            return {}
        
        roi_names = ["left_mirror", "right_mirror", "road", "rearmirror", "left_window", "right_window"]
        calibration_representative = {}
        for roi_name in roi_names:
            roi_data = self.calibrated_rois[roi_name]
            
            if classification_type == 'angle':
                calibration_representative[roi_name] = roi_data.get('angle', 0.0)*.7+(roi_data.get('left_mag', 0.0)+roi_data.get('right_mag', 0.0))*.3*360/60
                
            elif classification_type == 'centroid':
                # find a better way to represent this 
                cx = roi_data.get('centroid_x', 0)
                cy = roi_data.get('centroid_y', 0)
                calibration_representative[roi_name] = np.sqrt(cx**2 + cy**2)
                
        return calibration_representative

    def start(self):        
        num_metrics = len(self.classification_types)
        self.fig, self.axes = plt.subplots(num_metrics, 1, figsize=(12, 2*num_metrics))
        
        # Handle single metric case
        if num_metrics == 1:
            self.axes = [self.axes]
        
        self.fig.suptitle('Driver Gaze Live Graph', fontsize=14, fontweight='bold')
        
        # Setup subplots for each classifier
        for idx, metric in enumerate(self.classification_types):
            ax = self.axes[idx]
            
            # initialise plot
            line, = ax.plot([], [], 'o-', label='Metric unit', markersize=4)
            self.lines[metric] = {'line': line, 'ax': ax}
        
        self.is_running = True
        self.animation = FuncAnimation(self.fig,self._update_plot, interval=100,blit=False ,repeat=True, cache_frame_data=False)
        plt.tight_layout()
        plt.show(block=False)
    
    def _update_plot(self, frame):
        for classification_type in self.classification_types:
            if classification_type not in self.data_buffers:
                continue
            
            buffer = self.data_buffers[classification_type]
            times = list(buffer['times'])
            values = list(buffer['values'])
            cameras = list(buffer['cameras'])
            
            if not times:
                continue
            
            ax = self.lines[classification_type]['ax']
            ax.clear()
            
            roi_references = self._get_roi_marker(classification_type)
            roi_colours = ['pink', 'lime', 'magenta', 'purple', 'cyan', 'blue']
            
            # plot reference lines for calibration
            for i, (roi_name, ref_value) in enumerate(roi_references.items()):
                colour = roi_colours[i % len(roi_colours)]
                ax.axhline(y=ref_value, color=colour, linestyle='--', linewidth=2, label=f'{roi_name}')
            
            # to plot specific point depending on camera priority
            for camera_idx in set(cameras):
                mask = [i for i, c in enumerate(cameras) if c == camera_idx]
                cam_times = [times[i] for i in mask]
                cam_values = [values[i] for i in mask]
                
                colour, label = self.CAMERA_COLOURS.get(camera_idx, self.CAMERA_COLOURS[None])
                ax.plot(cam_times, cam_values, 'o-', color=colour, label=label,markersize=4, linewidth=2)

            # set axes limits and adjustments for x axis
            ax.set_xlim(max(0, max(times) - self.history_duration), max(times)+1)
            if classification_type == 'angle':
                ax.set_ylim(0, 400)
            elif classification_type == 'centroid':
                ax.set_ylim(0, 1500) 
            
            ax.set_xlabel('Time (seconds)', fontsize=10)
            ax.set_ylabel(f'{classification_type} unit', fontsize=10)
            ax.set_title(f'{classification_type.upper()}', fontsize=11, fontweight='bold')
            ax.legend(loc='upper left', fontsize=8)
    
    def stop(self):
        if self.animation:
            self.animation.event_source.stop()
        if self.fig:
            plt.close(self.fig)
        self.is_running = False