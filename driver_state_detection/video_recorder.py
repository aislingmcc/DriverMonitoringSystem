import os
import cv2

class VideoRecorder:
    def __init__(self, output_path, cam_indices):
        self.path =output_path
        self.cam_idx = list(cam_indices)
        self.writers=[None]*len(self.cam_idx)
        self.initialised =False

    def init(self, frame_sizes, fps):
        # initialise path
        file_paths = []
        if len(self.cam_idx) > 1:
            # treats path as directory
            base_dir = self.path
            os.makedirs(base_dir, exist_ok=True)
            file_paths = [os.path.join(base_dir, f"cam{i}.mp4") for i in range(len(self.cam_idx))]
        else:
            # one camera
            file_paths = [self.path]
        # initialise video
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        for idx,size in enumerate(frame_sizes):
            path=file_paths[idx]
            w,h=size
            self.writers[idx] = cv2.VideoWriter(path, fourcc, fps[idx], (w, h))
        self.initialised=True

    def write(self, cam_idx, frame):
        if self.writers[cam_idx] is None:
            return
        self.writers[cam_idx].write(frame)

    def release(self):
        for writer in self.writers:
            if writer is not None:
                writer.release()
        self.initialised = False

