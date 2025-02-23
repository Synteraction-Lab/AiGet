import datetime
import threading
from collections import deque
import cv2
import time
from PIL import Image

from src.Module.Gaze.pupil_video_recorder import PupilCameraVideoRecorder


class VideoRecorder:
    def __init__(self, display_width=320, display_height=240):
        self.frames_buffer = deque(maxlen=5 * 5)  # Example buffer setup for 5 seconds of footage at 5 FPS
        self.frame_rate = 5  # Target frame rate for recording
        self.display_width = display_width
        self.display_height = display_height
        self.frame_id = 0

    def run(self):
        camera = PupilCameraVideoRecorder(frame_format="bgr")
        threading.Thread(target=camera.start, daemon=True).start()
        while True:
            frame = camera.get_frame()
            if frame is not None:
                frame = self.preprocess_frame(frame)
                self.frames_buffer.append(frame)
                time.sleep(1 / self.frame_rate)

        # cap = cv2.VideoCapture(0)
        # while cap.isOpened():
        #     ret, frame = cap.read()
        #     if ret:
        #         # Resize and color conversion
        #         frame = self.preprocess_frame(frame)
        #         self.frames_buffer.append(frame)
        #         time.sleep(1 / self.frame_rate)
        #     else:
        #         break
        # cap.release()

    def preprocess_frame(self, frame):
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize the frame to fit the display area
        h, w, _ = frame.shape
        scale = min(self.display_width / w, self.display_height / h)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        # add frame id to the frame
        self.frame_id += 1
        time = datetime.datetime.now().strftime("%H:%M:%S:%f")
        cv2.putText(frame, f"Time: {time}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame

    def get_frames(self):
        return list(self.frames_buffer)  # Return a copy of the buffer
