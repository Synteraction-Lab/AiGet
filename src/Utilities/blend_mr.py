import threading
import time

import numpy as np
from PIL import Image, ImageEnhance
import mss
from flask import Flask, Response
import cv2

from src.Module.Gaze.pupil_video_recorder import PupilCameraVideoRecorder


class VideoFrameProcessor:
    def __init__(self):
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[1]  # Assume capturing the first monitor

    def blend_images(self, background, overlay, opacity=0.5):
        if overlay.mode != 'RGBA':
            overlay = overlay.convert('RGBA')
        overlay_with_opacity = ImageEnhance.Brightness(overlay.split()[-1]).enhance(opacity)
        overlay.putalpha(overlay_with_opacity)
        return Image.alpha_composite(background.convert('RGBA'), overlay)

    def resize_image_keep_ratio_fill_black(self, image, base_width, base_height):
        img_height, img_width = image.shape[:2]
        ratio = min(base_width / img_width, base_height / img_height)
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        new_image = np.zeros((base_height, base_width, 3), dtype=np.uint8)
        top = (base_height - new_height) // 2
        left = (base_width - new_width) // 2
        new_image[top:top + new_height, left:left + new_width] = resized_image
        return new_image

    def generate_video_frame(self, video_frame, ui_opacity=0.5):
        video_frame_pil = Image.fromarray(video_frame)
        sct_img = self.sct.grab(self.monitor)
        ui_image = np.array(sct_img)
        if ui_image.shape[2] > 3:
            ui_image = ui_image[..., :3]
        ui_image_resized = self.resize_image_keep_ratio_fill_black(ui_image, video_frame_pil.width,
                                                                   video_frame_pil.height)
        ui_image_pil = Image.fromarray(ui_image_resized)
        result_frame_pil = self.blend_images(video_frame_pil, ui_image_pil, opacity=ui_opacity)
        result_frame = np.array(result_frame_pil)
        return result_frame


class StreamServer:
    def __init__(self, device="0"):
        self.app = Flask(__name__)
        self.video_processor = VideoFrameProcessor()
        self.device = device
        self.define_routes()

    def generate_frames(self):
        if self.device == "pupil":
            camera = PupilCameraVideoRecorder(frame_format="bgr")
            threading.Thread(target=camera.start, daemon=True).start()
            while True:
                frame = camera.get_frame()
                if frame is not None:
                    # height, width = frame.shape[:2]
                    # frame = cv2.resize(frame, (int(width/2), int(height/2)))
                    result_frame = self.video_processor.generate_video_frame(frame, ui_opacity=0.6)
                    ret, buffer = cv2.imencode('.jpg', result_frame)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            video_capture = cv2.VideoCapture(0)
            while True:
                success, frame = video_capture.read()
                if not success:
                    break
                result_frame = self.video_processor.generate_video_frame(frame, ui_opacity=0.6)
                ret, buffer = cv2.imencode('.jpg', result_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    def define_routes(self):
        @self.app.route('/video')
        def video():
            return Response(self.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def run(self):
        self.app.run(host='0.0.0.0', port=5002)


if __name__ == '__main__':
    stream_server = StreamServer(device="pupil")
    stream_server.run()

