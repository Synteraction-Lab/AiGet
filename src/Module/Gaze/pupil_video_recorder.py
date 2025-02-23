import threading
import time
from datetime import datetime

import cv2
import numpy as np
import zmq
from msgpack import unpackb, packb
from pyplr.pupil import PupilCore


class PupilCameraVideoRecorder:
    def __init__(self, frame_format="bgr", recording=False):
        self.gaze_y_list = []
        self.gaze_x_list = []
        self.gaze_position = None
        self.context = zmq.Context()
        self.addr = "127.0.0.1"  # remote ip or localhost
        self.req_port = "50020"  # same as in the pupil remote gui
        self.req = self.context.socket(zmq.REQ)
        self.req.connect("tcp://{}:{}".format(self.addr, self.req_port))
        self.pupil_core = PupilCore()
        self.pupil_core.annotation_capture_plugin(should='start')

        # ask for the sub port
        self.req.send_string("SUB_PORT")
        self.sub_port = self.req.recv_string()

        # open a sub port to listen to pupil
        self.sub = self.context.socket(zmq.SUB)
        self.sub.connect("tcp://{}:{}".format(self.addr, self.sub_port))

        # set subscriptions to topics
        # recv just pupil/gaze/notifications
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "frame.world")
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "gaze.2d.0.")

        self.recent_world = None

        self.FRAME_FORMAT = frame_format

        # get current time in format: MM-DD-HH-MM-SS
        self.current_time = datetime.now().strftime("%m-%d-%H-%M-%S")
        if recording:
            print("Start recording")
            self.pupil_core.command("R {}".format(self.current_time))

        # Set the frame format via the Network API plugin
        self.notify({"subject": "frame_publishing.set_format", "format": self.FRAME_FORMAT})

    def recv_from_sub(self):
        """Recv a message with topic, payload.
        Topic is a utf-8 encoded string. Returned as unicode object.
        Payload is a msgpack serialized dict. Returned as a python dict.
        Any addional message frames will be added as a list
        in the payload dict with key: '__raw_data__' .
        """
        topic = self.sub.recv_string()
        payload = unpackb(self.sub.recv(), raw=False)
        extra_frames = []
        while self.sub.get(zmq.RCVMORE):
            extra_frames.append(self.sub.recv())
        if extra_frames:
            payload["__raw_data__"] = extra_frames
        return topic, payload

    def has_new_data_available(self):
        # Returns True as long subscription socket has received data queued for processing
        return self.sub.get(zmq.EVENTS) & zmq.POLLIN

    def notify(self, notification):
        """Sends ``notification`` to Pupil Remote"""
        topic = "notify." + notification["subject"]
        payload = packb(notification, use_bin_type=True)
        self.req.send_string(topic, flags=zmq.SNDMORE)
        self.req.send(payload)
        return self.req.recv_string()

    def start(self):
        try:
            while True:
                gaze_x_list = gaze_y_list = []
                recent_world = None
                while self.has_new_data_available():
                    topic, msg = self.recv_from_sub()
                    if topic == "frame.world":
                        recent_world = np.frombuffer(
                            msg["__raw_data__"][0], dtype=np.uint8
                        ).reshape(msg["height"], msg["width"], 3)
                    elif topic == "gaze.2d.0.":
                        # if msg['confidence'] > 0.1:
                        gaze_x, gaze_y = msg['norm_pos']
                        self.gaze_x_list.append(gaze_x)
                        self.gaze_y_list.append(gaze_y)
                if recent_world is not None:
                    frame = recent_world
                    frame_height, frame_width = frame.shape[:2]
                    if self.gaze_x_list and self.gaze_y_list:
                        gaze_x = float(np.array(self.gaze_x_list).mean())
                        gaze_y = 1 - float(np.array(self.gaze_y_list).mean())
                        self.gaze_x_list = self.gaze_y_list = []
                        self.gaze_position = (int(gaze_x * frame_width), int(gaze_y * frame_height))
                    if self.gaze_position is not None:
                        cv2.circle(frame, self.gaze_position, 20, (0, 0, 255), 4)
                    self.recent_world = frame


        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyAllWindows()

    def get_frame(self):
        return self.recent_world


if __name__ == '__main__':
    camera = PupilCameraVideoRecorder(frame_format="bgr")
    threading.Thread(target=camera.start, daemon=True).start()
    while True:
        frame = camera.get_frame()
        if frame is not None:
            cv2.imshow("world", frame)
            cv2.waitKey(1)

