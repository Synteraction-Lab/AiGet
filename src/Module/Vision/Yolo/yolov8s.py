import json
import multiprocessing
import ssl
import time
import urllib
from collections import deque
from multiprocessing.managers import BaseManager

import cv2
import numpy as np
from ultralytics import YOLO

from src.Module.Gaze.frame_stream import PupilCamera

from scipy.spatial import distance

from src.Module.Gaze.gaze_data import GazeData

ssl._create_default_https_context = ssl._create_unverified_context


class ObjectDetector:
    def __init__(self, simulate=False, debug_info=False, cv_imshow=True, record=False, video_path=None):
        self.recording = record
        self.last_time = None
        self.potential_interested_object = None
        self.original_frame = None
        self.norm_gaze_position = None
        self.zoom_in = False
        self.closest_object = None
        self.simulate = simulate
        self.debug_info = debug_info
        self.cv_imshow = cv_imshow
        self.person_count = 0
        self.frame_height = None
        self.frame_width = None
        self.GAZE_SLIDE_WINDOW_SIZE = 10  # Adjust the size according to your frame rate (on my macOS, 10 is ~2 seconds)
        self.gaze_positions_window = deque(maxlen=self.GAZE_SLIDE_WINDOW_SIZE)
        self.fixation_detected = False
        self.video_path = video_path

        self.prev_object = None
        self.gaze_data = None

        # load model
        self.model = YOLO('yolov8n.pt')

        # set model parameters
        self.model.overrides['conf'] = 0.7  # NMS confidence threshold
        self.model.overrides['iou'] = 0.45  # NMS IoU threshold
        self.model.overrides['agnostic_nms'] = False  # NMS class-agnostic
        self.model.overrides['max_det'] = 100  # maximum number of detections per image
        self.model.overrides['verbose'] = False  # print all detections

        self.distance_threshold = 100  # in pixel of the image, adjust as needed

        # load the mapping file
        self.class_idx = None
        # self.map_imagenet_id()

        # open webcam
        self.cap = None

        # Define a threshold for zoom detection
        self.zoom_threshold = 0.1  # adjust as needed

        # Initialize gaze position
        self.gaze_position = (0, 0)  # replace with actual gaze tracking data
        self.fixation_position = (0, 0)
        self.prev_size = 0
        self.gaze_x_list = []
        self.gaze_y_list = []

        if simulate:
            cv2.namedWindow('YOLO Object Detection')
            cv2.setMouseCallback('YOLO Object Detection', self.mouse_callback)
            self.total_frames = 0
            self.current_frame = 0

        self.interested_categories = ['bird', 'cat', 'dog', 'cow', 'elephant', 'bear']


    def mouse_callback(self, event, x, y, flags, param):
        # Update gaze_position with cursor position
        if event == cv2.EVENT_MOUSEMOVE:
            self.gaze_position = (x, y)

    def process_frame(self, frame):
        if not frame.flags.writeable:
            frame = frame.copy()  # Make a writable copy
        cv2.circle(frame, self.gaze_position, 20, (0, 0, 255), 4)
        # frame = self.find_gaze_object(frame)
        self.gaze_data.put_original_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if self.cv_imshow:
            # draw gaze position

            # draw yellow fixation position
            cv2.circle(frame, self.fixation_position, 10, (0, 255, 255), 3)

            cv2.imshow('YOLO Object Detection', frame)

    def find_gaze_object(self, frame):
        results = self.model.track(frame, persist=True)
        boxes = results[0].boxes
        closest_object = None
        closest_distance = float('inf')
        closest_size = 0
        person_count = 0
        potential_interested_object = None

        # Sort the boxes by whether they contain gaze_position and their area
        sorted_boxes = sorted(zip(boxes.xyxy, boxes.conf, boxes.cls), key=lambda box: (
            not (box[0][0] <= self.gaze_position[0] <= box[0][2] and box[0][1] <= self.gaze_position[1] <= box[0][3]),
            (box[0][2] - box[0][0]) * (box[0][3] - box[0][1])
        ))

        for xyxy, conf, cls in sorted_boxes:
            if cls == 0:
                person_count += 1

            object_label = self.model.model.names[int(cls)]
            dx = max(xyxy[0] - self.gaze_position[0], 0, self.gaze_position[0] - xyxy[2])
            dy = max(xyxy[1] - self.gaze_position[1], 0, self.gaze_position[1] - xyxy[3])
            distance = np.sqrt(dx ** 2 + dy ** 2)

            if distance < closest_distance:
                closest_object = object_label
                closest_distance = distance
                closest_size = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])

            # Check if the current object belongs to any interested category
            if object_label in self.interested_categories:
                potential_interested_object = object_label

        if closest_distance <= self.distance_threshold and closest_object is not None:
            self.closest_object = closest_object
            self.gaze_data.put_closest_object(closest_object)

            # Compare sizes only if the current and previous objects are the same
            # if self.prev_object == closest_object and closest_size > self.prev_size * (1 + self.zoom_threshold):
            #     self.zoom_in = True
            #     self.gaze_data.put_zoom_in(True)
            # else:
            #     self.zoom_in = False
            #     self.gaze_data.put_zoom_in(False)

            # Update the previous object and size
            self.prev_object = closest_object
            self.prev_size = closest_size
        else:
            self.closest_object = None
            self.gaze_data.put_closest_object(None)
            # self.zoom_in = False
            self.gaze_data.put_zoom_in(False)

        # self.detect_fixation(frame)

        if potential_interested_object != closest_object:
            self.potential_interested_object = potential_interested_object
            self.gaze_data.put_potential_interested_object(potential_interested_object)

        # render = render_result(model=self.model, image=frame, result=results[0])
        # frame = np.array(render.convert('RGB'))
        frame = results[0].plot(labels=False)


        if self.cv_imshow:
            if self.closest_object is not None:
                cv2.putText(frame, f"User is looking at a {closest_object}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2)
                # if self.zoom_in:
                #     cv2.putText(frame, f"User zoomed in / moved closer to the {closest_object}", (10, 90),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
            else:
                cv2.putText(frame, "User is not looking at any object", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 0, 255), 2)

            if self.fixation_detected:
                cv2.putText(frame, "Fixation Detected", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            self.person_count = person_count
            self.gaze_data.put_person_count(person_count)
            cv2.putText(frame, f"Person Count: {self.person_count}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 0), 2)

            cv2.putText(frame, f"gaze pos:{self.gaze_position}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        return frame

    def detect_zoom_in_with_pupil_core(self):
        camera = PupilCamera(frame_format="bgr", recording=self.recording)
        try:
            while True:
                recent_world = None
                gaze_x, gaze_y = None, None
                # gaze_x_list = []
                # gaze_y_list = []
                fixation_x, fixation_y = None, None
                fixation_conf = 0
                try:
                    while camera.has_new_data_available():
                        topic, msg = camera.recv_from_sub()

                        if topic == "fixations":
                            if msg["confidence"] > fixation_conf:
                                fixation_conf = msg["confidence"]
                                fixation_x, fixation_y = msg['norm_pos']
                        elif topic == "gaze.2d.0.":
                            # if msg['confidence'] > 0.1:
                            gaze_x, gaze_y = msg['norm_pos']
                            self.gaze_x_list.append(gaze_x)
                            self.gaze_y_list.append(gaze_y)
                        elif topic == "frame.world":
                            recent_world = np.frombuffer(
                                msg["__raw_data__"][0], dtype=np.uint8
                            ).reshape(msg["height"], msg["width"], 3)
                except Exception as e:
                    print(e)
                    continue

                if recent_world is not None:
                    frame = recent_world
                    frame_height, frame_width = frame.shape[:2]
                    self.original_frame = frame
                    # self.gaze_data.put_original_frame(frame)
                    self.frame_height = frame_height
                    self.frame_width = frame_width
                    if fixation_x is not None and fixation_y is not None:
                        fixation_y = 1 - fixation_y
                        self.fixation_position = (int(fixation_x * frame_width), int(fixation_y * frame_height))
                        self.fixation_detected = True
                        self.gaze_data.put_fixation_detected((True, time.time()))
                    else:
                        self.fixation_detected = False
                        self.gaze_data.put_fixation_detected((False, None))
                    if self.gaze_x_list and self.gaze_y_list:
                        gaze_x = float(np.array(self.gaze_x_list).mean())
                        gaze_y = 1 - float(np.array(self.gaze_y_list).mean())
                        self.gaze_x_list = []
                        self.gaze_y_list = []
                        # calculate time diff between current and previous gaze
                        # if self.last_time is not None:
                        #     print("time diff:", time.time() - self.last_time)
                        self.last_time = time.time()
                        self.norm_gaze_position = (gaze_x, gaze_y)
                        self.gaze_data.put_norm_gaze_position(self.norm_gaze_position)
                        self.gaze_position = (int(gaze_x * frame_width), int(gaze_y * frame_height))

                    self.process_frame(frame)
                    if self.cv_imshow:
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyAllWindows()

    def detect_zoom_in(self):
        self.cap = cv2.VideoCapture(0)
        while True:
            ret, frame = self.cap.read()

            if ret is None or frame is None:
                continue

            self.frame_height, self.frame_width = frame.shape[:2]
            self.original_frame = frame
            self.gaze_data.put_original_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.process_frame(frame)

            # press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def on_trackbar_change(self, value):
        self.current_frame = value
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)

    def detect_zoom_in_from_video(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)

        cv2.namedWindow('YOLO Object Detection')
        cv2.createTrackbar('Progress', 'YOLO Object Detection', 0, self.total_frames - 1, self.on_trackbar_change)

        while True:
            start_time = time.time()
            ret, frame = self.cap.read()

            if not ret:
                break

            self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            # cv2.setTrackbarPos('Progress', 'YOLO Object Detection', self.current_frame)

            # self.frame_height, self.frame_width = frame.shape[:2]
            # self.original_frame = frame
            self.gaze_data.put_original_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


            cv2.imshow('YOLO Object Detection', frame)

            spend_time = time.time() - start_time
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            #
            key = cv2.waitKey(max(0, int((1000 / fps-spend_time)/2))) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                cv2.waitKey(-1)  # Wait until any key is pressed

        self.cap.release()
        cv2.destroyAllWindows()
        print("\nVideo processing completed.")


    def run(self, gaze_data=None):
        # open webcam
        self.gaze_data = gaze_data
        if self.simulate:
            if self.video_path is not None:
                self.detect_zoom_in_from_video(self.video_path)
            self.detect_zoom_in()
        else:
            self.detect_zoom_in_with_pupil_core()


if __name__ == '__main__':
    # Set simulate to False if you use Pupil Core. Set to True to use mouse cursor.
    BaseManager.register('GazeData', GazeData)
    manager = BaseManager()
    manager.start()
    gaze_data = manager.GazeData()
    object_detector = ObjectDetector(simulate=True, cv_imshow=True,)
    thread_vision = multiprocessing.Process(target=object_detector.run, args=(gaze_data,))
    thread_vision.start()
    while True:
        norm_pose = gaze_data.get_norm_gaze_position()
