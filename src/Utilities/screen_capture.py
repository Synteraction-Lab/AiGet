import os.path
import platform
import threading
import time
from datetime import datetime

import cv2
import numpy as np
from mss import mss
from screeninfo import get_monitors

from src.Utilities.image_processor import compare_img
from pynput import keyboard

TIME_INTERVAL = 10  # in seconds


def print_monitors():
    for m in get_monitors():
        print(m)


def get_second_monitor_original_pos():
    if len(get_monitors()) == 1:
        selected_monitor_idx = 0
        return 0, 0, get_monitors()[selected_monitor_idx].width, get_monitors()[selected_monitor_idx].height
    else:
        selected_monitor_idx = 1
        if platform.uname().system == "Windows":
            y = get_monitors()[selected_monitor_idx].y
        else:
            y = -get_monitors()[selected_monitor_idx].y + get_monitors()[0].height - get_monitors()[1].height
        return get_monitors()[selected_monitor_idx].x, y, \
            get_monitors()[selected_monitor_idx].width, \
            get_monitors()[selected_monitor_idx].height


class ScreenCapture:
    def __init__(self, pid="p1_1"):
        self.last_time = None
        self.manual_change_slide = False
        self.prev_img = None
        self.thread = None
        self.is_thread_running = False
        try:
            self.pid = pid
            # save to project_root/data/pid
            self.path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", pid)
            if not os.path.exists(self.path):
                os.makedirs(self.path)
            print_monitors()

            # set main monitor
            self.main_monitor_idx = 0

            # set the monitor to capture screen
            if len(get_monitors()) == 1:
                self.selected_monitor_idx = 0
            else:
                self.selected_monitor_idx = 1
            self.SCREEN_SIZE = tuple((get_monitors()[self.selected_monitor_idx].width,
                                      get_monitors()[self.selected_monitor_idx].height))
            self.ORIGINAL_POINT = tuple((get_monitors()[self.selected_monitor_idx].x,
                                         get_monitors()[self.selected_monitor_idx].y
                                         + get_monitors()[self.selected_monitor_idx].height))

            self.is_recording = False
            self.out = None
            # self.listen_keyboard()
        except:
            raise RuntimeError("Please connect to the extended monitor")

    def start_monitoring(self):
        self.is_recording = True
        self.prev_img = None

    def stop_monitoring(self):
        self.is_recording = False
        self.prev_img = None

    # # use pynput to listen to keyboard
    # def listen_keyboard(self):
    #     listener = keyboard.Listener(on_press=self.on_press)
    #     listener.start()

    def on_press(self, key):
        if key == keyboard.Key.enter:
            self.manual_change_slide = True

    def monitor_screen(self, mode="manual"):
        while self.is_recording:
            img = self.get_current_frame()
            if self.prev_img is not None:
                if mode == "manual":
                    # detect if user press enter on keyboard
                    if self.manual_change_slide:
                        yield self.prev_img
                        self.prev_img = img
                        self.manual_change_slide = False
                        print("significant change detected", threading.current_thread().name)
                    else:
                        yield None
                elif mode == "timer":
                    if self.last_time is None:
                        self.last_time = time.time()
                        continue
                    if time.time() - self.last_time > TIME_INTERVAL:
                        self.last_time = time.time()
                        yield img
                        print("10 seconds passed", threading.current_thread().name)
                else:
                    img_diff = compare_img(img, self.prev_img)
                    print(img_diff, threading.current_thread().name)
                    if img_diff > 0.3:
                        yield self.prev_img
                        self.prev_img = img
                        print("significant change detected", threading.current_thread().name)
                    else:
                        yield None
            else:
                self.prev_img = img
            time.sleep(3)

    def get_current_frame(self):
        mon = mss().monitors[self.selected_monitor_idx + 1]

        img = mss().grab(mon)

        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2BGR)
        img = cv2.resize(img, self.SCREEN_SIZE)
        return img

    def get_partial_current_frame(self, x, y, w, h):
        mon = mss().monitors[self.selected_monitor_idx + 1]
        monitor = {
            "top": mon["top"] + y,
            "left": mon["left"] + x,
            "width": w,
            "height": h,
            "mon": self.selected_monitor_idx + 1,
        }
        output = "sct-mon{mon}_{top}x{left}_{width}x{height}.png".format(**monitor)
        img = mss().grab(monitor)
        size = img.size
        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2BGR)
        img = cv2.resize(img, size)
        print(output, size)
        return img

    def take_screenshot(self, time=None):
        img = self.get_current_frame()
        # if time is None:
        #     time = datetime.now().strftime("%H_%M_%S")
        # export_file_path = os.path.join(self.path, "{time}.png".format(time=time))
        # cv2.imwrite(export_file_path, img)
        return img

    def take_partial_screenshot(self, x, y, w, h, time=None):
        img = self.get_partial_current_frame(x, y, w, h)
        if time is None:
            time = datetime.now().strftime("%H_%M_%S")
        cv2.imwrite(os.path.join(self.path, "{time}.png".format(time=time)), img)

    def get_path(self):
        return self.path


if __name__ == '__main__':
    screen_capture = ScreenCapture()
    # Start monitoring the screen
    screen_capture.start_monitoring()

    try:
        for screenshot in screen_capture.monitor_screen():
            if screenshot is not None:
                # Process the screenshot here, e.g., save it or display it
                current_time = datetime.now().strftime("%H_%M_%S")
                save_path = os.path.join(screen_capture.get_path(), f"{current_time}.png")
                cv2.imwrite(save_path, screenshot)
                print(f"Significant screen change detected. Saved at {save_path}")
    except KeyboardInterrupt:
        print("Stopping screen monitoring...")
        screen_capture.stop_monitoring()
