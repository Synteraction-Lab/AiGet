import concurrent
import datetime
import json
import multiprocessing
import os
import threading
import time
from collections import deque
from multiprocessing.managers import BaseManager

import customtkinter
import customtkinter as ctk
import tkinter as tk
from tkinter import scrolledtext, filedialog

import pandas
from PIL import Image, ImageTk
from customtkinter import CTkTextbox

from src.Backend import Backend
from src.Data.SystemData import SystemData
from src.Data.UserData import UserData
from src.Module.Audio.live_transcriber import LiveTranscriber
from src.Module.Audio.music import MusicRecognition
from src.Module.Audio.text_to_speech import play_text_to_speech_audio
from src.Module.Gaze.gaze_data import GazeData
# from src.Module.Gaze.gaze_recording import EyeTracker
from src.Module.LLM.LLMPipeline import get_llm_response
from src.Module.Vision.Yolo.yolov8s import ObjectDetector
from src.Storage.reader import load_task_description
from src.UI.UI_config import MAIN_GREEN_COLOR
from src.UI.device_panel import DevicePanel
from src.UI.live_comments import LiveCommentsApp
from src.Utilities.constant import config_path
from src.Utilities.location import get_current_location
from src.Utilities.process_json import detect_json
from pynput.keyboard import Key, Listener as KeyboardListener

from src.Utilities.screen_capture import ScreenCapture

FPV_SIMILARITY_THRESHOLD = 0.6

MINIMUM_GPT_REQUEST_INTERVAL = 12

IMAGE_WINDOW_WIDTH = 391
IMAGE_WINDOW_HEIGHT = 220

SIMULATED_LOCATION = "NUS UTown, Singapore"


def draw_bounding_boxes(canvas, img_width, img_height, bounding_boxes):
    for box in bounding_boxes:
        relx1, rely1, relx2, rely2 = box
        x1 = int(relx1 * img_width)
        y1 = int(rely1 * img_height)
        x2 = int(relx2 * img_width)
        y2 = int(rely2 * img_height)
        canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2)


# Function to load AI suggestions from a JSON file
def load_ai_suggestions(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        ai_suggestions = json.load(json_file)
    return ai_suggestions


class PANDAExpFrontend(ctk.CTk):
    def __init__(self, simulate=False, mode="Desktop", language="en", video_path=None, auto=True,
                 knowledge_generation_model="gemini-pro-001"):
        super().__init__()
        self.audio_process = None
        DevicePanel(self, parent_object_save_command=self.update_config)
        self.last_gpt_request_time = None
        self.heatmap_image_path = None
        self.last_ai_response = None
        self.simulate = simulate
        self.response_language = language
        self.video_path = video_path
        self.auto = auto
        self.knowledge_generation_model = knowledge_generation_model

        # self.folder_path = os.path.join(project_root, "data", pid)
        # create the folder if it does not exist
        self.system_data = SystemData()
        self.user_data = UserData()
        self.backend = Backend(self.system_data, self.user_data)
        self.last_analyzed_frame_embedding = None

        self.title("PANDAExp")
        self.geometry("1500x900")

        self.instruction_prompt_display = scrolledtext.ScrolledText(self, wrap=tk.WORD, height=5)
        # self.instruction_prompt_display.pack(pady=10, fill='both')

        # Image selected canvas (placeholder for now)
        self.image_canvas = tk.Canvas(self, width=1000, height=600, bg="grey")
        self.image_canvas.pack(pady=5)

        # create another canvas at the right side of the image_canvas to display the image in the queue
        self.send_image_preview = tk.Canvas(self, width=120, height=600, bg="grey")
        self.send_image_preview.pack(pady=5, side='right')

        # Response box in horizontal layout
        self.response_frame = ctk.CTkFrame(self)
        self.response_frame.pack(pady=5, fill='x')
        self.response_text = scrolledtext.ScrolledText(self.response_frame, wrap=tk.WORD, height=18, width=100)
        self.response_text.pack(side='left', padx=2, fill='both', expand=True)

        # User input
        self.entry_frame = ctk.CTkFrame(self)
        self.entry_frame.pack(pady=5, fill='x')
        self.user_input = ctk.CTkEntry(self.entry_frame, placeholder_text="Enter your comments or questions here")
        self.user_input.pack(pady=5, fill='x')
        self.mode = mode
        self.allow_audio = True
        self.system_on = True

        if mode == "Desktop":
            # creat a top level window
            self.notification_window = ctk.CTkToplevel(self, fg_color="#1D4C22")
            self.notification_window.wm_attributes("-topmost", True)
            self.notification_window.title("Suggestion")
            self.notification_window.attributes("-alpha", 0.8)
        else:
            self.black_window = ctk.CTkToplevel(self, fg_color="black")
            # make the window full screen
            self.black_window.attributes("-fullscreen", True)
            self.black_window.focus_set()
            self.notification_window = customtkinter.CTkFrame(self.black_window, height=200, width=1220,
                                                              fg_color='black', bg_color='black')
            self.notification_window.pack(side=tk.LEFT, anchor=tk.N, padx=10, pady=10)
            self.notification_window.pack_forget()
            self.image_window = customtkinter.CTkFrame(self.black_window, height=IMAGE_WINDOW_HEIGHT,
                                                       width=IMAGE_WINDOW_WIDTH,
                                                       fg_color='black', bg_color='black')
            self.image_window.pack(side=tk.RIGHT, anchor=tk.N, padx=10, pady=10)
            self.image_window.pack_forget()
            self.image_window_canvas = customtkinter.CTkCanvas(self.image_window, width=IMAGE_WINDOW_WIDTH,
                                                               height=IMAGE_WINDOW_HEIGHT,
                                                               bg="black")
            self.image_window_canvas.pack(pady=5, fill='both', expand=True)

            # put the toplevel window on the middle right of the screen
            screen_width = self.notification_window.winfo_screenwidth()
            screen_height = self.notification_window.winfo_screenheight()
            window_width = 500
            window_height = 230
            # put notification window on the middle top of the screen
            x, y = screen_width / 2 - window_width / 2, 0
            # self.notification_window.geometry(f"{window_width}x{window_height}+{int(x)}+{int(y)}")
            self.danmaku_list = []

        self.notification_scroll_text = CTkTextbox(self.notification_window, height=180, width=450,
                                                   text_color=MAIN_GREEN_COLOR, font=('Robot Bold', 25),
                                                   # bg_color='systemTransparent',
                                                   wrap="word", padx=5,
                                                   border_color="#42AF74", border_width=3)
        self.notification_scroll_text.pack(pady=3, fill='both', expand=True)
        if mode == "Desktop":
            self.record_button = ctk.CTkButton(self.notification_window, text="Record", command=self.record_voice,
                                               fg_color=MAIN_GREEN_COLOR, width=15)
            self.record_button.pack(pady=2, side='left')

            # Submit button
            self.send_button = ctk.CTkButton(self.notification_window, text="Send",
                                             command=lambda init="user": self.send_data(init),
                                             fg_color=MAIN_GREEN_COLOR, width=10)
            self.send_button.pack(pady=2, side='right')
        else:
            self.record_button = ctk.CTkButton(self.black_window, text="Record", command=self.record_voice,
                                               border_color=MAIN_GREEN_COLOR, border_width=2, text_color= MAIN_GREEN_COLOR,
                                               fg_color="black", width=15, height=15, font=('Robot Bold', 20))
            self.record_button.pack(padx=10, side='right')

            self.mute_button = ctk.CTkButton(self.black_window, text="Mute", command=self.switch_audio,
                                             border_color=MAIN_GREEN_COLOR, border_width=2, text_color= MAIN_GREEN_COLOR,
                                             fg_color="black", width=15, height=15, font=('Robot Bold', 20))
            self.mute_button.pack(padx=10, side='left')

            # Submit button
            self.send_button = ctk.CTkButton(self.black_window, text="Send",
                                             command=lambda init="user": self.send_data(init),
                                             fg_color=MAIN_GREEN_COLOR, width=10)
            # self.send_button.pack(pady=2, side='right')

        self.current_feedback_mode = "Live Comments"  # "Live Comments" or "Single Comment"
        self.start_mouse_key_listener()

        self.music_recognition = None

        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.AI_suggestion_history = []

    def update_config(self):
        if not os.path.isfile(config_path):
            pid_num = os.path.join("p1", "01")
            audio_device_idx = 0
            # task_name = "daily_activity_learning"
            gpt_response_style = "Live Comments"
        else:
            try:
                df = pandas.read_csv(config_path)
                pid_num = df[df['item'] == 'pid']['details'].item()
                audio_device_idx = df[df['item'] == 'audio_device']['details'].item()
                gpt_response_style = df[df['item'] == 'gpt_response_style']['details'].item()
                # task_name = df[df['item'] == 'task']['details'].item()
            except Exception as e:
                print("Config file has an error!", e)
                pid_num = os.path.join("p1", "01")
                audio_device_idx = 0
                gpt_response_style = "Live Comments"
                # task_name = "daily_activity_learning"

        # Set up path
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        folder_path = os.path.join(project_root, os.path.join("data", "recordings"), pid_num)
        self.folder_path = folder_path
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
        self.gpt_response_style = gpt_response_style

        self.audio_device_idx = audio_device_idx

        # instruct_prompt = load_task_description(task_name)
        # self.conversation_history = [{"role": "system",
        #                               "content": [{"type": "text",
        #                                            "text": instruct_prompt}]}]
        # self.instruction_prompt_display.insert(tk.INSERT, instruct_prompt)

        ai_history_path = os.path.join(self.folder_path, 'ai_suggestions.json')
        if os.path.exists(ai_history_path):
            self.AI_suggestion_history = load_ai_suggestions(ai_history_path)
            print("AI suggestions loaded: ", self.AI_suggestion_history)

        if self.mode != "Desktop":
                self.danmaku = LiveCommentsApp(self.black_window)
        self.run()

    def start_mouse_key_listener(self):
        self.keyboard_listener = KeyboardListener(
            on_press=self.on_press)
        # self.mouse_listener.start()
        time.sleep(0.1)
        self.keyboard_listener.start()

    def on_press(self, key):
        if key == Key.f9:
            self.record_voice()
        elif key == Key.f8:
            self.switch_notification_window_visibility()
        elif key == Key.f6:
            # self.switch_default_response_style()
            self.switch_system_on_off()
        elif key == Key.f7:
            self.switch_audio()

    def switch_system_on_off(self):
        self.switch_notification_window_visibility()
        if self.system_on:
            self.system_on = False
            self.auto = False
            # unpack buttons
            self.record_button.pack_forget()
            self.mute_button.pack_forget()
            print("System off")
            self.log("System off")
        else:
            self.system_on = True
            self.auto = True
            # pack buttons
            self.record_button.pack(padx=10, side='right')
            self.mute_button.pack(padx=10, side='left')
            # release idle tasks

            print("System on")
            self.log("System on")
        self.update_idletasks()

    def switch_audio(self):
        if self.allow_audio:
            self.allow_audio = False
            self.mute_button.configure(text="Unmute")
            if self.system_data.is_playing_audio:
                self.audio_process.terminate()
                self.system_data.is_playing_audio = False
        else:
            self.allow_audio = True
            self.mute_button.configure(text="Mute")

    def switch_notification_window_visibility(self):
        if self.current_feedback_mode == "Live Comments":
            if self.danmaku.winfo_ismapped():
                self.danmaku.pack_forget()
                self.image_window.pack_forget()
                if self.audio_process is not None:
                    self.audio_process.terminate()
                    self.system_data.is_playing_audio = False
                self.log("Hide Live Comments")
            else:
                self.danmaku.pack()
                self.image_window.pack(side=tk.RIGHT, anchor=tk.N, padx=10, pady=10)
        else:
            if self.mode == "Desktop":
                if self.notification_window.winfo_ismapped():
                    self.notification_window.withdraw()
                else:
                    self.notification_window.deiconify()
            else:
                if self.notification_window.winfo_ismapped():
                    self.notification_window.pack_forget()
                    self.image_window.pack_forget()
                    if self.audio_process is not None:
                        self.audio_process.terminate()
                        self.system_data.is_playing_audio = False
                    self.log("Hide Single Comment")
                else:
                    self.notification_window.pack(side=tk.LEFT, anchor=tk.N, padx=10, pady=10)
                    self.image_window.pack(side=tk.RIGHT, anchor=tk.N, padx=10, pady=10)

    def record_voice(self):
        if not self.is_recording:
            self.record_button.configure(text="Stop Recording")
            # if self.system_data.is_playing_audio:
            #     self.audio_process.terminate()
            #     self.system_data.is_playing_audio = False
            self.black_window.update_idletasks()
            self.is_recording = True
            self.transcriber.start()
        else:
            voice_text = self.transcriber.stop()
            print(f"Voice text: {voice_text}")
            # Clear the user input box
            self.user_input.delete(0, tk.END)
            self.user_input.insert(tk.INSERT, voice_text)
            self.is_recording = False
            self.record_button.configure(text="Processing")
            self.send_data(initiation="Single Comment")

    def send_data(self, initiation="timer"):
        self.send_button.configure(state='disabled')  # Disable the button while processing
        self.send_button.configure(text="Processing...")

        self.last_gpt_request_time = time.time()
        user_input = self.user_input.get()
        self.user_input.delete(0, tk.END)
        image = self.system_data.video_sequence_queue.copy()
        self.system_data.last_sent_video_sequence_queue = image.copy()

        threading.Thread(target=self.put_all_images_in_queue_in_canvas, args=(image.copy(),), daemon=True).start()

        # Run send_data_thread in a separate thread to prevent UI blocking
        threading.Thread(target=self.send_data_thread, args=(image, user_input, initiation), daemon=True).start()

    def put_all_images_in_queue_in_canvas(self, images: deque):
        if len(images) == 0:
            return

        self.send_image_preview.delete("all")
        max_width, max_height = 120, 60
        y_position = 0  # Initialize vertical position

        def process_image(image_data, position):
            image = Image.fromarray(image_data)
            width, height = image.width, image.height
            scale_factor = max(width / max_width, height / max_height)
            new_width = int(width / scale_factor)
            new_height = int(height / scale_factor)

            resized_image = image.resize((new_width, new_height), Image.LANCZOS)
            heatmap_image = ImageTk.PhotoImage(resized_image)

            # Use a thread-safe way to update the canvas
            self.send_image_preview.create_image(0, position, anchor='nw', image=heatmap_image)

            return heatmap_image

        image_list = list(images)[:6]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for i, image_data in enumerate(image_list):
                futures.append(executor.submit(process_image, image_data, y_position))
                y_position += int(
                    image_data.shape[0] / max(image_data.shape[0] / max_height, 1))  # Increment the y position

            # Keep a reference to the images to prevent garbage collection
            if not hasattr(self, '_image_refs'):
                self._image_refs = []
            self._image_refs.extend(f.result() for f in concurrent.futures.as_completed(futures))

    def send_data_thread(self, image, user_input, initiation="timer"):
        try:
            if self.simulate:
                location = SIMULATED_LOCATION
            else:
                location = get_current_location()
            send_message = {"Request Mode": initiation, "user comments": user_input, "location": location,
                            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            if self.music_recognition is not None:
                if self.music_recognition.current_music:
                    send_message["ambient audio"] = self.music_recognition.current_music

            if initiation == "fixation" or initiation == "timer":
                send_message["Response Style"] = self.gpt_response_style
            else:
                send_message["Response Style"] = "Single Comment with Engaging Conversational Style"

            if self.last_ai_response:
                try:
                    history = self.AI_suggestion_history
                except Exception as e:
                    print(f"Error in sending last AI response: {e}")
                    history = "Filter out similar knowledge as the follow list:\n" + self.last_ai_response
            else:
                history = None

            response, location = get_llm_response(image_queue=image.copy(), text=send_message, user_profile=None,
                                                  history=history, user_comment=user_input,
                                                  language=self.response_language,
                                                  last_response=self.last_ai_response,
                                                  knowledge_generation_model=self.knowledge_generation_model)
            if location is None:
                self.log(response)
            else:
                main_image = {"image": image[max(0, int(location["frame_number"]) - 1)], "box": location["box"]}

                # Process the response on the main thread
                self.process_response(response, initiation, main_image)
                self.log(response)
        except Exception as e:
            print(f"An error occurred in thread: {e}")
            self.process_response('{"error": "An error occurred while getting response."}', initiation)
        finally:
            self.send_button.configure(state='normal')
            self.send_button.configure(text="Send")
            self.record_button.configure(text="Record")

    def switch_default_response_style(self):
        if self.gpt_response_style == "Single Comment with Engaging Conversational Style":
            self.gpt_response_style = "Danmaku"
            if self.mode == "Desktop":
                self.notification_window.withdraw()
            else:
                self.notification_window.pack_forget()
        else:
            self.gpt_response_style = "Single Comment with Engaging Conversational Style"
            self.danmaku.pack_forget()

    def process_response(self, response, initiation="timer", image=None):
        # This method processes the response and updates the UI
        if not response.startswith('{"error"') and response is not None:
            response_data = detect_json(response)

            self.response_text.delete('1.0', tk.END)  # Clear previous content
            self.response_text.insert(tk.INSERT, f"{json.dumps(response_data, indent=2, ensure_ascii=False)}")

            self.last_ai_response = {}
            if response_data.get("English AI Suggestion") is not None:
                self.last_ai_response["Last AI Suggestion"] = response_data.get("English AI Suggestion")
            if response_data.get("Context Description") is not None:
                self.last_ai_response["Last Moment's Context Description"] = response_data.get("Context Description")


            if isinstance(response_data.get("AI Suggestion").get("v"), str):
                text_ai_suggestions = [response_data.get("AI Suggestion").get("v")]
            elif isinstance(response_data.get("AI Suggestion").get("v"), list):
                text_ai_suggestions = response_data.get("AI Suggestion").get("v")
            else:
                text_ai_suggestions = None
            if isinstance(response_data.get("AI Suggestion").get("a"), str):
                audio_ai_suggestions = [response_data.get("AI Suggestion").get("a")]
            elif isinstance(response_data.get("AI Suggestion").get("a"), list):
                audio_ai_suggestions = response_data.get("AI Suggestion").get("a")
            else:
                audio_ai_suggestions = None
            if response_data.get("Suggestion Type") == "None":
                self.notification_window.pack_forget()
                return
            elif response_data.get("Suggestion Type") == "Danmaku" or response_data.get(
                    "Suggestion Type") == "Live Comments":
                self.danmaku.pack()
                self.danmaku_list.extend(list(text_ai_suggestions))
                self.AI_suggestion_history.extend(list(response_data.get("English AI Suggestion")))
                # hide the notification window
                self.notification_window.pack_forget()
                self.current_feedback_mode = "Live Comments"
            elif "Single Comment" in response_data.get("Suggestion Type"):
                if (response_data.get("AI Suggestion") is not None and len(response_data.get("AI Suggestion"))) != 0:
                    self.notification_window.pack(side=tk.LEFT, anchor=tk.N, padx=10, pady=10)
                    self.danmaku.pack_forget()
                    self.current_feedback_mode = "Single Comment"
                    self.AI_suggestion_history.extend(text_ai_suggestions)
                    # clear the notification text
                    self.notification_scroll_text.delete('1.0', tk.END)
                    self.notification_scroll_text.insert(tk.INSERT, text_ai_suggestions[0])
                    # Schedule a new auto scroll and store its ID
                    self.auto_scroll_id = self.after(6000, self.auto_scroll_text)

            # if response_data.get("Suggestion Type") == "Single Comment":
            if image is not None:
                #  resize the image to fit the canvas and put in canvas
                box = image["box"]
                image = Image.fromarray(image["image"])

                self.image_window_canvas.delete("all")
                width, height = image.width, image.height
                scale_factor = max(width / IMAGE_WINDOW_WIDTH, height / IMAGE_WINDOW_HEIGHT)
                new_width = int(width / scale_factor)
                new_height = int(height / scale_factor)

                image = image.resize((new_width, new_height), Image.LANCZOS)
                image = ImageTk.PhotoImage(image)
                # print("image size: ", image.width(), image.height())
                self.image_window_canvas.create_image(0, 0, anchor='nw', image=image)
                self.image_window_canvas.image = image
                draw_bounding_boxes(self.image_window_canvas, new_width, new_height, box)
                self.image_window.pack(side=tk.RIGHT, anchor=tk.N, padx=10, pady=10)

            if audio_ai_suggestions and self.allow_audio:
                while self.system_data.is_playing_audio:
                    time.sleep(0.1)
                self.system_data.is_playing_audio = True
                # self.AI_suggestion_history.extend(response_data.get("AI Suggestion"))
                self.audio_process = multiprocessing.Process(target=play_text_to_speech_audio,
                                                             args=(str(audio_ai_suggestions[0]),))
                self.audio_process.start()
                while self.audio_process.is_alive():
                    time.sleep(0.1)
                self.system_data.is_playing_audio = False

    def auto_scroll_text(self):
        if self.notification_scroll_text is not None:
            if self.notification_scroll_text.winfo_ismapped():
                self.notification_scroll_text.yview_scroll(1, "units")

                # Cancel the old auto scroll if it exists
                if self.auto_scroll_id is not None:
                    self.after_cancel(self.auto_scroll_id)

                # Schedule a new auto scroll and store its ID
                self.auto_scroll_id = self.after(3500, self.auto_scroll_text)

    def log(self, text):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file_path = os.path.join(self.folder_path, "response_log.txt")
        # create the folder if it does not exist
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
        print(f"Logging to: {log_file_path}")
        with open(log_file_path, "a") as log_file:
            log_file.write(f"Timestamp: {timestamp}\n")
            log_file.write(f"{text}\n")
            log_file.write("\n")

    def run(self):
        # Display the window
        self.setup_input_threads()
        self.backend_thread = threading.Thread(target=self.backend.run, daemon=True)
        self.backend_thread.start()
        self.start_main_func()

    def setup_input_threads(self):
        if self.mode == "Glasses":
            self.setup_pupil_core_thread()
        else:
            self.setup_screen_monitor_thread()
        # self.setup_music_thread()

        self.transcriber = LiveTranscriber(device_index=self.audio_device_idx)
        self.is_recording = False

    def setup_music_thread(self):
        self.music_recognition = MusicRecognition()
        self.music_recognition.start()

    def setup_pupil_core_thread(self):
        BaseManager.register('GazeData', GazeData)
        manager = BaseManager()
        manager.start()
        self.vision_detector = manager.GazeData()
        object_detector = ObjectDetector(simulate=self.simulate, record=False, cv_imshow=False,
                                         video_path=self.video_path)
        self.thread_vision = multiprocessing.Process(target=object_detector.run, args=(self.vision_detector,),
                                                     daemon=True)
        self.thread_vision.start()

    def setup_screen_monitor_thread(self):
        self.screen_monitor = ScreenCapture()
        # Start monitoring the screen
        self.screen_monitor.start_monitoring()

    def start_main_func(self):
        # if not self.simulate:
        #     self.get_gaze_data()
        if self.mode == "Glasses":
            self.get_pupil_core_data()
        else:
            self.get_screen_monitor_data()
        self.listen_backend_msg()
        if self.auto:
            if self.mode == "Glasses" and self.detect_fixation_request_trigger():
                self.send_data(initiation="fixation")
            elif self.detect_timer_request_trigger():
                self.send_data(initiation="timer")
        if self.mode == "Glasses":
            if self.danmaku_list:
                self.danmaku.add_new_item(self.danmaku_list)
                self.danmaku_list = []

        self.after(100, self.start_main_func)

    def detect_fixation_request_trigger(self):
        detected_fixation, fixation_time = self.vision_detector.get_fixation_detected()
        if detected_fixation and fixation_time is not None and time.time() - fixation_time < 2:
            if self.system_data.vision_frame is not None and self.no_other_request_running() and self.fpv_similarity_pass_threshold(
                    0.6):
                self.system_data.last_vision_frame = self.system_data.vision_frame
                return True
        return False

    def detect_timer_request_trigger(self):
        if self.last_gpt_request_time is None or time.time() - self.last_gpt_request_time > MINIMUM_GPT_REQUEST_INTERVAL:
            if self.system_data.vision_frame is not None and self.no_other_request_running() and self.fpv_similarity_pass_threshold():
                self.system_data.last_vision_frame = self.system_data.vision_frame
                return True
        return False

    def no_other_request_running(self):
        return (self.send_button.cget('state') != 'disabled' and
                self.record_button.cget('text') == "Record" and
                not self.system_data.is_playing_audio)

    def fpv_similarity_pass_threshold(self, threshold=FPV_SIMILARITY_THRESHOLD):
        return ((self.system_data.fpv_similarity is None or
                 self.system_data.fpv_similarity < threshold) and
                len(self.system_data.video_sequence_queue) >= 6)

    def get_pupil_core_data(self):
        self.system_data.vision_frame = self.vision_detector.get_original_frame()
        # put the vision frame in the queue every 0.5 second
        if self.system_data.vision_frame is not None:
            if time.time() - self.system_data.last_vision_frame_in_queue_time > 0.3125:
                self.system_data.last_vision_frame_in_queue_time = time.time()
                self.system_data.video_sequence_queue.append(self.system_data.vision_frame)
        self.update_heatmap()

    def get_screen_monitor_data(self):
        self.system_data.vision_frame = self.screen_monitor.get_current_frame()
        if self.system_data.vision_frame is not None:
            self.system_data.video_sequence_queue.append(self.system_data.vision_frame)
            if len(self.system_data.video_sequence_queue) > 6:
                self.system_data.video_sequence_queue.popleft()
        self.update_heatmap()

    def listen_backend_msg(self):
        if self.system_data.backend_msg:
            msg = self.system_data.backend_msg
            if msg["Type"] == "Gaze Heatmap":
                self.update_heatmap(msg["Data"])
            self.system_data.backend_msg = None

    def update_heatmap(self, heatmap_path=None):
        if heatmap_path:
            self.heatmap_image_path = heatmap_path
            heatmap_image = Image.open(heatmap_path)
        else:
            heatmap_image = self.system_data.vision_frame
            if heatmap_image is None:
                return
            heatmap_image = Image.fromarray(heatmap_image)

        width, height = heatmap_image.width, heatmap_image.height

        # Scaling the image using PIL's resize method
        max_width, max_height = 1000, 600
        scale_factor = max(width / max_width, height / max_height)
        new_width = int(width / scale_factor)
        new_height = int(height / scale_factor)

        heatmap_image = heatmap_image.resize((new_width, new_height), Image.LANCZOS)
        heatmap_image = ImageTk.PhotoImage(heatmap_image)

        self.image_canvas.delete("all")
        self.image_canvas.create_image(0, 0, anchor='nw', image=heatmap_image)
        self.image_canvas.image = heatmap_image
        # add text of fpv similarity to the top left corner of the canvas
        self.image_canvas.create_text(10, 10, anchor='nw', text=f"FPV Similarity: {self.system_data.fpv_similarity}")

    def on_closing(self):
        os._exit(0)  # Forcefully exit the process


if __name__ == '__main__':
    PANDAExpFrontend(simulate=True, mode="Glasses", language="English").mainloop()
