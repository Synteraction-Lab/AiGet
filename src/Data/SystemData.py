from collections import deque

class SystemData:
    def __init__(self):
        self.gaze_data = None
        self.user_comment = None
        self.notification = None
        self.backend_msg = None
        self.target_obj_img_path = None
        self.vision_frame = None
        self.last_vision_frame = None
        self.last_vision_frame_in_queue_time = 0
        self.fpv_similarity = 0
        self.video_sequence_queue = deque(maxlen=16)
        self.last_sent_video_sequence_queue = deque(maxlen=6)
        self.is_playing_audio = False
        self.norm_gaze_data = None
