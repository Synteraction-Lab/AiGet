import time
import os
from mediapipe.tasks.python.vision.image_embedder import ImageEmbedder

# from src.Module.Gaze.gaze_recording import overlay_heatmap_on_image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class Backend:
    def __init__(self, system_data, user_data):
        self.system_data = system_data
        self.user_data = user_data
        self.heat_map_generation_count = 0
        # Create options for Image Embedder
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        image_embedding_model_path = os.path.join(project_root, 'src', 'Module', 'Vision', 'mobilenet_v3_small_075_224_embedder.tflite')
        base_options = python.BaseOptions(model_asset_path=image_embedding_model_path)
        l2_normalize = True  # @param {type:"boolean"}
        quantize = True  # @param {type:"boolean"}
        options = vision.ImageEmbedderOptions(
            base_options=base_options, l2_normalize=l2_normalize, quantize=quantize)
        # Create Image Embedder
        self.embedder = vision.ImageEmbedder.create_from_options(options)

    def run(self):
        while True:
            # print current time
            # print(datetime.datetime.now())
            time.sleep(0.5)

            # self.process_gaze_data()
            # self.process_fpv()
            # start_time = time.time()
            self.calculate_fpv_sequence_similarity()
            # print(f"FPS: {1/(time.time() - start_time)}")
            # if self.trigger_GPT_request():
            #     GPT_response = self.send_GPT_request()
            #     self.parse_GPT_response(GPT_response)

    # def process_gaze_data(self):
    #     # count the gaze number within (0,1) for both x and y in [[x1,y1],[x2,y2],...,[xn,yn]]
    #     gaze_data = self.system_data.gaze_data
    #     gaze_number = 0
    #     for gaze in gaze_data:
    #         if 0 < gaze[0] < 1 and 0 < gaze[1] < 1:
    #             gaze_number += 1
    #
    #     # print(gaze_number)
    #
    #     # if gaze number is larger than 5, then trigger the action
    #     if gaze_number > 200 * (self.heat_map_generation_count + 1) and self.system_data.target_obj_img_path is not None:
    #         print("Trigger the gaze action")
    #         image = cv2.imread(self.system_data.target_obj_img_path)
    #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #         heatmap_path = overlay_heatmap_on_image(image, gaze_data)
    #         self.system_data.backend_msg = {"Type": "Gaze Heatmap", "Data": heatmap_path}
    #         self.heat_map_generation_count += 1

    def process_fpv(self):
        if self.system_data.vision_frame is None or self.system_data.last_vision_frame is None:
            return
        current_mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=self.system_data.vision_frame)
        image_embedding = self.embedder.embed(current_mp_image)
        last_mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=self.system_data.last_vision_frame)
        last_image_embedding = self.embedder.embed(last_mp_image)
        self.system_data.fpv_similarity = ImageEmbedder.cosine_similarity(image_embedding.embeddings[0],last_image_embedding.embeddings[0])

    def calculate_fpv_sequence_similarity(self):
        last_sent_video_sequence_queue = self.system_data.last_sent_video_sequence_queue.copy()
        video_sequence_queue = self.system_data.video_sequence_queue.copy()
        if len(last_sent_video_sequence_queue) == 0 or len(video_sequence_queue) == 0:
            return None

        last_sequence_embeddings = [
            self.embedder.embed(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)).embeddings[0] for frame in
            last_sent_video_sequence_queue]
        current_sequence_embeddings = [
            self.embedder.embed(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)).embeddings[0] for frame in
            video_sequence_queue]

        similarities = [vision.ImageEmbedder.cosine_similarity(last_emb, current_emb) for last_emb, current_emb in
                        zip(last_sequence_embeddings, current_sequence_embeddings)]


        # select the forth largest similarity
        self.system_data.fpv_similarity = sorted(similarities)[-1 - int(len(similarities) * 0.2)]
        # print(f"Average similarity: {self.system_data.fpv_similarity}")
        # return average_similarity


    def trigger_GPT_request(self):
        pass

    def send_GPT_request(self):
        return None

    def parse_GPT_response(self):
        pass
