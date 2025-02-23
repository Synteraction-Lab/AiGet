import base64
import datetime
import os
import time

import requests
from io import BytesIO
from PIL import Image
from collections import deque
import concurrent.futures

from dotenv import load_dotenv
load_dotenv()

def compress_image(image, max_size=512):
    image.thumbnail((max_size, max_size))
    compressed_image = BytesIO()
    image.save(compressed_image, format='PNG')
    image.seek(0)  # Seek back to the beginning after thumbnailing
    # save the image locally and use now time in d- h m s and ms as the name
    # image.save(f"image_{datetime.datetime.now().strftime('%m-%d-%H-%M-%S-%f')}.png")
    compressed_image.seek(0)  # Seek back to the beginning of the buffer
    return compressed_image

def encode_image(compressed_image):
    return base64.b64encode(compressed_image.getvalue()).decode('utf-8')

def process_image(image_frame):
    image = Image.fromarray(image_frame)
    compressed_image = compress_image(image)
    return encode_image(compressed_image)

def get_gpt_response(image_frame=None, text=None, conversation_history=None, max_tokens=4096, image_path=None,
                     image_queue=None, model="gpt-4o", temperature=0.0):
    new_message = None
    if image_queue is not None:
        base64_video_frames = []
        compress_start = time.time()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_image, image_queue.popleft()) for _ in range(len(image_queue))]
            for future in futures:
                base64_video_frames.append(future.result())
        print(f"Compressing {len(base64_video_frames)} frames took {time.time() - compress_start:.2f}s")

        new_message = [
            {
                "role": "user",
                "content": [
                    # {
                    #     "type": "text",
                    #      "text":text + " And here are the user's FPV frames with Gaze points in red.",
                    # },

                    # {
                    #     "type": "image_url",
                    #     "image_url": {
                    #         "url": f"data:image/jpeg;base64,{base64_video_frames}",
                    #         "detail": "low"
                    #     }
                    # }
                    str(text) + " Here are the user's FPV frames with Gaze in red.",
                    *map(lambda x: {"type": "image_url",
                                    "image_url": {"url": f'data:image/jpg;base64,{x}'}}, base64_video_frames[:5])
                ],
            },
        ]
    elif image_path is not None or image_frame is not None:
        if image_frame is not None:
            image = Image.fromarray(image_frame)
        elif image_path is not None:
            image = Image.open(image_path)
        compressed_image = compress_image(image)
        base64_image = encode_image(compressed_image)
        new_message = [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": text
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "low"
                    }
                }
            ]
        }]
    elif text is not None:
        new_message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": text
                }
            ]
        }
        print(new_message)

    api_key = os.environ["OPENAI_API_KEY_U1"]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    if conversation_history is None:
        conversation_history = []
    if new_message is not None:
        conversation_history.extend(new_message)

    print("GPT model:", model)
    payload = {
        "model": model,
        "messages": conversation_history,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "response_format": {"type": "json_object"}
    }

    # print("\n-----------------Conversation History-----------------\n")
    # for message in conversation_history:
    #     if isinstance(message["content"], list):
    #         print(message["content"][0])
    #     else:
    #         print(message["content"])
    # print("\n------------------------------------------------------\n")

    request_start = time.time()

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    print(f"GPT Request took {time.time() - request_start:.2f}s")

    if response.status_code == 200:
        response_data = response.json()
        return response_data['choices'][0]['message']['content']
    else:
        print(f"Failed to get caption: {response.text}")
        return None

def get_image_caption(image_path):
    return get_gpt_response(image_path=image_path,
                            text="Generate a brief caption for user's FPV. OCR content if needed", max_tokens=2000)


if __name__ == '__main__':
    image_path = "/Users/Vincent/Documents/GitHub/PandaExp/data/sample_data/di.png"
    caption = get_image_caption(image_path)
    print(caption)