import base64
import datetime
import json
import os
import time
import pathlib

import google.generativeai as genai
from io import BytesIO
from PIL import Image
from collections import deque
import concurrent.futures

from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.environ["GEMINI_API_KEY"])


def compress_image(image, max_size=512):
    image.thumbnail((max_size, max_size))
    compressed_image = BytesIO()
    image.save(compressed_image, format='PNG')
    image.seek(0)  # Seek back to the beginning after thumbnailing
    compressed_image.seek(0)  # Seek back to the beginning of the buffer
    return compressed_image


def encode_image(compressed_image):
    return base64.b64encode(compressed_image.getvalue()).decode('utf-8')


def process_image(image_frame, image_size=512):
    image = Image.fromarray(image_frame)
    compressed_image = compress_image(image, max_size=image_size)
    return encode_image(compressed_image)


def get_gemini_response(system_instruction=None, image_frame=None, image_path=None, image_queue=None, text=None,
                        max_tokens=8192, temperature=0.0, history=None, model="gemini-2.0-flash-exp", image_size=512,
                        image_process_notes="Above images are the user's 5 second-long fish-eye FPV frames with Gaze Circle/Dots in red. "):
    start_time = time.time()
    generation_config = {
        "temperature": temperature,
        "top_p": 0.95,
        "top_k": 32,
        "max_output_tokens": max_tokens,
        "response_mime_type": "application/json",
    }

    model = genai.GenerativeModel(
        model_name=model,
        generation_config=generation_config,
        system_instruction=system_instruction)

    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE"

        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE"
        }
    ]

    contents = []

    if image_queue is not None:
        compress_start = time.time()
        base64_video_frames = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_image, image_queue.popleft(), image_size) for _ in range(len(image_queue))]
            for future in futures:
                base64_video_frames.append(future.result())

        print(f"Compressing {len(base64_video_frames)} frames took {time.time() - compress_start:.2f}s")

        for base64_image in base64_video_frames:
            image_data = base64.b64decode(base64_image)
            contents.append({
                'mime_type': 'image/png',
                'data': image_data
            })

    elif image_path is not None or image_frame is not None:
        if image_frame is not None:
            image = Image.fromarray(image_frame)
        elif image_path is not None:
            image = Image.open(image_path)

        compressed_image = compress_image(image)
        base64_image = encode_image(compressed_image)
        image_data = base64.b64decode(base64_image)

        contents.append({
            'mime_type': 'image/png',
            'data': image_data
        })

    if text is not None:
        if contents:
            text = image_process_notes + text
        contents.append(text)
        print("moment input: " + text + "\n")

    if history:
        contents = [{"role": "model", "parts": ["Suggestion History (*avoid similar content in the following list*): "+str(history)]}, {"role": "user", "parts": contents}]
        print("history: " + history + "\n")

    response = model.generate_content(contents=contents, safety_settings=safety_settings)
    print(f"Gemini Request took {time.time() - start_time:.2f}s")
    try:
        print(json.dumps(response.text, indent=2))
    except:
        print(response)

    return response.text


def get_image_caption(image_path):
    return get_gemini_response(image_path=image_path,
                               system_instruction="Generate a brief caption for user's FPV. OCR content if needed",
                               text="Here is the user's FPV frame with Gaze in red.",
                               max_tokens=2000)


if __name__ == '__main__':
    image_path = "/Users/Vincent/Documents/GitHub/PandaExp/data/sample_data/t2.png"
    caption = get_image_caption(image_path)
    print(caption)
