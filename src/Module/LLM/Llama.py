# Use a pipeline as a high-level helper
import os
import time

from transformers import pipeline

from src.Storage.reader import load_task_description

token = os.environ["HUGGINGFACE_API_KEY"]

pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B", token=token)


def get_llama_response(instruction=None, text=None):
    return pipe(instruction + text)


if __name__ == '__main__':
    instruction = load_task_description("process_output")
    text = {"AI Suggestion": [
        "On the table in front of you, the black bag is made of ballistic nylon, a material originally developed for military body armor due to its high durability and resistance to abrasion.",
        "Behind you, the whiteboard was invented in the 1960s and became popular in the 1990s as a cleaner alternative to chalkboards."
    ]}
    start = time.time()
    print(get_llama_response(instruction, str(text)))
    print(f"Time: {time.time() - start:.2f}s")
