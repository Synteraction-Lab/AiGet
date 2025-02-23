import os
import time
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)


def get_groq_response(text=None, conversation_history=None, max_tokens=4096, model="llama-3.1-70b-versatile",
                      temperature=0.0):
    request_start = time.time()

    new_message = None
    if text is not None:
        new_message = {
            "role": "user",
            "content": text
        }
    if conversation_history is None:
        conversation_history = []
    if new_message is not None:
        conversation_history.append(new_message)

    response = client.chat.completions.create(
        messages=conversation_history,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=False,
        response_format={"type": "json_object"},
    )
    print(f"GROQ Request took: {time.time()-request_start}s")
    print(response, "\n")
    return response.choices[0].message.content


if __name__ == '__main__':
    print(get_groq_response(text="Hello"))
