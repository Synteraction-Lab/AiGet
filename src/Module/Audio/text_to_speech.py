import tempfile

from playsound import playsound
import openai
import os
import time


def play_text_to_speech_audio(txt):
    try:
        # Parse the response to get the AI suggestion
        start_time = time.time()
        if txt is not None and txt != "None":
            # Create the audio file using OpenAI's text-to-speech API
            client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY_U1"])
            tts_response = client.audio.speech.create(
                model="tts-1",
                voice="nova",
                input=txt
            )

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                temp_file.write(tts_response.content)
                # Play the audio file
                start_reading_time = time.time()
                playsound(temp_file.name)
                length = len(txt.split(" "))
                print(f"Reading time: {time.time() - start_reading_time}, wps: {length/ (time.time() - start_reading_time)}")
            # Delete the temporary file
            os.remove(temp_file.name)
        print(f"Total time to play the AI suggestion: {time.time() - start_time}, wps: {length / (time.time() - start_time)}")

    except Exception as e:
        print(f"An error occurred while playing the AI suggestion: {e}")


def play_text_to_speech_audio_save(txt):
    try:
        # Parse the response to get the AI suggestion
        start_time = time.time()
        if txt is not None and txt != "None":
            # Create the audio file using OpenAI's text-to-speech API
            client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY_U1"])
            tts_response = client.audio.speech.create(
                model="tts-1",
                voice="nova",
                input=txt
            )

            # Save the audio to 'voiceover.mp3' instead of using a temporary file
            file_path = "voiceover.mp3"
            with open(file_path, "wb") as audio_file:
                audio_file.write(tts_response.content)

            # Play the saved audio file
            start_reading_time = time.time()
            playsound(file_path)
            length = len(txt.split(" "))
            print(
                f"Reading time: {time.time() - start_reading_time}, wps: {length / (time.time() - start_reading_time)}")

        print(
            f"Total time to play the AI suggestion: {time.time() - start_time}, wps: {length / (time.time() - start_time)}")

    except Exception as e:
        print(f"An error occurred while playing the AI suggestion: {e}")

if __name__ == '__main__':
    txt = "The Spider Lily near the entrance is poisonous if eaten, especially for pets. They contain alkaloids that can cause vomiting, diarrhea, and other unpleasant symptoms."
    play_text_to_speech_audio_save(txt)
