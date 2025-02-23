import subprocess
import threading
import time


class MusicRecognition:
    def __init__(self):
        self.current_music = None

    def start(self):
        threading.Thread(target=self.main_thread, daemon=True).start()

    def main_thread(self):
        while True:
            self.current_music = self.get_music_type()
            time.sleep(15)

    @staticmethod
    def get_music_type(shortcut_name="Shazam"):
        try:
            # Run the shortcut and capture the output
            result = subprocess.run(
                ['shortcuts', 'run', shortcut_name],
                input='\n',  # Simulate pressing Enter
                text=True,
                capture_output=True,
                check=True,
                timeout=10
            )
            # Return the output
            return result.stdout.strip()
        except Exception as e:
            return None


if __name__ == "__main__":
    music_recognition = MusicRecognition()
    music_recognition.start()
    while True:
        print(music_recognition.current_music)
        time.sleep(5)
