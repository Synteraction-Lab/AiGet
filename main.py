from src.Frontend import PANDAExpFrontend
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == '__main__':
    pid = "p1_1"
    LANGUAGE = "English"  # "Chinese", "English"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Set a video path if you want to run the program in a simulated mode (i.e., without real-time FPV streaming)
    video_path = os.path.join(current_dir, "data", "sample_data", "SAMPLE_DATA_WITH_GAZE.mp4")
    # Replace the model name (e.g., "gemini-1.5-flash", "gemini-1.5-pro") depending on your need
    PANDAExpFrontend(simulate=False, mode="Glasses", language=LANGUAGE,
                     video_path=video_path, auto=True, knowledge_generation_model="gemini-1.5-flash").mainloop()
