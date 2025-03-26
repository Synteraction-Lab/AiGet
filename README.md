# AiGet

AiGet is a software system that enables seamless knowledge acquisition on Optical Head-Mounted Displays (OHMD) with the assistance of Large Language Models (LLMs) in everyday life.

Check the Demo Here: [Link](https://code.runzecai.com/aiget-demo/).

## Publications

- [AiGet: Transforming Everyday Moments into Hidden Knowledge Discovery with AI Assistance on Smart Glasses](https://doi.org/10.1145/3706598.3713953), CHI'2025
  - Arxiv: [PDF](https://arxiv.org/pdf/2501.16240).

```
Runze Cai, Nuwan Janaka, Hyeongcheol Kim, Yang Chen, Shengdong Zhao,
Yun Huang, and David Hsu. 2025. AiGet: Transforming Everyday Moments
into Hidden Knowledge Discovery with AI Assistance on Smart Glasses.
In CHI Conference on Human Factors in Computing Systems (CHI '25), April
26–May 01, 2025, Yokohama, Japan. ACM, New York, NY, USA, 26 pages.
https://doi.org/10.1145/3706598.3713953
```

## Contact person

- [Runze Cai](http://runzecai.com)

## System Overview

AiGet transforms everyday environments into learning opportunities by providing contextually relevant information through smart glasses. The system:

1. Captures first-person view (FPV) video with gaze tracking
2. Analyzes the user's environment and focus of attention
3. Generates contextual knowledge using LLMs
4. Delivers tailored information through either visual text display or audio

## Requirements

- Python 3.9.18 (recommended to create a new [conda env](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or [virtual environment](https://realpython.com/python-virtual-environments-a-primer/) first)
- [FFmpeg](https://ffmpeg.org/) (must be added to your environment path)
  - For macOS: `brew install ffmpeg`
  - For Windows: [Manually add to environment variables](https://phoenixnap.com/kb/ffmpeg-windows)
- API Keys:
  - Google AI Studio account for Gemini API
  - OpenAI account for GPT API (optional, for baseline testing)
- [Pupil Labs software](https://docs.pupil-labs.com/core/) for eye tracking
- **Note:** macOS is the preferred and verified OS, as many functions (e.g., GPS and text-to-speech) in the release code use macOS native APIs. These can be replaced with alternative APIs when using other operating systems.

## Installation and Setup

1. Clone the repository to your local machine
2. Install required Python packages:
   ```
   pip install -r requirements.txt
   ```
3. Set up API keys as environment variables:

   - macOS (using zsh):
     ```
     echo "export GEMINI_API_KEY='your_gemini_key'" >> ~/.zshrc
     echo "export OPENAI_API_KEY_U1='your_openai_key'" >> ~/.zshrc
     source ~/.zshrc
     ```
     Verify with:
     ```
     echo $GEMINI_API_KEY
     echo $OPENAI_API_KEY_U1
     ```
   
   - macOS (using bash):
     Follow the same steps as above, replacing `.zshrc` with `.bash_profile`
   
   - Windows:
     Follow [these instructions](https://www.architectryan.com/2018/08/31/how-to-change-environment-variables-on-windows-10/) to set environment variables

## System Workflow

AiGet works through the following pipeline:

1. **Input Collection**:
   - Captures real-time FPV with gaze tracking from Pupil Labs glasses
   - Records audio for user questions/commands
   - Tracks location data for context enrichment

2. **Context Analysis**:
   - Analyzes gaze patterns to determine user focus
   - Identifies primary and peripheral objects in view
   - Uses OCR to extract text from the environment
   - Predicts user intention based on behavior

3. **Knowledge Generation**:
   - Queries LLMs with contextual data
   - Filters responses to avoid repetitive information
   - Personalizes knowledge based on user profile/interests

4. **Content Delivery**:
   - Presents information in multiple modes:
     - Live Comments: Streaming-style text that appears as the user looks around
     - Image with Bounding Box: Highlighted objects of target knowledge
     - Audio narration for low cognitive load & engaging experience

## Usage

### Step 1: Start Pupil Capture
- Run the following command to start the Pupil Lab software:
  ```
  sudo -S "/Applications/Pupil Capture.app/Contents/MacOS/pupil_capture"
  ```

### Step 2: Launch AiGet
- Edit the parameters in `main.py` if needed (language, video path, LLM model)
- Run the application:
  ```
  python main.py
  ```

### Step 3: Configure the Application
- On first run, set up your device and task:
  - Enter a user ID
  - Select voice recording source
  - Click "Save" to save configuration

### Step 4: Interface Controls
- Use the interface controls to manage AiGet:
  - **Left Button/Arrow Key**: Mute/unmute the system
  - **Up Button/Arrow Key**: Stop knowledge display
  - **Down Button/Arrow Key**: Disable/enable proactive knowledge delivery
  - **Right Button/Arrow Key**: Ask questions
  - **Mouse Wheel**: Scroll through generated knowledge
  - For wearable use, you can map ring interactions to these controls using tools like Karabiner-Elements

### Step 5: Review History
- Check your full interaction history with the LLM in:
  ```
  data/recordings/{USER_ID}/response_log.txt
  ```

## Operating Modes

AiGet supports two primary modes:

1. **Glasses Mode**: For deployment on Optical Head-Mounted Displays with Pupil eye tracking
2. **Desktop Mode**: for fun usage on desktop.

## Troubleshooting

### Windows-specific issues
- Windows Defender might treat the keyboardListener in App.py as a threat and automatically delete the file
  - Solution: Add an exclusion in Windows Defender settings:
    1. Open Settings > Update & Security > Windows Security > Virus & Threat protection
    2. Click on "Manage Settings" under "Virus & threat protection settings"
    3. Scroll down to "Exclusions" and click "Add or remove exclusions"
    4. Add the AiGet directory or specific files as exclusions

### MacOS-specific issues
- pyttsx3 issues with `AttributeError: 'super' object has no attribute 'init'`
  - Solution: Edit the pyttsx3 driver file:
    ```
    # Add this line at the top of /path_to_your_venv/pyttsx3/drivers/nsss.py
    from objc import super
    ```

## Customization

- **LLM Models**: Change the `knowledge_generation_model` parameter in `main.py` to use different models:
  - "gemini series model" (requires Gemini API key)
  - "gpt series model" (requires OpenAI API key)

- **User Profile**: Modify the user profile in `src/Module/LLM/task_description/user_profile` to personalize knowledge

## Acknowledgments

References:
- https://realpython.com/python-virtual-environments-a-primer/
- https://phoenixnap.com/kb/ffmpeg-windows
- https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety
