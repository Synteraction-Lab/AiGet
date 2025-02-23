# AiGet
A software that enables you to gain knowledge seamlessly on OHMD with the assistance of LLM in everyday life.

## Publications
- [AiGet: Transforming Everyday Moments into Hidden Knowledge Discovery with AI Assistance on Smart Glasses](https://doi.org/10.1145/3706598.3713953), CHI'2025
  - Arxiv: [PDF](https://arxiv.org/pdf/2501.16240).

```
Runze Cai, Nuwan Janaka, Hyeongcheol Kim, Yang Chen, Shengdong Zhao,
Yun Huang, and David Hsu. 2025. AiGet: Transforming Everyday Moments
into Hidden Knowledge Discovery with AI Assistance on Smart Glasses.
In CHI Conference on Human Factors in Computing Systems (CHI ’25), April
26–May 01, 2025, Yokohama, Japan. ACM, New York, NY, USA, 26 pages.
https://doi.org/10.1145/3706598.3713953

```

## Contact person
- [Runze Cai](http://runzecai.com)


## Requirements
- Python 3.9.18 (better to create a new [conda env](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or [virtual environment](https://realpython.com/python-virtual-environments-a-primer/) first.)
- Install [FFmpeg](https://ffmpeg.org/) and add it to your environment path.
  - For macOS, you can use [`brew install ffmpeg`](https://formulae.brew.sh/formula/ffmpeg).
  - For Windows, you may need to [manually add it to the environment variable](https://phoenixnap.com/kb/ffmpeg-windows).
  - **Note: macOS is the preferred and verified OS, as many functions (e.g., GPS and text-to-speech) in the release code use the macOS native APIs.** But feel free to replace them with other APIs when you are using other OS.
- A Google AI Studio and a OpenAI (optional for baseline testing) account to access the GPT API.
- [Pupil Lab software](https://docs.pupil-labs.com/core/) for eye tracking.


## Installation and Setup

1. Clone the repository to your local machine.
2. Run `pip install -r requirements.txt` to install the necessary Python packages.
3. Set your environment variables with your [Gemini API](https://ai.google.dev/aistudio) and OpenAI API keys (optional for baseline testing, from [OpneAI Account](https://platform.openai.com/account/api-keys)), which can be set as follows:

   - MacOS:

      - Option 1: Set your ‘OPENAI_API_KEY’ Environment Variable using zsh:

         1. Run the following command in your terminal, replacing `<yourkey>` with your API key.

            ```echo "export GEMINI_API_KEY='your_gemini_key'" >> ~/.zshrc```
        
            ```echo "export OPENAI_API_KEY_U1='your_openai_key'" >> ~/.zshrc```

         2. Update the shell with the new variable:

            ```source ~/.zshrc```

         3. Confirm that you have set your environment variable using the following command.

            ```echo $GEMINI_API_KEY```
        
            ```echo $OPENAI_API_KEY_U1``` 

      - Option 2: Set your ‘OPENAI_API_KEY’ Environment Variable using bash:
        Follow the directions in Option 1, replacing `.zshrc` with `.bash_profile`.
   - Windows:

      - Please find tutorials on how to set environment variables in Windows [here](https://www.architectryan.com/2018/08/31/how-to-change-environment-variables-on-windows-10/).

## Windows-specific issues
- Windows Defender issue
  - Windows Defender will treat the keyboardListener in App.py as a threat and automatically delete the file 
    - To overcome this problem, follow these steps
      1. Press the Windows + I keys and open Settings
      2. Click on Update & Security
      3. Go to Windows Security
      4. Click on Virus & Threat protection
      5. Select Manage Settings
      6. Under Exclusions, click on Add or Remove exclusion
      7. Click on the + sign which says Add an exclusion
      8. Select File, Folder, File Type, or Process

## MacOS specific issues
- pyttsx3 issue
  - If you meet the issue with `AttributeError: 'super' object has no attribute 'init'` when using the pyttsx3 on macOS
    - Please follow the [instruction](https://github.com/RapidWareTech/pyttsx/pull/35/files) to add `from objc import super` at the top of the `/path_to_your_venv/pyttsx3/drivers/nsss.py` file.


## Manipulation

### Step 1
- Run ``sudo -S "/Applications/Pupil Capture.app/Contents/MacOS/pupil_capture"`` in your terminal to start the Pupil Lab software for macOS.

### Step 2
- Set attributes in ``python main.py`` and run it.

### Step 3
- Set up your device & task, including entering the user_id, selecting task type and output modality, and selecting your source for voice recording.
- Click "Save" to save the configuration.

### Step 4
- You can use our ring mouse to manipulate the menu. You can use your mouse and keyboard for desktop settings.
  - To mute/unmute the system, press the ``arrow_left`` key on your keyboard or click the left button in the GUI. 
  - To stop the new knowledge display, press the ``arrow_up`` key on your keyboard or click the top button in the GUI. 
  - To disable/enable proactive knowledge delivery, press the ``arrow_down`` key on your keyboard or click the bottom button in the GUI.
  - To ask questions, press the  ``arrow_right`` key on your keyboard or click the right button in the GUI. 
  - You can scroll your mouse wheel up and down the generated knowledge.
  - To map the ring interaction to the above settings, you can leverage tools, e.g., Karabiner-Elements.

### Step 5
To check your full history with LLM, you can check the history recording in the ``data/recordings/{USER_ID}/response_log.txt`` folder.

## References

- https://realpython.com/python-virtual-environments-a-primer/
- https://phoenixnap.com/kb/ffmpeg-windows
- https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety



