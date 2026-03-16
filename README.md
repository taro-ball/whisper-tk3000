# Whisper Transcriber GUI

A free, simple desktop app to turn your audio files into text. It runs entirely offline on your computer, ensuring your data remains completely private. 

This tool works on most Windows 10 or 11 computers. It uses your graphics card (supporting Intel, AMD, and NVIDIA) to speed up the process, without requiring any complex setup.

![App screenshot](docs/images/screenshot.jpg)

## How to Get Started (No installation required!)

1. Go to our **[Download Page](https://github.com/taro-ball/whisper-tk3000/releases)**.
1. Download the latest `.zip` file for Windows.
1. Open the extracted folder and double-click the application .exe file to run it.

*That's it! You don't need to install Python, command-line tools, or any other software.*

## How to Use

1. Open the app.
1. Select your audio file.
1. Choose a transcription model (you can download one directly within the app if you haven't already).
1. Click **Transcribe**.

## Credits

- `whisper.cpp` for transcription:
  https://github.com/jerryshell/whisper.cpp-windows-vulkan-bin
- `ffmpeg` for media conversion:
  https://ffmpeg.org/about.html
- Model downloads from the `whisper.cpp` Hugging Face repository:
  https://huggingface.co/ggerganov/whisper.cpp/tree/main

---

## For Developers

*The following section is for software developers looking to build or modify the app.*

### Dependencies

- ffmpeg: `bin/ffmpeg.exe`
- whispercpp: `bin/Vulkan/main64.exe`
- Whisper models are not bundled into the PyInstaller build.
- Download a model from the app, or place a `.bin` file under `models/`.


### Run locally

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

### Build

```bash
./build.sh
```
