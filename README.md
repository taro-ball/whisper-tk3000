# Whisper Transcriber GUI

Desktop app for local, offline audio transcription with whisper.cpp.
Currently supports Windows. Uses Vulkan for GPU acceleration, so it does not require CUDA and works on most modern GPUs with Vulkan support, including AMD, NVIDIA, and Intel.

## Dependencies

- ffmpeg: `bin/ffmpeg.exe`
- whispercpp: `bin/Vulkan/main64.exe`
- Whisper models are not bundled into the PyInstaller build.
- Download a model from the app, or place a `.bin` file under `models/`.


## Run locally

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

## Build

```bash
./build.sh
```
