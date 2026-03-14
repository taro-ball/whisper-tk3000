# Whisper Transcriber GUI

Windows desktop app for transcribing audio to text with `whisper.cpp`.


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
