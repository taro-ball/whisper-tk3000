# Whisper Transcriber GUI

Windows desktop app for converting one media file to WAV and transcribing it with `whisper.cpp`.

## Core flow

- Select one input audio or video file.
- Choose output format: `srt` or `txt`.
- Choose a Whisper model from `models/*.bin`.
- If no model is installed yet, download one from inside the app.
- Optionally provide an initial prompt.
- Click `Convert and Transcribe`.

## Behavior

- The app converts the selected input to a temporary `<input-name>_audio_<timestamp>.wav` in the same folder and removes it after transcription finishes.
- It writes the transcript beside that WAV file as:
  - `<audio-name>_transcript_<timestamp>.srt`
  - `<audio-name>_transcript_<timestamp>.txt`
- Plain text mode runs whisper with `-otxt -nt`.
- SRT mode runs whisper with `-osrt`.
- If a prompt is provided, the app passes it with `--prompt`.

## Dependencies

- ffmpeg: `bin/ffmpeg.exe`
- whispercpp: `bin/Vulkan/main64.exe`
- Whisper models are not bundled into the PyInstaller build.
- Download a model from the app, or place a `.bin` file under `models/`.
- Model downloads come from:
  `https://huggingface.co/ggerganov/whisper.cpp/tree/main`


## Files

- `app.py` - GUI app entry point
- `requirements.txt` - Python dependencies
- `whisper_transcriber.spec` - PyInstaller spec for a `onedir` build without bundled model files
- `build.sh` - optional Git Bash build script

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
