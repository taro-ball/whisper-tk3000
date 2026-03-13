# Whisper Transcriber GUI

Windows desktop app for converting one media file to WAV and transcribing it with `whisper.cpp`.

## Core flow

- Select one input audio or video file.
- Choose output format: `srt` or `txt`.
- Choose a Whisper model from `models/*.bin`.
- Optionally provide an initial prompt.
- Click `Convert and Transcribe`.

## Behavior

- The app converts the selected input to `<input-name>_audio_<timestamp>.wav` in the same folder.
- It writes the transcript beside that WAV file as:
  - `<audio-name>_transcript_<timestamp>.srt`
  - `<audio-name>_transcript_<timestamp>.txt`
- Plain text mode runs whisper with `-otxt -nt`.
- SRT mode runs whisper with `-osrt`.
- If a prompt is provided, the app passes it with `--prompt`.

## Dependencies

- ffmpeg: `bin/ffmpeg.exe`
- whispercpp: `bin/Vulkan/main64.exe`
- At least one model file in `models/*.bin`
  where to get the model:
https://huggingface.co/ggerganov/whisper.cpp/blob/main/ggml-large-v3-turbo.bin


## Files

- `app.py` - GUI app entry point
- `requirements.txt` - Python dependencies
- `whisper_transcriber.spec` - PyInstaller spec for a `onedir` build
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
