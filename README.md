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

- `bin/ffmpeg.exe`
- `bin/Vulkan/main64.exe`
- At least one model file in `models/*.bin`

## Logging and errors

- Console output is timestamped for app-generated status lines.
- `ffmpeg` and `whisper.cpp` stdout and stderr are streamed into the GUI console.
- The final output file path is printed on success.
- Clear errors are shown for missing input, missing dependencies, missing models, or non-zero exit codes.

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

The build script creates a timestamped output directory in `dist/`, for example:

```text
dist/20260312-173000_whisper_transcriber/
```

Run the executable inside that directory, for example:

```powershell
.\dist\20260312-173000_whisper_transcriber\whisper_transcriber.exe
```
