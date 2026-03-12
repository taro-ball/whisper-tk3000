# Python + CustomTkinter + PyInstaller boilerplate

Minimal Windows desktop app boilerplate with:

- Python
- CustomTkinter
- PyInstaller (`onedir`)

The app has one button that runs `ipconfig` and prints the output into a read-only text box.

## Files

- `app.py` - app entry point
- `requirements.txt` - Python dependencies
- `ipconfig_runner.spec` - PyInstaller spec for a `onedir` build
- `build.sh` - optional Git Bash build script

## Prerequisites

- Windows
- Python 3.11+ available as `python`

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
dist/20260312-173000_ipconfig_runner/
```

At the end of the build, it also opens that folder in Windows Explorer.

Run the executable inside the timestamped directory, for example:

```powershell
.\dist\20260312-173000_ipconfig_runner\ipconfig_runner.exe
```

Do not run the executable under `build/`; that is only an intermediate PyInstaller artifact.
