# Architecture

## Overview

This app is organized around a thin UI/controller layer in `whisper_tk3000/app.py` and a small set of focused modules that own execution, pure build logic, runtime policy, downloads, and telemetry.

The goal is to keep one obvious place to edit each concern:

- UI and widget behavior: `whisper_tk3000/app.py`
- Transcription and benchmark execution: `whisper_tk3000/transcription_service.py`
- Pure config and command building: `whisper_tk3000/core_logic.py`
- Runtime discovery and CPU/GPU policy: `whisper_tk3000/platform_runtime.py`
- Model download flow: `whisper_tk3000/model_downloads.py`
- Telemetry sending: `whisper_tk3000/telemetry.py`
- Build workflow and packaging entry points: `build.ps1`, `build.sh`, `whisper_transcriber.py`
- Python module entrypoint: `python -m whisper_tk3000`

The app expects `bin/` and `models/` to remain sibling directories in the runtime layout. Packaging should preserve that assumption.

## Module Boundaries

### `whisper_tk3000/app.py`

`whisper_tk3000/app.py` is the CustomTkinter controller.

It is responsible for:

- Building widgets and dialogs
- Holding UI state
- Translating user selections into execution requests
- Launching background worker threads
- Applying log/progress/result callbacks back onto the UI

It should not be the place to edit subprocess execution, runtime selection policy, download internals, or telemetry payload logic.

### `whisper_tk3000/transcription_service.py`

`whisper_tk3000/transcription_service.py` owns the execution layer.

It is responsible for:

- Single-file transcription flow
- Batch transcription flow
- Benchmark flow
- ffmpeg conversion
- whisper.cpp process execution
- Cancellation checks and process termination
- Temporary audio cleanup
- Emitting log, stdout, and progress callbacks

This is the main place to edit execution behavior.

### `whisper_tk3000/core_logic.py`

`whisper_tk3000/core_logic.py` contains pure, UI-free helpers.

It is responsible for:

- `RunConfig`
- Output-path generation
- ffmpeg command construction
- whisper command construction
- Run-config construction
- Small formatting and label helpers

This is the place to edit command-line construction and deterministic path/config logic.

### `whisper_tk3000/platform_runtime.py`

`whisper_tk3000/platform_runtime.py` owns runtime and hardware policy.

It is responsible for:

- CPU detection and CPU execution policy
- Runtime discovery
- Vulkan GPU detection
- GPU option population
- CPU/GPU runtime resolution
- Vulkan environment variable setup
- CPU fallback messages
- CPU slowdown warnings

This is the place to edit runtime selection rules and hardware-related behavior.

### `whisper_tk3000/model_downloads.py`

`whisper_tk3000/model_downloads.py` owns the bundled model catalog and download workflow.

It is responsible for:

- Downloadable model metadata
- Download progress reporting
- Download temp-file handling and final placement

### `whisper_tk3000/telemetry.py`

`whisper_tk3000/telemetry.py` owns telemetry sending.

It is responsible for:

- Session-scoped telemetry client state
- One-shot signal tracking
- Payload construction
- Async HTTP send behavior

## Runtime Flow

1. `python -m whisper_tk3000` enters through `whisper_tk3000/__main__.py`, which calls `whisper_tk3000.app.main()`.
2. `whisper_tk3000/app.py` gathers UI state and validates user inputs.
3. `whisper_tk3000/app.py` builds `RunConfig` objects via `whisper_tk3000/core_logic.py`.
4. `whisper_tk3000/app.py` builds an `ExecutionContext` and calls `TranscriptionService`.
5. `TranscriptionService` resolves runtime policy through `whisper_tk3000/platform_runtime.py`.
6. `TranscriptionService` builds ffmpeg and whisper commands through `whisper_tk3000/core_logic.py`.
7. `TranscriptionService` streams logs/progress/results back through callbacks supplied by `whisper_tk3000/app.py`.

## Build Workflow

- Source and development entrypoint: `python -m whisper_tk3000`
- Canonical Windows packaging entrypoint: `build.ps1`
- Optional Git Bash wrapper: `build.sh`
- Frozen-build launcher: `whisper_transcriber.py`

`build.ps1` is the first place to edit when the packaging workflow changes. `whisper_transcriber.py` exists only for the frozen Windows build and should stay separate from the normal module entrypoint.

The packaged app bundles `bin/` with the distribution, while models remain user-managed under `models/` rather than being baked into the executable.

## Testing Goals

- Keep the default test suite fast, deterministic, and easy for coding agents to run with `python -m unittest discover -s tests`.
- Prefer tests around `whisper_tk3000/core_logic.py`, `whisper_tk3000/platform_runtime.py`, and `whisper_tk3000/transcription_service.py`; avoid GUI-driven tests through `whisper_tk3000/app.py` unless the behavior cannot be covered elsewhere.
- Treat end-to-end CPU and Vulkan smoke tests as opt-in integration checks that catch ffmpeg/runtime/CLI drift. They should depend only on local binaries, models, and hardware, and skip cleanly when prerequisites are missing.

## Editing Guide

- If the app layout, dialogs, or button behavior changes, edit `whisper_tk3000/app.py`.
- If a transcription step or subprocess behavior changes, edit `whisper_tk3000/transcription_service.py`.
- If command arguments or output naming changes, edit `whisper_tk3000/core_logic.py`.
- If CPU/GPU choice behavior changes, edit `whisper_tk3000/platform_runtime.py`.
- If model catalog or download behavior changes, edit `whisper_tk3000/model_downloads.py`.
- If analytics behavior changes, edit `whisper_tk3000/telemetry.py`.
