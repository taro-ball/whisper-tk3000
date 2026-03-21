# Architecture

## Overview

This app is organized around a thin UI/controller layer in `app.py` and a small set of focused modules that own execution, pure build logic, runtime policy, downloads, and telemetry.

The goal is to keep one obvious place to edit each concern:

- UI and widget behavior: `app.py`
- Transcription and benchmark execution: `transcription_service.py`
- Pure config and command building: `core_logic.py`
- Runtime discovery and CPU/GPU policy: `platform_runtime.py`
- Model download flow: `model_downloads.py`
- Telemetry sending: `telemetry.py`
- Build workflow and packaging entry points: `build.ps1`, `build.sh`, `whisper_transcriber.spec`

## Module Boundaries

### `app.py`

`app.py` is the CustomTkinter controller.

It is responsible for:

- Building widgets and dialogs
- Holding UI state
- Translating user selections into execution requests
- Launching background worker threads
- Applying log/progress/result callbacks back onto the UI

It should not be the place to edit subprocess execution, runtime selection policy, download internals, or telemetry payload logic.

### `transcription_service.py`

`transcription_service.py` owns the execution layer.

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

### `core_logic.py`

`core_logic.py` contains pure, UI-free helpers.

It is responsible for:

- `RunConfig`
- Output-path generation
- ffmpeg command construction
- whisper command construction
- Run-config construction
- Small formatting and label helpers

This is the place to edit command-line construction and deterministic path/config logic.

### `platform_runtime.py`

`platform_runtime.py` owns runtime and hardware policy.

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

### `model_downloads.py`

`model_downloads.py` owns the bundled model catalog and download workflow.

It is responsible for:

- Downloadable model metadata
- Download progress reporting
- Download temp-file handling and final placement

### `telemetry.py`

`telemetry.py` owns telemetry sending.

It is responsible for:

- Session-scoped telemetry client state
- One-shot signal tracking
- Payload construction
- Async HTTP send behavior

## Runtime Flow

1. `app.py` gathers UI state and validates user inputs.
2. `app.py` builds `RunConfig` objects via `core_logic.py`.
3. `app.py` builds an `ExecutionContext` and calls `TranscriptionService`.
4. `TranscriptionService` resolves runtime policy through `platform_runtime.py`.
5. `TranscriptionService` builds ffmpeg and whisper commands through `core_logic.py`.
6. `TranscriptionService` streams logs/progress/results back through callbacks supplied by `app.py`.

## Build Workflow

- Canonical Windows build: `build.ps1`
- Optional Git Bash wrapper: `build.sh`
- PyInstaller spec: `whisper_transcriber.spec`

Prefer updating `build.ps1` first if the packaging workflow changes.

## Testing Goals

- Keep the default test suite fast, deterministic, and easy for coding agents to run with `python -m unittest discover -s tests`.
- Prefer tests around `core_logic.py`, `platform_runtime.py`, and `transcription_service.py`; avoid GUI-driven tests through `app.py` unless the behavior cannot be covered elsewhere.
- Treat end-to-end CPU and Vulkan smoke tests as opt-in integration checks that catch ffmpeg/runtime/CLI drift. They should depend only on local binaries, models, and hardware, and skip cleanly when prerequisites are missing.

## Editing Guide

- If the app layout, dialogs, or button behavior changes, edit `app.py`.
- If a transcription step or subprocess behavior changes, edit `transcription_service.py`.
- If command arguments or output naming changes, edit `core_logic.py`.
- If CPU/GPU choice behavior changes, edit `platform_runtime.py`.
- If model catalog or download behavior changes, edit `model_downloads.py`.
- If analytics behavior changes, edit `telemetry.py`.
