from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


MODEL_REPO_URL = "https://huggingface.co/ggerganov/whisper.cpp/tree/main"


@dataclass(frozen=True)
class ModelDownloadOption:
    name: str
    size_label: str
    size_bytes: int
    label: str
    url: str


MODEL_OPTIONS = [
    ModelDownloadOption(
        name="ggml-base.en.bin",
        size_label="148 MB",
        size_bytes=148 * 1024 * 1024,
        label="balanced (english only)",
        url="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin",
    ),
    ModelDownloadOption(
        name="ggml-large-v3-turbo.bin",
        size_label="1.62 GB",
        size_bytes=int(1.62 * 1024 * 1024 * 1024),
        label="precise, multilingual, but slower",
        url="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin",
    ),
    ModelDownloadOption(
        name="ggml-tiny.en.bin",
        size_label="77 MB",
        size_bytes=77 * 1024 * 1024,
        label="good enough, fast (english only)",
        url="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin",
    ),
]
MODEL_OPTIONS_BY_NAME = {option.name: option for option in MODEL_OPTIONS}


def download_model(
    model_option: ModelDownloadOption,
    models_dir: Path,
    *,
    log: Callable[[str], None],
) -> Path:
    destination = models_dir / model_option.name
    temp_destination = destination.with_suffix(destination.suffix + ".part")
    try:
        models_dir.mkdir(parents=True, exist_ok=True)
        log(f"Downloading model from {model_option.url}")
        log(f"Saving model to {destination}")
        _download_file(model_option.url, temp_destination, log=log)
        temp_destination.replace(destination)
        log(f"Model download complete: {destination.name}")
        return destination
    except Exception:
        if temp_destination.exists():
            try:
                temp_destination.unlink()
            except OSError:
                pass
        raise


def _download_file(
    url: str,
    destination: Path,
    *,
    log: Callable[[str], None],
) -> None:
    last_percent = -1

    def report_progress(blocks: int, block_size: int, total_size: int) -> None:
        nonlocal last_percent
        if total_size <= 0:
            return
        downloaded = min(blocks * block_size, total_size)
        percent = int((downloaded * 100) / total_size)
        if percent != last_percent and percent % 10 == 0:
            last_percent = percent
            log(f"Download progress: {percent}%")

    try:
        urllib.request.urlretrieve(url, destination, report_progress)
    except OSError as exc:
        raise RuntimeError(f"Could not download model: {exc}") from exc
