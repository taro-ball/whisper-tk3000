from __future__ import annotations

import json
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .platform_runtime import (
    detect_gpu_vendor_name,
    guess_best_gpu_index,
    is_cpu_inference,
    is_cpu_selection,
    resolve_whisper_runtime,
)
from .settings import SettingsStore


TRANSCRIPTION_EVENTS = frozenset(
    {
        "transcribe_start",
        "transcribe_success",
        "transcribe_fail",
    }
)


@dataclass
class TelemetryClient:
    app_id: str
    namespace: str
    app_version: str
    settings_store: SettingsStore

    @property
    def url(self) -> str:
        return f"https://nom.telemetrydeck.com/v2/namespace/{self.namespace}/"

    def send_async(self, signal_type: str, execution_class: str) -> None:
        if not self.app_id or signal_type not in TRANSCRIPTION_EVENTS:
            return
        settings = self.settings_store.load()
        if not settings.telemetry_enabled or settings.install_id is None:
            return
        threading.Thread(
            target=self._send_signal,
            args=(signal_type, execution_class, settings.install_id),
            daemon=True,
        ).start()

    def _send_signal(self, signal_type: str, execution_class: str, install_id: str) -> None:
        body = json.dumps(
            [
                {
                    "appID": self.app_id,
                    "clientUser": install_id,
                    "type": signal_type,
                    "payload": {
                        "app_version": self.app_version,
                        "execution_class": execution_class,
                    },
                }
            ]
        ).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=body,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=5):
                pass
        except OSError:
            pass


def build_execution_class(
    bin_dir: Path,
    selection_label: str,
    gpu_options: dict[str, int | str | None],
    gpu_devices: list[dict[str, Any]],
) -> str:
    if is_cpu_selection(selection_label, gpu_options):
        return "cpu"

    try:
        runtime = resolve_whisper_runtime(bin_dir, selection_label, gpu_options)
    except Exception:
        runtime = None

    if runtime is not None and is_cpu_inference(selection_label, runtime, gpu_options):
        return "cpu"

    try:
        vendor_name = _detect_selected_gpu_vendor(selection_label, gpu_options, gpu_devices)
    except (TypeError, ValueError):
        vendor_name = None
    return vendor_name or "gpu"


def _detect_selected_gpu_vendor(
    selection_label: str,
    gpu_options: dict[str, int | str | None],
    gpu_devices: list[dict[str, Any]],
) -> str | None:
    selected_gpu = gpu_options.get(selection_label)
    if selected_gpu is None:
        selected_gpu = guess_best_gpu_index(gpu_devices)
    if not isinstance(selected_gpu, int):
        return None
    for device in gpu_devices:
        if int(device.get("index", -1)) != selected_gpu:
            continue
        return detect_gpu_vendor_name(str(device.get("name", "")))
    return None

