from __future__ import annotations

import hashlib
import json
import threading
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Any

from platform_runtime import build_gpu_vendors_payload_value


@dataclass
class TelemetryClient:
    app_id: str
    namespace: str
    app_version: str
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    _sent_once_signals: set[str] = field(default_factory=set, init=False)

    @property
    def url(self) -> str:
        return f"https://nom.telemetrydeck.com/v2/namespace/{self.namespace}/"

    def send_once_async(self, signal_type: str, gpu_devices: list[dict[str, Any]]) -> None:
        if not self.app_id or signal_type in self._sent_once_signals:
            return
        self._sent_once_signals.add(signal_type)
        self.send_async(signal_type, gpu_devices)

    def send_async(self, signal_type: str, gpu_devices: list[dict[str, Any]]) -> None:
        if not self.app_id:
            return
        threading.Thread(
            target=self._send_signal,
            args=(signal_type, gpu_devices),
            daemon=True,
        ).start()

    def _send_signal(self, signal_type: str, gpu_devices: list[dict[str, Any]]) -> None:
        payload = {
            "App.version": self.app_version,
        }
        gpu_vendors = build_gpu_vendors_payload_value(gpu_devices)
        if gpu_vendors:
            payload["App.gpuVendors"] = gpu_vendors

        body = json.dumps(
            [
                {
                    "appID": self.app_id,
                    "clientUser": hashlib.sha256(str(uuid.getnode()).encode("utf-8")).hexdigest(),
                    "sessionID": self.session_id,
                    "type": signal_type,
                    "payload": payload,
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
