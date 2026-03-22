from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SETTINGS_PATH = Path(__file__).resolve().parents[1] / "whisper_tk3000.settings.json"


@dataclass(frozen=True)
class AppSettings:
    telemetry_enabled: bool = True
    install_id: str | None = None


class SettingsStore:
    def __init__(self, path: Path = SETTINGS_PATH) -> None:
        self.path = path

    def load(self) -> AppSettings:
        data = self._read_json()
        telemetry_enabled = bool(data.get("telemetry_enabled", True))
        install_id = self._normalize_install_id(data.get("install_id"))
        if telemetry_enabled and install_id is None:
            install_id = self._generate_install_id()
        if not telemetry_enabled:
            install_id = None

        settings = AppSettings(
            telemetry_enabled=telemetry_enabled,
            install_id=install_id,
        )
        self._write_json(settings)
        return settings

    def set_telemetry_enabled(self, enabled: bool) -> AppSettings:
        settings = AppSettings(
            telemetry_enabled=bool(enabled),
            install_id=self._generate_install_id() if enabled else None,
        )
        self._write_json(settings)
        return settings

    def _read_json(self) -> dict[str, Any]:
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return {}
        if isinstance(data, dict):
            return data
        return {}

    def _write_json(self, settings: AppSettings) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", encoding="utf-8") as handle:
                json.dump(asdict(settings), handle, indent=2)
        except OSError:
            pass

    @staticmethod
    def _generate_install_id() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def _normalize_install_id(value: object) -> str | None:
        if not isinstance(value, str):
            return None
        install_id = value.strip()
        return install_id or None
