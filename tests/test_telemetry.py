import json
import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from whisper_tk3000.settings import SettingsStore
from whisper_tk3000.telemetry import TelemetryClient


class _ImmediateThread:
    def __init__(self, target, args, daemon) -> None:
        self._target = target
        self._args = args
        self.daemon = daemon

    def start(self) -> None:
        self._target(*self._args)


class _FakeResponse:
    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def _make_temp_dir(prefix: str) -> Path:
    path = Path("tests") / f"{prefix}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


class TelemetryClientTests(unittest.TestCase):
    def test_disabled_telemetry_is_a_no_op(self) -> None:
        temp_dir = _make_temp_dir("tmp_telemetry")
        try:
            store = SettingsStore(temp_dir / "settings.json")
            store.set_telemetry_enabled(False)
            client = TelemetryClient(
                app_id="app-id",
                namespace="namespace",
                app_version="1.2.3",
                settings_store=store,
            )

            with patch("whisper_tk3000.telemetry.threading.Thread") as thread_mock:
                client.send_async("transcribe_start", "cpu")

            thread_mock.assert_not_called()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_payload_shape_is_minimal(self) -> None:
        captured_request = None

        def fake_urlopen(request, timeout):
            nonlocal captured_request
            captured_request = request
            return _FakeResponse()

        temp_dir = _make_temp_dir("tmp_telemetry")
        try:
            store = SettingsStore(temp_dir / "settings.json")
            settings = store.load()
            client = TelemetryClient(
                app_id="app-id",
                namespace="namespace",
                app_version="1.2.3",
                settings_store=store,
            )

            with patch(
                "whisper_tk3000.telemetry.threading.Thread",
                side_effect=lambda target, args, daemon: _ImmediateThread(target, args, daemon),
            ):
                with patch("whisper_tk3000.telemetry.urllib.request.urlopen", side_effect=fake_urlopen):
                    client.send_async("transcribe_success", "NVIDIA")

            self.assertIsNotNone(captured_request)
            body = json.loads(captured_request.data.decode("utf-8"))
            self.assertEqual(len(body), 1)
            event = body[0]
            self.assertEqual(set(event.keys()), {"appID", "clientUser", "type", "payload"})
            self.assertEqual(event["appID"], "app-id")
            self.assertEqual(event["clientUser"], settings.install_id)
            self.assertEqual(event["type"], "transcribe_success")
            self.assertEqual(set(event["payload"].keys()), {"app_version", "execution_class"})
            self.assertEqual(event["payload"]["app_version"], "1.2.3")
            self.assertEqual(event["payload"]["execution_class"], "NVIDIA")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
