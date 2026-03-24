import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

from whisper_tk3000.app import App
from whisper_tk3000.transcription_service import TranscriptionCancelled, TranscriptionOutcome


class _DummyApp:
    def __init__(self, *, return_value=None, side_effect=None) -> None:
        self.transcription_running = True
        self.logs: list[str] = []
        self.result_path = "unset"
        self.telemetry_client = SimpleNamespace(send_async=Mock())
        self.service_callbacks = object()
        self.transcription_service = SimpleNamespace(
            run_transcription=Mock(return_value=return_value, side_effect=side_effect),
            reset_cancellation=Mock(),
        )

    def _build_run_configs(self) -> list[object]:
        return [object()]

    def _build_execution_context(self) -> object:
        return object()

    def _build_transcription_execution_class(self, context: object) -> str:
        return "gpu"

    def _schedule_ui_update(self, callback) -> None:
        callback()

    def _show_transcription_result(self, path: Path | None) -> None:
        self.result_path = path

    def _set_result_path(self, path: Path | None) -> None:
        self.result_path = path

    def _restore_batch_input_summary(self) -> None:
        self.logs.append("restore")

    def set_running_state(self, running: bool) -> None:
        self.logs.append(f"running:{running}")

    def log(self, message: str) -> None:
        self.logs.append(message)


class AppTelemetryTests(unittest.TestCase):
    def test_transcription_success_emits_start_then_success(self) -> None:
        outcome = TranscriptionOutcome(last_output=Path("output.txt"))
        app = _DummyApp(return_value=outcome)

        App._execute_transcription(app)

        self.assertEqual(
            app.telemetry_client.send_async.call_args_list,
            [
                call("transcribe_start", "gpu"),
                call("transcribe_success", "gpu"),
            ],
        )
        self.assertEqual(app.result_path, Path("output.txt"))

    def test_transcription_failure_emits_start_then_fail(self) -> None:
        app = _DummyApp(side_effect=RuntimeError("boom"))

        App._execute_transcription(app)

        self.assertEqual(
            app.telemetry_client.send_async.call_args_list,
            [
                call("transcribe_start", "gpu"),
                call("transcribe_fail", "gpu"),
            ],
        )
        self.assertIsNone(app.result_path)

    def test_transcription_cancellation_does_not_emit_fail(self) -> None:
        app = _DummyApp(side_effect=TranscriptionCancelled())

        App._execute_transcription(app)

        self.assertEqual(
            app.telemetry_client.send_async.call_args_list,
            [
                call("transcribe_start", "gpu"),
            ],
        )
        self.assertIsNone(app.result_path)

    def test_benchmark_does_not_emit_transcription_telemetry(self) -> None:
        app = SimpleNamespace(
            telemetry_client=SimpleNamespace(send_async=Mock()),
            transcription_service=SimpleNamespace(
                run_benchmark=Mock(),
                reset_cancellation=Mock(),
            ),
            service_callbacks=object(),
            logs=[],
        )
        app._build_run_configs = lambda: [object()]
        app._build_execution_context = lambda: object()
        app._schedule_ui_update = lambda callback: callback()
        app.set_running_state = lambda running: app.logs.append(f"running:{running}")
        app.log = lambda message: app.logs.append(message)

        App._execute_benchmark(app)

        app.telemetry_client.send_async.assert_not_called()


if __name__ == "__main__":
    unittest.main()
