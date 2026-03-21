import unittest
from unittest.mock import patch

from transcription_service import ServiceCallbacks, TranscriptionService


class _FakeStdout:
    def __init__(self, lines: list[str]) -> None:
        self._lines = iter(lines)
        self.closed = False

    def __iter__(self) -> "_FakeStdout":
        return self

    def __next__(self) -> str:
        return next(self._lines)

    def __enter__(self) -> "_FakeStdout":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def close(self) -> None:
        self.closed = True


class _FakeProcess:
    def __init__(self, stdout: _FakeStdout, return_code: int) -> None:
        self.stdout = stdout
        self._return_code = return_code
        self.wait_called = False

    def wait(self) -> int:
        self.wait_called = True
        return self._return_code


class TranscriptionServiceTests(unittest.TestCase):
    @staticmethod
    def _build_callbacks(output_lines: list[str]) -> ServiceCallbacks:
        return ServiceCallbacks(log=lambda _message: None, emit_output=output_lines.append)

    def test_run_process_closes_stdout_after_success(self) -> None:
        service = TranscriptionService()
        output_lines: list[str] = []
        stdout = _FakeStdout(["line 1\n", "line 2\n"])
        process = _FakeProcess(stdout, 0)

        with patch("transcription_service.subprocess.Popen", return_value=process):
            service._run_process(
                ["fake-tool.exe"],
                "fake-tool",
                self._build_callbacks(output_lines),
                log_details=False,
            )

        self.assertEqual(output_lines, ["line 1\n", "line 2\n"])
        self.assertTrue(process.wait_called)
        self.assertTrue(stdout.closed)
        self.assertIsNone(service.current_process)

    def test_run_process_closes_stdout_after_failure(self) -> None:
        service = TranscriptionService()
        stdout = _FakeStdout(["problem line\n"])
        process = _FakeProcess(stdout, 7)

        with patch("transcription_service.subprocess.Popen", return_value=process):
            with self.assertRaisesRegex(RuntimeError, "fake-tool exited with code 7"):
                service._run_process(
                    ["fake-tool.exe"],
                    "fake-tool",
                    self._build_callbacks([]),
                    log_details=False,
                )

        self.assertTrue(process.wait_called)
        self.assertTrue(stdout.closed)
        self.assertIsNone(service.current_process)


if __name__ == "__main__":
    unittest.main()
