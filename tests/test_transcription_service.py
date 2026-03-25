import unittest
from pathlib import Path
from unittest.mock import patch

from whisper_tk3000.core_logic import RunConfig
from whisper_tk3000.platform_runtime import CpuExecutionPolicy
from whisper_tk3000.transcription_service import (
    ExecutionContext,
    ServiceCallbacks,
    TranscriptionService,
)


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
    def _build_callbacks(output_lines: list[str], logs: list[str] | None = None) -> ServiceCallbacks:
        return ServiceCallbacks(
            log=(logs.append if logs is not None else (lambda _message: None)),
            emit_output=output_lines.append,
        )

    @staticmethod
    def _build_context(*, ffmpeg_path: Path | None) -> ExecutionContext:
        cpu_policy = CpuExecutionPolicy(
            cpu_name="Test CPU",
            cpu_option_label="CPU only - Test CPU - 4 physical cores",
            cpu_thread_count=4,
            physical_core_count=4,
            thread_count_log_message="Using 4 physical core(s) for CPU thread count.",
        )
        return ExecutionContext(
            ffmpeg_path=ffmpeg_path,
            bin_dir=Path(r"C:\bin"),
            cpu_policy=cpu_policy,
            gpu_selection_label=cpu_policy.cpu_option_label,
            gpu_options={cpu_policy.cpu_option_label: "cpu"},
            gpu_devices=[],
            debug_enabled=False,
        )

    @staticmethod
    def _build_run_config(input_suffix: str) -> RunConfig:
        input_path = Path(rf"C:\input\clip{input_suffix}")
        audio_output = Path(r"C:\temp\clip.wav")
        transcript_output = Path(r"C:\temp\clip.transcript.txt")
        return RunConfig(
            input_path=input_path,
            format="txt",
            model_path=Path(r"C:\models\ggml-tiny.en.bin"),
            model_info=None,
            prompt="",
            audio_output=audio_output,
            output_base=transcript_output.with_suffix(""),
            transcript_output=transcript_output,
        )

    def test_run_process_closes_stdout_after_success(self) -> None:
        service = TranscriptionService()
        output_lines: list[str] = []
        stdout = _FakeStdout(["line 1\n", "line 2\n"])
        process = _FakeProcess(stdout, 0)

        with patch("whisper_tk3000.transcription_service.subprocess.Popen", return_value=process):
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

        with patch("whisper_tk3000.transcription_service.subprocess.Popen", return_value=process):
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

    def test_run_transcription_passes_supported_audio_directly_without_ffmpeg(self) -> None:
        service = TranscriptionService()
        logs: list[str] = []
        config = self._build_run_config(".wav")
        context = self._build_context(ffmpeg_path=None)
        runtime = {
            "label": "CPU",
            "cli_path": Path(r"C:\bin\whisper.cpu\whisper-cli.exe"),
            "supports_vulkan": False,
        }

        with patch("whisper_tk3000.transcription_service.resolve_whisper_runtime", return_value=runtime), patch(
            "whisper_tk3000.transcription_service.build_whisper_env",
            return_value={},
        ), patch.object(service, "_run_process") as run_process:
            outcome = service.run_transcription(
                [config],
                context,
                self._build_callbacks([], logs),
            )

        whisper_command = run_process.call_args_list[0].args[0]
        self.assertEqual(outcome.last_output, config.transcript_output)
        self.assertEqual(run_process.call_count, 1)
        self.assertEqual(whisper_command[0], str(runtime["cli_path"]))
        self.assertIn(str(config.input_path), whisper_command)
        self.assertNotIn(str(config.audio_output), whisper_command)

    def test_run_transcription_converts_unsupported_input_before_whisper(self) -> None:
        service = TranscriptionService()
        config = self._build_run_config(".mp4")
        context = self._build_context(ffmpeg_path=Path(r"C:\tools\ffmpeg.exe"))
        runtime = {
            "label": "CPU",
            "cli_path": Path(r"C:\bin\whisper.cpu\whisper-cli.exe"),
            "supports_vulkan": False,
        }

        with patch("whisper_tk3000.transcription_service.resolve_whisper_runtime", return_value=runtime), patch(
            "whisper_tk3000.transcription_service.build_whisper_env",
            return_value={},
        ), patch.object(service, "_run_process") as run_process:
            outcome = service.run_transcription(
                [config],
                context,
                self._build_callbacks([]),
            )

        ffmpeg_command = run_process.call_args_list[0].args[0]
        whisper_command = run_process.call_args_list[1].args[0]
        self.assertEqual(outcome.last_output, config.transcript_output)
        self.assertEqual(run_process.call_count, 2)
        self.assertEqual(ffmpeg_command[0], str(context.ffmpeg_path))
        self.assertIn(str(config.input_path), ffmpeg_command)
        self.assertIn(str(config.audio_output), ffmpeg_command)
        self.assertEqual(whisper_command[0], str(runtime["cli_path"]))
        self.assertIn(str(config.audio_output), whisper_command)

    def test_run_transcription_fails_clearly_when_conversion_requires_missing_ffmpeg(self) -> None:
        service = TranscriptionService()
        config = self._build_run_config(".mp4")
        context = self._build_context(ffmpeg_path=None)

        with self.assertRaisesRegex(RuntimeError, r"FFmpeg is required to process \.mp4 inputs"):
            service.run_transcription(
                [config],
                context,
                self._build_callbacks([]),
            )


if __name__ == "__main__":
    unittest.main()
