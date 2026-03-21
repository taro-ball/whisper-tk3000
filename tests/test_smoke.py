import os
import shutil
import unittest
import uuid
import wave
from pathlib import Path

from whisper_tk3000.core_logic import RunConfig
from whisper_tk3000.platform_runtime import build_cpu_execution_policy, load_gpu_selection_state
from whisper_tk3000.transcription_service import (
    ExecutionContext,
    ServiceCallbacks,
    TranscriptionService,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
BIN_DIR = REPO_ROOT / "bin"
MODELS_DIR = REPO_ROOT / "models"
FFMPEG_PATH = BIN_DIR / "ffmpeg.exe"
CPU_RUNTIME_PATH = BIN_DIR / "whisper.cpu" / "whisper-cli.exe"
VULKAN_RUNTIME_PATH = BIN_DIR / "whisper.vulkan" / "whisper-cli.exe"
TINY_MODEL_PATH = MODELS_DIR / "ggml-tiny.en.bin"
RUN_SMOKE_ENV = "WHISPER_TK3000_RUN_SMOKE"
RUN_VULKAN_SMOKE_ENV = "WHISPER_TK3000_RUN_VULKAN_SMOKE"
TESTS_DIR = Path(__file__).resolve().parent


def _make_temp_dir(prefix: str) -> Path:
    temp_dir = TESTS_DIR / f".{prefix}-{uuid.uuid4().hex}"
    temp_dir.mkdir()
    return temp_dir


def _write_silence_wav(path: Path, *, seconds: float = 2.0, sample_rate: int = 16000) -> None:
    frame_count = int(seconds * sample_rate)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * frame_count)


def _build_run_config(temp_dir: Path, model_path: Path) -> RunConfig:
    input_path = temp_dir / "silence-input.wav"
    audio_output = temp_dir / "silence-temp.wav"
    transcript_output = temp_dir / "silence.transcript.txt"
    _write_silence_wav(input_path)
    return RunConfig(
        input_path=input_path,
        format="txt",
        model_path=model_path,
        model_info=None,
        prompt="",
        audio_output=audio_output,
        output_base=transcript_output.with_suffix(""),
        transcript_output=transcript_output,
    )


def _build_callbacks(logs: list[str], output_lines: list[str]) -> ServiceCallbacks:
    return ServiceCallbacks(
        log=logs.append,
        emit_output=output_lines.append,
    )


class SmokeTests(unittest.TestCase):
    def test_cpu_transcription_smoke(self) -> None:
        if os.environ.get(RUN_SMOKE_ENV) != "1":
            self.skipTest(f"Set {RUN_SMOKE_ENV}=1 to run the CPU smoke test.")
        if not FFMPEG_PATH.exists():
            self.skipTest("Skipping CPU smoke test because bin/ffmpeg.exe is missing.")
        if not TINY_MODEL_PATH.exists():
            self.skipTest("Skipping CPU smoke test because models/ggml-tiny.en.bin is missing.")
        if not CPU_RUNTIME_PATH.exists() and not VULKAN_RUNTIME_PATH.exists():
            self.skipTest("Skipping CPU smoke test because no whisper runtime binary is available.")

        cpu_policy = build_cpu_execution_policy()
        context = ExecutionContext(
            ffmpeg_path=FFMPEG_PATH,
            bin_dir=BIN_DIR,
            cpu_policy=cpu_policy,
            gpu_selection_label=cpu_policy.cpu_option_label,
            gpu_options={cpu_policy.cpu_option_label: "cpu"},
            gpu_devices=[],
            debug_enabled=False,
        )

        temp_path = _make_temp_dir("tmp-smoke-cpu")
        self.addCleanup(lambda: shutil.rmtree(temp_path, ignore_errors=True))
        config = _build_run_config(temp_path, TINY_MODEL_PATH)
        logs: list[str] = []
        output_lines: list[str] = []

        outcome = TranscriptionService().run_transcription(
            [config],
            context,
            _build_callbacks(logs, output_lines),
        )

        self.assertEqual(outcome.last_output, config.transcript_output)
        self.assertTrue(config.transcript_output.exists())
        self.assertFalse(config.audio_output.exists())

    def test_vulkan_transcription_smoke(self) -> None:
        if os.environ.get(RUN_VULKAN_SMOKE_ENV) != "1":
            self.skipTest(f"Set {RUN_VULKAN_SMOKE_ENV}=1 to run the Vulkan smoke test.")
        if not FFMPEG_PATH.exists():
            self.skipTest("Skipping Vulkan smoke test because bin/ffmpeg.exe is missing.")
        if not VULKAN_RUNTIME_PATH.exists():
            self.skipTest("Skipping Vulkan smoke test because bin/whisper.vulkan/whisper-cli.exe is missing.")
        if not TINY_MODEL_PATH.exists():
            self.skipTest("Skipping Vulkan smoke test because models/ggml-tiny.en.bin is missing.")

        cpu_policy = build_cpu_execution_policy()
        runtime_state = load_gpu_selection_state(BIN_DIR, "", cpu_policy)
        if not runtime_state.devices:
            self.skipTest("Skipping Vulkan smoke test because no Vulkan device was detected.")
        if runtime_state.selected_value == cpu_policy.cpu_option_label:
            self.skipTest("Skipping Vulkan smoke test because GPU selection resolved to CPU only.")

        context = ExecutionContext(
            ffmpeg_path=FFMPEG_PATH,
            bin_dir=BIN_DIR,
            cpu_policy=cpu_policy,
            gpu_selection_label=runtime_state.selected_value,
            gpu_options=runtime_state.options,
            gpu_devices=runtime_state.devices,
            debug_enabled=True,
        )

        temp_path = _make_temp_dir("tmp-smoke-vulkan")
        self.addCleanup(lambda: shutil.rmtree(temp_path, ignore_errors=True))
        config = _build_run_config(temp_path, TINY_MODEL_PATH)
        logs: list[str] = []
        output_lines: list[str] = []

        outcome = TranscriptionService().run_transcription(
            [config],
            context,
            _build_callbacks(logs, output_lines),
        )

        whisper_command_log = next(
            message for message in logs if message.startswith("Running whisper.cpp: ")
        )

        self.assertEqual(outcome.last_output, config.transcript_output)
        self.assertTrue(config.transcript_output.exists())
        self.assertFalse(config.audio_output.exists())
        self.assertIn("Using whisper.cpp runtime: Vulkan (whisper.vulkan)", logs)
        self.assertIn(str(VULKAN_RUNTIME_PATH), whisper_command_log)
        self.assertNotIn("-ng", whisper_command_log)


if __name__ == "__main__":
    unittest.main()
