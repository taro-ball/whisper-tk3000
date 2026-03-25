import shutil
import unittest
import uuid
from pathlib import Path

from whisper_tk3000.core_logic import (
    build_ffmpeg_command,
    build_unique_output_path,
    build_whisper_command,
    requires_ffmpeg_conversion,
)


TESTS_DIR = Path(__file__).resolve().parent


def _make_temp_dir(prefix: str) -> Path:
    temp_dir = TESTS_DIR / f".{prefix}-{uuid.uuid4().hex}"
    temp_dir.mkdir()
    return temp_dir


class CoreLogicTests(unittest.TestCase):
    def test_build_unique_output_path_adds_numeric_suffix_after_collisions(self) -> None:
        temp_path = _make_temp_dir("tmp-core")
        self.addCleanup(lambda: shutil.rmtree(temp_path, ignore_errors=True))
        input_path = temp_path / "clip.mp4"
        (temp_path / "clip.wav").touch()
        (temp_path / "clip-1.wav").touch()

        output_path = build_unique_output_path(input_path, ".wav")

        self.assertEqual(output_path, temp_path / "clip-2.wav")

    def test_requires_ffmpeg_conversion_matches_supported_direct_input_types(self) -> None:
        self.assertFalse(requires_ffmpeg_conversion(Path(r"C:\input\clip.wav")))
        self.assertFalse(requires_ffmpeg_conversion(Path(r"C:\input\clip.mp3")))
        self.assertFalse(requires_ffmpeg_conversion(Path(r"C:\input\clip.ogg")))
        self.assertFalse(requires_ffmpeg_conversion(Path(r"C:\input\clip.flac")))
        self.assertTrue(requires_ffmpeg_conversion(Path(r"C:\input\clip.mp4")))
        self.assertTrue(
            requires_ffmpeg_conversion(
                Path(r"C:\input\clip.wav"),
                duration_seconds=120,
            )
        )

    def test_build_ffmpeg_command_includes_expected_conversion_flags(self) -> None:
        input_path = Path(r"C:\input\clip.mp4")
        audio_output = Path(r"C:\output\clip.wav")
        ffmpeg_path = Path(r"C:\tools\ffmpeg.exe")

        command = build_ffmpeg_command(
            input_path=input_path,
            audio_output=audio_output,
            ffmpeg_path=ffmpeg_path,
            include_stats=True,
            duration_seconds=12,
        )

        self.assertEqual(
            command,
            [
                str(ffmpeg_path),
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-stats",
                "-i",
                str(input_path),
                "-t",
                "12",
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                str(audio_output),
            ],
        )

    def test_build_ffmpeg_command_omits_optional_flags_when_not_requested(self) -> None:
        input_path = Path(r"C:\input\clip.mp4")
        audio_output = Path(r"C:\output\clip.wav")
        ffmpeg_path = Path(r"C:\tools\ffmpeg.exe")

        command = build_ffmpeg_command(
            input_path=input_path,
            audio_output=audio_output,
            ffmpeg_path=ffmpeg_path,
            include_stats=False,
        )

        self.assertNotIn("-stats", command)
        self.assertNotIn("-t", command)

    def test_build_whisper_command_for_gpu_txt_includes_prompt_without_cpu_flags(self) -> None:
        model_path = Path(r"C:\models\ggml-tiny.en.bin")
        audio_output = Path(r"C:\temp\clip.wav")
        output_base = Path(r"C:\temp\clip.transcript")
        whisper_cli_path = Path(r"C:\bin\whisper.vulkan\whisper-cli.exe")

        command = build_whisper_command(
            model_path=model_path,
            audio_output=audio_output,
            output_base=output_base,
            whisper_cli_path=whisper_cli_path,
            output_format="txt",
            cpu_thread_count=8,
            is_cpu_selection=False,
            supports_vulkan=True,
            prompt="Use product names exactly.",
        )

        self.assertEqual(
            command,
            [
                str(whisper_cli_path),
                "-m",
                str(model_path),
                "-f",
                str(audio_output),
                "-of",
                str(output_base),
                "-np",
                "-pp",
                "-otxt",
                "-nt",
                "--prompt",
                "Use product names exactly.",
            ],
        )

    def test_build_whisper_command_for_cpu_selection_on_vulkan_runtime_disables_gpu(self) -> None:
        model_path = Path(r"C:\models\ggml-tiny.en.bin")
        audio_output = Path(r"C:\temp\clip.wav")
        output_base = Path(r"C:\temp\clip.transcript")
        whisper_cli_path = Path(r"C:\bin\whisper.vulkan\whisper-cli.exe")

        command = build_whisper_command(
            model_path=model_path,
            audio_output=audio_output,
            output_base=output_base,
            whisper_cli_path=whisper_cli_path,
            output_format="srt",
            cpu_thread_count=6,
            is_cpu_selection=True,
            supports_vulkan=True,
            debug_enabled=True,
        )

        self.assertEqual(
            command,
            [
                str(whisper_cli_path),
                "-m",
                str(model_path),
                "-f",
                str(audio_output),
                "-of",
                str(output_base),
                "-np",
                "-pp",
                "-osrt",
                "-ng",
                "-t",
                "6",
            ],
        )

    def test_build_whisper_command_for_cpu_runtime_fallback_adds_threads_without_no_gpu_flag(self) -> None:
        model_path = Path(r"C:\models\ggml-tiny.en.bin")
        audio_output = Path(r"C:\temp\clip.wav")
        output_base = Path(r"C:\temp\clip.transcript")
        whisper_cli_path = Path(r"C:\bin\whisper.cpu\whisper-cli.exe")

        command = build_whisper_command(
            model_path=model_path,
            audio_output=audio_output,
            output_base=output_base,
            whisper_cli_path=whisper_cli_path,
            output_format="srt",
            cpu_thread_count=4,
            is_cpu_selection=False,
            supports_vulkan=False,
        )

        self.assertEqual(
            command,
            [
                str(whisper_cli_path),
                "-m",
                str(model_path),
                "-f",
                str(audio_output),
                "-of",
                str(output_base),
                "-np",
                "-osrt",
                "-t",
                "4",
            ],
        )
        self.assertNotIn("-ng", command)


if __name__ == "__main__":
    unittest.main()
