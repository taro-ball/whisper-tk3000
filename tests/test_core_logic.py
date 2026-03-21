"""Unit tests for core_logic module (pure functions, no GUI required)."""

from __future__ import annotations

import unittest
from pathlib import Path
from datetime import datetime

from core_logic import (
    RunConfig,
    build_unique_output_path,
    build_ffmpeg_command,
    build_whisper_command,
    build_run_configs,
    format_model_size_label,
    slugify_label,
    get_preferred_gpu_device,
    build_auto_gpu_label,
)
from tests.temp_env import temporary_directory


class TestRunConfig(unittest.TestCase):
    """Tests for RunConfig dataclass."""

    def test_run_config_creation(self) -> None:
        """RunConfig can be created and accessed."""
        config = RunConfig(
            input_path=Path("/tmp/audio.mp3"),
            format="srt",
            model_path=Path("/models/model.bin"),
            model_info={"name": "model.bin", "size_bytes": 100},
            prompt="test prompt",
            audio_output=Path("/tmp/audio.wav"),
            output_base=Path("/tmp/output"),
            transcript_output=Path("/tmp/output.srt"),
        )
        
        self.assertEqual(config.input_path, Path("/tmp/audio.mp3"))
        self.assertEqual(config.format, "srt")
        self.assertEqual(config.prompt, "test prompt")

    def test_run_config_to_dict(self) -> None:
        """RunConfig.to_dict() returns expected dict structure."""
        config = RunConfig(
            input_path=Path("/tmp/audio.mp3"),
            format="txt",
            model_path=Path("/models/model.bin"),
            model_info=None,
            prompt="",
            audio_output=Path("/tmp/audio.wav"),
            output_base=Path("/tmp/output"),
            transcript_output=Path("/tmp/output.txt"),
        )
        
        d = config.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["format"], "txt")
        self.assertIsNone(d["model_info"])


class TestBuildUniqueOutputPath(unittest.TestCase):
    """Tests for build_unique_output_path."""

    def test_no_collision(self) -> None:
        """Returns the candidate path when it doesn't exist."""
        with temporary_directory() as tmpdir:
            input_path = Path(tmpdir) / "input.mp3"
            input_path.touch()
            
            result = build_unique_output_path(input_path, ".wav")
            
            self.assertEqual(result.name, "input.wav")
            self.assertFalse(result.exists())

    def test_collision_adds_suffix(self) -> None:
        """Adds numeric suffix when file exists."""
        with temporary_directory() as tmpdir:
            input_path = Path(tmpdir) / "input.mp3"
            input_path.touch()
            
            existing = Path(tmpdir) / "input.wav"
            existing.touch()
            
            result = build_unique_output_path(input_path, ".wav")
            
            self.assertEqual(result.name, "input-1.wav")
            self.assertFalse(result.exists())

    def test_multiple_collisions(self) -> None:
        """Increments suffix for multiple existing files."""
        with temporary_directory() as tmpdir:
            input_path = Path(tmpdir) / "input.mp3"
            input_path.touch()
            
            (Path(tmpdir) / "input.wav").touch()
            (Path(tmpdir) / "input-1.wav").touch()
            (Path(tmpdir) / "input-2.wav").touch()
            
            result = build_unique_output_path(input_path, ".wav")
            
            self.assertEqual(result.name, "input-3.wav")

    def test_custom_stem(self) -> None:
        """Uses custom stem if provided."""
        with temporary_directory() as tmpdir:
            input_path = Path(tmpdir) / "input.mp3"
            input_path.touch()
            
            result = build_unique_output_path(input_path, ".srt", stem="transcript")
            
            self.assertEqual(result.name, "transcript.srt")


class TestBuildFFmpegCommand(unittest.TestCase):
    """Tests for build_ffmpeg_command."""

    def test_basic_command(self) -> None:
        """Builds correct ffmpeg command."""
        cmd = build_ffmpeg_command(
            input_path=Path("/input.mp4"),
            audio_output=Path("/output.wav"),
            ffmpeg_path=Path("/usr/bin/ffmpeg"),
            include_stats=True,
        )
        
        # Check key elements in the command
        self.assertTrue(any("ffmpeg" in str(c).lower() for c in cmd))
        self.assertIn("-i", cmd)
        self.assertIn("-stats", cmd)
        self.assertIn("-ar", cmd)
        self.assertIn("16000", cmd)
        self.assertIn("-ac", cmd)
        self.assertIn("1", cmd)
        # Check that input and output are somewhere in the command
        self.assertTrue(any("input.mp4" in str(c) for c in cmd))
        self.assertTrue(any("output.wav" in str(c) for c in cmd))

    def test_with_duration(self) -> None:
        """Includes duration limit when specified."""
        cmd = build_ffmpeg_command(
            input_path=Path("/input.mp4"),
            audio_output=Path("/output.wav"),
            ffmpeg_path=Path("/usr/bin/ffmpeg"),
            duration_seconds=120,
        )
        
        self.assertIn("-t", cmd)
        self.assertIn("120", cmd)

    def test_without_stats(self) -> None:
        """Excludes -stats flag when include_stats=False."""
        cmd = build_ffmpeg_command(
            input_path=Path("/input.mp4"),
            audio_output=Path("/output.wav"),
            ffmpeg_path=Path("/usr/bin/ffmpeg"),
            include_stats=False,
        )
        
        self.assertNotIn("-stats", cmd)


class TestBuildWhisperCommand(unittest.TestCase):
    """Tests for build_whisper_command."""

    def test_txt_output_format(self) -> None:
        """Builds correct whisper command for txt output."""
        cmd = build_whisper_command(
            model_path=Path("/models/tiny.bin"),
            audio_output=Path("/audio.wav"),
            output_base=Path("/output"),
            whisper_cli_path=Path("/bin/whisper-cli"),
            output_format="txt",
            cpu_thread_count=4,
            is_cpu_selection=True,
            supports_vulkan=True,
            debug_enabled=False,
        )
        
        self.assertTrue(any("whisper-cli" in str(c) for c in cmd))
        self.assertIn("-m", cmd)
        self.assertTrue(any("tiny.bin" in str(c) for c in cmd))
        self.assertIn("-otxt", cmd)
        self.assertIn("-nt", cmd)
        self.assertIn("-ng", cmd)  # Disable GPU when CPU selected
        self.assertIn("-t", cmd)
        self.assertIn("4", cmd)

    def test_srt_output_format(self) -> None:
        """Builds correct whisper command for srt output."""
        cmd = build_whisper_command(
            model_path=Path("/models/tiny.bin"),
            audio_output=Path("/audio.wav"),
            output_base=Path("/output"),
            whisper_cli_path=Path("/bin/whisper-cli"),
            output_format="srt",
            cpu_thread_count=4,
            is_cpu_selection=False,
            supports_vulkan=True,
            debug_enabled=False,
        )
        
        self.assertIn("-osrt", cmd)
        self.assertNotIn("-otxt", cmd)

    def test_gpu_selection_no_disable(self) -> None:
        """Doesn't disable GPU when GPU is selected."""
        cmd = build_whisper_command(
            model_path=Path("/models/tiny.bin"),
            audio_output=Path("/audio.wav"),
            output_base=Path("/output"),
            whisper_cli_path=Path("/bin/whisper-cli"),
            output_format="txt",
            cpu_thread_count=4,
            is_cpu_selection=False,
            supports_vulkan=True,
            debug_enabled=False,
        )
        
        self.assertNotIn("-ng", cmd)

    def test_unsupported_vulkan_uses_cpu(self) -> None:
        """Uses CPU threads even on GPU when Vulkan not supported."""
        cmd = build_whisper_command(
            model_path=Path("/models/tiny.bin"),
            audio_output=Path("/audio.wav"),
            output_base=Path("/output"),
            whisper_cli_path=Path("/bin/whisper-cli"),
            output_format="txt",
            cpu_thread_count=8,
            is_cpu_selection=False,
            supports_vulkan=False,
            debug_enabled=False,
        )
        
        self.assertIn("-t", cmd)
        self.assertIn("8", cmd)

    def test_with_prompt(self) -> None:
        """Includes prompt when provided."""
        cmd = build_whisper_command(
            model_path=Path("/models/tiny.bin"),
            audio_output=Path("/audio.wav"),
            output_base=Path("/output"),
            whisper_cli_path=Path("/bin/whisper-cli"),
            output_format="txt",
            cpu_thread_count=4,
            is_cpu_selection=True,
            supports_vulkan=True,
            debug_enabled=False,
            prompt="Hello world",
        )
        
        self.assertIn("--prompt", cmd)
        self.assertIn("Hello world", cmd)

    def test_debug_mode(self) -> None:
        """Includes debug flags when debug_enabled=True."""
        cmd = build_whisper_command(
            model_path=Path("/models/tiny.bin"),
            audio_output=Path("/audio.wav"),
            output_base=Path("/output"),
            whisper_cli_path=Path("/bin/whisper-cli"),
            output_format="srt",
            cpu_thread_count=4,
            is_cpu_selection=False,
            supports_vulkan=True,
            debug_enabled=True,
        )
        
        # When debug is enabled on SRT output, includes -pp
        self.assertIn("-pp", cmd)


class TestBuildRunConfigs(unittest.TestCase):
    """Tests for build_run_configs."""

    def test_single_input_file(self) -> None:
        """Creates one config for single input."""
        with temporary_directory() as tmpdir:
            input_path = Path(tmpdir) / "audio.mp3"
            input_path.touch()
            model_path = Path("/models/tiny.bin")
            
            configs = build_run_configs(
                input_paths=[input_path],
                model_path=model_path,
                model_info={"name": "tiny.bin", "size_bytes": 100},
                output_format="txt",
                prompt="test",
            )
            
            self.assertEqual(len(configs), 1)
            self.assertEqual(configs[0].input_path, input_path)
            self.assertEqual(configs[0].format, "txt")
            self.assertEqual(configs[0].prompt, "test")

    def test_multiple_input_files(self) -> None:
        """Creates multiple configs for multiple inputs."""
        with temporary_directory() as tmpdir:
            input1 = Path(tmpdir) / "audio1.mp3"
            input2 = Path(tmpdir) / "audio2.mp3"
            input1.touch()
            input2.touch()
            
            configs = build_run_configs(
                input_paths=[input1, input2],
                model_path=Path("/models/tiny.bin"),
                model_info=None,
                output_format="srt",
                prompt="",
            )
            
            self.assertEqual(len(configs), 2)
            self.assertEqual(configs[0].input_path, input1)
            self.assertEqual(configs[1].input_path, input2)

    def test_audio_output_paths_are_unique(self) -> None:
        """Each config has different audio output path."""
        with temporary_directory() as tmpdir:
            input1 = Path(tmpdir) / "audio1.mp3"
            input2 = Path(tmpdir) / "audio2.mp3"
            input1.touch()
            input2.touch()
            
            configs = build_run_configs(
                input_paths=[input1, input2],
                model_path=Path("/models/tiny.bin"),
                model_info=None,
                output_format="txt",
                prompt="",
            )
            
            self.assertNotEqual(
                configs[0].audio_output,
                configs[1].audio_output,
            )


class TestFormatModelSizeLabel(unittest.TestCase):
    """Tests for format_model_size_label."""

    def test_bytes(self) -> None:
        """Formats bytes correctly."""
        self.assertEqual(format_model_size_label(500), "0 MB")

    def test_megabytes(self) -> None:
        """Formats megabytes correctly."""
        result = format_model_size_label(148 * 1024 * 1024)
        self.assertIn("148", result)
        self.assertIn("MB", result)

    def test_gigabytes(self) -> None:
        """Formats gigabytes correctly."""
        result = format_model_size_label(int(1.62 * 1024 * 1024 * 1024))
        self.assertIn("1.62", result)
        self.assertIn("GB", result)


class TestSlugifyLabel(unittest.TestCase):
    """Tests for slugify_label."""

    def test_basic_slug(self) -> None:
        """Converts label to lowercase slug."""
        result = slugify_label("GPU 1 - Device Name")
        self.assertTrue(result.islower() or result == result.lower())

    def test_removes_spaces(self) -> None:
        """Replaces spaces with hyphens."""
        result = slugify_label("My Device Name")
        self.assertNotIn(" ", result)
        self.assertIn("-", result)

    def test_empty_string(self) -> None:
        """Returns 'benchmark' for empty input."""
        result = slugify_label("")
        self.assertEqual(result, "benchmark")


class TestGetPreferredGpuDevice(unittest.TestCase):
    """Tests for get_preferred_gpu_device."""

    def test_empty_devices(self) -> None:
        """Returns None for empty device list."""
        result = get_preferred_gpu_device([])
        self.assertIsNone(result)

    def test_prefers_uma_zero(self) -> None:
        """Prefers device with uma=0."""
        devices = [
            {"index": 0, "name": "Device A", "uma": 1},
            {"index": 1, "name": "Device B", "uma": 0},
            {"index": 2, "name": "Device C", "uma": 1},
        ]
        
        result = get_preferred_gpu_device(devices)
        
        self.assertEqual(result["index"], 1)
        self.assertEqual(result["uma"], 0)

    def test_fallback_to_first(self) -> None:
        """Returns first device if none have uma=0."""
        devices = [
            {"index": 0, "name": "Device A", "uma": 1},
            {"index": 1, "name": "Device B", "uma": 1},
        ]
        
        result = get_preferred_gpu_device(devices)
        
        self.assertEqual(result["index"], 0)


class TestBuildAutoGpuLabel(unittest.TestCase):
    """Tests for build_auto_gpu_label."""

    def test_empty_devices(self) -> None:
        """Returns base label for no devices."""
        result = build_auto_gpu_label([])
        self.assertEqual(result, "Auto (best guess)")

    def test_with_devices(self) -> None:
        """Builds label with device info."""
        devices = [
            {"index": 0, "name": "NVIDIA GPU", "uma": 0},
        ]
        
        result = build_auto_gpu_label(devices)
        
        self.assertIn("Auto (best guess)", result)
        self.assertIn("NVIDIA GPU", result)
        self.assertIn("GPU 1", result)  # Display index is 1-based

    def test_with_multiple_devices(self) -> None:
        """Selects preferred device from multiple."""
        devices = [
            {"index": 0, "name": "Device A", "uma": 1},
            {"index": 1, "name": "Device B", "uma": 0},
        ]
        
        result = build_auto_gpu_label(devices)
        
        self.assertIn("GPU 2", result)  # Device 1, but display as 2
        self.assertIn("Device B", result)


if __name__ == "__main__":
    unittest.main()
