"""Unit tests for platform_runtime module (pure functions, no GUI required)."""

from __future__ import annotations

import unittest
from pathlib import Path

from platform_runtime import (
    detect_gpu_vendor_name,
    build_gpu_vendors_payload_value,
    shorten_device_name,
    shorten_cpu_name,
    shorten_gpu_name,
    build_cpu_option_label,
    discover_whisper_runtimes,
    get_preferred_whisper_runtime,
)
from tests.temp_env import temporary_directory


class TestDetectGpuVendorName(unittest.TestCase):
    """Tests for detect_gpu_vendor_name."""

    def test_nvidia_detection(self) -> None:
        """Detects NVIDIA GPUs."""
        for name in ["NVIDIA GeForce RTX 3080", "nvidia quadro", "Tesla A100"]:
            result = detect_gpu_vendor_name(name)
            self.assertEqual(result, "NVIDIA")

    def test_amd_detection(self) -> None:
        """Detects AMD GPUs."""
        for name in ["AMD Radeon RX 6800", "ati radeon", "AMD Radeon Pro"]:
            result = detect_gpu_vendor_name(name)
            self.assertEqual(result, "AMD")

    def test_intel_detection(self) -> None:
        """Detects Intel GPUs."""
        for name in ["Intel Iris Xe", "Intel Arc A770", "Intel UHD"]:
            result = detect_gpu_vendor_name(name)
            self.assertEqual(result, "Intel")

    def test_unknown_vendor(self) -> None:
        """Returns None for unknown vendor."""
        result = detect_gpu_vendor_name("Unknown GPU Brand XYZ")
        self.assertIsNone(result)

    def test_case_insensitive(self) -> None:
        """Detection is case-insensitive."""
        result1 = detect_gpu_vendor_name("NVIDIA GeForce")
        result2 = detect_gpu_vendor_name("nvidia geforce")
        self.assertEqual(result1, result2)


class TestBuildGpuVendorsPayloadValue(unittest.TestCase):
    """Tests for build_gpu_vendors_payload_value."""

    def test_single_device(self) -> None:
        """Formats single GPU vendor."""
        devices = [{"name": "NVIDIA RTX 3080"}]
        result = build_gpu_vendors_payload_value(devices)
        self.assertEqual(result, "NVIDIA")

    def test_multiple_devices_same_vendor(self) -> None:
        """Deduplicates same vendor."""
        devices = [
            {"name": "NVIDIA RTX 3080"},
            {"name": "NVIDIA RTX 3090"},
        ]
        result = build_gpu_vendors_payload_value(devices)
        self.assertEqual(result, "NVIDIA")

    def test_multiple_vendors(self) -> None:
        """Lists multiple vendors."""
        devices = [
            {"name": "NVIDIA RTX 3080"},
            {"name": "AMD Radeon RX 6800"},
        ]
        result = build_gpu_vendors_payload_value(devices)
        self.assertIn("NVIDIA", result)
        self.assertIn("AMD", result)

    def test_unknown_vendor_excluded(self) -> None:
        """Excludes unknown vendors."""
        devices = [
            {"name": "Unknown GPU"},
            {"name": "NVIDIA RTX 3080"},
        ]
        result = build_gpu_vendors_payload_value(devices)
        self.assertEqual(result, "NVIDIA")


class TestShortenDeviceName(unittest.TestCase):
    """Tests for shorten_device_name."""

    def test_removes_parenthetical_content(self) -> None:
        """Removes content in parentheses."""
        result = shorten_device_name("Device Name (extra info)")
        self.assertNotIn("extra info", result)
        self.assertIn("Device Name", result)

    def test_removes_bracketed_content(self) -> None:
        """Removes content in brackets."""
        result = shorten_device_name("Device Name [v1.0]")
        self.assertNotIn("v1.0", result)

    def test_removes_suffix_patterns(self) -> None:
        """Removes lowercase suffix patterns."""
        result = shorten_device_name("NVIDIA GeForce RTX 3080 driver support info")
        # Should remove everything from lowercase word onwards
        self.assertIn("NVIDIA", result)
        self.assertNotIn("driver", result)

    def test_normalizes_whitespace(self) -> None:
        """Normalizes multiple spaces."""
        result = shorten_device_name("Device   Name   Weird   Spacing")
        self.assertNotIn("   ", result)


class TestShortenCpuName(unittest.TestCase):
    """Tests for shorten_cpu_name."""

    def test_uses_shorten_device_name(self) -> None:
        """Delegates to shorten_device_name."""
        long_name = "Intel(R) Core(TM) i7-10700K CPU @ 3.80GHz"
        result = shorten_cpu_name(long_name)
        # Should remove parenthetical content
        self.assertNotIn("(R)", result)
        self.assertNotIn("(TM)", result)


class TestShortenGpuName(unittest.TestCase):
    """Tests for shorten_gpu_name."""

    def test_uses_shorten_device_name(self) -> None:
        """Delegates to shorten_device_name."""
        long_name = "NVIDIA GeForce(TM) RTX 3080 [official]"
        result = shorten_gpu_name(long_name)
        # Should remove bracketed and parenthetical content
        self.assertNotIn("[official]", result)


class TestBuildCpuOptionLabel(unittest.TestCase):
    """Tests for build_cpu_option_label."""

    def test_with_physical_cores(self) -> None:
        """Uses physical cores when available."""
        result = build_cpu_option_label(
            cpu_name="Intel Core i7",
            cpu_thread_count=8,
            physical_core_count=4,
        )
        
        self.assertIn("CPU only", result)
        self.assertIn("Intel Core i7", result)
        self.assertIn("4", result)
        self.assertIn("physical cores", result)

    def test_single_physical_core(self) -> None:
        """Uses singular 'core' for one core."""
        result = build_cpu_option_label(
            cpu_name="Intel Core i7",
            cpu_thread_count=1,
            physical_core_count=1,
        )
        
        self.assertIn("1 physical core", result)
        self.assertNotIn("cores", result)

    def test_without_physical_cores(self) -> None:
        """Uses logical threads when physical cores unknown."""
        result = build_cpu_option_label(
            cpu_name="Generic CPU",
            cpu_thread_count=8,
            physical_core_count=None,
        )
        
        self.assertIn("CPU only", result)
        self.assertIn("8", result)
        self.assertIn("logical threads", result)

    def test_single_logical_thread(self) -> None:
        """Uses singular 'thread' for one thread."""
        result = build_cpu_option_label(
            cpu_name="Generic CPU",
            cpu_thread_count=1,
            physical_core_count=None,
        )
        
        self.assertIn("1 logical thread", result)


class TestDiscoverWhisperRuntimes(unittest.TestCase):
    """Tests for discover_whisper_runtimes."""

    def test_empty_bin_dir(self) -> None:
        """Returns empty list when no runtimes found."""
        with temporary_directory() as tmpdir:
            result = discover_whisper_runtimes(Path(tmpdir))
            self.assertEqual(result, [])

    def test_finds_vulkan_runtime(self) -> None:
        """Finds whisper.vulkan runtime."""
        with temporary_directory() as tmpdir:
            bin_dir = Path(tmpdir)
            runtime_dir = bin_dir / "whisper.vulkan"
            runtime_dir.mkdir()
            cli = runtime_dir / "whisper-cli.exe"
            cli.touch()
            
            result = discover_whisper_runtimes(bin_dir)
            
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["key"], "vulkan")
            self.assertEqual(result[0]["cli_path"], cli)
            self.assertTrue(result[0]["supports_vulkan"])

    def test_finds_cpu_runtime(self) -> None:
        """Finds whisper.cpu runtime."""
        with temporary_directory() as tmpdir:
            bin_dir = Path(tmpdir)
            runtime_dir = bin_dir / "whisper.cpu"
            runtime_dir.mkdir()
            cli = runtime_dir / "whisper-cli.exe"
            cli.touch()
            
            result = discover_whisper_runtimes(bin_dir)
            
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["key"], "cpu")
            self.assertFalse(result[0]["supports_vulkan"])

    def test_finds_multiple_runtimes(self) -> None:
        """Finds multiple runtimes in order."""
        with temporary_directory() as tmpdir:
            bin_dir = Path(tmpdir)
            
            for folder in ["whisper.vulkan", "whisper.cpu", "whisper.cpp"]:
                runtime_dir = bin_dir / folder
                runtime_dir.mkdir()
                cli = runtime_dir / "whisper-cli.exe"
                cli.touch()
            
            result = discover_whisper_runtimes(bin_dir)
            
            self.assertEqual(len(result), 3)
            keys = [r["key"] for r in result]
            self.assertEqual(keys, ["vulkan", "cpu", "legacy"])


class TestGetPreferredWhisperRuntime(unittest.TestCase):
    """Tests for get_preferred_whisper_runtime."""

    def test_empty_runtimes(self) -> None:
        """Returns None when no runtimes available."""
        result = get_preferred_whisper_runtime([], ("vulkan", "cpu"))
        self.assertIsNone(result)

    def test_selects_first_preferred(self) -> None:
        """Selects first available from preferences."""
        runtimes = [
            {"key": "cpu", "label": "CPU"},
            {"key": "vulkan", "label": "Vulkan"},
        ]
        
        result = get_preferred_whisper_runtime(runtimes, ("vulkan", "cpu"))
        
        self.assertEqual(result["key"], "vulkan")

    def test_skips_missing_in_preferences(self) -> None:
        """Skips keys not in available runtimes."""
        runtimes = [
            {"key": "cpu", "label": "CPU"},
        ]
        
        result = get_preferred_whisper_runtime(runtimes, ("vulkan", "cpu"))
        
        self.assertEqual(result["key"], "cpu")

    def test_fallback_when_enabled(self) -> None:
        """Returns first available as fallback."""
        runtimes = [
            {"key": "legacy", "label": "Legacy"},
            {"key": "cpu", "label": "CPU"},
        ]
        
        result = get_preferred_whisper_runtime(
            runtimes,
            ("vulkan",),
            allow_fallback=True,
        )
        
        self.assertEqual(result["key"], "legacy")

    def test_no_fallback_when_disabled(self) -> None:
        """Returns None when fallback disabled and no match."""
        runtimes = [
            {"key": "cpu", "label": "CPU"},
        ]
        
        result = get_preferred_whisper_runtime(
            runtimes,
            ("vulkan",),
            allow_fallback=False,
        )
        
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
