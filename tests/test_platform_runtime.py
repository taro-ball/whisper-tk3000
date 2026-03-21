import os
import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from platform_runtime import (
    AUTO_GPU_LABEL,
    CpuExecutionPolicy,
    build_benchmark_option_labels,
    build_cpu_inference_log_message,
    build_cpu_slow_warning,
    build_whisper_env,
    discover_whisper_runtimes,
    resolve_whisper_runtime,
)


TESTS_DIR = Path(__file__).resolve().parent


def _make_temp_dir(prefix: str) -> Path:
    temp_dir = TESTS_DIR / f".{prefix}-{uuid.uuid4().hex}"
    temp_dir.mkdir()
    return temp_dir


def _touch_runtime(bin_dir: Path, folder_name: str) -> None:
    runtime_dir = bin_dir / folder_name
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "whisper-cli.exe").touch()


class PlatformRuntimeTests(unittest.TestCase):
    def test_discover_whisper_runtimes_returns_known_runtimes_in_candidate_order(self) -> None:
        bin_dir = _make_temp_dir("tmp-runtime-discovery")
        self.addCleanup(lambda: shutil.rmtree(bin_dir, ignore_errors=True))
        _touch_runtime(bin_dir, "whisper.cpu")
        _touch_runtime(bin_dir, "whisper.vulkan")
        _touch_runtime(bin_dir, "whisper.cpp")

        runtimes = discover_whisper_runtimes(bin_dir)

        self.assertEqual([runtime["key"] for runtime in runtimes], ["vulkan", "cpu", "legacy"])

    def test_resolve_whisper_runtime_prefers_cpu_selection_then_vulkan(self) -> None:
        bin_dir = _make_temp_dir("tmp-runtime-selection")
        self.addCleanup(lambda: shutil.rmtree(bin_dir, ignore_errors=True))
        _touch_runtime(bin_dir, "whisper.cpu")
        _touch_runtime(bin_dir, "whisper.vulkan")

        cpu_label = "CPU only - Test CPU - 4 physical cores"
        gpu_options = {
            AUTO_GPU_LABEL: None,
            cpu_label: "cpu",
        }

        cpu_runtime = resolve_whisper_runtime(bin_dir, cpu_label, gpu_options)
        gpu_runtime = resolve_whisper_runtime(bin_dir, AUTO_GPU_LABEL, gpu_options)

        self.assertEqual(cpu_runtime["key"], "cpu")
        self.assertEqual(gpu_runtime["key"], "vulkan")

    def test_build_whisper_env_sets_visible_device_for_auto_and_explicit_gpu(self) -> None:
        cpu_label = "CPU only - Test CPU - 4 physical cores"
        gpu_options = {
            AUTO_GPU_LABEL: None,
            "GPU 1 - Integrated": 0,
            "GPU 2 - Discrete": 1,
            cpu_label: "cpu",
        }
        gpu_devices = [
            {"index": 0, "name": "Integrated", "uma": 1},
            {"index": 1, "name": "Discrete", "uma": 0},
        ]
        runtime = {"supports_vulkan": True}

        with patch.dict(os.environ, {"BASE_ENV": "1"}, clear=True):
            auto_env = build_whisper_env(AUTO_GPU_LABEL, runtime, gpu_options, gpu_devices)
            explicit_env = build_whisper_env("GPU 1 - Integrated", runtime, gpu_options, gpu_devices)
            cpu_env = build_whisper_env(cpu_label, runtime, gpu_options, gpu_devices)
            cpu_runtime_env = build_whisper_env(AUTO_GPU_LABEL, {"supports_vulkan": False}, gpu_options, gpu_devices)

        self.assertEqual(auto_env["GGML_VK_VISIBLE_DEVICES"], "1")
        self.assertEqual(explicit_env["GGML_VK_VISIBLE_DEVICES"], "0")
        self.assertEqual(auto_env["BASE_ENV"], "1")
        self.assertNotIn("GGML_VK_VISIBLE_DEVICES", cpu_env)
        self.assertNotIn("GGML_VK_VISIBLE_DEVICES", cpu_runtime_env)

    def test_build_cpu_inference_log_message_reflects_execution_mode(self) -> None:
        cpu_policy = CpuExecutionPolicy(
            cpu_name="Test CPU",
            cpu_option_label="CPU only - Test CPU - 4 physical cores",
            cpu_thread_count=4,
            physical_core_count=4,
            thread_count_log_message="Using 4 physical core(s) for CPU thread count.",
        )
        gpu_options = {
            AUTO_GPU_LABEL: None,
            cpu_policy.cpu_option_label: "cpu",
        }

        cpu_selection_message = build_cpu_inference_log_message(
            cpu_policy,
            cpu_policy.cpu_option_label,
            {"supports_vulkan": True},
            gpu_options,
        )
        cpu_fallback_message = build_cpu_inference_log_message(
            cpu_policy,
            AUTO_GPU_LABEL,
            {"supports_vulkan": False},
            gpu_options,
        )
        gpu_message = build_cpu_inference_log_message(
            cpu_policy,
            AUTO_GPU_LABEL,
            {"supports_vulkan": True},
            gpu_options,
        )

        self.assertEqual(cpu_selection_message, cpu_policy.thread_count_log_message)
        self.assertIn("Falling back to CPU runtime", cpu_fallback_message)
        self.assertIn(cpu_policy.thread_count_log_message, cpu_fallback_message)
        self.assertIsNone(gpu_message)

    def test_build_cpu_slow_warning_only_triggers_for_large_models_on_cpu(self) -> None:
        cpu_label = "CPU only - Test CPU - 4 physical cores"
        gpu_options = {
            AUTO_GPU_LABEL: None,
            cpu_label: "cpu",
        }
        large_model = {
            "name": "ggml-medium.bin",
            "size_bytes": 200 * 1024 * 1024,
            "size_label": "200 MB",
        }
        small_model = {
            "name": "ggml-tiny.en.bin",
            "size_bytes": 70 * 1024 * 1024,
            "size_label": "70 MB",
        }

        cpu_warning = build_cpu_slow_warning(
            large_model,
            cpu_label,
            {"supports_vulkan": True},
            gpu_options,
        )
        cpu_fallback_warning = build_cpu_slow_warning(
            large_model,
            AUTO_GPU_LABEL,
            {"supports_vulkan": False},
            gpu_options,
        )

        self.assertIn("may be slow", cpu_warning)
        self.assertIn("ggml-medium.bin", cpu_warning)
        self.assertIn("may be slow", cpu_fallback_warning)
        self.assertIsNone(
            build_cpu_slow_warning(
                large_model,
                AUTO_GPU_LABEL,
                {"supports_vulkan": True},
                gpu_options,
            )
        )
        self.assertIsNone(
            build_cpu_slow_warning(
                small_model,
                cpu_label,
                {"supports_vulkan": True},
                gpu_options,
            )
        )
        self.assertIsNone(
            build_cpu_slow_warning(
                None,
                cpu_label,
                {"supports_vulkan": True},
                gpu_options,
            )
        )

    def test_build_benchmark_option_labels_keeps_cpu_first_and_excludes_auto(self) -> None:
        cpu_label = "CPU only - Test CPU - 4 physical cores"
        gpu_options = {
            AUTO_GPU_LABEL: None,
            "GPU 1 - Integrated": 0,
            cpu_label: "cpu",
            "GPU 2 - Discrete": 1,
        }

        labels = build_benchmark_option_labels(gpu_options, cpu_label)

        self.assertEqual(labels, [cpu_label, "GPU 1 - Integrated", "GPU 2 - Discrete"])


if __name__ == "__main__":
    unittest.main()
