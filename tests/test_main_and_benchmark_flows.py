from __future__ import annotations

import unittest
from pathlib import Path
import importlib.util
import sys
import types

from core_logic import RunConfig
from tests.temp_env import temporary_directory


if importlib.util.find_spec("customtkinter") is None:
    ctk_stub = types.ModuleType("customtkinter")
    ctk_stub.CTk = type("CTk", (object,), {})
    ctk_stub.CTkToplevel = type("CTkToplevel", (object,), {})
    ctk_stub.CTkFont = lambda *args, **kwargs: None
    ctk_stub.CTkFrame = type("CTkFrame", (object,), {})
    ctk_stub.CTkLabel = type("CTkLabel", (object,), {})
    ctk_stub.CTkButton = type("CTkButton", (object,), {})
    ctk_stub.CTkTextbox = type("CTkTextbox", (object,), {})
    ctk_stub.CTkEntry = type("CTkEntry", (object,), {})
    ctk_stub.CTkOptionMenu = type("CTkOptionMenu", (object,), {})
    ctk_stub.CTkCheckBox = type("CTkCheckBox", (object,), {})
    ctk_stub.set_appearance_mode = lambda *args, **kwargs: None
    ctk_stub.set_default_color_theme = lambda *args, **kwargs: None
    sys.modules["customtkinter"] = ctk_stub

from app import APP_DIR, MEDIA_SUFFIXES, MODELS_DIR, App


class _Var:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


def _first_media_file_in_dist() -> Path:
    dist_dir = APP_DIR / "dist"
    if not dist_dir.exists():
        raise unittest.SkipTest(f"dist directory not found: {dist_dir}")

    media_files = sorted(
        p
        for p in dist_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {suffix.lower() for suffix in MEDIA_SUFFIXES}
    )
    if not media_files:
        raise unittest.SkipTest(f"No media files found in dist directory: {dist_dir}")

    return media_files[0]


def _tiny_model_path() -> Path:
    tiny = MODELS_DIR / "ggml-tiny.en.bin"
    if not tiny.exists():
        raise unittest.SkipTest(f"Tiny model not found: {tiny}")
    return tiny


def _build_single_config(media_path: Path, model_path: Path, work_dir: Path) -> RunConfig:
    transcript_output = work_dir / f"{media_path.stem}.transcript.txt"
    return RunConfig(
        input_path=media_path,
        format="txt",
        model_path=model_path,
        model_info={
            "name": model_path.name,
            "size_label": "77 MB",
            "size_bytes": 77 * 1024 * 1024,
        },
        prompt="",
        audio_output=work_dir / f"{media_path.stem}.wav",
        output_base=transcript_output.with_suffix(""),
        transcript_output=transcript_output,
    )


class TestMainAndBenchmarkFlows(unittest.TestCase):
    def _new_harness(self, config: RunConfig) -> tuple[App, str, str]:
        app = App.__new__(App)

        cpu_label = "CPU only - test"
        gpu_label = "GPU 1 - test"

        app.cpu_option_label = cpu_label
        app.gpu_options = {
            cpu_label: "cpu",
            gpu_label: 0,
            "Auto (best guess)": None,
        }
        app.gpu_devices = [{"index": 0, "name": "Fake Vulkan GPU", "uma": 0}]

        app.debug_var = _Var(False)
        app.gpu_var = _Var(cpu_label)

        app.batch_selected_files = []
        app.cancel_requested = False
        app.transcription_running = False
        app.is_running = False
        app.latest_result_path = None

        app.log_messages = []
        app.log = lambda message: app.log_messages.append(str(message))

        app._schedule_ui_update = lambda callback: callback()
        app._set_result_path = lambda path: setattr(app, "latest_result_path", path)
        app.reveal_result_file = lambda: None
        app.set_running_state = lambda running: setattr(app, "is_running", running)
        app._show_batch_progress = lambda completed, total: None
        app._restore_batch_input_summary = lambda: None

        app._build_run_configs = lambda: [config]
        app._convert_input_to_audio = lambda *args, **kwargs: None
        app._cleanup_audio_output = lambda *args, **kwargs: None
        app._log_cpu_inference_details = lambda *args, **kwargs: None
        app._warn_if_cpu_inference_may_be_slow = lambda *args, **kwargs: False

        cpu_runtime = {
            "label": "CPU",
            "supports_vulkan": False,
            "cli_path": Path("bin/whisper.cpu/whisper-cli.exe"),
        }
        vulkan_runtime = {
            "label": "Vulkan",
            "supports_vulkan": True,
            "cli_path": Path("bin/whisper.vulkan/whisper-cli.exe"),
        }

        app._resolve_whisper_runtime = (
            lambda selection_label: cpu_runtime if selection_label == cpu_label else vulkan_runtime
        )

        return app, cpu_label, gpu_label

    def test_main_flow_uses_tiny_model_with_cpu_and_vulkan(self) -> None:
        media_path = _first_media_file_in_dist()
        model_path = _tiny_model_path()

        with temporary_directory() as temp_dir:
            config = _build_single_config(media_path, model_path, Path(temp_dir))
            app, cpu_label, gpu_label = self._new_harness(config)

            captured_runs: list[tuple[list[str], dict[str, str] | None]] = []

            def fake_run_process(command, tool_name, log_details=True, env=None):
                if tool_name == "whisper.cpp":
                    captured_runs.append((command, env))

            app._run_process = fake_run_process

            app.gpu_var.set(cpu_label)
            app.cancel_requested = False
            App._execute_transcription(app)

            app.gpu_var.set(gpu_label)
            app.cancel_requested = False
            App._execute_transcription(app)

            self.assertEqual(len(captured_runs), 2)

            cpu_cmd, cpu_env = captured_runs[0]
            gpu_cmd, gpu_env = captured_runs[1]

            self.assertIn(str(model_path), cpu_cmd)
            self.assertIn(str(config.audio_output), cpu_cmd)
            self.assertNotIn("-ng", cpu_cmd)
            self.assertIn("-t", cpu_cmd)
            self.assertTrue(cpu_env is None or "GGML_VK_VISIBLE_DEVICES" not in cpu_env)

            self.assertIn(str(model_path), gpu_cmd)
            self.assertIn(str(config.audio_output), gpu_cmd)
            self.assertNotIn("-ng", gpu_cmd)
            self.assertEqual(gpu_env.get("GGML_VK_VISIBLE_DEVICES"), "0")

    def test_benchmark_flow_uses_tiny_model_for_cpu_and_vulkan(self) -> None:
        media_path = _first_media_file_in_dist()
        model_path = _tiny_model_path()

        with temporary_directory() as temp_dir:
            config = _build_single_config(media_path, model_path, Path(temp_dir))
            app, cpu_label, gpu_label = self._new_harness(config)

            captured_benchmarks: list[tuple[str, list[str], dict[str, str]]] = []

            def fake_run_benchmark_process(command, env, selection_label):
                captured_benchmarks.append((selection_label, command, env))
                return 0.01

            app._run_benchmark_process = fake_run_benchmark_process

            App._execute_benchmark(app)

            self.assertEqual([label for label, _, _ in captured_benchmarks], [cpu_label, gpu_label])

            cpu_label_seen, cpu_cmd, cpu_env = captured_benchmarks[0]
            gpu_label_seen, gpu_cmd, gpu_env = captured_benchmarks[1]

            self.assertEqual(cpu_label_seen, cpu_label)
            self.assertIn(str(model_path), cpu_cmd)
            self.assertNotIn("-ng", cpu_cmd)
            self.assertIn("-t", cpu_cmd)
            self.assertTrue("GGML_VK_VISIBLE_DEVICES" not in cpu_env)

            self.assertEqual(gpu_label_seen, gpu_label)
            self.assertIn(str(model_path), gpu_cmd)
            self.assertNotIn("-ng", gpu_cmd)
            self.assertEqual(gpu_env.get("GGML_VK_VISIBLE_DEVICES"), "0")


if __name__ == "__main__":
    unittest.main()
