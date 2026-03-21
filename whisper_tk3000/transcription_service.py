from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .core_logic import (
    RunConfig,
    build_ffmpeg_command,
    build_unique_output_path,
    build_whisper_command,
    slugify_label,
)
from .platform_runtime import (
    CpuExecutionPolicy,
    build_benchmark_option_labels,
    build_cpu_inference_log_message,
    build_cpu_slow_warning,
    build_hidden_subprocess_kwargs,
    build_whisper_env,
    is_cpu_selection,
    resolve_whisper_runtime,
)


class TranscriptionCancelled(Exception):
    pass


@dataclass(frozen=True)
class ExecutionContext:
    ffmpeg_path: Path
    bin_dir: Path
    cpu_policy: CpuExecutionPolicy
    gpu_selection_label: str
    gpu_options: dict[str, int | str | None]
    gpu_devices: list[dict[str, Any]]
    debug_enabled: bool


@dataclass(frozen=True)
class ServiceCallbacks:
    log: Callable[[str], None]
    emit_output: Callable[[str], None]
    on_batch_progress: Callable[[int, int], None] | None = None


@dataclass(frozen=True)
class TranscriptionOutcome:
    last_output: Path | None


class TranscriptionService:
    def __init__(self) -> None:
        self.cancel_requested = False
        self.current_process: subprocess.Popen[str] | None = None
        self.process_lock = threading.Lock()

    def reset_cancellation(self) -> None:
        self.cancel_requested = False

    def cancel(self) -> bool:
        already_requested = self.cancel_requested
        self.cancel_requested = True
        with self.process_lock:
            process = self.current_process
        if process is None or process.poll() is not None:
            return not already_requested
        try:
            process.terminate()
        except OSError:
            pass
        return not already_requested

    def kill_active_process(self) -> None:
        with self.process_lock:
            process = self.current_process
        if process is None or process.poll() is not None:
            return
        try:
            process.kill()
        except OSError:
            pass

    def run_transcription(
        self,
        configs: list[RunConfig],
        context: ExecutionContext,
        callbacks: ServiceCallbacks,
    ) -> TranscriptionOutcome:
        should_show_batch_progress = len(configs) > 1
        cpu_speed_warning_logged = False
        callbacks.log("")
        total = len(configs)
        last_output: Path | None = None

        if should_show_batch_progress and callbacks.on_batch_progress is not None:
            callbacks.on_batch_progress(1, total)

        for index, config in enumerate(configs, start=1):
            self._raise_if_cancelled()
            if total > 1:
                callbacks.log(f"========================= Batch item {index} of {total}")
            callbacks.log(f"========================= processing {config.input_path.name}\n")
            if context.debug_enabled:
                callbacks.log(f"Selected output format: {config.format.upper()}")
                callbacks.log(f"Selected model: {config.model_path.name}")

            try:
                self._convert_input_to_audio(
                    config,
                    context,
                    callbacks,
                    debug_enabled=context.debug_enabled,
                )
                self._raise_if_cancelled()

                selection_label = context.gpu_selection_label.strip()
                whisper_runtime = resolve_whisper_runtime(
                    context.bin_dir,
                    selection_label,
                    context.gpu_options,
                )
                whisper_env = build_whisper_env(
                    selection_label,
                    whisper_runtime,
                    context.gpu_options,
                    context.gpu_devices,
                )
                if context.debug_enabled or index == 1:
                    callbacks.log(
                        f"Using whisper.cpp runtime: {whisper_runtime['label']} "
                        f"({Path(whisper_runtime['cli_path']).parent.name})"
                    )

                cpu_log_message = build_cpu_inference_log_message(
                    context.cpu_policy,
                    selection_label,
                    whisper_runtime,
                    context.gpu_options,
                )
                if cpu_log_message:
                    callbacks.log(cpu_log_message)

                if not cpu_speed_warning_logged:
                    cpu_warning_message = build_cpu_slow_warning(
                        config.model_info,
                        selection_label,
                        whisper_runtime,
                        context.gpu_options,
                    )
                    if cpu_warning_message:
                        callbacks.log(cpu_warning_message)
                        cpu_speed_warning_logged = True

                whisper_command = self._build_whisper_command(
                    config,
                    context,
                    selection_label,
                    whisper_runtime,
                    debug_enabled=context.debug_enabled,
                )
                self._run_process(
                    whisper_command,
                    "whisper.cpp",
                    callbacks,
                    log_details=context.debug_enabled,
                    env=whisper_env,
                )

                last_output = config.transcript_output
                callbacks.log(f"Success. Output file: {config.transcript_output}")
                if should_show_batch_progress and callbacks.on_batch_progress is not None:
                    callbacks.on_batch_progress(index, total)
            finally:
                self._cleanup_audio_output(
                    config.audio_output,
                    callbacks,
                    log_removal=context.debug_enabled,
                )

        return TranscriptionOutcome(last_output=last_output)

    def run_benchmark(
        self,
        config: RunConfig,
        context: ExecutionContext,
        callbacks: ServiceCallbacks,
    ) -> None:
        audio_output: Path | None = None
        benchmark_outputs: list[Path] = []

        try:
            callbacks.log("")
            input_path = config.input_path
            timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
            audio_output = build_unique_output_path(
                input_path,
                ".wav",
                stem=f"{input_path.stem}{timestamp}.benchmark",
            )

            callbacks.log(f"========================= Benchmarking on {input_path.name}")
            callbacks.log(f"Selected model: {config.model_path.name}")
            self._convert_input_to_audio(
                config,
                context,
                callbacks,
                audio_output=audio_output,
                duration_seconds=120,
                debug_enabled=context.debug_enabled,
            )

            option_labels = build_benchmark_option_labels(
                context.gpu_options,
                context.cpu_policy.cpu_option_label,
            )
            for label in option_labels:
                transcript_output = build_unique_output_path(
                    input_path,
                    ".txt",
                    stem=f"{input_path.stem}{timestamp}.{slugify_label(label)}.benchmark",
                )
                output_base = transcript_output.with_suffix("")
                benchmark_config = RunConfig(
                    input_path=config.input_path,
                    format="txt",
                    model_path=config.model_path,
                    model_info=config.model_info,
                    prompt=config.prompt,
                    audio_output=audio_output,
                    output_base=output_base,
                    transcript_output=transcript_output,
                )
                benchmark_outputs.append(benchmark_config.transcript_output)

                whisper_runtime = resolve_whisper_runtime(
                    context.bin_dir,
                    label,
                    context.gpu_options,
                )
                whisper_env = build_whisper_env(
                    label,
                    whisper_runtime,
                    context.gpu_options,
                    context.gpu_devices,
                )
                whisper_command = self._build_whisper_command(
                    benchmark_config,
                    context,
                    label,
                    whisper_runtime,
                    debug_enabled=True,
                    force_txt=True,
                )
                callbacks.log(
                    f"Benchmarking {label} with {whisper_runtime['label']} "
                    f"({Path(whisper_runtime['cli_path']).parent.name})"
                )

                cpu_log_message = build_cpu_inference_log_message(
                    context.cpu_policy,
                    label,
                    whisper_runtime,
                    context.gpu_options,
                )
                if cpu_log_message:
                    callbacks.log(cpu_log_message)

                cpu_warning_message = build_cpu_slow_warning(
                    config.model_info,
                    label,
                    whisper_runtime,
                    context.gpu_options,
                )
                if cpu_warning_message:
                    callbacks.log(cpu_warning_message)

                elapsed_seconds = self._run_benchmark_process(
                    whisper_command,
                    whisper_env,
                    label,
                )
                callbacks.log(f"{elapsed_seconds:.2f} seconds")
        finally:
            if audio_output is not None:
                self._cleanup_audio_output(
                    audio_output,
                    callbacks,
                    log_removal=context.debug_enabled,
                )
            for output_path in benchmark_outputs:
                if output_path.exists():
                    try:
                        output_path.unlink()
                    except OSError:
                        pass

    def _convert_input_to_audio(
        self,
        config: RunConfig,
        context: ExecutionContext,
        callbacks: ServiceCallbacks,
        *,
        audio_output: Path | None = None,
        duration_seconds: int | None = None,
        debug_enabled: bool = False,
    ) -> None:
        ffmpeg_command = build_ffmpeg_command(
            input_path=config.input_path,
            audio_output=audio_output or config.audio_output,
            ffmpeg_path=context.ffmpeg_path,
            include_stats=debug_enabled,
            duration_seconds=duration_seconds,
        )
        self._run_process(
            ffmpeg_command,
            "ffmpeg",
            callbacks,
            log_details=debug_enabled,
        )

    def _build_whisper_command(
        self,
        config: RunConfig,
        context: ExecutionContext,
        selection_label: str,
        runtime: dict[str, Any],
        *,
        debug_enabled: bool,
        force_txt: bool = False,
    ) -> list[str]:
        return build_whisper_command(
            model_path=config.model_path,
            audio_output=config.audio_output,
            output_base=config.output_base,
            whisper_cli_path=Path(runtime["cli_path"]),
            output_format="txt" if force_txt else config.format,
            cpu_thread_count=context.cpu_policy.cpu_thread_count,
            is_cpu_selection=is_cpu_selection(selection_label, context.gpu_options),
            supports_vulkan=bool(runtime["supports_vulkan"]),
            debug_enabled=debug_enabled,
            prompt=config.prompt,
        )

    def _run_process(
        self,
        command: list[str],
        tool_name: str,
        callbacks: ServiceCallbacks,
        *,
        log_details: bool = True,
        env: dict[str, str] | None = None,
    ) -> None:
        self._raise_if_cancelled()
        if log_details:
            callbacks.log(
                f"Running {tool_name}: {' '.join(self._quote_argument(arg) for arg in command)}"
            )
        output_lines: list[str] = []
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                shell=False,
                env=env,
                **build_hidden_subprocess_kwargs(),
            )
        except OSError as exc:
            raise RuntimeError(f"Failed to start {tool_name}: {exc}") from exc

        with self.process_lock:
            self.current_process = process

        try:
            stdout = process.stdout
            if stdout is not None:
                with stdout:
                    for line in stdout:
                        output_lines.append(line)
                        callbacks.emit_output(line)

            exit_code = process.wait()
        finally:
            with self.process_lock:
                if self.current_process is process:
                    self.current_process = None

        if self.cancel_requested:
            raise TranscriptionCancelled()

        if exit_code != 0:
            if tool_name == "ffmpeg":
                combined_output = "".join(output_lines)
                if "Duration: N/A" in combined_output or "Invalid duration" in combined_output:
                    raise RuntimeError("ffmpeg reported an invalid duration for the selected input.")
            raise RuntimeError(f"{tool_name} exited with code {exit_code}")

        if log_details:
            callbacks.log(f"{tool_name} finished successfully.")

    def _run_benchmark_process(
        self,
        command: list[str],
        env: dict[str, str],
        selection_label: str,
    ) -> float:
        started_at = time.perf_counter()
        try:
            process = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                shell=False,
                env=env,
                **build_hidden_subprocess_kwargs(),
            )
        except OSError as exc:
            raise RuntimeError(f"Failed to start benchmark for {selection_label}: {exc}") from exc

        elapsed_seconds = time.perf_counter() - started_at
        if process.returncode != 0:
            raise RuntimeError(f"Benchmark failed for {selection_label} with code {process.returncode}")

        return elapsed_seconds

    def _raise_if_cancelled(self) -> None:
        if self.cancel_requested:
            raise TranscriptionCancelled()

    def _cleanup_audio_output(
        self,
        audio_output: Path,
        callbacks: ServiceCallbacks,
        *,
        log_removal: bool = True,
    ) -> None:
        if not audio_output.exists():
            return
        try:
            audio_output.unlink()
            if log_removal:
                callbacks.log(f"Removed temporary audio file: {audio_output}")
        except OSError as exc:
            callbacks.log(f"WARNING: Could not remove temporary audio file {audio_output}: {exc}")

    @staticmethod
    def _quote_argument(arg: str) -> str:
        if " " in arg or "\t" in arg:
            return f'"{arg}"'
        return arg

