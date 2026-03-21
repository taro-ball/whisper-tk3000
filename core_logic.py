"""Pure logic for transcription and benchmark command/config building.

This module contains dataclasses and functions for building transcription
configurations and whisper/ffmpeg commands. No GUI or I/O dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


# ============================================================================
# RunConfig Dataclass
# ============================================================================

@dataclass
class RunConfig:
    """Immutable configuration for a single transcription run."""
    
    input_path: Path
    format: str
    model_path: Path
    model_info: dict[str, Any] | None
    prompt: str
    audio_output: Path
    output_base: Path
    transcript_output: Path
    
    def to_dict(self) -> dict[str, Path | str]:
        """Convert to dict for backward compatibility with code expecting dicts."""
        return {
            "input_path": self.input_path,
            "format": self.format,
            "model_path": self.model_path,
            "model_info": self.model_info,
            "prompt": self.prompt,
            "audio_output": self.audio_output,
            "output_base": self.output_base,
            "transcript_output": self.transcript_output,
        }


# ============================================================================
# Output Path Building
# ============================================================================

def build_unique_output_path(
    input_path: Path,
    suffix: str,
    stem: str | None = None,
) -> Path:
    """Build a unique output path, adding numeric suffix if needed.
    
    Args:
        input_path: Source file to base the output path on
        suffix: File extension (e.g., '.wav', '.txt')
        stem: Optional custom stem; defaults to input_path.stem
    
    Returns:
        A Path that doesn't already exist
    """
    base_stem = stem or input_path.stem
    candidate = input_path.with_name(f"{base_stem}{suffix}")
    index = 1

    while candidate.exists():
        candidate = input_path.with_name(f"{base_stem}-{index}{suffix}")
        index += 1

    return candidate


# ============================================================================
# ffmpeg Command Building
# ============================================================================

def build_ffmpeg_command(
    input_path: Path,
    audio_output: Path,
    ffmpeg_path: Path,
    *,
    include_stats: bool = True,
    duration_seconds: int | None = None,
) -> list[str]:
    """Build an ffmpeg command to convert media to 16kHz mono WAV.
    
    Args:
        input_path: Source media file
        audio_output: Destination audio file
        ffmpeg_path: Path to ffmpeg executable
        include_stats: Whether to include progress stats
        duration_seconds: Optional duration limit in seconds
    
    Returns:
        Command as list of strings suitable for subprocess
    """
    command = [
        str(ffmpeg_path),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    if include_stats:
        command.append("-stats")
    
    command.extend(["-i", str(input_path)])
    
    if duration_seconds is not None:
        command.extend(["-t", str(duration_seconds)])
    
    command.extend(
        [
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(audio_output),
        ]
    )
    return command


# ============================================================================
# whisper.cpp Command Building
# ============================================================================

def build_whisper_command(
    model_path: Path,
    audio_output: Path,
    output_base: Path,
    whisper_cli_path: Path,
    output_format: str,
    cpu_thread_count: int,
    *,
    is_cpu_selection: bool,
    supports_vulkan: bool,
    debug_enabled: bool = False,
    prompt: str = "",
) -> list[str]:
    """Build a whisper.cpp command to transcribe audio.
    
    Args:
        model_path: Path to whisper model file
        audio_output: Audio file to transcribe
        output_base: Base path for output (without extension)
        whisper_cli_path: Path to whisper-cli executable
        output_format: Output format ('txt' or 'srt')
        cpu_thread_count: Number of CPU threads to use
        is_cpu_selection: Whether user explicitly selected CPU
        supports_vulkan: Whether the runtime supports Vulkan
        debug_enabled: Whether debug logging is enabled
        prompt: Optional --prompt text
    
    Returns:
        Command as list of strings suitable for subprocess
    """
    command = [
        str(whisper_cli_path),
        "-m",
        str(model_path),
        "-f",
        str(audio_output),
        "-of",
        str(output_base),
        "-np",
    ]

    # Output format configuration
    if output_format == "txt":
        command.extend(["-pp", "-otxt", "-nt"])
    else:  # srt
        if debug_enabled:
            command.append("-pp")
        command.append("-osrt")

    # CPU configuration
    if is_cpu_selection:
        if supports_vulkan:
            command.append("-ng")  # Disable GPU
        command.extend(["-t", str(cpu_thread_count)])
    elif not supports_vulkan:
        # Fallback to CPU if runtime doesn't support Vulkan
        command.extend(["-t", str(cpu_thread_count)])

    # Optional prompt
    if prompt:
        command.extend(["--prompt", prompt])

    return command


# ============================================================================
# Config Building
# ============================================================================

def build_run_configs(
    input_paths: list[Path],
    model_path: Path,
    model_info: dict[str, Any] | None,
    output_format: str,
    prompt: str,
) -> list[RunConfig]:
    """Build transcription configs for one or more input files.
    
    Args:
        input_paths: List of input media files
        model_path: Path to whisper model
        model_info: Model metadata dict
        output_format: Output format ('txt' or 'srt')
        prompt: Optional --prompt text
    
    Returns:
        List of RunConfig objects, one per input file
    """
    configs: list[RunConfig] = []
    
    for input_path in input_paths:
        timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
        audio_output = build_unique_output_path(input_path, ".wav")
        transcript_base_name = f"{input_path.stem}{timestamp}.transcript"
        transcript_output = build_unique_output_path(
            input_path,
            f".{output_format}",
            stem=transcript_base_name,
        )
        output_base = transcript_output.with_suffix("")
        
        configs.append(
            RunConfig(
                input_path=input_path,
                format=output_format,
                model_path=model_path,
                model_info=model_info,
                prompt=prompt,
                audio_output=audio_output,
                output_base=output_base,
                transcript_output=transcript_output,
            )
        )
    
    return configs


# ============================================================================
# Utility Functions
# ============================================================================

def format_model_size_label(size_bytes: int) -> str:
    """Format a file size in bytes as a human-readable label.
    
    Args:
        size_bytes: Size in bytes
    
    Returns:
        Human-readable size like '148 MB' or '1.62 GB'
    """
    gib = 1024 * 1024 * 1024
    mib = 1024 * 1024
    if size_bytes >= gib:
        return f"{size_bytes / gib:.2f} GB"
    return f"{size_bytes / mib:.0f} MB"


def slugify_label(label: str) -> str:
    """Convert a label to a slug suitable for filenames.
    
    Args:
        label: Label to slugify (e.g., 'GPU 1 - Device Name')
    
    Returns:
        Slugified version (lowercase, hyphens, alphanumeric only)
    """
    import re
    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    return slug or "benchmark"


def get_preferred_gpu_device(devices: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Select the preferred GPU device from available devices.
    
    Prefers devices with UMA (unified memory architecture) = 0, otherwise
    returns the first device.
    
    Args:
        devices: List of GPU devices with 'index' and 'uma' keys
    
    Returns:
        The preferred device dict, or None if no devices available
    """
    if not devices:
        return None
    for device in devices:
        if int(device.get("uma", 1)) == 0:
            return device
    return devices[0]


def build_auto_gpu_label(devices: list[dict[str, Any]]) -> str:
    """Build a label for the Auto GPU selection.
    
    Args:
        devices: List of GPU devices
    
    Returns:
        Label like 'Auto (best guess)' or
        'Auto (best guess) - GPU 1 - Device Name' if devices available
    """
    AUTO_GPU_LABEL = "Auto (best guess)"
    preferred_device = get_preferred_gpu_device(devices)
    if preferred_device is None:
        return AUTO_GPU_LABEL

    best_index = int(preferred_device["index"])
    best_name = str(preferred_device["name"])
    return f"{AUTO_GPU_LABEL} - GPU {best_index + 1} - {best_name}"
