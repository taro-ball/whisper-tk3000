"""GPU and runtime detection logic.

This module contains pure functions for detecting Vulkan GPUs, CPU info,
and available whisper.cpp runtimes. No GUI or I/O dependencies except
subprocess for detection.
"""

from __future__ import annotations

import os
import re
import shutil
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .core_logic import build_auto_gpu_label, get_preferred_gpu_device

try:
    import ctypes
    from ctypes import wintypes
except ImportError:
    ctypes = None
    wintypes = None

try:
    import winreg
except ImportError:
    winreg = None

try:
    import platform as platform_module
except ImportError:
    platform_module = None


AUTO_GPU_LABEL = "Auto (best guess)"
GPU_RUNTIME_MISSING_LABEL = "GPU runtime missing"
GPU_DETECTION_FAILED_LABEL = "Vulkan detection failed"
GPU_NO_DEVICES_LABEL = "No Vulkan devices"
SLOW_CPU_MODEL_WARNING_THRESHOLD_BYTES = 150 * 1024 * 1024
MISSING_VULKAN_RUNTIME_MESSAGE = (
    "INFO: Hiding GPU options because no Vulkan runtime binary was found. "
    "Expected bin\\whisper.vulkan\\whisper-cli.exe or "
    "bin\\whisper.cpp\\whisper-cli.exe."
)
MISSING_VULKAN_BACKEND_MESSAGE = (
    "INFO: Hiding GPU options because the Vulkan binary is present, "
    "but no Vulkan devices were detected. GPU acceleration is "
    "unavailable on this machine."
)
MISSING_WHISPER_RUNTIME_MESSAGE = (
    "Missing whisper.cpp runtime. Expected "
    "bin\\whisper.vulkan\\whisper-cli.exe and/or "
    "bin\\whisper.cpu\\whisper-cli.exe."
)
FFMPEG_EXECUTABLE_NAME = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"


def discover_bundled_ffmpeg(bin_dir: Path) -> Path | None:
    bundled_path = bin_dir / FFMPEG_EXECUTABLE_NAME
    if bundled_path.exists():
        return bundled_path
    return None


def discover_ffmpeg_path(bin_dir: Path) -> Path | None:
    bundled_path = discover_bundled_ffmpeg(bin_dir)
    if bundled_path is not None:
        return bundled_path

    discovered_path = shutil.which("ffmpeg")
    if not discovered_path:
        return None
    return Path(discovered_path)


def build_missing_ffmpeg_message(
    input_path: Path,
    *,
    duration_seconds: int | None = None,
) -> str:
    if duration_seconds is not None:
        return (
            "FFmpeg is required to prepare benchmark audio. Install ffmpeg on PATH "
            "or include bin\\ffmpeg.exe."
        )

    input_type = input_path.suffix.lower() or "selected"
    return (
        f"FFmpeg is required to process {input_type} inputs. Install ffmpeg on PATH "
        "or include bin\\ffmpeg.exe."
    )


@dataclass(frozen=True)
class CpuExecutionPolicy:
    cpu_name: str
    cpu_option_label: str
    cpu_thread_count: int
    physical_core_count: int | None
    thread_count_log_message: str


@dataclass(frozen=True)
class GpuAvailability:
    status: str
    devices: list[dict[str, Any]]
    log_message: str | None
    note_label: str
    controls_enabled: bool


@dataclass(frozen=True)
class GpuSelectionState:
    runtimes: list[dict[str, Any]]
    runtime_lookup: dict[str, dict[str, Any]]
    devices: list[dict[str, Any]]
    options: dict[str, int | str | None]
    values: list[str]
    selected_value: str
    log_message: str | None
    note_label: str
    controls_enabled: bool
    cpu_option_label: str

# ============================================================================
# CPU Detection
# ============================================================================

def detect_physical_cpu_core_count() -> tuple[int | None, str | None]:
    """Detect the number of physical CPU cores on Windows.
    
    Returns:
        (core_count, error_message) where one is None
    """
    if os.name != "nt" or ctypes is None or wintypes is None:
        return None, "Physical core detection unavailable on this platform."

    relation_processor_core = 0
    error_insufficient_buffer = 122

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_logical_processor_information_ex = kernel32.GetLogicalProcessorInformationEx
        get_logical_processor_information_ex.argtypes = [
            wintypes.DWORD,
            ctypes.c_void_p,
            ctypes.POINTER(wintypes.DWORD),
        ]
        get_logical_processor_information_ex.restype = wintypes.BOOL

        buffer_size = wintypes.DWORD(0)
        get_logical_processor_information_ex(
            relation_processor_core,
            None,
            ctypes.byref(buffer_size),
        )
        if ctypes.get_last_error() != error_insufficient_buffer or buffer_size.value <= 0:
            raise OSError(f"buffer probe failed with Win32 error {ctypes.get_last_error()}")

        buffer = ctypes.create_string_buffer(buffer_size.value)
        if not get_logical_processor_information_ex(
            relation_processor_core,
            buffer,
            ctypes.byref(buffer_size),
        ):
            raise OSError(f"processor query failed with Win32 error {ctypes.get_last_error()}")

        core_count = 0
        offset = 0
        while offset < buffer_size.value:
            _, entry_size = struct.unpack_from("II", buffer, offset)
            if entry_size <= 0:
                raise ValueError("invalid processor info record size")
            core_count += 1
            offset += entry_size

        if core_count <= 0:
            raise ValueError("Windows returned zero physical cores")

        return core_count, None
    except (AttributeError, OSError, struct.error, ValueError) as exc:
        return None, f"Physical core detection failed: {exc}"


def detect_cpu_name() -> str:
    """Detect the CPU name from registry (Windows) or environment.
    
    Returns:
        CPU name string, or 'CPU' if detection fails
    """
    if winreg is not None:
        try:
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                name = " ".join(str(value).split())
                if name:
                    return name
        except OSError:
            pass

    candidates = [
        os.environ.get("PROCESSOR_IDENTIFIER", ""),
    ]
    
    if platform_module is not None:
        candidates.extend([
            platform_module.processor(),
            platform_module.uname().processor,
        ])
    
    for candidate in candidates:
        name = " ".join(str(candidate).split())
        if name:
            return name

    return "CPU"


# ============================================================================
# Device Name Formatting
# ============================================================================

def shorten_device_name(device_name: str) -> str:
    """Clean up device name by removing parenthetical and bracketed content.
    
    Args:
        device_name: Raw device name
    
    Returns:
        Shortened device name
    """
    cleaned = re.sub(r"\([^)]*\)|\[[^\]]*\]|\{[^}]*\}", " ", device_name)
    cleaned = " ".join(cleaned.split())
    cleaned = re.sub(r"\s+[a-z]+\b.*$", "", cleaned).strip(" -")
    return cleaned or " ".join(device_name.split()) or device_name


def shorten_cpu_name(cpu_name: str) -> str:
    """Shorten a CPU name for display.
    
    Args:
        cpu_name: Full CPU name
    
    Returns:
        Shortened CPU name
    """
    return shorten_device_name(cpu_name)


def shorten_gpu_name(gpu_name: str) -> str:
    """Shorten a GPU name for display.
    
    Args:
        gpu_name: Full GPU name
    
    Returns:
        Shortened GPU name
    """
    return shorten_device_name(gpu_name)


def detect_gpu_vendor_name(gpu_name: str) -> str | None:
    """Detect the GPU vendor from the device name.
    
    Args:
        gpu_name: GPU device name
    
    Returns:
        Vendor name like 'NVIDIA', 'AMD', etc., or None if unknown
    """
    normalized = f" {' '.join(gpu_name.split()).lower()} "
    vendor_patterns = (
        ("NVIDIA", ("nvidia", "geforce", "quadro", "tesla")),
        ("AMD", ("amd", "radeon", "ati")),
        ("Intel", ("intel", "iris", "uhd", "arc")),
        ("Qualcomm", ("qualcomm", "adreno")),
        ("Apple", ("apple",)),
        ("ARM", ("arm", "mali")),
        ("Imagination", ("imagination", "powervr")),
        ("Microsoft", ("microsoft", "warp")),
    )
    for vendor_name, patterns in vendor_patterns:
        if any(f" {pattern} " in normalized for pattern in patterns):
            return vendor_name
    return None


def build_gpu_vendors_payload_value(devices: list[dict[str, Any]]) -> str:
    """Build a comma-separated list of unique GPU vendors.
    
    Args:
        devices: List of GPU device dicts with 'name' key
    
    Returns:
        String like 'NVIDIA, AMD' or empty string
    """
    vendor_names: list[str] = []
    seen_vendor_names: set[str] = set()
    for device in devices:
        vendor_name = detect_gpu_vendor_name(str(device.get("name", "")))
        if vendor_name and vendor_name not in seen_vendor_names:
            seen_vendor_names.add(vendor_name)
            vendor_names.append(vendor_name)
    return ", ".join(vendor_names)


def build_cpu_option_label(cpu_name: str, cpu_thread_count: int, physical_core_count: int | None = None) -> str:
    """Build a label for the CPU option in the GPU dropdown.
    
    Args:
        cpu_name: Short CPU name
        cpu_thread_count: Number of threads to use
        physical_core_count: Number of physical cores, or None
    
    Returns:
        Label like 'CPU only - Intel Core i7 - 8 physical cores'
    """
    cpu_name = shorten_cpu_name(cpu_name)
    if physical_core_count is not None:
        core_label = "physical core" if physical_core_count == 1 else "physical cores"
        return f"CPU only - {cpu_name} - {physical_core_count} {core_label}"

    thread_label = "logical thread" if cpu_thread_count == 1 else "logical threads"
    return f"CPU only - {cpu_name} - {cpu_thread_count} {thread_label}"


def build_cpu_execution_policy() -> CpuExecutionPolicy:
    """Build the CPU-related execution policy used by UI and services."""
    cpu_name = detect_cpu_name()
    logical_thread_count = max(1, os.cpu_count() or 1)
    physical_core_count, fallback_message = detect_physical_cpu_core_count()
    cpu_thread_count = physical_core_count or logical_thread_count

    if physical_core_count is not None:
        thread_count_log_message = (
            f"Using {cpu_thread_count} physical core(s) for CPU thread count."
        )
    else:
        thread_count_log_message = (
            f"{fallback_message} Using {cpu_thread_count} logical thread(s)."
        )

    return CpuExecutionPolicy(
        cpu_name=cpu_name,
        cpu_option_label=build_cpu_option_label(
            cpu_name,
            cpu_thread_count,
            physical_core_count,
        ),
        cpu_thread_count=cpu_thread_count,
        physical_core_count=physical_core_count,
        thread_count_log_message=thread_count_log_message,
    )


# ============================================================================
# GPU Detection (Vulkan)
# ============================================================================

GPU_LINE_RE = re.compile(r"^ggml_vulkan:\s+(\d+)\s+=\s+(.*?)\s+\|\s+uma:\s+(\d+)\b")


def build_hidden_subprocess_kwargs() -> dict[str, Any]:
    """Build kwargs to hide subprocess window on Windows.
    
    Returns:
        Dict with 'creationflags' and 'startupinfo' on Windows, empty dict otherwise
    """
    WINDOWS_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    if WINDOWS_NO_WINDOW == 0:
        return {}

    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = 0
    return {
        "creationflags": WINDOWS_NO_WINDOW,
        "startupinfo": startupinfo,
    }


def detect_vulkan_devices(
    whisper_cli_path: Path,
) -> tuple[list[dict[str, Any]], str | None]:
    """Detect available Vulkan GPUs by parsing whisper-cli --help output.
    
    Args:
        whisper_cli_path: Path to whisper-cli executable
    
    Returns:
        (devices_list, error_message) where one is None.
        devices_list contains dicts with 'index', 'name', 'uma' keys
    """
    try:
        result = subprocess.run(
            [str(whisper_cli_path), "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
            **build_hidden_subprocess_kwargs(),
        )
    except OSError as exc:
        return [], f"WARNING: Could not detect Vulkan GPUs: {exc}"

    devices: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        match = GPU_LINE_RE.match(line.strip())
        if match is None:
            continue
        devices.append(
            {
                "index": int(match.group(1)),
                "name": shorten_gpu_name(match.group(2).strip()),
                "uma": int(match.group(3)),
            }
        )

    return devices, None


# ============================================================================
# Runtime Discovery
# ============================================================================

WHISPER_RUNTIME_CANDIDATES = (
    {
        "key": "vulkan",
        "folder": "whisper.vulkan",
        "label": "Vulkan",
        "supports_vulkan": True,
    },
    {
        "key": "cpu",
        "folder": "whisper.cpu",
        "label": "CPU",
        "supports_vulkan": False,
    },
    {
        "key": "legacy",
        "folder": "whisper.cpp",
        "label": "Legacy Vulkan",
        "supports_vulkan": True,
    },
)


def discover_whisper_runtimes(bin_dir: Path) -> list[dict[str, Any]]:
    """Discover available whisper.cpp runtimes in bin_dir.
    
    Args:
        bin_dir: Path to bin directory containing runtime folders
    
    Returns:
        List of runtime dicts with 'key', 'label', 'supports_vulkan',
        'dir', and 'cli_path' keys
    """
    runtimes: list[dict[str, Any]] = []
    for candidate in WHISPER_RUNTIME_CANDIDATES:
        runtime_dir = bin_dir / str(candidate["folder"])
        cli_path = runtime_dir / "whisper-cli.exe"
        if not cli_path.exists():
            continue
        runtimes.append(
            {
                "key": candidate["key"],
                "label": candidate["label"],
                "supports_vulkan": candidate["supports_vulkan"],
                "dir": runtime_dir,
                "cli_path": cli_path,
            }
        )
    return runtimes


def get_preferred_whisper_runtime(
    runtimes: list[dict[str, Any]],
    keys: tuple[str, ...],
    *,
    allow_fallback: bool = True,
) -> dict[str, Any] | None:
    """Select a preferred runtime from available runtimes.
    
    Args:
        runtimes: List of available runtime dicts
        keys: Tuple of keys to try in order ('vulkan', 'cpu', 'legacy', etc.)
        allow_fallback: If True, return first available runtime if no match in keys
    
    Returns:
        The selected runtime dict, or None
    """
    runtime_by_key = {str(rt["key"]): rt for rt in runtimes}
    
    for key in keys:
        if key in runtime_by_key:
            return runtime_by_key[key]
    
    if allow_fallback and runtimes:
        return runtimes[0]
    
    return None


def get_vulkan_gpu_availability(runtimes: list[dict[str, Any]]) -> GpuAvailability:
    """Describe whether GPU-backed Vulkan execution is available."""
    runtime = get_preferred_whisper_runtime(
        runtimes,
        ("vulkan", "legacy"),
        allow_fallback=False,
    )
    if runtime is None:
        return GpuAvailability(
            status="runtime_missing",
            devices=[],
            log_message=MISSING_VULKAN_RUNTIME_MESSAGE,
            note_label=GPU_RUNTIME_MISSING_LABEL,
            controls_enabled=False,
        )

    devices, detection_message = detect_vulkan_devices(Path(runtime["cli_path"]))
    if detection_message is not None:
        return GpuAvailability(
            status="detection_failed",
            devices=[],
            log_message=detection_message,
            note_label=GPU_DETECTION_FAILED_LABEL,
            controls_enabled=True,
        )
    if not devices:
        return GpuAvailability(
            status="no_devices",
            devices=[],
            log_message=MISSING_VULKAN_BACKEND_MESSAGE,
            note_label=GPU_NO_DEVICES_LABEL,
            controls_enabled=True,
        )

    return GpuAvailability(
        status="available",
        devices=devices,
        log_message=None,
        note_label="",
        controls_enabled=True,
    )


def load_gpu_selection_state(
    bin_dir: Path,
    current_selection: str,
    cpu_policy: CpuExecutionPolicy,
) -> GpuSelectionState:
    """Build the UI-facing GPU selection state from runtime policy."""
    runtimes = discover_whisper_runtimes(bin_dir)
    runtime_lookup = {str(runtime["key"]): runtime for runtime in runtimes}
    availability = get_vulkan_gpu_availability(runtimes)
    devices = list(availability.devices)
    values: list[str] = []
    options: dict[str, int | str | None] = {}

    if availability.status == "available" and devices:
        auto_label = build_auto_gpu_label(devices)
        values.append(auto_label)
        options[auto_label] = None

    if availability.status == "available":
        for display_index, device in enumerate(devices, start=1):
            label = f"GPU {display_index} - {device['name']}"
            values.append(label)
            options[label] = int(device["index"])

    values.append(cpu_policy.cpu_option_label)
    options[cpu_policy.cpu_option_label] = "cpu"

    if availability.status == "runtime_missing":
        selected_value = cpu_policy.cpu_option_label
    elif current_selection in options:
        selected_value = current_selection
    elif devices:
        selected_value = values[0]
    else:
        selected_value = cpu_policy.cpu_option_label

    return GpuSelectionState(
        runtimes=runtimes,
        runtime_lookup=runtime_lookup,
        devices=devices,
        options=options,
        values=values,
        selected_value=selected_value,
        log_message=availability.log_message,
        note_label=availability.note_label,
        controls_enabled=availability.controls_enabled,
        cpu_option_label=cpu_policy.cpu_option_label,
    )


def is_cpu_selection(
    selection_label: str,
    gpu_options: dict[str, int | str | None],
) -> bool:
    return gpu_options.get(selection_label) == "cpu"


def is_cpu_inference(
    selection_label: str,
    runtime: dict[str, Any],
    gpu_options: dict[str, int | str | None],
) -> bool:
    return is_cpu_selection(selection_label, gpu_options) or not bool(runtime["supports_vulkan"])


def resolve_whisper_runtime(
    bin_dir: Path,
    selection_label: str,
    gpu_options: dict[str, int | str | None],
) -> dict[str, Any]:
    runtimes = discover_whisper_runtimes(bin_dir)
    if is_cpu_selection(selection_label, gpu_options):
        runtime = get_preferred_whisper_runtime(
            runtimes,
            ("cpu", "vulkan", "legacy"),
        )
    else:
        runtime = get_preferred_whisper_runtime(
            runtimes,
            ("vulkan", "legacy", "cpu"),
        )

    if runtime is None:
        raise FileNotFoundError(MISSING_WHISPER_RUNTIME_MESSAGE)

    return runtime


def guess_best_gpu_index(devices: list[dict[str, Any]]) -> int | None:
    preferred_device = get_preferred_gpu_device(devices)
    if preferred_device is None:
        return None
    return int(preferred_device["index"])


def build_whisper_env(
    selection_label: str,
    runtime: dict[str, Any],
    gpu_options: dict[str, int | str | None],
    gpu_devices: list[dict[str, Any]],
) -> dict[str, str]:
    env = dict(os.environ)
    if not bool(runtime["supports_vulkan"]):
        return env

    selected_gpu = gpu_options.get(selection_label)
    if selected_gpu == "cpu":
        return env
    if selected_gpu is None:
        selected_gpu = guess_best_gpu_index(gpu_devices)
    if isinstance(selected_gpu, int):
        env["GGML_VK_VISIBLE_DEVICES"] = str(selected_gpu)
    return env


def build_cpu_inference_log_message(
    cpu_policy: CpuExecutionPolicy,
    selection_label: str,
    runtime: dict[str, Any],
    gpu_options: dict[str, int | str | None],
) -> str | None:
    if is_cpu_selection(selection_label, gpu_options):
        return cpu_policy.thread_count_log_message

    if not bool(runtime["supports_vulkan"]):
        return (
            "WARNING: Vulkan runtime not available. Falling back to CPU runtime. "
            f"{cpu_policy.thread_count_log_message}"
        )

    return None


def build_cpu_slow_warning(
    model_info: object,
    selection_label: str,
    runtime: dict[str, Any],
    gpu_options: dict[str, int | str | None],
) -> str | None:
    if not is_cpu_inference(selection_label, runtime, gpu_options):
        return None

    if not isinstance(model_info, dict):
        return None

    model_size_bytes = int(model_info["size_bytes"])
    if model_size_bytes <= SLOW_CPU_MODEL_WARNING_THRESHOLD_BYTES:
        return None

    model_size_label = str(model_info["size_label"])
    model_name = str(model_info["name"])
    return f"WARNING: CPU inference with model {model_name} [{model_size_label}] may be slow."


def build_benchmark_option_labels(
    gpu_options: dict[str, int | str | None],
    cpu_option_label: str,
) -> list[str]:
    labels = [cpu_option_label]
    labels.extend(
        label
        for label in gpu_options.keys()
        if label != cpu_option_label and not label.startswith(AUTO_GPU_LABEL)
    )
    return labels


