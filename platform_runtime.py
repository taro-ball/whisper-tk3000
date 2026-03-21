"""GPU and runtime detection logic.

This module contains pure functions for detecting Vulkan GPUs, CPU info,
and available whisper.cpp runtimes. No GUI or I/O dependencies except
subprocess for detection.
"""

from __future__ import annotations

import os
import re
import struct
import subprocess
from pathlib import Path
from typing import Any

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
