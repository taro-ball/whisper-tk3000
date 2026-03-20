from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import struct
import subprocess
import threading
import time
import urllib.request
import uuid
import webbrowser
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, ttk
from tkinter import messagebox

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


APP_DIR = Path(__file__).resolve().parent
APP_NAME = "whisper-tk3000"
APP_VERSION = "0.3.1"
APP_TITLE = "whisper-tk3000 audio to text transcriber"
TELEMETRY_APP_ID = "5FD59222-E42C-4491-AD54-9A8FA5088609"
TELEMETRY_NAMESPACE = "com.gr"
TELEMETRY_URL = f"https://nom.telemetrydeck.com/v2/namespace/{TELEMETRY_NAMESPACE}/"
BIN_DIR = APP_DIR / "bin"
FFMPEG_PATH = BIN_DIR / "ffmpeg.exe"
MODELS_DIR = APP_DIR / "models"
MODEL_REPO_URL = "https://huggingface.co/ggerganov/whisper.cpp/tree/main"
MEDIA_SUFFIXES = (
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".mp3",
    ".wav",
    ".m4a",
    ".flac",
    ".aac",
    ".ogg",
    ".webm",
)
MODEL_OPTIONS = [
    {
        "name": "ggml-base.en.bin",
        "size_label": "148 MB",
        "size_bytes": 148 * 1024 * 1024,
        "label": "balanced (english only)",
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin",
    },
    {
        "name": "ggml-large-v3-turbo.bin",
        "size_label": "1.62 GB",
        "size_bytes": int(1.62 * 1024 * 1024 * 1024),
        "label": "precise, multilingual, but slower",
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin",
    },
    {
        "name": "ggml-tiny.en.bin",
        "size_label": "77 MB",
        "size_bytes": 77 * 1024 * 1024,
        "label": "good enough, fast (english only)",
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin",
    },
]
MODEL_OPTIONS_BY_NAME = {str(option["name"]): option for option in MODEL_OPTIONS}
FORMAT_OPTIONS = {
    "srt subtitle": "srt",
    "plain text": "txt",
}
NO_MODELS_LABEL = "No models found"
SUPPORTED_MEDIA_TYPES = [
    ("Media files", " ".join(f"*{suffix}" for suffix in MEDIA_SUFFIXES)),
    ("All files", "*.*"),
]
WINDOWS_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
AUTO_GPU_LABEL = "Auto (best guess)"
GPU_RUNTIME_MISSING_LABEL = "GPU runtime missing"
GPU_DETECTION_FAILED_LABEL = "Vulkan detection failed"
GPU_NO_DEVICES_LABEL = "No Vulkan devices"
SLOW_CPU_MODEL_WARNING_THRESHOLD_BYTES = 150 * 1024 * 1024
GPU_LINE_RE = re.compile(r"^ggml_vulkan:\s+(\d+)\s+=\s+(.*?)\s+\|\s+uma:\s+(\d+)\b")
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


def detect_physical_cpu_core_count() -> tuple[int | None, str | None]:
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


CPU_LOGICAL_THREAD_COUNT = max(1, os.cpu_count() or 1)
CPU_PHYSICAL_CORE_COUNT, CPU_THREAD_FALLBACK_DEBUG_MESSAGE = detect_physical_cpu_core_count()
CPU_THREAD_COUNT = CPU_PHYSICAL_CORE_COUNT or CPU_LOGICAL_THREAD_COUNT
if CPU_PHYSICAL_CORE_COUNT is not None:
    CPU_THREAD_COUNT_LOG_MESSAGE = (
        f"Using {CPU_THREAD_COUNT} physical core(s) for CPU thread count."
    )
else:
    CPU_THREAD_COUNT_LOG_MESSAGE = (
        f"{CPU_THREAD_FALLBACK_DEBUG_MESSAGE} "
        f"Using {CPU_THREAD_COUNT} logical thread(s)."
    )


def build_hidden_subprocess_kwargs() -> dict[str, object]:
    if WINDOWS_NO_WINDOW == 0:
        return {}

    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = 0
    return {
        "creationflags": WINDOWS_NO_WINDOW,
        "startupinfo": startupinfo,
    }


def detect_cpu_name() -> str:
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

    for candidate in (
        os.environ.get("PROCESSOR_IDENTIFIER", ""),
        platform.processor(),
        platform.uname().processor,
    ):
        name = " ".join(str(candidate).split())
        if name:
            return name

    return "CPU"


def shorten_device_name(device_name: str) -> str:
    cleaned = re.sub(r"\([^)]*\)|\[[^\]]*\]|\{[^}]*\}", " ", device_name)
    cleaned = " ".join(cleaned.split())
    cleaned = re.sub(r"\s+[a-z]+\b.*$", "", cleaned).strip(" -")
    return cleaned or " ".join(device_name.split()) or device_name


def shorten_cpu_name(cpu_name: str) -> str:
    return shorten_device_name(cpu_name)


def shorten_gpu_name(gpu_name: str) -> str:
    return shorten_device_name(gpu_name)


def detect_gpu_vendor_name(gpu_name: str) -> str | None:
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


def build_gpu_vendors_payload_value(devices: list[dict[str, object]]) -> str:
    vendor_names: list[str] = []
    seen_vendor_names: set[str] = set()
    for device in devices:
        vendor_name = detect_gpu_vendor_name(str(device.get("name", "")))
        if vendor_name and vendor_name not in seen_vendor_names:
            seen_vendor_names.add(vendor_name)
            vendor_names.append(vendor_name)
    return ", ".join(vendor_names)


def build_cpu_option_label(cpu_name: str) -> str:
    cpu_name = shorten_cpu_name(cpu_name)
    if CPU_PHYSICAL_CORE_COUNT is not None:
        core_label = "physical core" if CPU_THREAD_COUNT == 1 else "physical cores"
        return f"CPU only - {cpu_name} - {CPU_THREAD_COUNT} {core_label}"

    thread_label = "logical thread" if CPU_THREAD_COUNT == 1 else "logical threads"
    return f"CPU only - {cpu_name} - {CPU_THREAD_COUNT} {thread_label}"


class TranscriptionCancelled(Exception):
    pass


class App(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("green")

        self.title(APP_NAME)
        self.geometry("920x680")
        self.minsize(840, 620)

        self.output_queue: queue.Queue[str] = queue.Queue()
        self.is_running = False
        self.transcription_running = False
        self.cancel_requested = False
        self.current_process: subprocess.Popen[str] | None = None
        self.process_lock = threading.Lock()

        self.input_path_var = tk.StringVar()
        self.format_var = tk.StringVar(value="srt subtitle")
        self.model_var = tk.StringVar()
        self.gpu_var = tk.StringVar(value=AUTO_GPU_LABEL)
        self.prompt_var = tk.StringVar()
        self.debug_var = tk.BooleanVar(value=False)
        self.download_model_var = tk.StringVar(value=MODEL_OPTIONS[0]["name"])
        self.available_models: list[dict[str, object]] = []
        self.available_models_by_name: dict[str, dict[str, object]] = {}
        self.model_display_lookup: dict[str, str] = {}
        self.latest_result_path: Path | None = None
        self.download_dialog: ctk.CTkToplevel | None = None
        self.batch_dialog: ctk.CTkToplevel | None = None
        self.batch_folder_var = tk.StringVar()
        self.batch_selected_files: list[Path] = []
        self.batch_file_rows: list[Path] = []
        self.batch_tree: ttk.Treeview | None = None
        self.download_telemetry_sent = False
        self.telemetry_session_id = str(uuid.uuid4())
        self._suspend_input_path_tracking = False
        self._is_closing = False
        self.cpu_name = detect_cpu_name()
        self.cpu_option_label = build_cpu_option_label(self.cpu_name)
        self.gpu_devices: list[dict[str, object]] = []
        self.gpu_options: dict[str, int | str | None] = {AUTO_GPU_LABEL: None, self.cpu_option_label: "cpu"}
        self.gpu_controls_enabled = True
        self.whisper_runtimes: list[dict[str, object]] = []
        self.whisper_runtime_lookup: dict[str, dict[str, object]] = {}

        self.input_path_var.trace_add("write", self._on_input_path_changed)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        self.header_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.header_frame.grid(row=0, column=0, padx=16, pady=(16, 8), sticky="ew")
        self.header_frame.grid_columnconfigure(0, weight=1)

        self.header_label = ctk.CTkLabel(
            self.header_frame,
            text=APP_TITLE,
            font=ctk.CTkFont(size=26, weight="bold"),
        )
        self.header_label.grid(row=0, column=0, padx=0, pady=0, sticky="w")

        self.help_button = ctk.CTkButton(
            self.header_frame,
            text="?",
            width=1,
            height=22,
            corner_radius=12,
            command=self.show_about_dialog,
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.help_button.grid(row=0, column=1, padx=(8, 0), pady=0, sticky="e")

        self.controls_frame = ctk.CTkFrame(self)
        self.controls_frame.grid(row=1, column=0, padx=16, pady=(0, 12), sticky="ew")
        self.controls_frame.grid_columnconfigure(1, weight=1)

        self._build_controls()

        self.console = ctk.CTkTextbox(self, wrap="word", font=("Consolas", 13))
        self.console.grid(row=2, column=0, padx=16, pady=(0, 16), sticky="nsew")
        self.console.configure(state="disabled")

        self.reload_models()
        self.reload_gpu_options()
        self.after(100, self.flush_output)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_controls(self) -> None:
        row = 0

        ctk.CTkLabel(self.controls_frame, text="Input file").grid(
            row=row, column=0, padx=12, pady=(12, 6), sticky="w"
        )
        self.input_entry = ctk.CTkEntry(
            self.controls_frame,
            textvariable=self.input_path_var,
            placeholder_text="Select one audio or video file",
        )
        self.input_entry.grid(row=row, column=1, padx=12, pady=(12, 6), sticky="ew")
        self.browse_button = ctk.CTkButton(
            self.controls_frame, text="Browse", width=100, command=self.select_input_file
        )
        self.browse_button.grid(row=row, column=2, padx=(0, 12), pady=(12, 6), sticky="e")
        self.batch_button = ctk.CTkButton(
            self.controls_frame, text="Batch", width=100, command=self.open_batch_dialog
        )
        self.batch_button.grid(row=row, column=3, padx=(0, 12), pady=(12, 6), sticky="e")

        row += 1
        ctk.CTkLabel(self.controls_frame, text="Output format").grid(
            row=row, column=0, padx=12, pady=6, sticky="w"
        )
        self.format_menu = ctk.CTkOptionMenu(
            self.controls_frame,
            values=list(FORMAT_OPTIONS.keys()),
            variable=self.format_var,
        )
        self.format_menu.grid(row=row, column=1, padx=12, pady=6, sticky="w")

        row += 1
        ctk.CTkLabel(self.controls_frame, text="Whisper model").grid(
            row=row, column=0, padx=12, pady=6, sticky="w"
        )
        self.model_menu = ctk.CTkOptionMenu(
            self.controls_frame,
            values=[NO_MODELS_LABEL],
            variable=self.model_var,
        )
        self.model_menu.grid(row=row, column=1, padx=12, pady=6, sticky="ew")
        self.refresh_models_button = ctk.CTkButton(
            self.controls_frame, text="Refresh", width=100, command=self.reload_models
        )
        self.refresh_models_button.grid(row=row, column=2, padx=(0, 12), pady=6, sticky="e")
        self.download_model_button = ctk.CTkButton(
            self.controls_frame,
            text="Download",
            width=100,
            command=self.show_download_dialog,
        )
        self.download_model_button.grid(row=row, column=3, padx=(0, 12), pady=6, sticky="e")

        row += 1
        ctk.CTkLabel(self.controls_frame, text="GPU").grid(
            row=row, column=0, padx=12, pady=6, sticky="w"
        )
        self.gpu_menu = ctk.CTkOptionMenu(
            self.controls_frame,
            values=[AUTO_GPU_LABEL],
            variable=self.gpu_var,
        )
        self.gpu_menu.grid(row=row, column=1, padx=12, pady=6, sticky="ew")
        self.gpu_note_label = ctk.CTkLabel(
            self.controls_frame,
            text="",
            font=ctk.CTkFont(size=12),
            text_color=("gray40", "gray70"),
        )
        self.gpu_note_label.grid(row=row, column=3, padx=(0, 12), pady=6, sticky="w")
        self.refresh_gpu_button = ctk.CTkButton(
            self.controls_frame,
            text="Benchmark",
            width=100,
            command=self.run_benchmark,
        )
        self.refresh_gpu_button.grid(row=row, column=2, padx=(0, 12), pady=6, sticky="e")

        row += 1
        ctk.CTkLabel(self.controls_frame, text="Initial prompt").grid(
            row=row, column=0, padx=12, pady=6, sticky="nw"
        )
        self.prompt_entry = ctk.CTkEntry(
            self.controls_frame,
            textvariable=self.prompt_var,
            placeholder_text="Optional --prompt text",
        )
        self.prompt_entry.grid(row=row, column=1, columnspan=2, padx=12, pady=6, sticky="ew")

        row += 1
        self.run_button = ctk.CTkButton(
            self.controls_frame,
            text="Transcribe",
            command=self.run_transcription,
            width=200,
        )
        self.run_button.grid(row=row, column=0, padx=12, pady=(8, 12), sticky="w")

        self.result_row = ctk.CTkFrame(self.controls_frame, fg_color="transparent")
        self.result_row.grid(row=row, column=1, columnspan=3, padx=12, pady=(8, 12), sticky="ew")
        self.result_row.grid_columnconfigure(1, weight=1)
        self.result_row.grid_columnconfigure(2, weight=0)
        self.result_label = ctk.CTkLabel(self.result_row, text="Transcribed text:")
        self.result_label.grid(row=0, column=0, padx=(0, 8), pady=0, sticky="w")
        self.result_link_font = ctk.CTkFont(underline=True)
        self.result_link_disabled_font = ctk.CTkFont(underline=False)
        self.result_link_button = ctk.CTkButton(
            self.result_row,
            text="No result yet",
            command=self.reveal_result_file,
            fg_color="transparent",
            text_color=("blue", "#7fb3ff"),
            hover_color=self.controls_frame.cget("fg_color"),
            anchor="w",
            width=0,
            border_width=0,
            corner_radius=0,
            font=self.result_link_font,
            cursor="hand2",
        )
        self.result_link_button.grid(row=0, column=1, padx=0, pady=0, sticky="w")
        self.debug_checkbox = ctk.CTkCheckBox(
            self.result_row,
            text="debug",
            variable=self.debug_var,
            onvalue=True,
            offvalue=False,
            width=56,
            checkbox_width=14,
            checkbox_height=14,
            border_width=2,
            font=ctk.CTkFont(size=12),
        )
        self.debug_checkbox.grid(row=0, column=2, padx=(12, 0), pady=0, sticky="e")
        self._set_result_path(None)

    def append_output(self, text: str) -> None:
        self.console.configure(state="normal")
        self.console.insert("end", text)
        self.console.see("end")
        self.console.configure(state="disabled")

    def flush_output(self) -> None:
        while not self.output_queue.empty():
            self.append_output(self.output_queue.get_nowait())
        self.after(100, self.flush_output)

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.output_queue.put(f"[{timestamp}] {message}\n")

    def select_input_file(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select input media file",
            filetypes=SUPPORTED_MEDIA_TYPES,
        )
        if selected:
            self.batch_selected_files = []
            self._set_input_path_text(selected)

    def open_batch_dialog(self) -> None:
        if self.is_running:
            return

        folder = filedialog.askdirectory(title="Select folder with media files")
        if not folder:
            return

        media_files = self._find_media_files(Path(folder))
        if not media_files:
            self.log(f"No supported media files found in {folder}")
            return

        if self.batch_dialog is not None and self.batch_dialog.winfo_exists():
            self.batch_dialog.destroy()

        dialog = ctk.CTkToplevel(self)
        dialog.title("Batch Selection")
        dialog.geometry("760x480")
        dialog.minsize(680, 420)
        dialog.transient(self)
        dialog.grab_set()
        dialog.grid_columnconfigure(0, weight=1)
        dialog.grid_rowconfigure(2, weight=1)
        self.batch_dialog = dialog
        self.batch_folder_var.set(folder)
        self.batch_file_rows = media_files

        preselected = set(self.batch_selected_files)
        selected_files = {path for path in media_files if path in preselected}
        if not selected_files:
            selected_files = set(media_files)

        ctk.CTkLabel(
            dialog,
            text="Choose files to process",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).grid(row=0, column=0, padx=20, pady=(20, 8), sticky="w")

        ctk.CTkLabel(
            dialog,
            textvariable=self.batch_folder_var,
            justify="left",
            wraplength=700,
        ).grid(row=1, column=0, padx=20, pady=(0, 12), sticky="w")

        table_frame = ctk.CTkFrame(dialog)
        table_frame.grid(row=2, column=0, padx=20, pady=0, sticky="nsew")
        table_frame.grid_columnconfigure(0, weight=1)
        table_frame.grid_rowconfigure(0, weight=1)

        columns = ("selected", "name", "type")
        tree = ttk.Treeview(table_frame, columns=columns, show="headings", selectmode="none")
        tree.heading("selected", text="Use")
        tree.heading("name", text="File")
        tree.heading("type", text="Type")
        tree.column("selected", width=70, anchor="center", stretch=False)
        tree.column("name", width=500, anchor="w")
        tree.column("type", width=110, anchor="w", stretch=False)

        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        for index, path in enumerate(media_files):
            checked = "[x]" if path in selected_files else "[ ]"
            tree.insert("", "end", iid=str(index), values=(checked, path.name, path.suffix.lower()))

        tree.bind("<Button-1>", self._on_batch_tree_click)
        tree.bind("<space>", self._on_batch_tree_space)
        self.batch_tree = tree

        actions_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        actions_frame.grid(row=3, column=0, padx=20, pady=(16, 20), sticky="ew")
        actions_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkButton(
            actions_frame,
            text="Select all",
            width=100,
            command=lambda: self._set_all_batch_rows(True),
        ).grid(row=0, column=0, padx=(0, 12), pady=0, sticky="w")
        ctk.CTkButton(
            actions_frame,
            text="Clear",
            width=100,
            command=lambda: self._set_all_batch_rows(False),
        ).grid(row=0, column=1, padx=(0, 12), pady=0, sticky="w")
        ctk.CTkButton(
            actions_frame,
            text="Cancel",
            width=100,
            command=self.close_batch_dialog,
        ).grid(row=0, column=2, padx=(12, 12), pady=0, sticky="e")
        ctk.CTkButton(
            actions_frame,
            text="Use selected",
            width=120,
            command=self.apply_batch_selection,
        ).grid(row=0, column=3, padx=(0, 0), pady=0, sticky="e")

        dialog.protocol("WM_DELETE_WINDOW", self.close_batch_dialog)

    def close_batch_dialog(self) -> None:
        if self.batch_dialog is not None and self.batch_dialog.winfo_exists():
            self.batch_dialog.destroy()
        self.batch_dialog = None
        self.batch_tree = None

    def apply_batch_selection(self) -> None:
        selected = self._get_checked_batch_files()
        if not selected:
            self.log("No batch files selected.")
            return

        self.batch_selected_files = selected
        if len(selected) == 1:
            self._set_input_path_text(str(selected[0]))
        else:
            self._set_input_path_text(f"{len(selected)} files selected from {selected[0].parent}")
        self.close_batch_dialog()

    def _find_media_files(self, folder: Path) -> list[Path]:
        media_files = [
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in MEDIA_SUFFIXES
        ]
        return sorted(media_files, key=lambda path: path.name.lower())

    def _on_batch_tree_click(self, event: tk.Event) -> str | None:
        if self.batch_tree is None:
            return None
        region = self.batch_tree.identify("region", event.x, event.y)
        column = self.batch_tree.identify_column(event.x)
        row_id = self.batch_tree.identify_row(event.y)
        if region == "cell" and column == "#1" and row_id:
            self._toggle_batch_row(row_id)
            return "break"
        return None

    def _on_batch_tree_space(self, _event: tk.Event) -> str | None:
        if self.batch_tree is None:
            return None
        selected_item = self.batch_tree.focus()
        if selected_item:
            self._toggle_batch_row(selected_item)
            return "break"
        return None

    def _toggle_batch_row(self, row_id: str) -> None:
        if self.batch_tree is None:
            return
        values = list(self.batch_tree.item(row_id, "values"))
        if not values:
            return
        values[0] = "[ ]" if values[0] == "[x]" else "[x]"
        self.batch_tree.item(row_id, values=values)

    def _set_all_batch_rows(self, checked: bool) -> None:
        if self.batch_tree is None:
            return
        marker = "[x]" if checked else "[ ]"
        for row_id in self.batch_tree.get_children():
            values = list(self.batch_tree.item(row_id, "values"))
            values[0] = marker
            self.batch_tree.item(row_id, values=values)

    def _get_checked_batch_files(self) -> list[Path]:
        if self.batch_tree is None:
            return list(self.batch_selected_files)
        selected: list[Path] = []
        for row_id in self.batch_tree.get_children():
            values = self.batch_tree.item(row_id, "values")
            if values and values[0] == "[x]":
                selected.append(self.batch_file_rows[int(row_id)])
        return selected

    def _set_input_path_text(self, value: str) -> None:
        self._suspend_input_path_tracking = True
        try:
            self.input_path_var.set(value)
        finally:
            self._suspend_input_path_tracking = False

    def _on_input_path_changed(self, *_args: object) -> None:
        if self._suspend_input_path_tracking:
            return
        if self.batch_selected_files:
            self.batch_selected_files = []

    def reload_models(self) -> None:
        current_name = self._get_selected_model_name()
        available_models: list[dict[str, object]] = []
        for path in sorted(MODELS_DIR.glob("*.bin"), key=lambda candidate: candidate.name.lower()):
            try:
                size_bytes = path.stat().st_size
            except OSError:
                continue
            available_models.append(
                {
                    "name": path.name,
                    "path": path,
                    "size_bytes": size_bytes,
                    "size_label": self._format_model_size_label(size_bytes),
                }
            )

        self.available_models = available_models
        self.available_models_by_name = {str(model["name"]): model for model in available_models}
        self.model_display_lookup = {}

        if not available_models:
            self.model_menu.configure(values=[NO_MODELS_LABEL])
            self.model_var.set(NO_MODELS_LABEL)
            return

        model_labels = [self._build_model_display_label(model) for model in available_models]
        self.model_display_lookup = {
            self._build_model_display_label(model): str(model["name"]) for model in available_models
        }
        self.model_menu.configure(values=model_labels)

        if current_name and current_name in self.available_models_by_name:
            self._set_selected_model_name(current_name)
        else:
            self._set_selected_model_name(str(available_models[0]["name"]))

    def _build_model_display_label(self, model: dict[str, object]) -> str:
        return f"{model['name']} [{model['size_label']}]"

    def _get_selected_model_name(self) -> str:
        selected_value = self.model_var.get().strip()
        if not selected_value or selected_value == NO_MODELS_LABEL:
            return selected_value
        return self.model_display_lookup.get(selected_value, selected_value)

    def _set_selected_model_name(self, model_name: str) -> None:
        model = self.available_models_by_name.get(model_name)
        if model is None:
            self.model_var.set(model_name)
            return
        self.model_var.set(self._build_model_display_label(model))

    def _get_model_info(self, model_name: str) -> dict[str, object] | None:
        return self.available_models_by_name.get(model_name)

    def reload_gpu_options(self) -> None:
        self.reload_whisper_runtimes()
        gpu_availability = self._get_vulkan_gpu_availability()
        devices = list(gpu_availability["devices"])
        values: list[str] = []
        options: dict[str, int | str | None] = {}

        if gpu_availability["status"] == "available" and devices:
            auto_label = self._build_auto_gpu_label(devices)
            values.append(auto_label)
            options[auto_label] = None

        if gpu_availability["status"] == "available":
            for display_index, device in enumerate(devices, start=1):
                label = f"GPU {display_index} - {device['name']}"
                values.append(label)
                options[label] = int(device["index"])

        values.append(self.cpu_option_label)
        options[self.cpu_option_label] = "cpu"

        current = self.gpu_var.get()
        self.gpu_devices = devices
        self.gpu_options = options
        self.gpu_menu.configure(values=values)
        self.gpu_controls_enabled = bool(gpu_availability["controls_enabled"])
        log_message = str(gpu_availability["log_message"] or "").strip()
        if log_message:
            self.log(log_message)
        if gpu_availability["status"] == "runtime_missing":
            self.gpu_var.set(self.cpu_option_label)
        elif current in options:
            self.gpu_var.set(current)
        elif devices:
            self.gpu_var.set(values[0])
        else:
            self.gpu_var.set(self.cpu_option_label)
        self.gpu_note_label.configure(text=str(gpu_availability["note_label"]))
        self._sync_gpu_controls_state()

    def _sync_gpu_controls_state(self) -> None:
        if self.is_running:
            gpu_menu_state = "disabled"
            benchmark_state = "disabled"
        elif self.gpu_controls_enabled:
            gpu_menu_state = "normal"
            benchmark_state = "normal"
        else:
            gpu_menu_state = "disabled"
            benchmark_state = "disabled"
        self.gpu_menu.configure(state=gpu_menu_state)
        self.refresh_gpu_button.configure(state=benchmark_state)

    def set_running_state(self, running: bool) -> None:
        self.is_running = running
        state = "disabled" if running else "normal"
        if self.transcription_running:
            self.run_button.configure(text="Cancel", command=self.cancel_transcription, state="normal")
        else:
            self.run_button.configure(
                text="Transcribe",
                command=self.run_transcription,
                state=state,
            )
        for widget in (
            self.browse_button,
            self.batch_button,
            self.refresh_models_button,
            self.download_model_button,
            self.format_menu,
            self.model_menu,
            self.prompt_entry,
            self.input_entry,
            self.debug_checkbox,
        ):
            widget.configure(state=state)
        self._sync_gpu_controls_state()

    def show_download_dialog(self) -> None:
        if self.is_running:
            return

        if self.download_dialog is not None and self.download_dialog.winfo_exists():
            self.download_dialog.focus()
            return

        dialog = ctk.CTkToplevel(self)
        dialog.title("Download Whisper Model")
        dialog.geometry("520x260")
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.grab_set()
        dialog.grid_columnconfigure(0, weight=1)
        self.download_dialog = dialog

        ctk.CTkLabel(
            dialog,
            text="Choose a model to download",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).grid(row=0, column=0, padx=20, pady=(20, 12), sticky="w")

        options_frame = ctk.CTkFrame(dialog)
        options_frame.grid(row=1, column=0, padx=20, pady=0, sticky="ew")
        options_frame.grid_columnconfigure(0, weight=1)

        for index, option in enumerate(MODEL_OPTIONS):
            text = f"{option['name']} [{option['size_label']}] - {option['label']}"
            ctk.CTkRadioButton(
                options_frame,
                text=text,
                variable=self.download_model_var,
                value=option["name"],
            ).grid(row=index, column=0, padx=16, pady=8, sticky="w")

        actions_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        actions_frame.grid(row=2, column=0, padx=20, pady=(16, 20), sticky="ew")
        actions_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkButton(
            actions_frame,
            text="More models - Download manually.",
            command=self.open_manual_download_page,
            fg_color="transparent",
            text_color=("blue", "#7fb3ff"),
            hover_color=self.controls_frame.cget("fg_color"),
            border_width=0,
        ).grid(row=0, column=0, padx=0, pady=0, sticky="w")

        ctk.CTkButton(
            actions_frame,
            text="Cancel",
            width=100,
            command=self.close_download_dialog,
        ).grid(row=0, column=1, padx=(12, 12), pady=0, sticky="e")
        ctk.CTkButton(
            actions_frame,
            text="Download",
            width=100,
            command=self.download_selected_model,
        ).grid(row=0, column=2, padx=(0, 0), pady=0, sticky="e")

        dialog.protocol("WM_DELETE_WINDOW", self.close_download_dialog)

    def close_download_dialog(self) -> None:
        if self.download_dialog is not None and self.download_dialog.winfo_exists():
            self.download_dialog.destroy()
        self.download_dialog = None

    def show_about_dialog(self) -> None:
        about_text = (
            f"{APP_NAME}\n"
            f"Version {APP_VERSION}\n"
            "https://github.com/taro-ball/whisper-tk3000\n\n"
            "Desktop app for transcribing audio to text with whisper.cpp.\n"
            "Dedicated to my wife, who's love for science always inspires me.\n\n"
            "Credits\n"
            "- whisper.cpp project: \n   https://github.com/ggml-org/whisper.cpp\n"
            "- ffmpeg for media conversion: \n   https://ffmpeg.org/about.html\n"
            f"- Model downloads from the whisper.cpp Hugging Face repository: \n   {MODEL_REPO_URL}\n\n"
        )
        dialog = ctk.CTkToplevel(self)
        dialog.title(f"About {APP_NAME}")
        dialog.geometry("520x240")
        dialog.transient(self)
        dialog.grab_set()
        text = ctk.CTkTextbox(dialog, wrap="word")
        text.pack(fill="both", expand=True, padx=12, pady=12)
        text.insert("1.0", about_text)
        text.configure(state="disabled")

    def open_manual_download_page(self) -> None:
        self._open_url(MODEL_REPO_URL, f"Opened manual download page: {MODEL_REPO_URL}")

    def download_selected_model(self) -> None:
        if self.is_running:
            return

        selected_name = self.download_model_var.get().strip()
        selected_option = MODEL_OPTIONS_BY_NAME.get(selected_name)
        if selected_option is None:
            self.log("ERROR: No download model selected.")
            return

        self._send_download_telemetry_once()
        self.close_download_dialog()
        self.set_running_state(True)
        worker = threading.Thread(
            target=self._download_model,
            args=(selected_option,),
            daemon=True,
        )
        worker.start()

    def _download_model(self, model_option: dict[str, str | int]) -> None:
        destination = MODELS_DIR / str(model_option["name"])
        temp_destination = destination.with_suffix(destination.suffix + ".part")
        try:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            self.log(f"Downloading model from {model_option['url']}")
            self.log(f"Saving model to {destination}")
            self._download_file(str(model_option["url"]), temp_destination)
            temp_destination.replace(destination)
            self.log(f"Model download complete: {destination.name}")
            self._schedule_ui_update(self.reload_models)
            self._schedule_ui_update(lambda: self._set_selected_model_name(destination.name))
        except Exception as exc:
            self.log(f"ERROR: Failed to download model: {exc}")
            if temp_destination.exists():
                try:
                    temp_destination.unlink()
                except OSError:
                    pass
        finally:
            self._schedule_ui_update(lambda: self.set_running_state(False))

    def run_transcription(self) -> None:
        if self.is_running:
            return

        self._set_result_path(None)
        self.cancel_requested = False
        self.transcription_running = True
        self.set_running_state(True)
        worker = threading.Thread(target=self._execute_transcription, daemon=True)
        worker.start()

    def run_benchmark(self) -> None:
        if self.is_running:
            return

        self.reload_gpu_options()
        self._set_result_path(None)
        self.set_running_state(True)
        worker = threading.Thread(target=self._execute_benchmark, daemon=True)
        worker.start()

    def cancel_transcription(self) -> None:
        if not self.transcription_running:
            return

        should_cancel = messagebox.askyesno(
            "Cancel transcription",
            "Stop the current transcription job?",
            parent=self,
        )
        if not should_cancel:
            return

        if not self.cancel_requested:
            self.cancel_requested = True
            self.log("Cancellation requested. Stopping the current process...")

        with self.process_lock:
            process = self.current_process

        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except OSError as exc:
                self.log(f"WARNING: Failed to terminate the current process: {exc}")

    def on_close(self) -> None:
        self._is_closing = True
        with self.process_lock:
            process = self.current_process

        if process is not None and process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass

        self.destroy()

    def _execute_transcription(self) -> None:
        should_show_batch_progress = len(self.batch_selected_files) > 1
        debug_enabled = self.debug_var.get()
        cpu_speed_warning_logged = False
        try:
            configs = self._build_run_configs()
            total = len(configs)
            last_output: Path | None = None
            self.log(CPU_THREAD_COUNT_LOG_MESSAGE)

            if should_show_batch_progress:
                self._schedule_ui_update(lambda: self._show_batch_progress(1, total))

            for index, config in enumerate(configs, start=1):
                self._raise_if_cancelled()
                if total > 1:
                    self.log(f"========================= Batch item {index} of {total}")
                self.log(f"========================= processing {config['input_path'].name} =========================\n")
                if debug_enabled:
                    self.log(f"Selected output format: {config['format'].upper()}")
                    self.log(f"Selected model: {config['model_path'].name}")
                try:
                    self._convert_input_to_audio(config, debug_enabled=debug_enabled)
                    self._raise_if_cancelled()

                    selection_label = self.gpu_var.get().strip()
                    whisper_runtime = self._resolve_whisper_runtime(selection_label)
                    whisper_env = self._build_whisper_env(selection_label, whisper_runtime)
                    if debug_enabled or index == 1:
                        self.log(
                            f"Using whisper.cpp runtime: {whisper_runtime['label']} "
                            f"({Path(whisper_runtime['cli_path']).parent.name})"
                        )
                    if not self._is_cpu_selection(selection_label) and not bool(whisper_runtime["supports_vulkan"]):
                        self.log("WARNING: Vulkan runtime not available. Falling back to CPU runtime.")
                    if not cpu_speed_warning_logged:
                        cpu_speed_warning_logged = self._warn_if_cpu_inference_may_be_slow(
                            config.get("model_info"),
                            selection_label,
                            whisper_runtime,
                        )
                    whisper_command = self._build_whisper_command(
                        config,
                        selection_label,
                        whisper_runtime,
                        debug_enabled=debug_enabled,
                    )
                    self._run_process(
                        whisper_command,
                        "whisper.cpp",
                        log_details=debug_enabled,
                        env=whisper_env,
                    )

                    last_output = config["transcript_output"]
                    self.log(f"Success. Output file: {config['transcript_output']}")
                    if should_show_batch_progress:
                        self._schedule_ui_update(
                            lambda completed=index, count=total: self._show_batch_progress(completed, count)
                        )
                finally:
                    self._cleanup_audio_output(config["audio_output"], log_removal=debug_enabled)

            self._schedule_ui_update(lambda: self._set_result_path(last_output))
        except TranscriptionCancelled:
            self.log("Transcription cancelled.")
            self._schedule_ui_update(lambda: self._set_result_path(None))
        except Exception as exc:
            self.log(f"ERROR: {exc}")
            self._schedule_ui_update(lambda: self._set_result_path(None))
        finally:
            self.transcription_running = False
            self.cancel_requested = False
            if should_show_batch_progress:
                self._schedule_ui_update(self._restore_batch_input_summary)
            self._schedule_ui_update(lambda: self.set_running_state(False))

    def _execute_benchmark(self) -> None:
        audio_output: Path | None = None
        benchmark_outputs: list[Path] = []
        debug_enabled = self.debug_var.get()
        try:
            configs = self._build_run_configs()
            config = configs[0]
            input_path = Path(config["input_path"])
            model_path = Path(config["model_path"])
            prompt = str(config["prompt"])
            timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
            audio_output = self._build_unique_output_path(
                input_path,
                ".wav",
                stem=f"{input_path.stem}{timestamp}.benchmark",
            )

            self.log(f"Benchmarking first 2 minutes of {input_path.name}")
            self.log(CPU_THREAD_COUNT_LOG_MESSAGE)
            self.log(f"Selected model: {model_path.name}")
            self._convert_input_to_audio(
                config,
                audio_output=audio_output,
                duration_seconds=120,
                debug_enabled=debug_enabled,
            )

            option_labels = [self.cpu_option_label]
            option_labels.extend(
                label
                for label in self.gpu_options.keys()
                if label != self.cpu_option_label and not label.startswith(AUTO_GPU_LABEL)
            )
            for label in option_labels:
                transcript_output = self._build_unique_output_path(
                    input_path,
                    ".txt",
                    stem=f"{input_path.stem}{timestamp}.{self._slugify_label(label)}.benchmark",
                )
                output_base = transcript_output.with_suffix("")
                benchmark_outputs.append(transcript_output)

                whisper_runtime = self._resolve_whisper_runtime(label)
                whisper_env = self._build_whisper_env(label, whisper_runtime)
                whisper_command = self._build_whisper_command(
                    {
                        "model_path": model_path,
                        "audio_output": audio_output,
                        "output_base": output_base,
                        "prompt": prompt,
                        "format": "txt",
                    },
                    label,
                    whisper_runtime,
                    debug_enabled=True,
                    force_txt=True,
                )
                self.log(
                    f"Benchmarking {label} with {whisper_runtime['label']} "
                    f"({Path(whisper_runtime['cli_path']).parent.name})"
                )
                self._warn_if_cpu_inference_may_be_slow(config.get("model_info"), label, whisper_runtime)
                elapsed_seconds = self._run_benchmark_process(whisper_command, whisper_env, label)
                self.log(f"{elapsed_seconds:.2f} seconds")
        except Exception as exc:
            self.log(f"ERROR: {exc}")
        finally:
            if audio_output is not None:
                self._cleanup_audio_output(audio_output, log_removal=debug_enabled)
            for output_path in benchmark_outputs:
                if output_path.exists():
                    try:
                        output_path.unlink()
                    except OSError:
                        pass
            self._schedule_ui_update(lambda: self.set_running_state(False))

    def _build_run_configs(self) -> list[dict[str, Path | str]]:
        if not FFMPEG_PATH.exists():
            raise FileNotFoundError(f"Missing dependency: {FFMPEG_PATH}")

        self._resolve_whisper_runtime(self.gpu_var.get().strip())

        selected_model = self._get_selected_model_name()
        if not selected_model or selected_model == NO_MODELS_LABEL:
            raise FileNotFoundError("No model selected. Use download button or put .bin file under models manually\\.")

        model_path = MODELS_DIR / selected_model
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model file: {model_path}")
        model_info = self._get_model_info(selected_model)

        selected_format_label = self.format_var.get().strip().lower()
        selected_format = FORMAT_OPTIONS.get(selected_format_label)
        if selected_format is None:
            raise ValueError(f"Unsupported output format: {selected_format_label}")

        prompt = self.prompt_var.get().strip()
        input_paths = self._get_input_paths()
        configs: list[dict[str, Path | str]] = []

        for input_path in input_paths:
            timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
            audio_output = self._build_unique_output_path(input_path, ".wav")
            transcript_base_name = f"{input_path.stem}{timestamp}.transcript"
            transcript_output = self._build_unique_output_path(
                input_path,
                f".{selected_format}",
                stem=transcript_base_name,
            )
            output_base = transcript_output.with_suffix("")

            configs.append(
                {
                    "input_path": input_path,
                    "format": selected_format,
                    "model_path": model_path,
                    "model_info": model_info,
                    "prompt": prompt,
                    "audio_output": audio_output,
                    "output_base": output_base,
                    "transcript_output": transcript_output,
                }
            )

        return configs

    def _get_input_paths(self) -> list[Path]:
        if self.batch_selected_files:
            missing = [path for path in self.batch_selected_files if not path.exists() or not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Batch file does not exist: {missing[0]}")
            return list(self.batch_selected_files)

        input_raw = self.input_path_var.get().strip()
        if not input_raw:
            raise ValueError("Missing input file.")

        input_path = Path(input_raw)
        if not input_path.exists() or not input_path.is_file():
            raise FileNotFoundError(f"Input file does not exist: {input_path}")

        return [input_path]

    def _build_unique_output_path(
        self,
        input_path: Path,
        suffix: str,
        stem: str | None = None,
    ) -> Path:
        base_stem = stem or input_path.stem
        candidate = input_path.with_name(f"{base_stem}{suffix}")
        index = 1

        while candidate.exists():
            candidate = input_path.with_name(f"{base_stem}-{index}{suffix}")
            index += 1

        return candidate

    def _build_ffmpeg_command(
        self,
        config: dict[str, Path | str],
        *,
        audio_output: Path | None = None,
        include_stats: bool = True,
        duration_seconds: int | None = None,
    ) -> list[str]:
        command = [
            str(FFMPEG_PATH),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
        ]
        if include_stats:
            command.append("-stats")
        command.extend(["-i", str(config["input_path"])])
        if duration_seconds is not None:
            command.extend(["-t", str(duration_seconds)])
        command.extend(
            [
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                str(audio_output or config["audio_output"]),
            ]
        )
        return command

    def _convert_input_to_audio(
        self,
        config: dict[str, Path | str],
        *,
        audio_output: Path | None = None,
        duration_seconds: int | None = None,
        debug_enabled: bool = False,
    ) -> None:
        ffmpeg_command = self._build_ffmpeg_command(
            config,
            audio_output=audio_output,
            include_stats=debug_enabled,
            duration_seconds=duration_seconds,
        )
        self._run_process(ffmpeg_command, "ffmpeg", log_details=debug_enabled)

    def _build_whisper_command(
        self,
        config: dict[str, Path | str],
        selection_label: str,
        runtime: dict[str, object],
        *,
        debug_enabled: bool,
        force_txt: bool = False,
    ) -> list[str]:
        output_format = "txt" if force_txt else str(config["format"])
        command = [
            str(runtime["cli_path"]),
            "-m",
            str(config["model_path"]),
            "-f",
            str(config["audio_output"]),
            "-of",
            str(config["output_base"]),
            "-np",
        ]

        if output_format == "txt":
            command.extend(["-pp", "-otxt", "-nt"])
        else:
            if debug_enabled:
                command.append("-pp")
            command.append("-osrt")

        if self._is_cpu_selection(selection_label):
            if bool(runtime["supports_vulkan"]):
                command.append("-ng")
            command.extend(["-t", str(CPU_THREAD_COUNT)])
        elif not bool(runtime["supports_vulkan"]):
            command.extend(["-t", str(CPU_THREAD_COUNT)])

        prompt = str(config["prompt"])
        if prompt:
            command.extend(["--prompt", prompt])

        return command

    def _run_process(
        self,
        command: list[str],
        tool_name: str,
        log_details: bool = True,
        env: dict[str, str] | None = None,
    ) -> None:
        self._raise_if_cancelled()
        if log_details:
            self.log(f"Running {tool_name}: {' '.join(self._quote_argument(arg) for arg in command)}")
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
            if process.stdout is not None:
                for line in process.stdout:
                    output_lines.append(line)
                    self.output_queue.put(line)

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
            self.log(f"{tool_name} finished successfully.")

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

    def reload_whisper_runtimes(self) -> None:
        runtimes = self._discover_whisper_runtimes()
        self.whisper_runtimes = runtimes
        self.whisper_runtime_lookup = {str(runtime["key"]): runtime for runtime in runtimes}

    def _discover_whisper_runtimes(self) -> list[dict[str, object]]:
        runtimes: list[dict[str, object]] = []
        for candidate in WHISPER_RUNTIME_CANDIDATES:
            runtime_dir = BIN_DIR / str(candidate["folder"])
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

    def _get_whisper_runtime(self, key: str) -> dict[str, object] | None:
        return self.whisper_runtime_lookup.get(key)

    def _get_preferred_whisper_runtime(
        self,
        keys: tuple[str, ...],
        *,
        allow_fallback: bool = True,
    ) -> dict[str, object] | None:
        for key in keys:
            runtime = self._get_whisper_runtime(key)
            if runtime is not None:
                return runtime
        if allow_fallback and self.whisper_runtimes:
            return self.whisper_runtimes[0]
        return None

    def _build_missing_vulkan_runtime_message(self) -> str:
        return (
            "INFO: Hiding GPU options because no Vulkan runtime binary was found. "
            "Expected "
            "bin\\whisper.vulkan\\whisper-cli.exe or "
            "bin\\whisper.cpp\\whisper-cli.exe."
        )

    def _build_missing_vulkan_backend_message(self) -> str:
        return (
            "INFO: Hiding GPU options because the Vulkan binary is present, "
            "but no Vulkan devices were detected. GPU acceleration is "
            "unavailable on this machine."
        )

    def _get_vulkan_gpu_availability(self) -> dict[str, object]:
        runtime = self._get_preferred_whisper_runtime(("vulkan", "legacy"), allow_fallback=False)
        if runtime is None:
            return {
                "status": "runtime_missing",
                "devices": [],
                "log_message": self._build_missing_vulkan_runtime_message(),
                "note_label": GPU_RUNTIME_MISSING_LABEL,
                "controls_enabled": False,
            }

        devices, detection_message = self._detect_vulkan_devices(runtime)
        if detection_message is not None:
            return {
                "status": "detection_failed",
                "devices": [],
                "log_message": detection_message,
                "note_label": GPU_DETECTION_FAILED_LABEL,
                "controls_enabled": True,
            }
        if not devices:
            return {
                "status": "no_devices",
                "devices": [],
                "log_message": self._build_missing_vulkan_backend_message(),
                "note_label": GPU_NO_DEVICES_LABEL,
                "controls_enabled": True,
            }
        return {
            "status": "available",
            "devices": devices,
            "log_message": None,
            "note_label": "",
            "controls_enabled": True,
        }

    def _resolve_whisper_runtime(self, selection_label: str) -> dict[str, object]:
        self.reload_whisper_runtimes()
        if self._is_cpu_selection(selection_label):
            runtime = self._get_preferred_whisper_runtime(("cpu", "vulkan", "legacy"))
        else:
            runtime = self._get_preferred_whisper_runtime(("vulkan", "legacy", "cpu"))

        if runtime is None:
            raise FileNotFoundError(
                "Missing whisper.cpp runtime. Expected "
                "bin\\whisper.vulkan\\whisper-cli.exe and/or "
                "bin\\whisper.cpu\\whisper-cli.exe."
            )

        return runtime

    def _detect_vulkan_devices(
        self,
        runtime: dict[str, object],
    ) -> tuple[list[dict[str, object]], str | None]:
        try:
            result = subprocess.run(
                [str(runtime["cli_path"]), "--help"],
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

        devices: list[dict[str, object]] = []
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

    def _guess_best_gpu_index(self) -> int | None:
        preferred_device = self._get_preferred_gpu_device(self.gpu_devices)
        if preferred_device is None:
            return None
        return int(preferred_device["index"])

    def _build_whisper_env(self, selection_label: str, runtime: dict[str, object]) -> dict[str, str]:
        env = dict(os.environ)
        if not bool(runtime["supports_vulkan"]):
            return env
        selected_gpu = self.gpu_options.get(selection_label)
        if selected_gpu == "cpu":
            return env
        if selected_gpu is None:
            selected_gpu = self._guess_best_gpu_index()
        if isinstance(selected_gpu, int):
            env["GGML_VK_VISIBLE_DEVICES"] = str(selected_gpu)
        return env

    def _build_auto_gpu_label(self, devices: list[dict[str, object]]) -> str:
        preferred_device = self._get_preferred_gpu_device(devices)
        if preferred_device is None:
            return AUTO_GPU_LABEL

        best_index = int(preferred_device["index"])
        best_name = str(preferred_device["name"])
        return f"{AUTO_GPU_LABEL} - GPU {best_index + 1} - {best_name}"

    def _is_cpu_selection(self, selection_label: str) -> bool:
        return self.gpu_options.get(selection_label) == "cpu"

    def _warn_if_cpu_inference_may_be_slow(
        self,
        model_info: object,
        selection_label: str,
        runtime: dict[str, object],
    ) -> bool:
        is_cpu_inference = self._is_cpu_selection(selection_label) or not bool(runtime["supports_vulkan"])
        if not is_cpu_inference:
            return False

        if not isinstance(model_info, dict):
            return False
        model_size_bytes = int(model_info["size_bytes"])
        model_size_label = str(model_info["size_label"])
        model_name = str(model_info["name"])

        if model_size_bytes <= SLOW_CPU_MODEL_WARNING_THRESHOLD_BYTES:
            return False

        self.log(
            f"WARNING: CPU inference with model {model_name} [{model_size_label}] may be slow."
        )
        return True

    @staticmethod
    def _format_model_size_label(size_bytes: int) -> str:
        gib = 1024 * 1024 * 1024
        mib = 1024 * 1024
        if size_bytes >= gib:
            return f"{size_bytes / gib:.2f} GB"
        return f"{size_bytes / mib:.0f} MB"

    @staticmethod
    def _get_preferred_gpu_device(devices: list[dict[str, object]]) -> dict[str, object] | None:
        if not devices:
            return None
        for device in devices:
            if int(device["uma"]) == 0:
                return device
        return devices[0]

    @staticmethod
    def _slugify_label(label: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
        return slug or "benchmark"

    def _raise_if_cancelled(self) -> None:
        if self.cancel_requested:
            raise TranscriptionCancelled()

    def _cleanup_audio_output(self, audio_output: Path, log_removal: bool = True) -> None:
        if not audio_output.exists():
            return
        try:
            audio_output.unlink()
            if log_removal:
                self.log(f"Removed temporary audio file: {audio_output}")
        except OSError as exc:
            self.log(f"WARNING: Could not remove temporary audio file {audio_output}: {exc}")

    def _schedule_ui_update(self, callback: Callable[[], None]) -> None:
        if self._is_closing:
            return
        try:
            if self.winfo_exists():
                self.after(0, callback)
        except tk.TclError:
            pass

    def _download_file(self, url: str, destination: Path) -> None:
        last_percent = -1

        def report_progress(blocks: int, block_size: int, total_size: int) -> None:
            nonlocal last_percent
            if total_size <= 0:
                return
            downloaded = min(blocks * block_size, total_size)
            percent = int((downloaded * 100) / total_size)
            if percent != last_percent and percent % 10 == 0:
                last_percent = percent
                self.log(f"Download progress: {percent}%")

        try:
            urllib.request.urlretrieve(url, destination, report_progress)
        except OSError as exc:
            raise RuntimeError(f"Could not download model: {exc}") from exc

    def _send_download_telemetry_once(self) -> None:
        if self.download_telemetry_sent or not TELEMETRY_APP_ID:
            return

        self.download_telemetry_sent = True
        self._send_telemetry_async("model_download_pressed")

    def _send_telemetry_async(self, signal_type: str) -> None:
        threading.Thread(
            target=self._send_telemetry_signal,
            args=(signal_type,),
            daemon=True,
        ).start()

    def _send_telemetry_signal(self, signal_type: str) -> None:
        payload = {
            "App.version": APP_VERSION,
        }
        gpu_vendors = build_gpu_vendors_payload_value(self.gpu_devices)
        if gpu_vendors:
            payload["App.gpuVendors"] = gpu_vendors

        body = json.dumps(
            [
                {
                    "appID": TELEMETRY_APP_ID,
                    "clientUser": hashlib.sha256(str(uuid.getnode()).encode("utf-8")).hexdigest(),
                    "sessionID": self.telemetry_session_id,
                    "type": signal_type,
                    "payload": payload,
                }
            ]
        ).encode("utf-8")
        request = urllib.request.Request(
            TELEMETRY_URL,
            data=body,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=5):
                pass
        except OSError:
            pass

    @staticmethod
    def _quote_argument(arg: str) -> str:
        if " " in arg or "\t" in arg:
            return f'"{arg}"'
        return arg

    @staticmethod
    def _truncate_result_label(name: str, max_length: int = 48) -> str:
        if len(name) <= max_length:
            return name
        return f"{name[: max_length - 3]}..."

    def _set_result_path(self, path: Path | None) -> None:
        self.latest_result_path = path
        if path is None:
            self.result_link_button.configure(
                text="No result yet",
                state="disabled",
                font=self.result_link_disabled_font,
            )
            return

        self.result_link_button.configure(
            text=f"{self._truncate_result_label(path.name)} >>",
            state="normal",
            font=self.result_link_font,
        )

    def _show_batch_progress(self, completed: int, total: int) -> None:
        self._set_input_path_text(f"{completed} of {total} files processed")

    def _restore_batch_input_summary(self) -> None:
        if not self.batch_selected_files:
            return
        if len(self.batch_selected_files) == 1:
            self._set_input_path_text(str(self.batch_selected_files[0]))
            return
        self._set_input_path_text(
            f"{len(self.batch_selected_files)} files selected from {self.batch_selected_files[0].parent}"
        )

    def reveal_result_file(self) -> None:
        path = self.latest_result_path
        if path is None:
            return
        if not path.exists():
            self.log(f"Result file no longer exists: {path}")
            self._set_result_path(None)
            return

        try:
            subprocess.Popen(["explorer.exe", "/select,", str(path)], shell=False)
        except OSError as exc:
            self.log(f"ERROR: Could not reveal result file: {exc}")

    def _open_url(self, url: str, success_message: str | None = None) -> None:
        try:
            webbrowser.open(url)
            if success_message is not None:
                self.log(success_message)
        except OSError as exc:
            self.log(f"ERROR: Could not open browser: {exc}")


if __name__ == "__main__":
    app = App()
    app.mainloop()
