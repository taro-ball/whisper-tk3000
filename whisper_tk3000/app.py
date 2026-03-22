from __future__ import annotations

import os
import queue
import threading
import webbrowser
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

try:
    import ctypes
except ImportError:
    ctypes = None

from .core_logic import (
    RunConfig,
    build_run_configs,
    format_model_size_label,
)
from .platform_runtime import (
    AUTO_GPU_LABEL,
    build_cpu_execution_policy,
    load_gpu_selection_state,
    resolve_whisper_runtime,
)
from .model_downloads import (
    MODEL_OPTIONS,
    MODEL_OPTIONS_BY_NAME,
    MODEL_REPO_URL,
    ModelDownloadOption,
    download_model,
)
from .telemetry import TelemetryClient
from .transcription_service import (
    ExecutionContext,
    ServiceCallbacks,
    TranscriptionCancelled,
    TranscriptionService,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_NAME = "whisper-tk3000"
APP_VERSION = "0.4.1"
APP_TITLE = "whisper-tk3000 audio to text transcriber"
TELEMETRY_APP_ID = "5FD59222-E42C-4491-AD54-9A8FA5088609"
TELEMETRY_NAMESPACE = "com.gr"
BIN_DIR = REPO_ROOT / "bin"
FFMPEG_PATH = BIN_DIR / "ffmpeg.exe"
MODELS_DIR = REPO_ROOT / "models"
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
FORMAT_OPTIONS = {
    "srt subtitle": "srt",
    "plain text": "txt",
}
NO_MODELS_LABEL = "No models found"
SUPPORTED_MEDIA_TYPES = [
    ("Media files", " ".join(f"*{suffix}" for suffix in MEDIA_SUFFIXES)),
    ("All files", "*.*"),
]


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
        self.transcription_service = TranscriptionService()

        self.input_path_var = tk.StringVar()
        self.format_var = tk.StringVar(value="srt subtitle")
        self.model_var = tk.StringVar()
        self.gpu_var = tk.StringVar(value=AUTO_GPU_LABEL)
        self.prompt_var = tk.StringVar()
        self.debug_var = tk.BooleanVar(value=False)
        self.download_model_var = tk.StringVar(value=MODEL_OPTIONS[0].name)
        self.available_models: list[dict[str, object]] = []
        self.available_models_by_name: dict[str, dict[str, object]] = {}
        self.model_display_lookup: dict[str, str] = {}
        self.latest_result_path: Path | None = None
        self.download_dialog: ctk.CTkToplevel | None = None
        self.about_dialog: ctk.CTkToplevel | None = None
        self.batch_dialog: ctk.CTkToplevel | None = None
        self.batch_folder_var = tk.StringVar()
        self.batch_selected_files: list[Path] = []
        self.batch_file_rows: list[Path] = []
        self.batch_file_vars: dict[Path, tk.BooleanVar] = {}
        self.telemetry_client = TelemetryClient(
            app_id=TELEMETRY_APP_ID,
            namespace=TELEMETRY_NAMESPACE,
            app_version=APP_VERSION,
        )
        self._suspend_input_path_tracking = False
        self._is_closing = False
        self.cpu_policy = build_cpu_execution_policy()
        self.cpu_name = self.cpu_policy.cpu_name
        self.cpu_option_label = self.cpu_policy.cpu_option_label
        self.gpu_devices: list[dict[str, object]] = []
        self.gpu_options: dict[str, int | str | None] = {AUTO_GPU_LABEL: None, self.cpu_option_label: "cpu"}
        self.gpu_controls_enabled = True
        self.service_callbacks = ServiceCallbacks(
            log=self.log,
            emit_output=self.output_queue.put,
            on_batch_progress=lambda completed, total: self._schedule_ui_update(
                lambda completed=completed, total=total: self._show_batch_progress(completed, total)
            ),
        )

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

        self.batch_folder_var.set(folder)
        dialog, _header_frame, content_frame, actions_frame = self._create_dialog_shell(
            title="Batch Selection",
            header_title="Choose files to process",
            header_text=self.batch_folder_var.get(),
            geometry="760x540",
            minsize=(680, 420),
        )
        self.batch_dialog = dialog
        self.batch_file_rows = media_files

        preselected = set(self.batch_selected_files)
        selected_files = {path for path in media_files if path in preselected}
        if not selected_files:
            selected_files = set(media_files)
        self.batch_file_vars = {}

        list_frame = ctk.CTkFrame(content_frame)
        list_frame.grid(row=0, column=0, sticky="nsew")
        list_frame.grid_columnconfigure(0, weight=1)
        list_frame.grid_rowconfigure(1, weight=1)

        header_row = ctk.CTkFrame(list_frame, fg_color="transparent")
        header_row.grid(row=0, column=0, padx=12, pady=(12, 4), sticky="ew")
        header_row.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(
            header_row,
            text="Use",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=("gray40", "gray70"),
        ).grid(row=0, column=0, padx=(4, 12), pady=0, sticky="w")
        ctk.CTkLabel(
            header_row,
            text="File",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=("gray40", "gray70"),
        ).grid(row=0, column=1, padx=0, pady=0, sticky="w")
        ctk.CTkLabel(
            header_row,
            text="Type",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=("gray40", "gray70"),
        ).grid(row=0, column=2, padx=(12, 4), pady=0, sticky="e")

        file_list = ctk.CTkScrollableFrame(list_frame, corner_radius=10)
        file_list.grid(row=1, column=0, padx=12, pady=(0, 12), sticky="nsew")
        file_list.grid_columnconfigure(0, weight=1)

        for index, path in enumerate(media_files):
            row_frame = ctk.CTkFrame(file_list)
            row_frame.grid(row=index, column=0, padx=0, pady=(0, 8), sticky="ew")
            row_frame.grid_columnconfigure(1, weight=1)

            selected_var = tk.BooleanVar(value=path in selected_files)
            self.batch_file_vars[path] = selected_var

            checkbox = ctk.CTkCheckBox(row_frame, text="", variable=selected_var, width=24)
            checkbox.grid(row=0, column=0, padx=(12, 8), pady=10, sticky="w")

            name_label = ctk.CTkLabel(row_frame, text=path.name, anchor="w")
            name_label.grid(row=0, column=1, padx=0, pady=10, sticky="ew")

            type_label = ctk.CTkLabel(
                row_frame,
                text=path.suffix.lower().lstrip(".").upper(),
                width=72,
                anchor="center",
                corner_radius=999,
                fg_color=("gray86", "gray22"),
                text_color=("gray20", "gray90"),
            )
            type_label.grid(row=0, column=2, padx=(12, 12), pady=10, sticky="e")

            for widget in (row_frame, name_label, type_label):
                widget.bind(
                    "<Button-1>",
                    lambda _event, batch_path=path: self._toggle_batch_row(batch_path),
                )

        actions_frame.grid_columnconfigure(0, weight=0)
        actions_frame.grid_columnconfigure(1, weight=0)
        actions_frame.grid_columnconfigure(2, weight=1)
        actions_frame.grid_columnconfigure(3, weight=0)
        actions_frame.grid_columnconfigure(4, weight=0)

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
        ).grid(row=0, column=3, padx=(12, 12), pady=0, sticky="e")
        ctk.CTkButton(
            actions_frame,
            text="Use selected",
            width=120,
            command=self.apply_batch_selection,
        ).grid(row=0, column=4, padx=0, pady=0, sticky="e")

        dialog.protocol("WM_DELETE_WINDOW", self.close_batch_dialog)

    def close_batch_dialog(self) -> None:
        if self.batch_dialog is not None and self.batch_dialog.winfo_exists():
            self.batch_dialog.destroy()
        self.batch_dialog = None
        self.batch_file_vars = {}

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

    def _toggle_batch_row(self, path: Path) -> None:
        variable = self.batch_file_vars.get(path)
        if variable is None:
            return
        variable.set(not variable.get())

    def _set_all_batch_rows(self, checked: bool) -> None:
        for variable in self.batch_file_vars.values():
            variable.set(checked)

    def _get_checked_batch_files(self) -> list[Path]:
        if not self.batch_file_vars:
            return list(self.batch_selected_files)
        selected: list[Path] = []
        for path in self.batch_file_rows:
            variable = self.batch_file_vars.get(path)
            if variable is not None and variable.get():
                selected.append(path)
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
        runtime_state = load_gpu_selection_state(
            BIN_DIR,
            self.gpu_var.get().strip(),
            self.cpu_policy,
        )
        self.gpu_devices = runtime_state.devices
        self.gpu_options = runtime_state.options
        self.gpu_menu.configure(values=runtime_state.values)
        self.gpu_controls_enabled = runtime_state.controls_enabled
        log_message = str(runtime_state.log_message or "").strip()
        if log_message:
            self.log(log_message)
        self.gpu_var.set(runtime_state.selected_value)
        self.gpu_note_label.configure(text=runtime_state.note_label)
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

    def _create_dialog_shell(
        self,
        *,
        title: str,
        header_title: str,
        header_text: str | None = None,
        geometry: str,
        minsize: tuple[int, int] | None = None,
        resizable: tuple[bool, bool] | None = None,
    ) -> tuple[ctk.CTkToplevel, ctk.CTkFrame, ctk.CTkFrame, ctk.CTkFrame]:
        dialog = ctk.CTkToplevel(self)
        dialog.title(title)
        dialog.geometry(geometry)
        if minsize is not None:
            dialog.minsize(*minsize)
        if resizable is not None:
            dialog.resizable(*resizable)
        dialog.transient(self)
        dialog.grab_set()
        dialog.grid_columnconfigure(0, weight=1)
        dialog.grid_rowconfigure(0, weight=1)

        outer_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        outer_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        outer_frame.grid_columnconfigure(0, weight=1)
        outer_frame.grid_rowconfigure(1, weight=1)

        header_frame = ctk.CTkFrame(outer_frame, fg_color="transparent")
        header_frame.grid(row=0, column=0, padx=0, pady=(0, 12), sticky="ew")
        header_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header_frame,
            text=header_title,
            font=ctk.CTkFont(size=18, weight="bold"),
        ).grid(row=0, column=0, padx=0, pady=0, sticky="w")

        if header_text:
            ctk.CTkLabel(
                header_frame,
                text=header_text,
                justify="left",
                wraplength=700,
                text_color=("gray40", "gray70"),
            ).grid(row=1, column=0, padx=0, pady=(6, 0), sticky="w")

        content_frame = ctk.CTkFrame(outer_frame, fg_color="transparent")
        content_frame.grid(row=1, column=0, padx=0, pady=0, sticky="nsew")
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_rowconfigure(0, weight=1)

        actions_frame = ctk.CTkFrame(outer_frame, fg_color="transparent")
        actions_frame.grid(row=2, column=0, padx=0, pady=(16, 0), sticky="ew")
        actions_frame.grid_columnconfigure(0, weight=1)
        self._apply_windows_titlebar_theme(dialog)

        return dialog, header_frame, content_frame, actions_frame

    def _apply_windows_titlebar_theme(self, window: tk.Misc) -> None:
        if ctypes is None or os.name != "nt":
            return

        def apply_theme() -> None:
            if not window.winfo_exists():
                return

            try:
                hwnd = ctypes.windll.user32.GetParent(window.winfo_id())
                if hwnd == 0:
                    return

                dark_mode_enabled = ctk.get_appearance_mode().lower() == "dark"
                value = ctypes.c_int(1 if dark_mode_enabled else 0)
                dwm_set_window_attribute = ctypes.windll.dwmapi.DwmSetWindowAttribute
                for attribute in (20, 19):
                    result = dwm_set_window_attribute(
                        hwnd,
                        attribute,
                        ctypes.byref(value),
                        ctypes.sizeof(value),
                    )
                    if result == 0:
                        break
            except Exception:
                pass

        window.after(20, apply_theme)

    def show_download_dialog(self) -> None:
        if self.is_running:
            return

        if self.download_dialog is not None and self.download_dialog.winfo_exists():
            self.download_dialog.focus()
            return

        dialog, _header_frame, content_frame, actions_frame = self._create_dialog_shell(
            title="Download Whisper Model",
            header_title="Choose a model to download",
            geometry="520x260",
            resizable=(False, False),
        )
        self.download_dialog = dialog

        options_frame = ctk.CTkFrame(content_frame)
        options_frame.grid(row=0, column=0, padx=0, pady=0, sticky="nsew")
        options_frame.grid_columnconfigure(0, weight=1)

        for index, option in enumerate(MODEL_OPTIONS):
            text = f"{option.name} [{option.size_label}] - {option.label}"
            ctk.CTkRadioButton(
                options_frame,
                text=text,
                variable=self.download_model_var,
                value=option.name,
            ).grid(row=index, column=0, padx=16, pady=8, sticky="w")

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
        if self.about_dialog is not None and self.about_dialog.winfo_exists():
            self.about_dialog.focus()
            return

        dialog, _header_frame, content_frame, actions_frame = self._create_dialog_shell(
            title=f"About {APP_NAME}",
            header_title=APP_NAME,
            header_text="Desktop app for transcribing audio and video to text with whisper.cpp.",
            geometry="560x420",
            minsize=(520, 380),
        )
        self.about_dialog = dialog

        summary_frame = ctk.CTkFrame(content_frame)
        summary_frame.grid(row=0, column=0, pady=(0, 12), sticky="ew")
        summary_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            summary_frame,
            text=f"Version {APP_VERSION}",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=("gray35", "gray75"),
        ).grid(row=0, column=0, padx=16, pady=(14, 4), sticky="w")
        ctk.CTkLabel(
            summary_frame,
            text="A local-first desktop transcription app with a CustomTkinter UI over whisper.cpp.",
            justify="left",
            wraplength=480,
        ).grid(row=1, column=0, padx=16, pady=(0, 14), sticky="w")

        resources_frame = ctk.CTkFrame(content_frame)
        resources_frame.grid(row=1, column=0, pady=(0, 12), sticky="ew")
        resources_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            resources_frame,
            text="Resources",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=0, column=0, padx=16, pady=(14, 6), sticky="w")

        about_links = (
            (
                "Project source",
                "https://github.com/taro-ball/whisper-tk3000",
                "Opened project page: https://github.com/taro-ball/whisper-tk3000",
            ),
            (
                "whisper.cpp",
                "https://github.com/ggml-org/whisper.cpp",
                "Opened whisper.cpp project page: https://github.com/ggml-org/whisper.cpp",
            ),
            (
                "FFmpeg",
                "https://ffmpeg.org/about.html",
                "Opened FFmpeg about page: https://ffmpeg.org/about.html",
            ),
            (
                "Model repository",
                MODEL_REPO_URL,
                f"Opened model repository page: {MODEL_REPO_URL}",
            ),
        )
        for index, (label, url, success_message) in enumerate(about_links, start=1):
            ctk.CTkButton(
                resources_frame,
                text=label,
                anchor="w",
                command=lambda link=url, message=success_message: self._open_url(link, message),
                fg_color="transparent",
                text_color=("blue", "#7fb3ff"),
                hover_color=self.controls_frame.cget("fg_color"),
                border_width=0,
            ).grid(row=index, column=0, padx=12, pady=2, sticky="ew")

        credits_frame = ctk.CTkFrame(content_frame)
        credits_frame.grid(row=2, column=0, sticky="ew")
        credits_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            credits_frame,
            text="Credits",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=0, column=0, padx=16, pady=(14, 6), sticky="w")
        ctk.CTkLabel(
            credits_frame,
            text=(
                "Built on top of whisper.cpp for speech recognition and FFmpeg for media conversion.\n"
                "Dedicated to my wife, whose love for science always inspires me."
            ),
            justify="left",
            wraplength=480,
        ).grid(row=1, column=0, padx=16, pady=(0, 14), sticky="w")

        ctk.CTkButton(
            actions_frame,
            text="Close",
            width=100,
            command=self.close_about_dialog,
        ).grid(row=0, column=1, padx=0, pady=0, sticky="e")

        dialog.protocol("WM_DELETE_WINDOW", self.close_about_dialog)

    def close_about_dialog(self) -> None:
        if self.about_dialog is not None and self.about_dialog.winfo_exists():
            self.about_dialog.destroy()
        self.about_dialog = None

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

        self.telemetry_client.send_once_async("model_download_pressed", self.gpu_devices)
        self.close_download_dialog()
        self.set_running_state(True)
        worker = threading.Thread(
            target=self._download_model,
            args=(selected_option,),
            daemon=True,
        )
        worker.start()

    def _download_model(self, model_option: ModelDownloadOption) -> None:
        try:
            destination = download_model(model_option, MODELS_DIR, log=self.log)
            self._schedule_ui_update(self.reload_models)
            self._schedule_ui_update(lambda: self._set_selected_model_name(destination.name))
        except Exception as exc:
            self.log(f"ERROR: Failed to download model: {exc}")
        finally:
            self._schedule_ui_update(lambda: self.set_running_state(False))

    def run_transcription(self) -> None:
        if self.is_running:
            return

        self._set_result_path(None)
        self.transcription_service.reset_cancellation()
        self.transcription_running = True
        self.set_running_state(True)
        worker = threading.Thread(target=self._execute_transcription, daemon=True)
        worker.start()

    def run_benchmark(self) -> None:
        if self.is_running:
            return

        self.reload_gpu_options()
        self._set_result_path(None)
        self.transcription_service.reset_cancellation()
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

        if self.transcription_service.cancel():
            self.log("Cancellation requested. Stopping the current process...")

    def on_close(self) -> None:
        self._is_closing = True
        self.transcription_service.kill_active_process()
        self.destroy()

    def _execute_transcription(self) -> None:
        should_show_batch_progress = False
        try:
            configs = self._build_run_configs()
            should_show_batch_progress = len(configs) > 1
            outcome = self.transcription_service.run_transcription(
                configs,
                self._build_execution_context(),
                self.service_callbacks,
            )
            self._schedule_ui_update(
                lambda path=outcome.last_output: self._show_transcription_result(path)
            )
        except TranscriptionCancelled:
            self.log("Transcription cancelled.")
            self._schedule_ui_update(lambda: self._set_result_path(None))
        except Exception as exc:
            self.log(f"ERROR: {exc}")
            self._schedule_ui_update(lambda: self._set_result_path(None))
        finally:
            self.transcription_running = False
            self.transcription_service.reset_cancellation()
            if should_show_batch_progress:
                self._schedule_ui_update(self._restore_batch_input_summary)
            self._schedule_ui_update(lambda: self.set_running_state(False))

    def _execute_benchmark(self) -> None:
        try:
            configs = self._build_run_configs()
            self.transcription_service.run_benchmark(
                configs[0],
                self._build_execution_context(),
                self.service_callbacks,
            )
        except TranscriptionCancelled:
            self.log("Benchmark cancelled.")
        except Exception as exc:
            self.log(f"ERROR: {exc}")
        finally:
            self.transcription_service.reset_cancellation()
            self._schedule_ui_update(lambda: self.set_running_state(False))

    def _build_run_configs(self) -> list[RunConfig]:
        if not FFMPEG_PATH.exists():
            raise FileNotFoundError(f"Missing dependency: {FFMPEG_PATH}")

        resolve_whisper_runtime(BIN_DIR, self.gpu_var.get().strip(), self.gpu_options)

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
        
        return build_run_configs(
            input_paths=input_paths,
            model_path=model_path,
            model_info=model_info,
            output_format=selected_format,
            prompt=prompt,
        )

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

    def _build_execution_context(self) -> ExecutionContext:
        return ExecutionContext(
            ffmpeg_path=FFMPEG_PATH,
            bin_dir=BIN_DIR,
            cpu_policy=self.cpu_policy,
            gpu_selection_label=self.gpu_var.get().strip(),
            gpu_options=dict(self.gpu_options),
            gpu_devices=list(self.gpu_devices),
            debug_enabled=self.debug_var.get(),
        )

    @staticmethod
    def _format_model_size_label(size_bytes: int) -> str:
        return format_model_size_label(size_bytes)

    def _schedule_ui_update(self, callback: Callable[[], None]) -> None:
        if self._is_closing:
            return
        try:
            if self.winfo_exists():
                self.after(0, callback)
        except tk.TclError:
            pass

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

    def _show_transcription_result(self, path: Path | None) -> None:
        self._set_result_path(path)
        if path is not None:
            self.reveal_result_file()

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
        if path is None or not path.exists():
            self.log(f"Result file no longer exists: {path}")
            self._set_result_path(None)
            return

        select_arg = f'/select,"{os.path.normpath(str(path.resolve()))}"'
        try:
            shell32 = ctypes.windll.shell32
            result = shell32.ShellExecuteW(None, "open", "explorer.exe", select_arg, None, 1)
            if result <= 32:
                raise OSError(f"ShellExecuteW failed with code {result}")
        except OSError as exc:
            self.log(f"ERROR: Could not reveal result file: {exc}")

    def _open_url(self, url: str, success_message: str | None = None) -> None:
        try:
            webbrowser.open(url)
            if success_message is not None:
                self.log(success_message)
        except OSError as exc:
            self.log(f"ERROR: Could not open browser: {exc}")


def main() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
