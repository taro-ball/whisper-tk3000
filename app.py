from __future__ import annotations

import queue
import subprocess
import threading
import urllib.request
import webbrowser
from datetime import datetime
from pathlib import Path

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, ttk


APP_DIR = Path(__file__).resolve().parent
FFMPEG_PATH = APP_DIR / "bin" / "ffmpeg.exe"
WHISPER_PATH = APP_DIR / "bin" / "Vulkan" / "main64.exe"
MODELS_DIR = APP_DIR / "models"
MODEL_REPO_URL = "https://huggingface.co/ggerganov/whisper.cpp/tree/main"
MODEL_OPTIONS = [
    {
        "name": "ggml-base.en.bin",
        "size": "148Mb",
        "label": "balanced",
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin",
    },
    {
        "name": "ggml-large-v3-turbo.bin",
        "size": "1.62Gb",
        "label": "precise",
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin",
    },
    {
        "name": "ggml-tiny.en.bin",
        "size": "77Mb",
        "label": "fast",
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin",
    },
]
FORMAT_OPTIONS = {
    "srt subtitle": "srt",
    "plain text": "txt",
}
SUPPORTED_MEDIA_TYPES = [
    ("Media files", "*.mp4 *.mkv *.mov *.avi *.mp3 *.wav *.m4a *.flac *.aac *.ogg *.webm"),
    ("All files", "*.*"),
]


class App(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("blue")

        self.title("Whisper Transcriber")
        self.geometry("920x680")
        self.minsize(840, 620)

        self.output_queue: queue.Queue[str] = queue.Queue()
        self.is_running = False

        self.input_path_var = tk.StringVar()
        self.format_var = tk.StringVar(value="srt subtitle")
        self.model_var = tk.StringVar()
        self.prompt_var = tk.StringVar()
        self.download_model_var = tk.StringVar(value=MODEL_OPTIONS[0]["name"])
        self.latest_result_path: Path | None = None
        self.download_dialog: ctk.CTkToplevel | None = None
        self.batch_dialog: ctk.CTkToplevel | None = None
        self.batch_folder_var = tk.StringVar()
        self.batch_selected_files: list[Path] = []
        self.batch_file_rows: list[Path] = []
        self.batch_tree: ttk.Treeview | None = None
        self._suspend_input_path_tracking = False

        self.input_path_var.trace_add("write", self._on_input_path_changed)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        self.header_label = ctk.CTkLabel(
            self,
            text="Whisper Media Transcriber",
            font=ctk.CTkFont(size=26, weight="bold"),
        )
        self.header_label.grid(row=0, column=0, padx=16, pady=(16, 8), sticky="w")

        self.controls_frame = ctk.CTkFrame(self)
        self.controls_frame.grid(row=1, column=0, padx=16, pady=(0, 12), sticky="ew")
        self.controls_frame.grid_columnconfigure(1, weight=1)

        self._build_controls()

        self.console = ctk.CTkTextbox(self, wrap="word", font=("Consolas", 13))
        self.console.grid(row=2, column=0, padx=16, pady=(0, 16), sticky="nsew")
        self.console.configure(state="disabled")

        self.reload_models()
        self.after(100, self.flush_output)

    def _build_controls(self) -> None:
        row = 0

        ctk.CTkLabel(self.controls_frame, text="Input file").grid(
            row=row, column=0, padx=12, pady=(12, 6), sticky="w"
        )
        ctk.CTkEntry(
            self.controls_frame,
            textvariable=self.input_path_var,
            placeholder_text="Select one audio or video file",
        ).grid(row=row, column=1, padx=12, pady=(12, 6), sticky="ew")
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
            values=["No models found"],
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
            text="Convert and Transcribe",
            command=self.run_transcription,
            width=200,
        )
        self.run_button.grid(row=row, column=0, padx=12, pady=(8, 12), sticky="w")

        self.result_row = ctk.CTkFrame(self.controls_frame, fg_color="transparent")
        self.result_row.grid(row=row, column=1, columnspan=2, padx=12, pady=(8, 12), sticky="ew")
        self.result_row.grid_columnconfigure(1, weight=1)
        self.result_label = ctk.CTkLabel(self.result_row, text="Transcribed text:")
        self.result_label.grid(row=0, column=0, padx=(0, 8), pady=0, sticky="w")
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
        )
        self.result_link_button.grid(row=0, column=1, padx=0, pady=0, sticky="w")
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
        patterns = ["*.mp4", "*.mkv", "*.mov", "*.avi", "*.mp3", "*.wav", "*.m4a", "*.flac", "*.aac", "*.ogg", "*.webm"]
        media_files: list[Path] = []
        for pattern in patterns:
            media_files.extend(path for path in folder.glob(pattern) if path.is_file())
            media_files.extend(path for path in folder.glob(pattern.upper()) if path.is_file())
        return sorted(set(media_files), key=lambda path: path.name.lower())

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
        models = sorted(path.name for path in MODELS_DIR.glob("*.bin"))
        if not models:
            models = ["No models found"]
        self.model_menu.configure(values=models)
        current = self.model_var.get()
        if current not in models:
            self.model_var.set(models[0])

    def set_running_state(self, running: bool) -> None:
        self.is_running = running
        state = "disabled" if running else "normal"
        self.run_button.configure(state=state)
        self.browse_button.configure(state=state)
        self.batch_button.configure(state=state)
        self.refresh_models_button.configure(state=state)
        self.download_model_button.configure(state=state)
        self.format_menu.configure(state=state)
        self.model_menu.configure(state=state)
        self.prompt_entry.configure(state=state)

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
            text = f"{option['name']} ({option['size']}) - {option['label']}"
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
            text="Download manually",
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

    def open_manual_download_page(self) -> None:
        try:
            webbrowser.open(MODEL_REPO_URL)
            self.log(f"Opened manual download page: {MODEL_REPO_URL}")
        except OSError as exc:
            self.log(f"ERROR: Could not open browser: {exc}")

    def download_selected_model(self) -> None:
        if self.is_running:
            return

        selected_name = self.download_model_var.get().strip()
        selected_option = next((option for option in MODEL_OPTIONS if option["name"] == selected_name), None)
        if selected_option is None:
            self.log("ERROR: No download model selected.")
            return

        self.close_download_dialog()
        self.set_running_state(True)
        worker = threading.Thread(
            target=self._download_model,
            args=(selected_option,),
            daemon=True,
        )
        worker.start()

    def _download_model(self, model_option: dict[str, str]) -> None:
        destination = MODELS_DIR / model_option["name"]
        temp_destination = destination.with_suffix(destination.suffix + ".part")
        try:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            self.log(f"Downloading model from {model_option['url']}")
            self.log(f"Saving model to {destination}")
            self._download_file(model_option["url"], temp_destination)
            temp_destination.replace(destination)
            self.log(f"Model download complete: {destination.name}")
            self.after(0, self.reload_models)
            self.after(0, lambda: self.model_var.set(destination.name))
        except Exception as exc:
            self.log(f"ERROR: Failed to download model: {exc}")
            if temp_destination.exists():
                try:
                    temp_destination.unlink()
                except OSError:
                    pass
        finally:
            self.after(0, lambda: self.set_running_state(False))

    def clear_console(self) -> None:
        self.console.configure(state="normal")
        self.console.delete("1.0", "end")
        self.console.configure(state="disabled")

    def run_transcription(self) -> None:
        if self.is_running:
            return

        self.clear_console()
        self._set_result_path(None)
        self.set_running_state(True)
        worker = threading.Thread(target=self._execute_transcription, daemon=True)
        worker.start()

    def _execute_transcription(self) -> None:
        try:
            configs = self._build_run_configs()
            total = len(configs)
            last_output: Path | None = None

            for index, config in enumerate(configs, start=1):
                if total > 1:
                    self.log(f"Batch item {index} of {total}")
                self.log(f"Starting transcription for {config['input_path'].name}")
                self.log(f"Selected output format: {config['format'].upper()}")
                self.log(f"Selected model: {config['model_path'].name}")

                self._run_process(
                    [
                        str(FFMPEG_PATH),
                        "-y",
                        "-i",
                        str(config["input_path"]),
                        "-vn",
                        "-ac",
                        "1",
                        "-ar",
                        "16000",
                        str(config["audio_output"]),
                    ],
                    "ffmpeg",
                )

                whisper_command = [
                    str(WHISPER_PATH),
                    "-m",
                    str(config["model_path"]),
                    "-f",
                    str(config["audio_output"]),
                    "-of",
                    str(config["output_base"]),
                ]

                if config["format"] == "txt":
                    whisper_command.extend(["-otxt", "-nt"])
                else:
                    whisper_command.append("-osrt")

                if config["prompt"]:
                    whisper_command.extend(["--prompt", config["prompt"]])

                self._run_process(whisper_command, "whisper.cpp")

                last_output = config["transcript_output"]
                self.log(f"Success. Output file: {config['transcript_output']}")

            self.after(0, lambda: self._set_result_path(last_output))
        except Exception as exc:
            self.log(f"ERROR: {exc}")
            self.after(0, lambda: self._set_result_path(None))
        finally:
            self.after(0, lambda: self.set_running_state(False))

    def _build_run_configs(self) -> list[dict[str, Path | str]]:
        if not FFMPEG_PATH.exists():
            raise FileNotFoundError(f"Missing dependency: {FFMPEG_PATH}")

        if not WHISPER_PATH.exists():
            raise FileNotFoundError(f"Missing dependency: {WHISPER_PATH}")

        selected_model = self.model_var.get().strip()
        if not selected_model or selected_model == "No models found":
            raise FileNotFoundError("No model selected. Add at least one .bin file under models\\.")

        model_path = MODELS_DIR / selected_model
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model file: {model_path}")

        selected_format_label = self.format_var.get().strip().lower()
        selected_format = FORMAT_OPTIONS.get(selected_format_label)
        if selected_format is None:
            raise ValueError(f"Unsupported output format: {selected_format_label}")

        prompt = self.prompt_var.get().strip()
        input_paths = self._get_input_paths()
        configs: list[dict[str, Path | str]] = []

        for input_path in input_paths:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            audio_output = input_path.with_name(f"{input_path.stem}_audio_{timestamp}.wav")
            output_base = audio_output.with_name(f"{audio_output.stem}_transcript_{timestamp}")
            transcript_output = output_base.with_suffix(f".{selected_format}")

            configs.append(
                {
                    "input_path": input_path,
                    "format": selected_format,
                    "model_path": model_path,
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

    def _run_process(self, command: list[str], tool_name: str) -> None:
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
            )
        except OSError as exc:
            raise RuntimeError(f"Failed to start {tool_name}: {exc}") from exc

        if process.stdout is not None:
            for line in process.stdout:
                output_lines.append(line)
                self.output_queue.put(line)

        exit_code = process.wait()
        if exit_code != 0:
            if tool_name == "ffmpeg":
                combined_output = "".join(output_lines)
                if "Duration: N/A" in combined_output or "Invalid duration" in combined_output:
                    raise RuntimeError("ffmpeg reported an invalid duration for the selected input.")
            raise RuntimeError(f"{tool_name} exited with code {exit_code}")

        self.log(f"{tool_name} finished successfully.")

    def _download_file(self, url: str, destination: Path) -> None:
        class DownloadProgressBar:
            def __init__(self, app: App) -> None:
                self.app = app
                self.last_percent = -1

            def __call__(self, blocks: int, block_size: int, total_size: int) -> None:
                if total_size <= 0:
                    return
                downloaded = min(blocks * block_size, total_size)
                percent = int((downloaded * 100) / total_size)
                if percent != self.last_percent and percent % 10 == 0:
                    self.last_percent = percent
                    self.app.log(f"Download progress: {percent}%")

        try:
            urllib.request.urlretrieve(url, destination, DownloadProgressBar(self))
        except OSError as exc:
            raise RuntimeError(f"Could not download model: {exc}") from exc

    @staticmethod
    def _quote_argument(arg: str) -> str:
        if " " in arg or "\t" in arg:
            return f'"{arg}"'
        return arg

    def _set_result_path(self, path: Path | None) -> None:
        self.latest_result_path = path
        if path is None:
            self.result_link_button.configure(text="No result yet", state="disabled")
            return

        self.result_link_button.configure(
            text=path.name,
            state="normal",
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


if __name__ == "__main__":
    app = App()
    app.mainloop()
