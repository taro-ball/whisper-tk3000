import queue
import subprocess
import threading

import tkinter as tk

import customtkinter as ctk


class App(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        self.title("IPConfig Runner")
        self.geometry("840x520")
        self.minsize(720, 420)

        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("blue")

        self.output_queue: queue.Queue[str] = queue.Queue()
        self.is_running = False

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        self.header_label = ctk.CTkLabel(
            self,
            text="Windows Network Info",
            font=ctk.CTkFont(size=24, weight="bold"),
        )
        self.header_label.grid(row=0, column=0, padx=16, pady=(16, 8), sticky="w")

        self.run_button = ctk.CTkButton(
            self,
            text="Run ipconfig",
            command=self.run_ipconfig,
            width=140,
        )
        self.run_button.grid(row=1, column=0, padx=16, pady=(0, 8), sticky="w")

        self.console = ctk.CTkTextbox(self, wrap="word", font=("Consolas", 13))
        self.console.grid(row=2, column=0, padx=16, pady=(0, 16), sticky="nsew")
        self.console.configure(state="disabled")

        self.after(100, self.flush_output)

    def append_output(self, text: str) -> None:
        self.console.configure(state="normal")
        self.console.insert("end", text)
        self.console.see("end")
        self.console.configure(state="disabled")

    def flush_output(self) -> None:
        while not self.output_queue.empty():
            self.append_output(self.output_queue.get_nowait())
        self.after(100, self.flush_output)

    def run_ipconfig(self) -> None:
        if self.is_running:
            return

        self.is_running = True
        self.run_button.configure(state="disabled")
        self.console.configure(state="normal")
        self.console.delete("1.0", "end")
        self.console.configure(state="disabled")
        self.output_queue.put("Running ipconfig...\n\n")

        worker = threading.Thread(target=self._execute_ipconfig, daemon=True)
        worker.start()

    def _execute_ipconfig(self) -> None:
        try:
            process = subprocess.Popen(
                ["ipconfig"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                shell=False,
            )

            if process.stdout is not None:
                for line in process.stdout:
                    self.output_queue.put(line)

            exit_code = process.wait()
            self.output_queue.put(f"\nProcess finished with exit code {exit_code}.\n")
        except Exception as exc:
            self.output_queue.put(f"Failed to run ipconfig: {exc}\n")
        finally:
            self.after(0, self._reset_button)

    def _reset_button(self) -> None:
        self.is_running = False
        self.run_button.configure(state="normal")


if __name__ == "__main__":
    app = App()
    app.mainloop()
