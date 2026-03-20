"""Desktop GUI for running the unified ICP pipeline locally."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from src.config import INPUT_DIR, OUTPUT_DIR
from src.pipeline import UnifiedPipeline


class ICPPipelineApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Unified ICP Pipeline")
        self.root.geometry("900x620")
        self.root.minsize(820, 540)

        self.input_dir_var = tk.StringVar(value=str(INPUT_DIR))
        self.output_dir_var = tk.StringVar(value=str(OUTPUT_DIR))
        self.semantic_enabled_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Ready.")

        self._log_queue: queue.Queue[str] = queue.Queue()
        self._worker_thread: threading.Thread | None = None

        self._build_ui()
        self._drain_queue()

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=14)
        main.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(
            main,
            text="Unified Data Pipeline (Local)",
            font=("TkDefaultFont", 14, "bold"),
        )
        title.pack(anchor=tk.W, pady=(0, 10))

        input_row = ttk.Frame(main)
        input_row.pack(fill=tk.X, pady=4)
        ttk.Label(input_row, text="Source folder", width=15).pack(side=tk.LEFT)
        ttk.Entry(input_row, textvariable=self.input_dir_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8)
        )
        ttk.Button(input_row, text="Browse", command=self._choose_input).pack(side=tk.LEFT)

        output_row = ttk.Frame(main)
        output_row.pack(fill=tk.X, pady=4)
        ttk.Label(output_row, text="Destination", width=15).pack(side=tk.LEFT)
        ttk.Entry(output_row, textvariable=self.output_dir_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8)
        )
        ttk.Button(output_row, text="Browse", command=self._choose_output).pack(
            side=tk.LEFT
        )

        options = ttk.Frame(main)
        options.pack(fill=tk.X, pady=(8, 6))
        ttk.Checkbutton(
            options,
            text="Enable AI semantic scoring (SentenceTransformer + FAISS)",
            variable=self.semantic_enabled_var,
        ).pack(side=tk.LEFT)

        actions = ttk.Frame(main)
        actions.pack(fill=tk.X, pady=(2, 10))

        self.run_button = ttk.Button(actions, text="Run Pipeline", command=self._run_pipeline)
        self.run_button.pack(side=tk.LEFT)
        ttk.Button(actions, text="Open Source Folder", command=self._open_input_folder).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        ttk.Button(
            actions, text="Open Destination Folder", command=self._open_output_folder
        ).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(actions, text="Clear Logs", command=self._clear_logs).pack(
            side=tk.RIGHT, padx=(8, 0)
        )

        status = ttk.Label(main, textvariable=self.status_var)
        status.pack(anchor=tk.W, pady=(0, 8))

        log_frame = ttk.LabelFrame(main, text="Run Logs", padding=8)
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(log_frame, wrap="word")
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scroll = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.configure(yscrollcommand=scroll.set, state=tk.DISABLED)

    def _choose_input(self) -> None:
        selected = filedialog.askdirectory(title="Select source folder")
        if selected:
            self.input_dir_var.set(selected)

    def _choose_output(self) -> None:
        selected = filedialog.askdirectory(title="Select destination folder")
        if selected:
            self.output_dir_var.set(selected)

    def _append_log(self, message: str) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _clear_logs(self) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _log(self, message: str) -> None:
        self._log_queue.put(message)

    def _drain_queue(self) -> None:
        try:
            while True:
                msg = self._log_queue.get_nowait()
                self._append_log(msg)
        except queue.Empty:
            pass
        self.root.after(120, self._drain_queue)

    def _set_running(self, running: bool) -> None:
        self.run_button.configure(state=tk.DISABLED if running else tk.NORMAL)
        self.status_var.set("Running..." if running else "Ready.")

    def _run_pipeline(self) -> None:
        if self._worker_thread and self._worker_thread.is_alive():
            messagebox.showinfo("Pipeline Running", "A pipeline run is already in progress.")
            return

        input_dir = Path(self.input_dir_var.get()).expanduser()
        output_dir = Path(self.output_dir_var.get()).expanduser()
        if not input_dir.exists() or not input_dir.is_dir():
            messagebox.showerror("Invalid Source Folder", "Please select a valid source folder.")
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        self._set_running(True)
        self._log("=" * 72)
        self._log("Starting unified pipeline...")
        self._log(f"Source: {input_dir}")
        self._log(f"Destination: {output_dir}")
        self._log(
            f"Semantic scoring: {'enabled' if self.semantic_enabled_var.get() else 'disabled'}"
        )

        def worker() -> None:
            try:
                pipeline = UnifiedPipeline(input_dir=input_dir, output_dir=output_dir)
                result = pipeline.run(
                    enable_semantic=self.semantic_enabled_var.get(),
                    logger=self._log,
                )
                self._log(
                    f"Completed. Input rows: {result.total_input_rows}, "
                    f"deduplicated rows: {result.deduplicated_rows}."
                )
                for file in result.output_files:
                    self._log(f"Output: {file}")
                self.root.after(
                    0,
                    lambda: messagebox.showinfo(
                        "Pipeline Complete",
                        "Run completed successfully. Check the destination folder.",
                    ),
                )
            except Exception as exc:
                self._log(f"ERROR: {exc}")
                self.root.after(
                    0,
                    lambda: messagebox.showerror(
                        "Pipeline Failed", f"Pipeline failed with error:\n{exc}"
                    ),
                )
            finally:
                self.root.after(0, lambda: self._set_running(False))

        self._worker_thread = threading.Thread(target=worker, daemon=True)
        self._worker_thread.start()

    def _open_input_folder(self) -> None:
        self._open_folder(self.input_dir_var.get())

    def _open_output_folder(self) -> None:
        self._open_folder(self.output_dir_var.get())

    @staticmethod
    def _open_folder(path: str) -> None:
        candidate = Path(path).expanduser()
        if not candidate.exists():
            messagebox.showerror("Folder Not Found", f"Folder does not exist:\n{candidate}")
            return

        try:
            import subprocess

            subprocess.run(["open", str(candidate)], check=False)
        except Exception as exc:
            messagebox.showerror("Open Folder Failed", str(exc))


def launch_gui() -> None:
    root = tk.Tk()
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass
    ICPPipelineApp(root)
    root.mainloop()

