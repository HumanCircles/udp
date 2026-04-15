"""Comprehensive ICP filter dashboard for sales team.

Layout:
  Left  — tabbed filter panel (Industries | Company & Seniority | Contact Quality)
  Right — live log + results breakdown table

Run via:
    python launch_dashboard.py
"""
from __future__ import annotations

import queue
import re
import subprocess
import threading
import time
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk

from src.config import INDUSTRY_GROUPS, INPUT_DIR, OUTPUT_DIR
from src.pipeline import UnifiedPipeline
from src.scorer import ICPConfig


# ────────────────────────────────────────────────────────────────────────────
# Preset configs shown in the Quick Preset dropdown
# ────────────────────────────────────────────────────────────────────────────
PRESETS: dict[str, dict] = {
    "Custom (manual)": {},
    "Narrow ICP  (1000–2000 emp | Health+Const+Mfg+Retail)": {
        "emp_min": 1000, "emp_max": 2000,
        "industries": ["Healthcare & Medical", "Construction & Infrastructure",
                       "Manufacturing & Industrial", "Retail & Consumer Goods"],
        "excl_staffing": True,
    },
    "Broad ICP  (20–5000 emp | all industries)": {
        "emp_min": 20, "emp_max": 5000,
        "industries": [],          # empty = all
        "excl_staffing": False,
    },
    "Decision Makers only  (C-Suite + VP + Director + Founder)": {
        "emp_min": 20, "emp_max": 9_999_999,
        "seniority": ["c suite", "vp", "director", "founder", "owner", "partner", "head"],
        "industries": [],
        "excl_staffing": False,
    },
    "Healthcare  (all sizes)": {
        "emp_min": 20, "emp_max": 9_999_999,
        "industries": ["Healthcare & Medical"],
        "excl_staffing": True,
    },
    "Construction + Manufacturing  (200–5000 emp)": {
        "emp_min": 200, "emp_max": 5000,
        "industries": ["Construction & Infrastructure", "Manufacturing & Industrial"],
        "excl_staffing": True,
    },
    "Software / Tech  (all sizes)": {
        "emp_min": 20, "emp_max": 9_999_999,
        "industries": ["Software & Technology"],
        "excl_staffing": True,
    },
}

SENIORITY_OPTIONS = [
    ("C-Suite",  "c suite"),
    ("Founder",  "founder"),
    ("Owner",    "owner"),
    ("VP",       "vp"),
    ("Director", "director"),
    ("Head",     "head"),
    ("Manager",  "manager"),
    ("Senior",   "senior"),
    ("Partner",  "partner"),
    ("Entry",    "entry"),
]


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────
def _estimate_rows(path: Path) -> int:
    try:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            return sum(1 for _ in fh) - 1
    except Exception:
        return 0


def _group_files(files: list[Path], batch_size: int) -> list[list[Path]]:
    batches: list[list[Path]] = []
    cur: list[Path] = []
    cur_rows = 0
    for f in files:
        rows = _estimate_rows(f)
        if cur and cur_rows + rows > batch_size:
            batches.append(cur)
            cur = [f]
            cur_rows = rows
        else:
            cur.append(f)
            cur_rows += rows
    if cur:
        batches.append(cur)
    return batches


# ────────────────────────────────────────────────────────────────────────────
# App
# ────────────────────────────────────────────────────────────────────────────
class DashboardApp:
    BATCH_SIZE = 150_000

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("HireQuotient — ICP Filter Dashboard")
        self.root.geometry("1200x780")
        self.root.minsize(960, 620)

        # ── state ────────────────────────────────────────────────────────
        self.input_dir_var   = tk.StringVar(value=str(INPUT_DIR))
        self.output_dir_var  = tk.StringVar(value=str(OUTPUT_DIR))
        self.emp_min_var     = tk.StringVar(value="20")
        self.emp_max_var     = tk.StringVar(value="5000")
        self.excl_staffing   = tk.BooleanVar(value=True)
        self.semantic_var    = tk.BooleanVar(value=False)
        self.require_email   = tk.BooleanVar(value=False)
        self.require_phone   = tk.BooleanVar(value=False)
        self.require_linkedin = tk.BooleanVar(value=False)
        self.status_var      = tk.StringVar(value="Ready.")
        self.preset_var      = tk.StringVar(value=list(PRESETS)[0])

        self._ind_vars: dict[str, tk.BooleanVar] = {
            name: tk.BooleanVar(value=False) for name in INDUSTRY_GROUPS
        }
        self._all_ind_var = tk.BooleanVar(value=True)

        self._sen_vars: dict[str, tk.BooleanVar] = {
            label: tk.BooleanVar(value=True) for label, _ in SENIORITY_OPTIONS
        }
        self._all_sen_var = tk.BooleanVar(value=True)

        self._log_q: queue.Queue[str] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._last_output: Path | None = None

        self._build_ui()
        self._drain_queue()

    # ── UI ───────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=0, minsize=400)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        self._build_left()
        self._build_right()

    # ── Left panel ───────────────────────────────────────────────────────

    def _build_left(self) -> None:
        left = ttk.Frame(self.root, padding=(12, 12, 0, 12))
        left.grid(row=0, column=0, sticky="nsew")
        left.rowconfigure(3, weight=0)

        # Header
        ttk.Label(left, text="ICP Filter Dashboard",
                  font=("TkDefaultFont", 13, "bold")).pack(anchor=tk.W, pady=(0, 8))

        # Preset picker
        pf = ttk.LabelFrame(left, text="Quick Preset", padding=6)
        pf.pack(fill=tk.X, pady=(0, 8))
        preset_cb = ttk.Combobox(pf, textvariable=self.preset_var,
                                 values=list(PRESETS), state="readonly", width=48)
        preset_cb.pack(fill=tk.X)
        preset_cb.bind("<<ComboboxSelected>>", self._apply_preset)

        # Folders
        ff = ttk.LabelFrame(left, text="Data Folders", padding=6)
        ff.pack(fill=tk.X, pady=(0, 8))
        self._folder_row(ff, "Source (raw CSVs)", self.input_dir_var, self._pick_input)
        self._folder_row(ff, "Output folder",    self.output_dir_var, self._pick_output)

        # Tabs for filters
        nb = ttk.Notebook(left)
        nb.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        nb.add(self._build_industry_tab(nb),  text="  Industries  ")
        nb.add(self._build_company_tab(nb),   text="  Company & Seniority  ")
        nb.add(self._build_quality_tab(nb),   text="  Contact Quality  ")

        # Run
        ttk.Separator(left).pack(fill=tk.X, pady=6)
        bf = ttk.Frame(left)
        bf.pack(fill=tk.X)
        self.run_btn = ttk.Button(bf, text="▶  Run Filter", command=self._run)
        self.run_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=5)
        ttk.Button(bf, text="Open Output", command=self._open_output,
                   width=12).pack(side=tk.LEFT, padx=(6, 0), ipady=5)
        ttk.Label(left, textvariable=self.status_var,
                  foreground="gray").pack(anchor=tk.W, pady=(4, 0))

    def _build_industry_tab(self, parent: ttk.Notebook) -> ttk.Frame:
        tab = ttk.Frame(parent, padding=8)

        # "All" toggle
        top = ttk.Frame(tab)
        top.pack(fill=tk.X, pady=(0, 4))
        ttk.Checkbutton(top, text="All industries (no filter)",
                        variable=self._all_ind_var,
                        command=self._toggle_all_ind).pack(side=tk.LEFT)
        ttk.Button(top, text="None", width=5,
                   command=lambda: self._set_all_ind(False)).pack(side=tk.RIGHT)
        ttk.Button(top, text="All", width=5,
                   command=lambda: self._set_all_ind(True)).pack(side=tk.RIGHT, padx=(0, 4))

        ttk.Separator(tab).pack(fill=tk.X, pady=4)

        # Scrollable checkbox list
        canvas = tk.Canvas(tab, height=300, highlightthickness=0)
        sb = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        self._ind_frame = ttk.Frame(canvas)
        self._ind_frame.bind("<Configure>",
                             lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self._ind_frame, anchor="nw")
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        for name, var in self._ind_vars.items():
            row = ttk.Frame(self._ind_frame)
            row.pack(fill=tk.X, pady=1)
            cb = ttk.Checkbutton(row, text=name, variable=var,
                                 command=self._ind_changed)
            cb.pack(side=tk.LEFT)
            # small pattern preview
            short_pat = INDUSTRY_GROUPS[name][:50].rstrip("|") + "…"
            ttk.Label(row, text=f"({short_pat})",
                      foreground="gray", font=("TkDefaultFont", 8)).pack(side=tk.LEFT, padx=(4, 0))

        return tab

    def _build_company_tab(self, parent: ttk.Notebook) -> ttk.Frame:
        tab = ttk.Frame(parent, padding=8)

        # Employee count
        ef = ttk.LabelFrame(tab, text="Employee Count Range", padding=8)
        ef.pack(fill=tk.X, pady=(0, 10))
        er = ttk.Frame(ef)
        er.pack(fill=tk.X)
        ttk.Label(er, text="Min", width=5).pack(side=tk.LEFT)
        ttk.Entry(er, textvariable=self.emp_min_var, width=9).pack(side=tk.LEFT, padx=(0, 14))
        ttk.Label(er, text="Max", width=5).pack(side=tk.LEFT)
        ttk.Entry(er, textvariable=self.emp_max_var, width=9).pack(side=tk.LEFT)
        # Quick range buttons
        qr = ttk.Frame(ef)
        qr.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(qr, text="Quick:").pack(side=tk.LEFT)
        for label, lo, hi in [("<200", 1, 200), ("200-1K", 200, 1000),
                               ("1K-2K", 1000, 2000), ("2K-5K", 2000, 5000), ("Any", 1, 9_999_999)]:
            ttk.Button(qr, text=label, width=7,
                       command=lambda l=lo, h=hi: (self.emp_min_var.set(str(l)),
                                                   self.emp_max_var.set(str(h)))).pack(side=tk.LEFT, padx=2)

        # Seniority
        sf = ttk.LabelFrame(tab, text="Seniority Level", padding=8)
        sf.pack(fill=tk.X, pady=(0, 10))
        st = ttk.Frame(sf)
        st.pack(fill=tk.X, pady=(0, 6))
        ttk.Checkbutton(st, text="All seniority levels",
                        variable=self._all_sen_var,
                        command=self._toggle_all_sen).pack(side=tk.LEFT)
        ttk.Button(st, text="DM only",
                   command=self._preset_dm_seniority).pack(side=tk.RIGHT)

        grid = ttk.Frame(sf)
        grid.pack(fill=tk.X)
        for i, (label, _) in enumerate(SENIORITY_OPTIONS):
            var = self._sen_vars[label]
            cb = ttk.Checkbutton(grid, text=label, variable=var,
                                 command=self._sen_changed)
            cb.grid(row=i // 2, column=i % 2, sticky=tk.W, padx=4, pady=1)

        # Options
        of = ttk.LabelFrame(tab, text="Options", padding=8)
        of.pack(fill=tk.X)
        ttk.Checkbutton(of, text="Exclude staffing & recruiting agencies",
                        variable=self.excl_staffing).pack(anchor=tk.W)
        ttk.Checkbutton(of, text="AI semantic scoring (slower, more accurate)",
                        variable=self.semantic_var).pack(anchor=tk.W, pady=(4, 0))

        return tab

    def _build_quality_tab(self, parent: ttk.Notebook) -> ttk.Frame:
        tab = ttk.Frame(parent, padding=12)

        ttk.Label(tab, text="Require these fields to be present:",
                  font=("TkDefaultFont", 10)).pack(anchor=tk.W, pady=(0, 10))

        qf = ttk.LabelFrame(tab, text="Contact Fields", padding=10)
        qf.pack(fill=tk.X, pady=(0, 10))
        ttk.Checkbutton(qf, text="Must have Email address",
                        variable=self.require_email).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(qf, text="Must have Phone number  (direct / mobile)",
                        variable=self.require_phone).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(qf, text="Must have LinkedIn URL",
                        variable=self.require_linkedin).pack(anchor=tk.W, pady=2)

        ttk.Label(tab,
                  text="Note: email is present for ~99% of Apollo records.\n"
                       "Phone is sparse — requiring it will significantly reduce output.",
                  foreground="gray", font=("TkDefaultFont", 9),
                  justify=tk.LEFT).pack(anchor=tk.W, pady=(6, 0))

        return tab

    def _folder_row(self, parent, label, var, cmd) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label, width=16).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        ttk.Button(row, text="Browse", command=cmd, width=7).pack(side=tk.LEFT)

    # ── Right panel ──────────────────────────────────────────────────────

    def _build_right(self) -> None:
        right = ttk.Frame(self.root, padding=(8, 12, 12, 12))
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.rowconfigure(1, weight=0)
        right.columnconfigure(0, weight=1)

        # Log
        lf = ttk.LabelFrame(right, text="Run Log", padding=6)
        lf.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        lf.rowconfigure(0, weight=1)
        lf.columnconfigure(0, weight=1)

        self.log_text = tk.Text(lf, wrap="word", state=tk.DISABLED,
                                font=("Courier", 10),
                                background="#1e1e1e", foreground="#d4d4d4")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(lf, command=self.log_text.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=sb.set)

        br = ttk.Frame(lf)
        br.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        ttk.Button(br, text="Clear log", command=self._clear_log).pack(side=tk.RIGHT)

        # Results table
        rf = ttk.LabelFrame(right, text="Results  (populated after run)", padding=6)
        rf.grid(row=1, column=0, sticky="ew")

        cols = ("Industry Group", "ACCEPT", "% of total", "Reject (emp)", "Reject (title)")
        self.tree = ttk.Treeview(rf, columns=cols, show="headings", height=10)
        widths = {cols[0]: 230, cols[1]: 80, cols[2]: 80, cols[3]: 100, cols[4]: 100}
        for col in cols:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=widths[col],
                             anchor=tk.W if col == cols[0] else tk.CENTER)
        self.tree.pack(fill=tk.X)
        self.tree.tag_configure("hit", foreground="#116611")
        self.tree.tag_configure("zero", foreground="#999999")

        self.summary_var = tk.StringVar(value="")
        ttk.Label(rf, textvariable=self.summary_var,
                  font=("TkDefaultFont", 10, "bold")).pack(anchor=tk.W, pady=(6, 0))

    # ── Callbacks ────────────────────────────────────────────────────────

    def _pick_input(self):
        d = filedialog.askdirectory(title="Select source folder")
        if d:
            self.input_dir_var.set(d)

    def _pick_output(self):
        d = filedialog.askdirectory(title="Select output folder")
        if d:
            self.output_dir_var.set(d)

    def _log(self, msg: str) -> None:
        self._log_q.put(msg)

    def _drain_queue(self) -> None:
        try:
            while True:
                msg = self._log_q.get_nowait()
                self.log_text.configure(state=tk.NORMAL)
                self.log_text.insert(tk.END, msg + "\n")
                self.log_text.see(tk.END)
                self.log_text.configure(state=tk.DISABLED)
        except queue.Empty:
            pass
        self.root.after(100, self._drain_queue)

    def _clear_log(self):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _open_output(self):
        p = self._last_output or Path(self.output_dir_var.get())
        if p.exists():
            subprocess.run(["open", str(p)], check=False)
        else:
            messagebox.showerror("Not found", str(p))

    # ── Industry checkbox logic ──────────────────────────────────────────

    def _set_all_ind(self, val: bool):
        for v in self._ind_vars.values():
            v.set(val)
        self._all_ind_var.set(False)

    def _toggle_all_ind(self):
        is_all = self._all_ind_var.get()
        for v in self._ind_vars.values():
            v.set(False)
        for child in self._ind_frame.winfo_children():
            for w in child.winfo_children():
                try:
                    w.configure(state=tk.DISABLED if is_all else tk.NORMAL)
                except Exception:
                    pass

    def _ind_changed(self):
        if any(v.get() for v in self._ind_vars.values()):
            self._all_ind_var.set(False)

    # ── Seniority logic ──────────────────────────────────────────────────

    def _toggle_all_sen(self):
        if self._all_sen_var.get():
            for v in self._sen_vars.values():
                v.set(False)

    def _sen_changed(self):
        if any(v.get() for v in self._sen_vars.values()):
            self._all_sen_var.set(False)

    def _preset_dm_seniority(self):
        dm_labels = {"C-Suite", "VP", "Director", "Founder", "Owner", "Head", "Partner"}
        for label, var in self._sen_vars.items():
            var.set(label in dm_labels)
        self._all_sen_var.set(False)

    # ── Preset ──────────────────────────────────────────────────────────

    def _apply_preset(self, _event=None):
        preset = PRESETS.get(self.preset_var.get(), {})
        if not preset:
            return
        if "emp_min" in preset:
            self.emp_min_var.set(str(preset["emp_min"]))
        if "emp_max" in preset:
            self.emp_max_var.set(str(preset["emp_max"]))
        if "excl_staffing" in preset:
            self.excl_staffing.set(preset["excl_staffing"])

        # industries
        selected_ind = preset.get("industries", None)
        if selected_ind is not None:
            use_all = (len(selected_ind) == 0)
            self._all_ind_var.set(use_all)
            for name, var in self._ind_vars.items():
                var.set(not use_all and name in selected_ind)

        # seniority
        selected_sen = preset.get("seniority", None)
        if selected_sen is not None:
            self._all_sen_var.set(False)
            sen_values = {v for _, v in SENIORITY_OPTIONS}
            for label, var in self._sen_vars.items():
                api_val = dict(SENIORITY_OPTIONS)[label]
                var.set(api_val in selected_sen)
        else:
            self._all_sen_var.set(True)
            for var in self._sen_vars.values():
                var.set(False)

    # ── Build ICPConfig from UI state ────────────────────────────────────

    def _build_icp_config(self) -> ICPConfig:
        try:
            emp_min = int(self.emp_min_var.get() or "1")
        except ValueError:
            emp_min = 1
        try:
            emp_max = int(self.emp_max_var.get() or "9999999")
        except ValueError:
            emp_max = 9_999_999

        # Industries
        use_all_ind = self._all_ind_var.get()
        if use_all_ind:
            tgt_pat = None
            apply_bonus = False
        else:
            selected = [n for n, v in self._ind_vars.items() if v.get()]
            if selected:
                # Exclude "Staffing" from bonus if it's selected — keep it separate
                tgt_pat = "|".join(f"(?:{INDUSTRY_GROUPS[n]})" for n in selected
                                   if n != "Staffing & Recruiting")
                apply_bonus = bool(tgt_pat)
            else:
                tgt_pat = None
                apply_bonus = False

        # Seniority
        use_all_sen = self._all_sen_var.get()
        if use_all_sen or not any(v.get() for v in self._sen_vars.values()):
            seniority_include = None
        else:
            seniority_include = [dict(SENIORITY_OPTIONS)[lbl]
                                 for lbl, var in self._sen_vars.items() if var.get()]

        return ICPConfig(
            emp_min=emp_min,
            emp_max=emp_max,
            apply_industry_filter=self.excl_staffing.get(),
            apply_industry_bonus=apply_bonus,
            target_industries_pat=tgt_pat,
            seniority_include=seniority_include,
            require_email=self.require_email.get(),
            require_phone=self.require_phone.get(),
            require_linkedin=self.require_linkedin.get(),
        )

    # ── Run ──────────────────────────────────────────────────────────────

    def _run(self):
        if self._worker and self._worker.is_alive():
            messagebox.showinfo("Running", "A run is already in progress.")
            return

        input_dir  = Path(self.input_dir_var.get()).expanduser()
        output_dir = Path(self.output_dir_var.get()).expanduser()
        if not input_dir.exists():
            messagebox.showerror("Error", f"Source folder not found:\n{input_dir}")
            return

        icp = self._build_icp_config()
        self.run_btn.configure(state=tk.DISABLED)
        self.status_var.set("Running…")

        # Log config summary
        self._log("=" * 70)
        self._log(f"Source  : {input_dir}")
        self._log(f"Output  : {output_dir}")
        self._log(f"Employees: {icp.emp_min}–{icp.emp_max}")
        selected_ind = [n for n, v in self._ind_vars.items() if v.get()]
        self._log(f"Industries: {'ALL' if self._all_ind_var.get() else (', '.join(selected_ind) or 'none')}")
        if icp.seniority_include:
            self._log(f"Seniority: {', '.join(icp.seniority_include)}")
        else:
            self._log("Seniority: all levels")
        self._log(f"Require email={icp.require_email}  phone={icp.require_phone}  linkedin={icp.require_linkedin}")
        self._log(f"Exclude staffing={icp.apply_industry_filter}  Semantic={self.semantic_var.get()}")
        self._log("")

        def worker():
            try:
                self._run_batched(input_dir, output_dir, icp)
            except Exception as exc:
                import traceback
                self._log(f"ERROR: {exc}\n{traceback.format_exc()}")
                self.root.after(0, lambda: messagebox.showerror("Error", str(exc)))
            finally:
                self.root.after(0, lambda: self.run_btn.configure(state=tk.NORMAL))
                self.root.after(0, lambda: self.status_var.set("Done."))

        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()

    def _run_batched(self, input_dir: Path, output_dir: Path, icp: ICPConfig):
        import pandas as pd

        csv_files = sorted(input_dir.rglob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {input_dir}")

        self._log(f"Found {len(csv_files)} CSV files — grouping into batches…")
        batches = _group_files(csv_files, self.BATCH_SIZE)
        self._log(f"{len(batches)} batch(es) of ~{self.BATCH_SIZE:,} rows each\n")

        batch_dirs: list[Path] = []
        t_start = time.time()

        for i, batch_files in enumerate(batches, 1):
            bd = output_dir / f"batch_{i:03d}"
            bd.mkdir(parents=True, exist_ok=True)
            self._log(f"[Batch {i}/{len(batches)}]  {len(batch_files)} files…")
            t0 = time.time()
            pipeline = UnifiedPipeline(input_dir=input_dir, output_dir=bd)
            result = pipeline.run(
                enable_semantic=self.semantic_var.get(),
                file_list=batch_files,
                icp_config=icp,
                logger=self._log,
            )
            self._log(f"  ✓ {time.time()-t0:.1f}s  input {result.total_input_rows:,} → deduped {result.deduplicated_rows:,}\n")
            batch_dirs.append(bd)

        # Merge
        self._log("Merging batches…")
        final_dir = output_dir / "FINAL"
        final_dir.mkdir(parents=True, exist_ok=True)
        totals: dict[str, int] = {}
        reject_df_parts: list[pd.DataFrame] = []

        for bucket in ("ACCEPT", "REVIEW", "REJECT"):
            frames = []
            for bd in batch_dirs:
                fp = bd / f"MASTER_{bucket}.csv"
                if fp.exists() and fp.stat().st_size > 100:
                    try:
                        frames.append(pd.read_csv(fp, low_memory=False))
                    except Exception:
                        pass
            if not frames:
                totals[bucket] = 0
                continue
            merged = pd.concat(frames, ignore_index=True)
            if bucket != "REJECT":
                from src.cleaner import DataCleaner
                merged = DataCleaner.deduplicate(merged)
            merged.to_csv(final_dir / f"MASTER_{bucket}.csv", index=False)
            totals[bucket] = len(merged)
            self._log(f"  {bucket}: {len(merged):,}")
            if bucket == "REJECT":
                reject_df_parts = frames   # keep for table

        self._last_output = final_dir
        elapsed = time.time() - t_start
        self._log(f"\nDone in {elapsed/60:.1f} min  →  {final_dir}")

        # Build results table on main thread
        accept_file = final_dir / "MASTER_ACCEPT.csv"
        reject_file = final_dir / "MASTER_REJECT.csv"
        self.root.after(0, lambda: self._populate_table(accept_file, reject_file, totals))

    # ── Results table ────────────────────────────────────────────────────

    def _populate_table(self, accept_file: Path, reject_file: Path, totals: dict):
        import pandas as pd

        self.tree.delete(*self.tree.get_children())

        if not accept_file.exists():
            return

        acc = pd.read_csv(accept_file, usecols=["industry"], low_memory=False)
        acc["industry"] = acc["industry"].fillna("").astype(str).str.lower()

        rej_by_reason: dict[str, pd.Series] = {}
        if reject_file.exists():
            try:
                rej = pd.read_csv(reject_file, usecols=["industry", "reject_reason"], low_memory=False)
                rej["industry"] = rej["industry"].fillna("").astype(str).str.lower()
                for reason in ("employee_count", "bad_title"):
                    sub = rej[rej["reject_reason"] == reason]
                    rej_by_reason[reason] = sub["industry"]
            except Exception:
                pass

        total_accept = totals.get("ACCEPT", 0)
        rows = []
        for group, pat in INDUSTRY_GROUPS.items():
            a = int(acc["industry"].str.contains(pat, regex=True, na=False).sum())
            re_emp = re_ttl = 0
            if "employee_count" in rej_by_reason:
                re_emp = int(rej_by_reason["employee_count"].str.contains(pat, regex=True, na=False).sum())
            if "bad_title" in rej_by_reason:
                re_ttl = int(rej_by_reason["bad_title"].str.contains(pat, regex=True, na=False).sum())
            rows.append((group, a, re_emp, re_ttl))

        # Sort by ACCEPT desc
        rows.sort(key=lambda r: r[1], reverse=True)

        for group, a, re_emp, re_ttl in rows:
            pct = f"{a/total_accept*100:.1f}%" if total_accept else "—"
            tag = "hit" if a > 0 else "zero"
            self.tree.insert("", tk.END,
                             values=(group, f"{a:,}", pct,
                                     f"{re_emp:,}" if re_emp else "—",
                                     f"{re_ttl:,}" if re_ttl else "—"),
                             tags=(tag,))

        total_rej = totals.get("REJECT", 0)
        rate = f"{total_accept/(total_accept+total_rej)*100:.1f}%" if (total_accept + total_rej) else "—"
        self.summary_var.set(
            f"ACCEPT: {total_accept:,}   REJECT: {total_rej:,}   Accept rate: {rate}"
        )


# ────────────────────────────────────────────────────────────────────────────
# Entry
# ────────────────────────────────────────────────────────────────────────────
def launch_dashboard() -> None:
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except Exception:
        pass
    DashboardApp(root)
    root.mainloop()
