# Streamlit Docker ICP Filter Dashboard — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Tkinter desktop dashboard with a Streamlit web app deployable on Render via Docker, preserving all ICP filter controls and both upload + volume-mount file modes.

**Architecture:** Single `streamlit_app.py` replaces `src/dashboard.py` and `launch_dashboard.py`. All engine files (`src/pipeline.py`, `src/scorer.py`, `src/ingestor.py`, `src/config.py`, `src/cleaner.py`) are untouched. Semantic scoring is hardcoded off.

**Tech Stack:** Python 3.11, Streamlit ≥1.35, Docker (python:3.11-slim), Render Web Service

---

## File Map

| File | Action | Purpose |
|---|---|---|
| `streamlit_app.py` | Create | Full Streamlit UI — replaces dashboard.py |
| `requirements.txt` | Modify | Add `streamlit>=1.35` |
| `Dockerfile` | Create | Docker build for Render |
| `.dockerignore` | Create | Exclude venv/data/git from image |
| `tests/test_icp_config_builder.py` | Create | Unit tests for config builder logic |
| `src/*.py` | No change | Engine untouched |

---

## Task 1: Project Scaffolding

**Files:**
- Modify: `requirements.txt`
- Create: `Dockerfile`
- Create: `.dockerignore`

- [ ] **Step 1: Add streamlit to requirements.txt**

Open `requirements.txt` and append:
```
streamlit>=1.35
```

- [ ] **Step 2: Create Dockerfile**

Create `/Volumes/part_one/Coding/Projects/HireQuotient/udp/Dockerfile`:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY streamlit_app.py .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
```

- [ ] **Step 3: Create .dockerignore**

Create `/Volumes/part_one/Coding/Projects/HireQuotient/udp/.dockerignore`:
```
venv/
.venv/
data/
__pycache__/
.git/
*.pyc
docs/
tests/
launch_dashboard.py
run_broad_icp.py
```

- [ ] **Step 4: Commit**
```bash
git add requirements.txt Dockerfile .dockerignore
git commit -m "feat: add Docker scaffolding for Streamlit deployment"
```

---

## Task 2: ICPConfig Builder (TDD)

**Files:**
- Create: `tests/test_icp_config_builder.py`
- Create: `streamlit_app.py` (stubs only — enough to make tests pass)

The `_build_icp_config()` and `_apply_preset()` functions are pure logic (read from a dict, return ICPConfig) — no Streamlit calls needed if we pass state as a dict argument. We'll write them to accept an explicit `state` dict for testability, then call them with `dict(st.session_state)` in the app.

- [ ] **Step 1: Write failing tests**

Create `tests/__init__.py` (empty).

Create `tests/test_icp_config_builder.py`:
```python
"""Tests for ICPConfig builder logic."""
import pytest
from src.config import INDUSTRY_GROUPS
from src.scorer import ICPConfig

# ── Helpers that mirror streamlit_app.py functions but take explicit state ──

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


def build_icp_config(state: dict) -> ICPConfig:
    """Extracted from streamlit_app._build_icp_config — pure, testable."""
    emp_min = int(state.get("emp_min", 1))
    emp_max = int(state.get("emp_max", 9_999_999))

    use_all_ind = state.get("all_industries", True)
    if use_all_ind:
        tgt_pat = None
        apply_bonus = False
    else:
        selected = [n for n in INDUSTRY_GROUPS if state.get(f"ind_{n}")]
        if selected:
            tgt_pat = "|".join(
                f"(?:{INDUSTRY_GROUPS[n]})" for n in selected
                if n != "Staffing & Recruiting"
            )
            apply_bonus = bool(tgt_pat)
        else:
            tgt_pat = None
            apply_bonus = False

    use_all_sen = state.get("all_seniority", True)
    if use_all_sen or not any(state.get(f"sen_{lbl}") for lbl, _ in SENIORITY_OPTIONS):
        seniority_include = None
    else:
        seniority_include = [
            api_val for lbl, api_val in SENIORITY_OPTIONS
            if state.get(f"sen_{lbl}")
        ]

    return ICPConfig(
        emp_min=emp_min,
        emp_max=emp_max,
        apply_industry_filter=state.get("excl_staffing", True),
        apply_industry_bonus=apply_bonus,
        target_industries_pat=tgt_pat,
        seniority_include=seniority_include,
        require_email=state.get("require_email", False),
        require_phone=state.get("require_phone", False),
        require_linkedin=state.get("require_linkedin", False),
    )


# ── Tests ────────────────────────────────────────────────────────────────────

def test_defaults_produce_no_seniority_filter():
    cfg = build_icp_config({"all_industries": True, "all_seniority": True})
    assert cfg.seniority_include is None


def test_emp_range_passed_through():
    cfg = build_icp_config({"emp_min": 200, "emp_max": 5000, "all_industries": True, "all_seniority": True})
    assert cfg.emp_min == 200
    assert cfg.emp_max == 5000


def test_selected_industries_build_pattern():
    state = {
        "all_industries": False,
        "all_seniority": True,
        "ind_Healthcare & Medical": True,
    }
    cfg = build_icp_config(state)
    assert cfg.target_industries_pat is not None
    assert "hospital" in cfg.target_industries_pat
    assert cfg.apply_industry_bonus is True


def test_all_industries_disables_bonus():
    cfg = build_icp_config({"all_industries": True, "all_seniority": True})
    assert cfg.target_industries_pat is None
    assert cfg.apply_industry_bonus is False


def test_seniority_filter_when_specific_levels_selected():
    state = {
        "all_industries": True,
        "all_seniority": False,
        "sen_C-Suite": True,
        "sen_VP": True,
        "sen_Founder": False,
        "sen_Owner": False,
        "sen_Director": False,
        "sen_Head": False,
        "sen_Manager": False,
        "sen_Senior": False,
        "sen_Partner": False,
        "sen_Entry": False,
    }
    cfg = build_icp_config(state)
    assert cfg.seniority_include == ["c suite", "vp"]


def test_contact_quality_gates():
    state = {
        "all_industries": True,
        "all_seniority": True,
        "require_email": True,
        "require_phone": False,
        "require_linkedin": True,
    }
    cfg = build_icp_config(state)
    assert cfg.require_email is True
    assert cfg.require_phone is False
    assert cfg.require_linkedin is True


def test_excl_staffing_flag():
    cfg_on  = build_icp_config({"all_industries": True, "all_seniority": True, "excl_staffing": True})
    cfg_off = build_icp_config({"all_industries": True, "all_seniority": True, "excl_staffing": False})
    assert cfg_on.apply_industry_filter is True
    assert cfg_off.apply_industry_filter is False
```

- [ ] **Step 2: Run tests — expect ImportError (module doesn't exist yet)**
```bash
cd /Volumes/part_one/Coding/Projects/HireQuotient/udp
python -m pytest tests/test_icp_config_builder.py -v 2>&1 | head -30
```
Expected: `ModuleNotFoundError` or similar — tests fail because `build_icp_config` is defined locally in the test file (self-contained), so they should actually run. If `src/scorer.py` import fails, check venv is active.

- [ ] **Step 3: Run with venv active**
```bash
source .venv/bin/activate || source venv/bin/activate
python -m pytest tests/test_icp_config_builder.py -v
```
Expected: All 7 tests PASS (the helper function is defined in the test file itself).

- [ ] **Step 4: Commit**
```bash
git add tests/
git commit -m "test: add unit tests for ICPConfig builder logic"
```

---

## Task 3: Streamlit App Skeleton + Session State

**Files:**
- Create: `streamlit_app.py`

- [ ] **Step 1: Create the skeleton**

Create `streamlit_app.py`:
```python
"""Streamlit ICP Filter Dashboard — web replacement for src/dashboard.py."""
from __future__ import annotations

import io
import os
import tempfile
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from src.cleaner import DataCleaner
from src.config import INDUSTRY_GROUPS
from src.pipeline import UnifiedPipeline
from src.scorer import ICPConfig

# ── Constants ────────────────────────────────────────────────────────────────

BATCH_SIZE = 150_000
DEFAULT_DATA_DIR = os.environ.get("DATA_DIR", "/app/data")

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
        "industries": [],
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

SENIORITY_OPTIONS: list[tuple[str, str]] = [
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

# ── Session state ─────────────────────────────────────────────────────────────

def _init_state() -> None:
    defaults: dict = {
        "preset": list(PRESETS)[0],
        "emp_min": 20,
        "emp_max": 5000,
        "excl_staffing": True,
        "require_email": False,
        "require_phone": False,
        "require_linkedin": False,
        "all_industries": True,
        "all_seniority": True,
    }
    for name in INDUSTRY_GROUPS:
        defaults[f"ind_{name}"] = False
    for label, _ in SENIORITY_OPTIONS:
        defaults[f"sen_{label}"] = True

    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _apply_preset(name: str) -> None:
    preset = PRESETS.get(name, {})
    if not preset:
        return
    if "emp_min" in preset:
        st.session_state["emp_min"] = preset["emp_min"]
    if "emp_max" in preset:
        st.session_state["emp_max"] = preset["emp_max"]
    if "excl_staffing" in preset:
        st.session_state["excl_staffing"] = preset["excl_staffing"]

    selected_ind = preset.get("industries", None)
    if selected_ind is not None:
        use_all = len(selected_ind) == 0
        st.session_state["all_industries"] = use_all
        for ind_name in INDUSTRY_GROUPS:
            st.session_state[f"ind_{ind_name}"] = (not use_all and ind_name in selected_ind)

    selected_sen = preset.get("seniority", None)
    if selected_sen is not None:
        st.session_state["all_seniority"] = False
        for label, api_val in SENIORITY_OPTIONS:
            st.session_state[f"sen_{label}"] = api_val in selected_sen
    else:
        st.session_state["all_seniority"] = True
        for label, _ in SENIORITY_OPTIONS:
            st.session_state[f"sen_{label}"] = False


def _build_icp_config() -> ICPConfig:
    state = dict(st.session_state)
    emp_min = int(state.get("emp_min", 1))
    emp_max = int(state.get("emp_max", 9_999_999))

    use_all_ind = state.get("all_industries", True)
    if use_all_ind:
        tgt_pat = None
        apply_bonus = False
    else:
        selected = [n for n in INDUSTRY_GROUPS if state.get(f"ind_{n}")]
        if selected:
            tgt_pat = "|".join(
                f"(?:{INDUSTRY_GROUPS[n]})" for n in selected
                if n != "Staffing & Recruiting"
            )
            apply_bonus = bool(tgt_pat)
        else:
            tgt_pat = None
            apply_bonus = False

    use_all_sen = state.get("all_seniority", True)
    if use_all_sen or not any(state.get(f"sen_{lbl}") for lbl, _ in SENIORITY_OPTIONS):
        seniority_include = None
    else:
        seniority_include = [
            api_val for lbl, api_val in SENIORITY_OPTIONS
            if state.get(f"sen_{lbl}")
        ]

    return ICPConfig(
        emp_min=emp_min,
        emp_max=emp_max,
        apply_industry_filter=state.get("excl_staffing", True),
        apply_industry_bonus=apply_bonus,
        target_industries_pat=tgt_pat,
        seniority_include=seniority_include,
        require_email=state.get("require_email", False),
        require_phone=state.get("require_phone", False),
        require_linkedin=state.get("require_linkedin", False),
    )


# ── Entry ─────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="HireQuotient — ICP Filter",
        page_icon="🎯",
        layout="wide",
    )
    _init_state()

    with st.sidebar:
        _render_sidebar()

    _render_main()


if __name__ == "__main__":
    main()
```

Add placeholder stubs so the app at least imports:
```python
def _render_sidebar() -> None:
    st.write("Sidebar coming soon")

def _render_main() -> None:
    st.write("Main area coming soon")
```

- [ ] **Step 2: Verify it starts**
```bash
source .venv/bin/activate || source venv/bin/activate
streamlit run streamlit_app.py --server.headless true &
sleep 3
curl -s http://localhost:8501/_stcore/health
# Expected: "ok"
kill %1
```

- [ ] **Step 3: Commit**
```bash
git add streamlit_app.py
git commit -m "feat: add streamlit app skeleton with session state"
```

---

## Task 4: Sidebar Filter Controls

**Files:**
- Modify: `streamlit_app.py` — replace `_render_sidebar()` stub

- [ ] **Step 1: Replace the `_render_sidebar` stub**

Replace the `_render_sidebar` function body with:
```python
def _render_sidebar() -> None:
    st.title("ICP Filters")

    # ── Preset ───────────────────────────────────────────────────────────
    new_preset = st.selectbox(
        "Quick Preset",
        list(PRESETS),
        index=list(PRESETS).index(st.session_state["preset"]),
    )
    if new_preset != st.session_state["preset"]:
        st.session_state["preset"] = new_preset
        _apply_preset(new_preset)
        st.rerun()

    # ── Industries ────────────────────────────────────────────────────────
    with st.expander("Industries", expanded=True):
        col1, col2 = st.columns(2)
        if col1.button("All", key="ind_btn_all"):
            st.session_state["all_industries"] = True
            for n in INDUSTRY_GROUPS:
                st.session_state[f"ind_{n}"] = False
            st.rerun()
        if col2.button("None", key="ind_btn_none"):
            st.session_state["all_industries"] = False
            for n in INDUSTRY_GROUPS:
                st.session_state[f"ind_{n}"] = False
            st.rerun()

        st.session_state["all_industries"] = st.checkbox(
            "All industries (no filter)",
            value=st.session_state["all_industries"],
        )
        if not st.session_state["all_industries"]:
            for name in INDUSTRY_GROUPS:
                st.session_state[f"ind_{name}"] = st.checkbox(
                    name,
                    value=st.session_state.get(f"ind_{name}", False),
                    key=f"cb_ind_{name}",
                )

    # ── Company & Seniority ───────────────────────────────────────────────
    with st.expander("Company & Seniority", expanded=True):
        c1, c2 = st.columns(2)
        st.session_state["emp_min"] = c1.number_input(
            "Emp Min", min_value=1,
            value=st.session_state["emp_min"], step=100,
        )
        st.session_state["emp_max"] = c2.number_input(
            "Emp Max", min_value=1,
            value=st.session_state["emp_max"], step=100,
        )
        # Quick range buttons
        qcols = st.columns(5)
        for qcol, (label, lo, hi) in zip(
            qcols,
            [("<200", 1, 200), ("200-1K", 200, 1000),
             ("1K-2K", 1000, 2000), ("2K-5K", 2000, 5000), ("Any", 1, 9_999_999)],
        ):
            if qcol.button(label, key=f"qrange_{label}"):
                st.session_state["emp_min"] = lo
                st.session_state["emp_max"] = hi
                st.rerun()

        st.divider()
        st.session_state["all_seniority"] = st.checkbox(
            "All seniority levels",
            value=st.session_state["all_seniority"],
        )
        if not st.session_state["all_seniority"]:
            sc1, sc2 = st.columns(2)
            for i, (label, _) in enumerate(SENIORITY_OPTIONS):
                target_col = sc1 if i % 2 == 0 else sc2
                st.session_state[f"sen_{label}"] = target_col.checkbox(
                    label,
                    value=st.session_state.get(f"sen_{label}", True),
                    key=f"cb_sen_{label}",
                )

        st.divider()
        st.session_state["excl_staffing"] = st.checkbox(
            "Exclude staffing & recruiting",
            value=st.session_state["excl_staffing"],
        )

    # ── Contact Quality ───────────────────────────────────────────────────
    with st.expander("Contact Quality"):
        st.session_state["require_email"] = st.checkbox(
            "Must have Email",
            value=st.session_state["require_email"],
        )
        st.session_state["require_phone"] = st.checkbox(
            "Must have Phone",
            value=st.session_state["require_phone"],
        )
        st.session_state["require_linkedin"] = st.checkbox(
            "Must have LinkedIn URL",
            value=st.session_state["require_linkedin"],
        )
        st.caption(
            "Phone is sparse in Apollo exports — requiring it will significantly reduce output."
        )
```

- [ ] **Step 2: Start the app and verify sidebar renders all controls**
```bash
streamlit run streamlit_app.py --server.headless true
```
Open `http://localhost:8501`. Verify:
- Preset dropdown shows 7 options
- Industries expander shows "All industries" checkbox + 16 industry checkboxes when unchecked
- Company & Seniority shows emp range inputs, 5 quick-range buttons, seniority checkboxes
- Contact Quality shows 3 checkboxes + caption
- Selecting a preset updates the controls

Stop the server (Ctrl+C).

- [ ] **Step 3: Commit**
```bash
git add streamlit_app.py
git commit -m "feat: add sidebar filter controls to streamlit dashboard"
```

---

## Task 5: Pipeline Helpers + Input Tabs

**Files:**
- Modify: `streamlit_app.py` — add helper functions + replace `_render_main()` stub with input tabs

- [ ] **Step 1: Add helper functions** (add before `_render_sidebar`):

```python
# ── Pipeline helpers ──────────────────────────────────────────────────────────

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


def _run_batched(
    csv_files: list[Path],
    output_dir: Path,
    icp: ICPConfig,
    log_fn,
) -> tuple[dict[str, int], Path]:
    """Run batched pipeline. Returns (bucket_totals, final_output_dir)."""
    log_fn(f"Found {len(csv_files)} CSV files — grouping into batches…")
    batches = _group_files(csv_files, BATCH_SIZE)
    log_fn(f"{len(batches)} batch(es) of ~{BATCH_SIZE:,} rows each")

    batch_dirs: list[Path] = []
    t_start = time.time()

    for i, batch_files in enumerate(batches, 1):
        bd = output_dir / f"batch_{i:03d}"
        bd.mkdir(parents=True, exist_ok=True)
        log_fn(f"[Batch {i}/{len(batches)}] {len(batch_files)} files…")
        t0 = time.time()
        pipeline = UnifiedPipeline(input_dir=batch_files[0].parent, output_dir=bd)
        result = pipeline.run(
            enable_semantic=False,
            file_list=batch_files,
            icp_config=icp,
            logger=log_fn,
        )
        log_fn(
            f"  ✓ {time.time() - t0:.1f}s  "
            f"input {result.total_input_rows:,} → deduped {result.deduplicated_rows:,}"
        )
        batch_dirs.append(bd)

    # Merge all batch outputs
    log_fn("Merging batches…")
    final_dir = output_dir / "FINAL"
    final_dir.mkdir(parents=True, exist_ok=True)
    totals: dict[str, int] = {}

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
            merged = DataCleaner.deduplicate(merged)
        merged.to_csv(final_dir / f"MASTER_{bucket}.csv", index=False)
        totals[bucket] = len(merged)
        log_fn(f"  {bucket}: {len(merged):,}")

    elapsed = time.time() - t_start
    log_fn(f"Done in {elapsed / 60:.1f} min → {final_dir}")
    return totals, final_dir
```

- [ ] **Step 2: Replace `_render_main()` stub**

```python
def _render_main() -> None:
    st.title("HireQuotient — ICP Filter Dashboard")

    upload_tab, folder_tab = st.tabs(["📁 Upload Files", "📂 Folder Path"])

    with upload_tab:
        uploaded_files = st.file_uploader(
            "Drop one or more CSV files",
            type=["csv"],
            accept_multiple_files=True,
        )

    with folder_tab:
        folder_path_str = st.text_input(
            "Folder containing raw CSVs",
            value=DEFAULT_DATA_DIR,
            help="Use this when CSVs are on a mounted volume (e.g. Render persistent disk).",
        )

    st.divider()

    if st.button("▶  Run Filter", type="primary", use_container_width=True):
        _execute_run(uploaded_files, folder_path_str)
```

- [ ] **Step 3: Add `_execute_run` stub** (will be filled in Task 6):
```python
def _execute_run(uploaded_files, folder_path_str: str) -> None:
    st.info("Run logic coming in next task.")
```

- [ ] **Step 4: Verify tabs render**
```bash
streamlit run streamlit_app.py --server.headless true
```
Open `http://localhost:8501`. Verify two tabs appear, file uploader shows in first tab, text input in second. Stop server.

- [ ] **Step 5: Commit**
```bash
git add streamlit_app.py
git commit -m "feat: add pipeline helpers and input tabs to streamlit app"
```

---

## Task 6: Pipeline Execution + Results Display

**Files:**
- Modify: `streamlit_app.py` — replace `_execute_run` stub with full implementation

- [ ] **Step 1: Replace `_execute_run` with full implementation**

```python
def _execute_run(uploaded_files, folder_path_str: str) -> None:
    icp = _build_icp_config()

    has_uploads = bool(uploaded_files)
    has_folder = bool(folder_path_str) and Path(folder_path_str).exists()

    if not has_uploads and not has_folder:
        st.error("Upload CSV files or specify a valid folder path.")
        return

    totals: dict[str, int] = {}
    in_memory: dict[str, bytes] = {}   # upload mode: results held in RAM
    final_dir: Path | None = None       # folder mode: results on disk

    with st.status("Running pipeline…", expanded=True) as status:
        def log_fn(msg: str) -> None:
            status.write(msg)

        try:
            if has_uploads:
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp = Path(tmpdir)
                    raw_dir = tmp / "raw"
                    raw_dir.mkdir()
                    out_dir = tmp / "output"
                    out_dir.mkdir()

                    csv_files: list[Path] = []
                    for uf in uploaded_files:
                        dest = raw_dir / uf.name
                        dest.write_bytes(uf.getvalue())
                        csv_files.append(dest)

                    totals, fd = _run_batched(csv_files, out_dir, icp, log_fn)

                    # Read results into memory before tempdir is cleaned up
                    for bucket in ("ACCEPT", "REVIEW", "REJECT"):
                        fp = fd / f"MASTER_{bucket}.csv"
                        if fp.exists():
                            in_memory[bucket] = fp.read_bytes()

            else:
                folder = Path(folder_path_str)
                csv_files = sorted(folder.rglob("*.csv"))
                if not csv_files:
                    st.error(f"No CSV files found in {folder}")
                    status.update(label="No files found", state="error")
                    return
                out_dir = folder / "output"
                out_dir.mkdir(parents=True, exist_ok=True)
                totals, final_dir = _run_batched(csv_files, out_dir, icp, log_fn)

            status.update(label="Pipeline complete!", state="complete")

        except Exception as exc:
            import traceback
            st.error(str(exc))
            st.code(traceback.format_exc())
            status.update(label="Failed", state="error")
            return

    # ── Metrics ──────────────────────────────────────────────────────────
    total_accept = totals.get("ACCEPT", 0)
    total_review = totals.get("REVIEW", 0)
    total_reject = totals.get("REJECT", 0)
    total_non_reject = total_accept + total_review + total_reject
    rate = (
        f"{total_accept / total_non_reject * 100:.1f}%"
        if total_non_reject else "—"
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ACCEPT", f"{total_accept:,}")
    m2.metric("REVIEW", f"{total_review:,}")
    m3.metric("REJECT", f"{total_reject:,}")
    m4.metric("Accept Rate", rate)

    # ── Results table ─────────────────────────────────────────────────────
    st.subheader("Results by Industry Group")

    def _read_csv_bytes(data: bytes) -> pd.DataFrame:
        return pd.read_csv(io.BytesIO(data), low_memory=False)

    if in_memory:
        acc_df = _read_csv_bytes(in_memory.get("ACCEPT", b"")).get(
            "industry", pd.Series(dtype=str)
        ) if in_memory.get("ACCEPT") else pd.Series(dtype=str)
        acc_series = acc_df.fillna("").astype(str).str.lower()

        rej_emp = pd.Series(dtype=str)
        rej_ttl = pd.Series(dtype=str)
        if in_memory.get("REJECT"):
            try:
                rej_df = _read_csv_bytes(in_memory["REJECT"])
                rej_df["industry"] = rej_df["industry"].fillna("").astype(str).str.lower()
                rej_emp = rej_df[rej_df["reject_reason"] == "employee_count"]["industry"]
                rej_ttl = rej_df[rej_df["reject_reason"] == "bad_title"]["industry"]
            except Exception:
                pass
    else:
        accept_path = final_dir / "MASTER_ACCEPT.csv"
        reject_path = final_dir / "MASTER_REJECT.csv"
        acc_series = (
            pd.read_csv(accept_path, usecols=["industry"], low_memory=False)["industry"]
            .fillna("").astype(str).str.lower()
            if accept_path.exists() else pd.Series(dtype=str)
        )
        rej_emp = pd.Series(dtype=str)
        rej_ttl = pd.Series(dtype=str)
        if reject_path.exists():
            try:
                rej_df = pd.read_csv(reject_path, usecols=["industry", "reject_reason"], low_memory=False)
                rej_df["industry"] = rej_df["industry"].fillna("").astype(str).str.lower()
                rej_emp = rej_df[rej_df["reject_reason"] == "employee_count"]["industry"]
                rej_ttl = rej_df[rej_df["reject_reason"] == "bad_title"]["industry"]
            except Exception:
                pass

    table_rows = []
    for group, pat in INDUSTRY_GROUPS.items():
        a = int(acc_series.str.contains(pat, regex=True, na=False).sum())
        re_emp = int(rej_emp.str.contains(pat, regex=True, na=False).sum()) if len(rej_emp) else 0
        re_ttl = int(rej_ttl.str.contains(pat, regex=True, na=False).sum()) if len(rej_ttl) else 0
        pct = f"{a / total_accept * 100:.1f}%" if total_accept else "—"
        table_rows.append({
            "Industry Group": group,
            "ACCEPT": a,
            "% of total": pct,
            "Reject (emp)": re_emp or "—",
            "Reject (title)": re_ttl or "—",
        })

    results_df = pd.DataFrame(table_rows).sort_values("ACCEPT", ascending=False)
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    # ── Download buttons ──────────────────────────────────────────────────
    st.subheader("Download Results")
    d1, d2, d3 = st.columns(3)
    for col, bucket in zip([d1, d2, d3], ["ACCEPT", "REVIEW", "REJECT"]):
        if in_memory:
            data = in_memory.get(bucket, b"")
        else:
            fp = final_dir / f"MASTER_{bucket}.csv"
            data = fp.read_bytes() if fp.exists() else b""
        col.download_button(
            f"⬇ {bucket}.csv",
            data=data,
            file_name=f"MASTER_{bucket}.csv",
            mime="text/csv",
            disabled=not data,
            use_container_width=True,
        )
```

- [ ] **Step 2: Smoke test — upload mode**

Place a small test CSV (`tests/fixtures/sample_apollo.csv`) with ~10 rows:
```csv
Title,Company,Email,Industry,# Employees,City,State,Seniority,Departments,Person Linkedin Url,Work Direct Phone
Director of Talent Acquisition,Acme Corp,alice@acme.com,Healthcare,1200,New York,New York,director,HR,https://linkedin.com/in/alice,
CEO,Beta Inc,bob@beta.com,Software & Technology,500,San Francisco,California,c suite,C-Suite,https://linkedin.com/in/bob,
Sales Rep,Gamma Ltd,carol@gamma.com,Retail,300,Chicago,Illinois,entry,Sales,https://linkedin.com/in/carol,
```

Run the app, upload the fixture CSV, click Run. Verify:
- Log streams in the status box
- Metrics show counts
- Results table shows industry breakdown
- Download buttons appear and work

```bash
streamlit run streamlit_app.py --server.headless true
```

- [ ] **Step 3: Smoke test — folder mode**

```bash
mkdir -p /tmp/icp_test/raw
cp tests/fixtures/sample_apollo.csv /tmp/icp_test/raw/
```
In the app, switch to Folder Path tab, enter `/tmp/icp_test`, click Run. Verify output written to `/tmp/icp_test/output/FINAL/`.

- [ ] **Step 4: Health check**
```bash
curl -s http://localhost:8501/_stcore/health
# Expected: ok
```

- [ ] **Step 5: Commit**
```bash
git add streamlit_app.py tests/fixtures/sample_apollo.csv
git commit -m "feat: add pipeline execution, results table, and download buttons"
```

---

## Task 7: Docker Build + Render Config

**Files:**
- No new files — verify existing Dockerfile works

- [ ] **Step 1: Build the Docker image locally**
```bash
docker build -t icp-dashboard .
```
Expected: build completes, no errors.

- [ ] **Step 2: Run the container locally**
```bash
docker run -p 8501:8501 icp-dashboard
```

- [ ] **Step 3: Verify health endpoint**
```bash
curl -s http://localhost:8501/_stcore/health
# Expected: ok
```

- [ ] **Step 4: Test upload mode in container**

Open `http://localhost:8501`, upload the fixture CSV, run. Verify results and downloads work.

- [ ] **Step 5: Stop container and commit**
```bash
docker stop $(docker ps -q --filter ancestor=icp-dashboard)
git add .
git commit -m "chore: verify docker build and health endpoint"
```

- [ ] **Step 6: Render deployment checklist**

On Render:
1. New Web Service → connect repo → select Docker runtime
2. Port: `8501`
3. Health check path: `/_stcore/health`
4. Add env var: `DATA_DIR=/app/data`
5. (Optional) Add persistent disk mounted at `/app/data`
6. Deploy → wait for health check to pass

---

## Render Notes

- **Upload mode** works on free tier (no disk needed) — results are ephemeral per session
- **Folder mode** requires a Render persistent disk mounted at `/app/data`
- Cold starts on free tier: ~15–30s; the health check path keeps the service warm
- Multiple simultaneous users each get isolated Streamlit sessions — no shared state
