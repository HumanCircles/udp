# Streamlit Docker ICP Filter Dashboard — Design Spec
_Date: 2026-04-16_

## Overview

Port the existing Tkinter desktop dashboard (`src/dashboard.py`) to a Streamlit web app deployed on Render via Docker. All ICP filtering capabilities are preserved. The pipeline engine (`src/pipeline.py`, `src/scorer.py`, `src/ingestor.py`, `src/config.py`, `src/cleaner.py`) is **not modified**.

---

## Architecture

```
udp/
├── streamlit_app.py        # NEW — replaces src/dashboard.py + launch_dashboard.py
├── Dockerfile              # NEW
├── .dockerignore           # NEW
├── requirements.txt        # UPDATE — add streamlit>=1.35
└── src/                    # UNCHANGED
    ├── config.py
    ├── pipeline.py
    ├── scorer.py
    ├── ingestor.py
    └── cleaner.py
```

---

## UI Layout

**Sidebar** (always visible):
- Quick Preset dropdown (same 6 presets as Tkinter version)
- `st.expander` — Industries (all 16 groups, All/None toggles)
- `st.expander` — Company & Seniority (emp range, quick buttons, seniority checkboxes, exclude staffing toggle)
- `st.expander` — Contact Quality (require email / phone / LinkedIn)
- Run Filter button

**Main area**:
- Input mode via `st.tabs(["Upload Files", "Folder Path"])`:
  - Upload tab: `st.file_uploader(accept_multiple_files=True, type=["csv"])`
  - Folder tab: text input defaulting to `DATA_DIR` env var (fallback: `/app/data`)
- Run log: `st.status()` expander, log lines streamed via `st.write()` during pipeline execution
- Results: `st.dataframe` (industry breakdown table, sortable)
- Download buttons: ACCEPT / REVIEW / REJECT CSVs

---

## File I/O

**Upload mode:**
1. User uploads CSVs via browser
2. Files written to `tempfile.TemporaryDirectory` for the run duration
3. `UnifiedPipeline` runs against temp dir
4. Output CSVs read into memory as bytes
5. `st.download_button` serves them — no files persist after session

**Folder path mode (volume mount):**
1. User specifies a folder path (pre-filled from `DATA_DIR` env var)
2. Pipeline reads `{path}/*.csv` (rglob)
3. Output written to `{path}/output/`
4. `st.download_button` reads output files from disk

---

## ICP Filters (all preserved)

| Filter | Control |
|---|---|
| Quick preset | `st.selectbox` — 6 presets |
| Industry groups | 16 `st.checkbox` items + All/None buttons |
| Employee count | Min/Max `st.number_input` + quick-range `st.button` |
| Seniority | 10 `st.checkbox` items + DM-only shortcut |
| Exclude staffing | `st.checkbox` |
| Require email/phone/LinkedIn | 3 `st.checkbox` items |

Semantic scoring: **hardcoded `False`** — removed from UI entirely.

---

## Docker

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

`.dockerignore`:
```
venv/
.venv/
data/
__pycache__/
.git/
*.pyc
```

---

## Render Deployment

| Setting | Value |
|---|---|
| Service type | Web Service (Docker) |
| Port | 8501 |
| Health check path | `/_stcore/health` |
| Env var | `DATA_DIR=/app/data` |
| Persistent disk (optional) | Mounted at `/app/data` |

The health endpoint `/_stcore/health` is built into Streamlit — no extra code required.

---

## Constraints

- No authentication (open URL)
- No semantic scoring in cloud version
- Render free tier: upload mode always works; folder mode requires a paid persistent disk
- Each Streamlit session is isolated — no shared state between users
