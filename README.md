# Unified ICP Pipeline

One project to ingest PhantomBuster, Apollo, and Aggregated lead CSV exports, normalize to a universal schema, deduplicate records, and score ICP fit into `ACCEPT` / `REVIEW` / `REJECT`.

## Structure

- `data/raw/`: source CSVs (mixed schemas allowed)
- `data/output/`: pipeline outputs
- `src/config.py`: keywords, model config, default paths
- `src/ingestor.py`: source auto-detection and mapping
- `src/cleaner.py`: normalization and deduplication
- `src/scorer.py`: rule + optional semantic scoring
- `src/pipeline.py`: orchestration
- `run.py`: command-line entrypoint

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Always run the app from the activated `.venv`.

## Hardware support (auto-detected)

The scorer automatically selects the best available backend:

- **NVIDIA GPU**: uses CUDA via PyTorch, and uses FAISS GPU index if installed.
- **Apple Silicon**: uses MPS acceleration automatically.
- **CPU fallback**: used when no GPU backend is available.

### Optional acceleration packages

For NVIDIA GPUs you can use max speed:

```bash
source .venv/bin/activate
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu121
pip install faiss-gpu-cu12
```

For Apple Silicon users:

```bash
source .venv/bin/activate
pip install --upgrade torch
```

If FAISS is not installed, semantic similarity still works via a NumPy fallback.

### Desktop app (recommended)

```bash
source .venv/bin/activate
python run.py
```

This opens a local GUI where users can:

- select a **Source folder** containing mixed CSV files
- select a **Destination folder** for outputs
- toggle semantic scoring on/off
- run the full pipeline with live logs

### CLI mode

Drop all CSV files into `data/raw/`, then run:

```bash
source .venv/bin/activate
python run.py --cli
```

Rules-only mode (no semantic model/FAISS):

```bash
source .venv/bin/activate
python run.py --cli --no-semantic
```

Custom local folders:

```bash
source .venv/bin/activate
python run.py --cli --input-dir /path/to/raw --output-dir /path/to/output
```

