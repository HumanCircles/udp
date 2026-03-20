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
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Desktop app (recommended)

```bash
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
python run.py --cli
```

Rules-only mode (no semantic model/FAISS):

```bash
python run.py --cli --no-semantic
```

Custom local folders:

```bash
python run.py --cli --input-dir /path/to/raw --output-dir /path/to/output
```

