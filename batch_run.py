"""Batch processing runner for the ICP pipeline.

Each batch produces its own output files:
  data/output/batches/BATCH_001_ACCEPT.csv
  data/output/batches/BATCH_001_REVIEW.csv
  data/output/batches/BATCH_001_REJECT.csv
  data/output/batches/BATCH_002_ACCEPT.csv
  ...

Progress is tracked in data/output/batch_state.json so you can stop and
resume at any time.

Usage
-----
  # Process next 1 batch (1L rows):
  python batch_run.py

  # Process next 4 batches (4L rows):
  python batch_run.py --num-batches 4

  # Process ALL remaining rows:
  python batch_run.py --all

  # Skip ML model (rules only — much faster):
  python batch_run.py --no-semantic --num-batches 4

  # Rebuild checkpoint from scratch (re-ingest + re-dedupe + clear progress):
  python batch_run.py --reset
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from src.config import INPUT_DIR, OUTPUT_DIR
from src.ingestor import DataIngestor
from src.cleaner import DataCleaner
from src.scorer import ICPEngine

CHECKPOINT  = OUTPUT_DIR / "checkpoint_deduped.csv"
STATE_FILE  = OUTPUT_DIR / "batch_state.json"
BATCHES_DIR = OUTPUT_DIR / "batches"
BATCH_SIZE_DEFAULT = 100_000  # 1L rows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def build_checkpoint(reset: bool = False) -> pd.DataFrame:
    """Ingest all raw CSVs, deduplicate, save to checkpoint CSV."""
    if CHECKPOINT.exists() and not reset:
        log(f"Loading existing checkpoint: {CHECKPOINT}")
        df = pd.read_csv(CHECKPOINT, low_memory=False)
        log(f"  → {len(df):,} deduplicated rows ready.")
        return df

    log("Building checkpoint: ingesting all raw CSVs …")
    csv_files = sorted(INPUT_DIR.rglob("*.csv"))
    if not csv_files:
        sys.exit(f"No CSVs found in {INPUT_DIR}")

    frames = []
    for i, f in enumerate(csv_files, 1):
        log(f"  [{i}/{len(csv_files)}] {f.name}")
        frames.append(DataIngestor.load_and_map(f))

    master = pd.concat(frames, ignore_index=True)
    log(f"Total rows after ingestion: {len(master):,}")

    clean = DataCleaner.deduplicate(master)
    log(f"After deduplication: {len(clean):,} rows  (removed {len(master)-len(clean):,} dupes)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clean.to_csv(CHECKPOINT, index=False)
    log(f"Checkpoint saved → {CHECKPOINT}")
    return clean


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"next_offset": 0, "batches_done": 0, "total_rows": None}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


def write_batch_results(batch_df: pd.DataFrame, batch_num: int) -> dict[str, int]:
    """Write each bucket to its own per-batch CSV file."""
    BATCHES_DIR.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for bucket in ("ACCEPT", "REVIEW", "REJECT"):
        subset = batch_df[batch_df["bucket"] == bucket]
        counts[bucket] = len(subset)
        out_path = BATCHES_DIR / f"BATCH_{batch_num:03d}_{bucket}.csv"
        subset.to_csv(out_path, index=False)
        log(f"   → {out_path.name}  ({len(subset):,} rows)")
    return counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    num_batches: int | None,
    process_all: bool,
    no_semantic: bool,
    reset: bool,
    batch_size: int,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if reset:
        # Clear old batch files and state so everything starts fresh
        if BATCHES_DIR.exists():
            for f in BATCHES_DIR.glob("BATCH_*.csv"):
                f.unlink()
            log("Cleared old batch output files.")
        STATE_FILE.unlink(missing_ok=True)

    df = build_checkpoint(reset=reset)

    state = load_state() if not reset else {"next_offset": 0, "batches_done": 0, "total_rows": len(df)}
    state["total_rows"] = len(df)
    save_state(state)

    total  = len(df)
    offset = state["next_offset"]

    if offset >= total:
        log(f"All {total:,} rows already processed ({state['batches_done']} batches). Nothing to do.")
        log("Run with --reset to start over.")
        return

    remaining = total - offset
    log(f"Rows remaining: {remaining:,}  |  offset={offset:,}  |  batch_size={batch_size:,}")

    if process_all:
        batches_to_run = (remaining + batch_size - 1) // batch_size
    elif num_batches is not None:
        batches_to_run = num_batches
    else:
        batches_to_run = 1

    log(f"Will process {batches_to_run} batch(es) this run.")
    log(f"Output folder: {BATCHES_DIR}")

    log(f"Initialising ICP scorer (semantic={'enabled' if not no_semantic else 'disabled'}) …")
    engine = ICPEngine(enable_semantic=not no_semantic)

    grand_total: dict[str, int] = {"ACCEPT": 0, "REVIEW": 0, "REJECT": 0}

    for _ in range(batches_to_run):
        if offset >= total:
            log("All rows processed.")
            break

        chunk     = df.iloc[offset: offset + batch_size]
        batch_num = state["batches_done"] + 1
        t0        = time.time()
        log(f"── Batch {batch_num}  rows {offset:,}–{offset+len(chunk)-1:,}  ({len(chunk):,} rows) ──")

        scored = engine.process(chunk.copy())
        counts = write_batch_results(scored, batch_num)

        elapsed = time.time() - t0
        log(
            f"   ACCEPT={counts['ACCEPT']:,}  REVIEW={counts['REVIEW']:,}  "
            f"REJECT={counts['REJECT']:,}   ({elapsed:.1f}s)"
        )

        for k in grand_total:
            grand_total[k] += counts.get(k, 0)

        offset += len(chunk)
        state["next_offset"] = offset
        state["batches_done"] = batch_num
        save_state(state)

    pct = offset / total * 100
    log(
        f"\nDone. Processed {offset:,}/{total:,} rows ({pct:.1f}%)  |  "
        f"Batches completed: {state['batches_done']}"
    )
    log(
        f"This run total → "
        f"ACCEPT={grand_total['ACCEPT']:,}  "
        f"REVIEW={grand_total['REVIEW']:,}  "
        f"REJECT={grand_total['REJECT']:,}"
    )
    if offset < total:
        log(f"{total-offset:,} rows left — run again to continue.")
    else:
        log("All batches complete!")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch ICP pipeline runner")
    p.add_argument(
        "--num-batches", type=int, default=None,
        help="Number of batches (1L rows each) to process in this run (default: 1).",
    )
    p.add_argument(
        "--all", dest="process_all", action="store_true",
        help="Process all remaining rows.",
    )
    p.add_argument(
        "--no-semantic", action="store_true",
        help="Disable ML/semantic scoring — rules only, much faster.",
    )
    p.add_argument(
        "--reset", action="store_true",
        help="Re-ingest, re-dedupe, and clear all progress. Starts fresh.",
    )
    p.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE_DEFAULT,
        help=f"Rows per batch (default: {BATCH_SIZE_DEFAULT:,}).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        num_batches=args.num_batches,
        process_all=args.process_all,
        no_semantic=args.no_semantic,
        reset=args.reset,
        batch_size=args.batch_size,
    )
