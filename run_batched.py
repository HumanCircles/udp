"""Batched pipeline runner.

Processes all CSVs in data/raw/ in file-groups of ~BATCH_SIZE rows each,
then merges and re-deduplicates across all batches into data/output/FINAL/.

Usage:
    python run_batched.py                    # all files, 150k rows/batch, semantic on
    python run_batched.py --batch 100000     # 100k rows per batch
    python run_batched.py --no-semantic      # rules-only (faster)
    python run_batched.py --start 5          # resume from batch 5
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on the path when run directly.
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from src.cleaner import DataCleaner
from src.config import INPUT_DIR, OUTPUT_DIR
from src.pipeline import UnifiedPipeline


def estimate_rows(filepath: Path) -> int:
    """Fast line-count estimate (header counts as 1, so subtract 1)."""
    try:
        with open(filepath, encoding="utf-8", errors="ignore") as fh:
            return sum(1 for _ in fh) - 1
    except Exception:
        return 0


def group_into_batches(files: list[Path], batch_size: int) -> list[list[Path]]:
    """Group files into batches whose combined estimated row count ≈ batch_size."""
    batches: list[list[Path]] = []
    current: list[Path] = []
    current_rows = 0

    for f in files:
        rows = estimate_rows(f)
        if current and current_rows + rows > batch_size:
            batches.append(current)
            current = [f]
            current_rows = rows
        else:
            current.append(f)
            current_rows += rows

    if current:
        batches.append(current)

    return batches


def merge_final(batch_dirs: list[Path], output_dir: Path) -> None:
    """Concatenate batch outputs, re-deduplicate, write to FINAL/."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*64}")
    print(f"Merging {len(batch_dirs)} batch(es) → {output_dir}")

    for bucket in ("ACCEPT", "REVIEW", "REJECT"):
        frames: list[pd.DataFrame] = []
        for bd in batch_dirs:
            fpath = bd / f"MASTER_{bucket}.csv"
            if fpath.exists() and fpath.stat().st_size > 100:
                try:
                    frames.append(pd.read_csv(fpath, low_memory=False))
                except Exception as exc:
                    print(f"  [warn] could not read {fpath}: {exc}")

        if not frames:
            print(f"  {bucket}: no data")
            continue

        merged = pd.concat(frames, ignore_index=True)
        if bucket != "REJECT":
            merged = DataCleaner.deduplicate(merged)

        out_path = output_dir / f"MASTER_{bucket}.csv"
        merged.to_csv(out_path, index=False)
        print(f"  {bucket}: {len(merged):,} records  →  {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batched ICP pipeline runner")
    parser.add_argument("--batch", type=int, default=150_000, metavar="N",
                        help="Target rows per batch (default: 150000)")
    parser.add_argument("--no-semantic", action="store_true",
                        help="Disable semantic scoring (faster, rules only)")
    parser.add_argument("--start", type=int, default=1, metavar="N",
                        help="Resume from batch N (1-indexed, skips earlier batches)")
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    semantic = not args.no_semantic
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    csv_files = sorted(input_dir.rglob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV files in {input_dir}")
    print(f"Estimating row counts (this takes ~10s for large datasets)...")
    total_rows = sum(estimate_rows(f) for f in csv_files)
    print(f"Estimated total rows: {total_rows:,}")

    batches = group_into_batches(csv_files, args.batch)
    print(f"Split into {len(batches)} batch(es) of ~{args.batch:,} rows each")
    print(f"Semantic scoring: {'ON' if semantic else 'OFF'}")
    if args.start > 1:
        print(f"Resuming from batch {args.start}")

    completed_batch_dirs: list[Path] = []
    # Include already-completed batches in the final merge if resuming.
    for i in range(1, args.start):
        bd = output_dir / f"batch_{i:03d}"
        if bd.exists():
            completed_batch_dirs.append(bd)

    overall_start = time.time()

    for batch_idx, batch_files in enumerate(batches, start=1):
        if batch_idx < args.start:
            continue

        est_rows = sum(estimate_rows(f) for f in batch_files)
        print(f"\n{'='*64}")
        print(f"Batch {batch_idx}/{len(batches)}  |  {len(batch_files)} files  |  ~{est_rows:,} rows")

        batch_output = output_dir / f"batch_{batch_idx:03d}"
        batch_output.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        try:
            pipeline = UnifiedPipeline(input_dir=input_dir, output_dir=batch_output)
            result = pipeline.run(
                enable_semantic=semantic,
                file_list=batch_files,
            )
            elapsed = time.time() - t0
            print(
                f"  Done in {elapsed:.1f}s  |  input: {result.total_input_rows:,}"
                f"  deduped: {result.deduplicated_rows:,}"
            )
            completed_batch_dirs.append(batch_output)
        except Exception as exc:
            print(f"  ERROR in batch {batch_idx}: {exc}")
            print("  Skipping batch — fix the issue and re-run with --start", batch_idx)

    total_elapsed = time.time() - overall_start
    print(f"\nAll batches done in {total_elapsed/60:.1f} min")

    if completed_batch_dirs:
        merge_final(completed_batch_dirs, output_dir / "FINAL")
    else:
        print("No completed batches to merge.")


if __name__ == "__main__":
    main()
