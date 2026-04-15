"""Broad ICP batch runner.

Applies relaxed criteria to the full raw dataset:
  - Employee count: 20–5000
  - Industry: ALL (no exclusions, no target bonus)
  - Decision Makers: same title logic as narrow ICP

Outputs to  data/output BROAD ICP/FINAL/
Usage:
    python run_broad_icp.py
    python run_broad_icp.py --batch 150000
    python run_broad_icp.py --no-semantic
    python run_broad_icp.py --start 5          # resume from batch 5
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from src.cleaner import DataCleaner
from src.config import INPUT_DIR
from src.pipeline import UnifiedPipeline
from src.scorer import ICPConfig

BROAD_CONFIG = ICPConfig(
    emp_min=20,
    emp_max=5_000,
    apply_industry_filter=False,
    apply_industry_bonus=False,
)

OUTPUT_DIR = Path("data/output BROAD ICP")


def estimate_rows(filepath: Path) -> int:
    try:
        with open(filepath, encoding="utf-8", errors="ignore") as fh:
            return sum(1 for _ in fh) - 1
    except Exception:
        return 0


def group_into_batches(files: list[Path], batch_size: int) -> list[list[Path]]:
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
                    print(f"  [warn] {fpath}: {exc}")
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
    parser = argparse.ArgumentParser(description="Broad ICP pipeline runner (20-5000 emp, all industries)")
    parser.add_argument("--batch", type=int, default=150_000)
    parser.add_argument("--no-semantic", action="store_true")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    semantic = not args.no_semantic

    csv_files = sorted(input_dir.rglob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV files")
    print(f"Estimating row counts...")
    total_rows = sum(estimate_rows(f) for f in csv_files)
    print(f"Estimated total rows: {total_rows:,}")

    batches = group_into_batches(csv_files, args.batch)
    print(f"Split into {len(batches)} batch(es) of ~{args.batch:,} rows each")
    print(f"ICP: emp {BROAD_CONFIG.emp_min}–{BROAD_CONFIG.emp_max}, all industries, semantic={'ON' if semantic else 'OFF'}")

    completed_batch_dirs: list[Path] = []
    for i in range(1, args.start):
        bd = OUTPUT_DIR / f"batch_{i:03d}"
        if bd.exists():
            completed_batch_dirs.append(bd)

    overall_start = time.time()

    for batch_idx, batch_files in enumerate(batches, start=1):
        if batch_idx < args.start:
            continue

        est_rows = sum(estimate_rows(f) for f in batch_files)
        print(f"\n{'='*64}")
        print(f"Batch {batch_idx}/{len(batches)}  |  {len(batch_files)} files  |  ~{est_rows:,} rows")

        batch_output = OUTPUT_DIR / f"batch_{batch_idx:03d}"
        batch_output.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        try:
            pipeline = UnifiedPipeline(input_dir=input_dir, output_dir=batch_output)
            result = pipeline.run(
                enable_semantic=semantic,
                file_list=batch_files,
                icp_config=BROAD_CONFIG,
            )
            elapsed = time.time() - t0
            print(
                f"  Done in {elapsed:.1f}s  |  input: {result.total_input_rows:,}"
                f"  deduped: {result.deduplicated_rows:,}"
            )
            completed_batch_dirs.append(batch_output)
        except Exception as exc:
            print(f"  ERROR in batch {batch_idx}: {exc}")
            print(f"  Re-run with --start {batch_idx} to resume")

    total_elapsed = time.time() - overall_start
    print(f"\nAll batches done in {total_elapsed/60:.1f} min")

    if completed_batch_dirs:
        merge_final(completed_batch_dirs, OUTPUT_DIR / "FINAL")
    else:
        print("No completed batches to merge.")


if __name__ == "__main__":
    main()
