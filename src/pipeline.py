"""Main orchestration for ingestion, cleaning, scoring, and export."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd

from src.cleaner import DataCleaner
from src.config import INPUT_DIR, OUTPUT_DIR
from src.ingestor import DataIngestor
from src.scorer import ICPEngine


@dataclass
class PipelineResult:
    total_input_rows: int
    deduplicated_rows: int
    output_files: List[Path]


class UnifiedPipeline:
    def __init__(self, input_dir: Path = INPUT_DIR, output_dir: Path = OUTPUT_DIR):
        self.input_dir = input_dir
        self.output_dir = output_dir

    def discover_csvs(self) -> List[Path]:
        return sorted(self.input_dir.rglob("*.csv"))

    @staticmethod
    def _log(logger: Optional[Callable[[str], None]], message: str) -> None:
        if logger:
            logger(message)
        else:
            print(message)

    def run(
        self,
        enable_semantic: bool = True,
        logger: Optional[Callable[[str], None]] = None,
        file_list: Optional[List[Path]] = None,
    ) -> PipelineResult:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        csv_files = file_list if file_list is not None else self.discover_csvs()
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.input_dir}")

        self._log(logger, f"Found {len(csv_files)} CSV file(s). Mapping to universal schema...")
        mapped_frames = [DataIngestor.load_and_map(file, logger=logger) for file in csv_files]
        master_df = pd.concat(mapped_frames, ignore_index=True)
        total_input_rows = len(master_df)

        clean_df = DataCleaner.deduplicate(master_df)
        self._log(
            logger, f"Deduplicated from {len(master_df)} to {len(clean_df)} unique records."
        )

        engine = ICPEngine(enable_semantic=enable_semantic, logger=logger)
        final_df = engine.process(clean_df)

        output_files: List[Path] = []
        for bucket in ["ACCEPT", "REVIEW", "REJECT"]:
            output_path = self.output_dir / f"MASTER_{bucket}.csv"
            final_df[final_df["bucket"] == bucket].to_csv(output_path, index=False)
            output_files.append(output_path)

        counts = final_df["bucket"].value_counts()
        self._log(logger, "Bucket counts:")
        for bucket, count in counts.items():
            self._log(logger, f"  {bucket}: {count}")
        self._log(logger, f"Pipeline complete. Output written to {self.output_dir}")

        return PipelineResult(
            total_input_rows=total_input_rows,
            deduplicated_rows=len(clean_df),
            output_files=output_files,
        )

