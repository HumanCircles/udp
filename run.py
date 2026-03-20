"""CLI entrypoint for unified ICP data pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline import UnifiedPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run unified ingestion, cleaning, dedupe, and ICP scoring (GUI or CLI)."
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--gui",
        action="store_true",
        help="Launch desktop GUI app.",
    )
    mode_group.add_argument(
        "--cli",
        action="store_true",
        help="Force CLI mode (default if any CLI options are provided).",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory containing source CSV files (default: data/raw).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where bucketed outputs are saved (default: data/output).",
    )
    parser.add_argument(
        "--no-semantic",
        action="store_true",
        help="Disable FAISS/SentenceTransformer semantic scoring (rules only).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    should_launch_gui = args.gui or (
        not args.cli
        and args.input_dir is None
        and args.output_dir is None
        and not args.no_semantic
    )
    if should_launch_gui:
        try:
            from src.gui import launch_gui
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "GUI dependencies are unavailable in this Python build. "
                "Install Python with Tk support or run in CLI mode with --cli."
            ) from exc
        launch_gui()
        return

    pipeline = UnifiedPipeline(
        input_dir=args.input_dir if args.input_dir else UnifiedPipeline().input_dir,
        output_dir=args.output_dir if args.output_dir else UnifiedPipeline().output_dir,
    )
    pipeline.run(enable_semantic=not args.no_semantic)


if __name__ == "__main__":
    main()

