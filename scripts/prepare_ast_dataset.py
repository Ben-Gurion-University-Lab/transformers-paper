"""Prepare 5-second AST feature datasets from local audio files."""

import argparse

from breathe_transformers.ast import prepare_ast_dataset_5sec


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare 5-second AST feature datasets from local audio files."
    )
    parser.add_argument("--metadata_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--label_column", type=str, default="target_label")
    parser.add_argument("--sampling_rate", type=int, default=16000)
    parser.add_argument("--overlap_percent", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    """Run the AST dataset preparation workflow."""
    args = parse_args()
    prepare_ast_dataset_5sec(
        metadata_csv=args.metadata_csv,
        output_dir=args.output_dir,
        sampling_rate=args.sampling_rate,
        dataset_name=args.dataset_name,
        label_column=args.label_column,
        overlap_percent=args.overlap_percent,
    )


if __name__ == "__main__":
    main()
