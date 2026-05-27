"""Evaluate AST predictions."""

import argparse
import json

import pandas as pd

from breathe_transformers.metrics import binary_metrics_from_columns


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate AST prediction CSV files.")
    parser.add_argument("--predictions_csv", type=str, required=True)
    parser.add_argument("--metadata_csv", type=str, required=True)
    parser.add_argument("--prediction_column", type=str, default="predicted_label")
    parser.add_argument("--label_column", type=str, default="target_label")
    parser.add_argument("--sample_id_column", type=str, default="sample_id")
    parser.add_argument("--positive_label", type=str, default="asthma")
    parser.add_argument("--negative_label", type=str, default="not asthma")
    parser.add_argument("--json_field", type=str)
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> dict[str, float | int]:
    """Evaluate clip-level AST predictions."""
    predictions = pd.read_csv(args.predictions_csv)
    metadata = pd.read_csv(args.metadata_csv)
    merged = predictions.merge(  # ty: ignore
        metadata[[args.sample_id_column, args.label_column]],
        on=args.sample_id_column,
        how="inner",
        validate="many_to_one",
    )
    return binary_metrics_from_columns(
        merged,
        label_column=args.label_column,
        prediction_column=args.prediction_column,
        positive_label=args.positive_label,
        negative_label=args.negative_label,
        json_field=args.json_field,
    )


def main() -> None:
    """Run AST metrics evaluation."""
    args = parse_args()
    metrics = evaluate(args)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
