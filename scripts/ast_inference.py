"""Run AST inference over raw audio files."""

import argparse
import csv
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import ASTFeatureExtractor

from breathe_transformers.ast import (
    extract_ast_audio_features,
    load_ast_model,
)
from breathe_transformers.torch_utils import get_default_device


RAW_OUTPUT_CSV_FIELDNAMES = [
    "sample_id",
    "audio_path",
    "clip_index",
    "start_seconds",
    "end_seconds",
    "predicted_label",
    "predicted_score",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference with a trained AST model"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model directory"
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--audio_path", type=str, help="Path to one WAV file")
    input_group.add_argument("--audio_dir", type=str, help="Directory of WAV files")
    parser.add_argument(
        "--output_mode",
        choices=["stdout", "csv"],
        default="stdout",
        help="Output format. Use csv for machine-readable rows on stdout.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for inference"
    )
    parser.add_argument("--sampling_rate", type=int, default=16000)
    parser.add_argument("--overlap_percent", type=int, default=25)
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "mps", "cpu"],
        help="Device to run inference on",
    )
    return parser.parse_args()


def resolve_audio_paths(args: argparse.Namespace) -> list[Path]:
    """Resolve raw audio input paths."""
    if args.audio_path:
        return [Path(args.audio_path)]

    audio_dir = Path(args.audio_dir)
    return sorted(audio_dir.glob("*.wav"))


def write_rows(
    rows: list[dict[str, object]],
    fieldnames: list[str],
) -> None:
    """Write prediction rows to stdout as CSV."""
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def write_stdout_rows(rows: list[dict[str, object]]) -> None:
    """Write human-readable prediction rows to stdout."""
    for row in rows:
        print(
            f"{row['sample_id']} clip {row['clip_index']} "
            f"{row['start_seconds']}-{row['end_seconds']}s: "
            f"{row['predicted_label']} ({row['predicted_score']:.4f})"
        )


def run_raw_audio_inference(
    args: argparse.Namespace,
    model: torch.nn.Module,
    device: str,
) -> list[dict[str, object]]:
    """Run inference over raw WAV files."""
    feature_extractor = ASTFeatureExtractor(
        sampling_rate=args.sampling_rate,
        num_mel_bins=128,
        max_length=1024,  # This is the default for AST
        padding_value=0.0,
        do_normalize=True,
    )
    clip_rows = []
    for audio_path in tqdm(
        resolve_audio_paths(args), desc="Preparing audio", file=sys.stderr
    ):
        for clip_row in extract_ast_audio_features(
            audio_path=audio_path,
            feature_extractor=feature_extractor,
            sampling_rate=args.sampling_rate,
            overlap_percent=args.overlap_percent,
        ):
            clip_rows.append(
                {
                    "sample_id": audio_path.stem,
                    "audio_path": str(audio_path),
                    **clip_row,
                }
            )

    if not clip_rows:
        raise SystemExit("No five-second clips were produced from the audio input.")

    output_rows = []
    batch_ranges = range(0, len(clip_rows), args.batch_size)
    id2label = {int(label_id): label for label_id, label in model.config.id2label.items()}
    for start in tqdm(batch_ranges, desc="Running inference", file=sys.stderr):
        batch_rows = clip_rows[start : start + args.batch_size]
        input_values = torch.stack([row["input_values"] for row in batch_rows]).to(
            device
        )
        with torch.no_grad():
            logits = model(input_values).logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_ids = torch.argmax(probabilities, dim=-1)

        for row_idx, (row, predicted_id) in enumerate(
            zip(batch_rows, predicted_ids.tolist())
        ):
            output_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "audio_path": row["audio_path"],
                    "clip_index": row["clip_index"],
                    "start_seconds": round(float(row["start_seconds"]), 3),
                    "end_seconds": round(float(row["end_seconds"]), 3),
                    "predicted_label": id2label[predicted_id],
                    "predicted_score": float(probabilities[row_idx, predicted_id]),
                }
            )
    return output_rows


def main() -> None:
    """Run AST inference and write prediction rows."""
    args = parse_args()
    device = args.device or get_default_device()
    print(f"Using device: {device}", file=sys.stderr)

    model, _ = load_ast_model(args.model_path, device=device)
    rows = run_raw_audio_inference(args, model, device)
    if args.output_mode == "csv":
        write_rows(rows, RAW_OUTPUT_CSV_FIELDNAMES)
    else:
        write_stdout_rows(rows)


if __name__ == "__main__":
    main()
