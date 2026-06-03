"""Run Moondream2 adapter inference over local audio metadata."""

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from breathe_transformers.moondream import (
    BASE_MODEL,
    MODEL_REVISION,
    SAMPLING_RATE,
    answer_sample,
    extract_moondream_audio_samples,
    load_moondream_model,
)


OUTPUT_CSV_FIELDNAMES = [
    "sample_id",
    "audio_path",
    "clip_index",
    "start_seconds",
    "end_seconds",
    "target_label",
    "prompt",
    "model_answer",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Moondream2 adapter inference on local audio metadata."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--metadata_csv",
        type=str,
        help=(
            "CSV with audio_path and prompt metadata columns. Relative audio_path "
            "values are resolved from the CSV location."
        ),
    )
    input_group.add_argument(
        "--data_json",
        type=str,
        help="Moondream2 data.json file",
    )
    parser.add_argument(
        "--images_folder",
        type=str,
        help="Image folder for data.json inference.",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="models/moondream_asthma_adapter",
    )
    parser.add_argument(
        "--output_mode",
        choices=["stdout", "csv"],
        default="stdout",
        help="Output format. Use csv for machine-readable rows on stdout.",
    )
    parser.add_argument("--base_model", type=str, default=BASE_MODEL)
    parser.add_argument("--revision", type=str, default=MODEL_REVISION)
    parser.add_argument("--device", type=str, choices=["cuda", "mps", "cpu"])
    parser.add_argument("--cache_dir", type=str)
    parser.add_argument("--base_weights_path", type=str)
    parser.add_argument("--sampling_rate", type=int, default=SAMPLING_RATE)
    parser.add_argument("--overlap_percent", type=int, default=50)
    args = parser.parse_args()
    if args.data_json and not args.images_folder:
        parser.error("--images_folder is required with --data_json")
    return args


def write_csv_rows(rows: list[dict[str, object]]) -> None:
    """Write prediction rows to stdout as CSV."""
    writer = csv.DictWriter(sys.stdout, fieldnames=OUTPUT_CSV_FIELDNAMES)
    writer.writeheader()
    writer.writerows(rows)


def write_stdout_rows(rows: list[dict[str, object]]) -> None:
    """Write human-readable prediction rows to stdout."""
    for row in rows:
        print(
            f"{row['sample_id']} clip {row['clip_index']} "
            f"{row['start_seconds']}-{row['end_seconds']}s: {row['model_answer']}"
        )


def load_model(args: argparse.Namespace):
    """Load the Moondream2 model for inference."""
    model, tokenizer = load_moondream_model(
        adapter_path=args.adapter_path,
        base_model=args.base_model,
        revision=args.revision,
        device=args.device,
        cache_dir=args.cache_dir,
        base_weights_path=args.base_weights_path,
    )
    return model, tokenizer


def run_audio_inference(args: argparse.Namespace, model, tokenizer) -> list[dict[str, object]]:
    """Run Moondream2 inference over audio rows."""
    metadata_csv = Path(args.metadata_csv)
    metadata = pd.read_csv(metadata_csv)

    rows = []
    for _, metadata_row in tqdm(
        metadata.iterrows(),
        total=len(metadata),
        desc="Running inference",
        file=sys.stderr,
    ):
        for sample in extract_moondream_audio_samples(
            metadata_row,
            metadata_dir=str(metadata_csv.parent),
            sampling_rate=args.sampling_rate,
            overlap_percent=args.overlap_percent,
        ):
            model_answer = answer_sample(
                model,
                tokenizer,
                {
                    "image": sample["image"],
                    "qa": [{"question": sample["prompt"], "answer": ""}],
                },
            )
            rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "audio_path": sample["audio_path"],
                    "clip_index": sample["clip_index"],
                    "start_seconds": round(float(sample["start_seconds"]), 3),
                    "end_seconds": round(float(sample["end_seconds"]), 3),
                    "target_label": sample["target_label"],
                    "prompt": sample["prompt"],
                    "model_answer": model_answer,
                }
            )
    return rows


def answer_label(sample: dict[str, object]) -> str:
    """Return the target label from a Moondream2 conversation row."""
    conversations = sample["conversations"]
    answer = conversations[1]["value"]  # ty: ignore
    try:
        parsed = json.loads(answer)
    except json.JSONDecodeError:
        return str(answer)
    return str(parsed.get("diagnosis", answer))


def run_image_inference(args: argparse.Namespace, model, tokenizer) -> list[dict[str, object]]:
    """Run Moondream2 inference over image rows."""
    data_json = Path(args.data_json)
    images_folder = Path(args.images_folder)
    samples = json.loads(data_json.read_text())

    rows = []
    for sample in tqdm(samples, desc="Running inference", file=sys.stderr):
        metadata = sample.get("metadata", {})
        prompt = sample["conversations"][0]["value"]
        image = Image.open(images_folder / sample["image"]).convert("RGB")
        model_answer = answer_sample(
            model,
            tokenizer,
            {"image": image, "qa": [{"question": prompt, "answer": ""}]},
        )
        rows.append(
            {
                "sample_id": metadata.get("sample_id", sample.get("id", "")),
                "audio_path": metadata.get("audio_path", ""),
                "clip_index": metadata.get("clip_index", ""),
                "start_seconds": metadata.get("start_seconds", ""),
                "end_seconds": metadata.get("end_seconds", ""),
                "target_label": metadata.get("target_label", answer_label(sample)),
                "prompt": prompt,
                "model_answer": model_answer,
            }
        )
    return rows


def run_inference(args: argparse.Namespace) -> list[dict[str, object]]:
    """Run Moondream2 inference over the selected input mode."""
    model, tokenizer = load_model(args)
    if getattr(args, "data_json", None):
        return run_image_inference(args, model, tokenizer)
    return run_audio_inference(args, model, tokenizer)


def main() -> None:
    """Run Moondream2 inference and write predictions."""
    args = parse_args()
    rows = run_inference(args)
    if args.output_mode == "csv":
        write_csv_rows(rows)
    else:
        write_stdout_rows(rows)


if __name__ == "__main__":
    main()
