"""AST data preparation and model helpers."""

from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torchaudio
from tqdm import tqdm
from transformers import ASTFeatureExtractor, ASTForAudioClassification

from breathe_transformers.audio_utils import (
    get_5sec_clips,
    load_audio_waveform,
    trim_and_norm,
)
from breathe_transformers.datasets import ASTDataset, ASTDatasetConfig
from breathe_transformers.metadata import (
    metadata_value,
    resolve_audio_path,
    validate_metadata_columns,
)
from breathe_transformers.torch_utils import get_default_device

__all__ = [
    "ASTDataset",
    "ASTDatasetConfig",
    "custom_collate",
    "extract_ast_audio_features",
    "load_ast_model",
    "prepare_ast_dataset_5sec",
    "resolve_audio_path",
]


def custom_collate(batch):
    """Collate AST feature tensors into a batch."""
    try:
        processed_tensors = []
        for item in batch:
            tensor = item["input_values"]
            tensor = tensor.squeeze()  # Remove singleton dimensions
            processed_tensors.append(tensor)

        input_values = torch.stack(processed_tensors)  # [B, 1024, 128]
        labels = torch.tensor([item["labels"] for item in batch])
        return {"input_values": input_values, "labels": labels}
    except Exception as e:
        shapes = [item["input_values"].shape for item in batch]
        raise ValueError(
            f"Failed to stack batch. Tensor shapes: {shapes}. Error: {str(e)}"
        )


def prepare_ast_dataset_5sec(
    metadata_csv: str,
    output_dir: str,
    sampling_rate: int = 16000,
    dataset_name: str = "ast_dataset",
    label_column: str = "target_label",
    overlap_percent: int = 25,
) -> str:
    """Process audio files for AST fine-tuning using 5-second clips.

    This mimics the ESC-50 dataset format which uses 5-second audio samples.
    """
    # Load dataset metadata
    metadata_csv_path = Path(metadata_csv)
    metadata = pd.read_csv(metadata_csv_path)
    required_columns = ["sample_id", "audio_path", "target_label"]
    if label_column not in required_columns:
        required_columns.append(label_column)
    validate_metadata_columns(metadata, required_columns, metadata_csv_path)
    csv_metadata_dir = metadata_csv_path.resolve().parent

    # Initialize AST feature extractor
    feature_extractor = ASTFeatureExtractor(
        sampling_rate=sampling_rate,
        num_mel_bins=128,
        max_length=1024,  # This is the default for AST
        padding_value=0.0,
        do_normalize=True,
    )

    # Create dataset-specific output directory
    dataset_output_dir = Path(output_dir) / dataset_name
    features_dir = dataset_output_dir / "features"
    output_metadata_dir = dataset_output_dir / "metadata"
    features_dir.mkdir(parents=True, exist_ok=True)
    output_metadata_dir.mkdir(parents=True, exist_ok=True)

    processed_rows: list[dict[str, Any]] = []
    for _, row in tqdm(metadata.iterrows(), total=len(metadata)):
        audio_path = resolve_audio_path(row, csv_metadata_dir)

        # Load and preprocess audio
        waveform, original_sample_rate = load_audio_waveform(audio_path)

        # Resample if needed
        if original_sample_rate != sampling_rate:
            resampler = torchaudio.transforms.Resample(
                original_sample_rate, sampling_rate
            )
            waveform = resampler(waveform)

        # Convert to mono and trim silence
        waveform = waveform.mean(dim=0).unsqueeze(0)  # Convert to mono
        trimmed = trim_and_norm(
            waveform.numpy()[0],
            sample_rate=sampling_rate,
            max_deviations=int(feature_extractor.std),
        )

        # Split into 5-second clips (matching ESC-50 format)
        clips = get_5sec_clips(trimmed, sampling_rate, overlap_percent)

        row_id = str(metadata_value(row, "sample_id"))
        for clip_idx, clip in enumerate(clips):
            # Extract AST features - the feature extractor will handle the 5-second input
            inputs = feature_extractor(
                clip, sampling_rate=sampling_rate, return_tensors="pt"
            )

            # Save features and metadata
            clip_filename = f"{row_id}_{clip_idx}.pt"
            torch.save(inputs["input_values"], features_dir / clip_filename)

            row_metadata = {
                "file_path": clip_filename,
                "label": metadata_value(row, label_column),
                "sample_id": row_id,
                "audio_path": str(audio_path),
                "clip_index": clip_idx,
                "target_label": metadata_value(row, "target_label"),
            }
            processed_rows.append(row_metadata)

    # Save metadata with dataset name included in the filename
    metadata_file = output_metadata_dir / f"{dataset_name}_dataset.csv"
    pd.DataFrame(processed_rows).to_csv(metadata_file, index=False)
    return str(dataset_output_dir)


def load_ast_model(
    model_path: str,
    device: str | None = None,
) -> tuple[ASTForAudioClassification, dict[str, dict]]:
    """Load an AST model and label mappings."""
    if device is None:
        device = get_default_device()
    # Load model directly - it will load the label mappings from config.json
    model = ASTForAudioClassification.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    # Get label mappings from the model's config
    label_mapping = {
        "label2id": model.config.label2id,
        "id2label": model.config.id2label,
    }
    return model, label_mapping


def extract_ast_audio_features(
    audio_path: str | Path,
    feature_extractor: ASTFeatureExtractor,
    sampling_rate: int = 16000,
    overlap_percent: int = 25,
    trimmed_seconds: int = 2,
) -> list[dict[str, Any]]:
    """Resample, trim, split, and featurize one audio file into timed AST rows."""
    audio_path = Path(audio_path)
    waveform, original_sample_rate = load_audio_waveform(audio_path)

    if original_sample_rate != sampling_rate:
        resampler = torchaudio.transforms.Resample(original_sample_rate, sampling_rate)
        waveform = resampler(waveform)

    waveform = waveform.mean(dim=0).unsqueeze(0)  # Convert to mono
    trimmed = trim_and_norm(
        waveform.numpy()[0],
        sample_rate=sampling_rate,
        trimmed_seconds=trimmed_seconds,
        max_deviations=int(feature_extractor.std),
    )

    rows = []
    for clip_idx, (clip, start_seconds, end_seconds) in enumerate(
        get_5sec_clips(
            trimmed,
            sampling_rate,
            overlap_percent,
            return_segments=True,
        )
    ):
        inputs = feature_extractor(clip, sampling_rate=sampling_rate, return_tensors="pt")
        rows.append(
            {
                "input_values": inputs["input_values"].squeeze(0),
                "clip_index": clip_idx,
                "start_seconds": start_seconds + trimmed_seconds,
                "end_seconds": end_seconds + trimmed_seconds,
            }
        )
    return rows
