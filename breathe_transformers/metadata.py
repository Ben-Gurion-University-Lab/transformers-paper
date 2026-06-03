"""Helpers for metadata CSV files."""

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def metadata_value(row: pd.Series, column: str) -> Any:
    """Return a pandas metadata value as a plain Python scalar when possible."""
    value = row[column]
    return value.item() if isinstance(value, np.generic) else value  # ty: ignore


def validate_metadata_columns(
    metadata: pd.DataFrame,
    required_columns: Iterable[str],
    metadata_csv: str | Path,
) -> None:
    """Raise a clear error when a metadata CSV is missing required columns."""
    missing = [column for column in required_columns if column not in metadata.columns]
    if missing:
        missing_list = ", ".join(missing)
        raise ValueError(f"{metadata_csv} is missing required columns: {missing_list}")


def resolve_audio_path(row: pd.Series, metadata_dir: str | Path) -> Path:
    """Resolve the row's audio_path relative to the metadata CSV directory."""
    value = metadata_value(row, "audio_path")
    if pd.isna(value) or str(value).strip() == "":
        raise FileNotFoundError(f"Missing audio_path for row: {row.to_dict()}")

    audio_path = Path(str(value))
    if not audio_path.is_absolute():
        audio_path = Path(metadata_dir) / audio_path
    if audio_path.exists():
        return audio_path
    raise FileNotFoundError(f"Could not resolve audio file: {audio_path}")
