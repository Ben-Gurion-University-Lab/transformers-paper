"""Reusable Moondream2 preparation and model helpers."""

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
import torchaudio
from peft.peft_model import PeftModel  # ty: ignore
from PIL import Image
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.generation.utils import GenerationMixin

from breathe_transformers.ast import resolve_audio_path
from breathe_transformers.audio_utils import get_5sec_clips, trim_and_norm
from breathe_transformers.torch_utils import get_default_device


BASE_MODEL = "vikhyatk/moondream2"
MODEL_REVISION = "2024-08-26"
WINDOW_WIDTHS = [0.025, 0.1, 0.175]
SAMPLING_RATE = 16000
RECORD_POINT_TRANSLATIONS = {
    "второе межреберье": "second intercostal space",
    "грудная клетка сзади": "posterior thoracic rib",
    "ротовая полость": "oral cavity and pharynx",
    "трахея": "trachea",
    "second intercostal space": "second intercostal space",
    "chest from behind": "posterior thoracic rib",
    "posterior thoracic rib": "posterior thoracic rib",
    "oral cavity": "oral cavity and pharynx",
    "trachea": "trachea",
}


def _first_available(row: pd.Series, columns: tuple[str, ...]) -> Any:
    for column in columns:
        if column in row and pd.notna(row[column]):  # ty: ignore
            return row[column]
    return None


def build_prompt(row: pd.Series) -> dict[str, Any]:
    """Build a Moondream2 diagnostic prompt from a metadata row."""
    prompt = {
        "task": (
            "Write a diagnosis for this patient by analyzing the respiratory sound "
            "spectrogram."
        ),
        "patient_sex": _first_available(row, ("sex", "patient_sex")),
        "patient_age": _first_available(row, ("age", "age_yrs", "patient_age")),
    }

    record_point = _first_available(row, ("record_point", "recording_point"))
    if record_point is not None:
        normalized_record_point = str(record_point).strip().lower()
        prompt["record_point"] = RECORD_POINT_TRANSLATIONS.get(
            normalized_record_point,
            str(record_point).strip(),
        )

    prompt["spectrogram_bins_width"] = json.dumps(WINDOW_WIDTHS)
    return {key: value for key, value in prompt.items() if value is not None}


def extract_spectrogram_channels(
    clip: np.ndarray,
    sampling_rate: int,
    window_widths: list[float] | None = None,
) -> list[np.ndarray]:
    """Extract three resized log-mel spectrogram channels."""
    if window_widths is None:
        window_widths = WINDOW_WIDTHS

    clip_tensor = torch.as_tensor(clip, dtype=torch.float32)
    specs = []
    for window_width in window_widths:
        power_of_2 = round(math.log(window_width * sampling_rate, 2))
        fft_section_width = 2**power_of_2
        spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sampling_rate,
            n_fft=fft_section_width,
            win_length=fft_section_width,
            hop_length=int(fft_section_width / 2),
            n_mels=128,
            f_min=0,
            f_max=None,
        )(clip_tensor)
        spec = torch.log(spec + 1e-6)
        spec = functional.interpolate(
            spec.unsqueeze(0).unsqueeze(0),
            size=(128, 250),
            mode="bilinear",
            align_corners=False,
        ).squeeze()
        specs.append(spec.numpy())
    return specs


def spectrogram_channels_to_image(specs: list[np.ndarray]) -> Image.Image:
    """Convert three spectrogram channels into one RGB TIFF image."""
    # Given: specs is a list of 3 numpy arrays, each [H, W]
    normalized_specs = []
    for spec in specs:
        spec_min = spec.min()
        spec_max = spec.max()
        spec_norm = ((spec - spec_min) / (spec_max - spec_min + 1e-6)) * 255
        normalized_specs.append(spec_norm.astype(np.uint8))
    rgb = np.stack(normalized_specs, axis=-1)  # shape [H, W, 3]
    return Image.fromarray(rgb)


def prepare_moondream_dataset_5sec(
    csv_path: str,
    audio_base_path: str,
    output_dir: str,
    sampling_rate: int = SAMPLING_RATE,
    dataset_name: str = "asthma_test",
    classification_column: str = "asthma_key",
    overlap_percent: int = 50,
) -> str:
    """Process audio files for Moondream LoRA fine-tuning using 5-second clips.

    This mimics the ESC-50 dataset format which uses 5-second audio samples.
    """
    # Load dataset metadata
    metadata = pd.read_csv(csv_path)

    # Create dataset-specific output directory
    dataset_output_dir = Path(output_dir) / dataset_name
    image_dir = dataset_output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for _, row in tqdm(metadata.iterrows(), total=len(metadata)):
        audio_path = resolve_audio_path(row, audio_base_path)

        # Load and preprocess audio
        waveform, original_sample_rate = torchaudio.load(audio_path)

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
            max_deviations=6,
        )

        # Split into 5-second clips (matching ESC-50 format)
        clips = get_5sec_clips(trimmed, sampling_rate, overlap_percent)
        row_id = row["id"] if "id" in row and pd.notna(row["id"]) else audio_path.stem  # ty: ignore
        mis_id = row.get("mis_id", "local")

        # Create prompt
        prompt = build_prompt(row)

        # Process each clip
        for clip_idx, clip in enumerate(clips):
            sample_id = f"{row_id}_{mis_id}_{clip_idx}"
            image_path = image_dir / f"{sample_id}.tiff"

            # Extract spectrograms for each window width
            image = spectrogram_channels_to_image(
                extract_spectrogram_channels(clip, sampling_rate)
            )
            image.save(image_path, format="TIFF")

            # Create data entry
            entries.append(
                {
                    "id": sample_id,
                    "image": f"images/{sample_id}.tiff",
                    "conversations": [
                        {"from": "human", "value": json.dumps(prompt)},
                        {
                            "from": "gpt",
                            "value": json.dumps(
                                {"diagnosis": row[classification_column]}
                            ),
                        },
                    ],
                }
            )

    with open(dataset_output_dir / "data.json", "w") as file:
        json.dump(entries, file, indent=2)
    return str(dataset_output_dir)


def patch_generation_mixin(model: Any) -> None:
    """Patch Moondream's text model so generation APIs are available."""
    # As of Transformers v4.50+, PreTrainedModel no longer includes
    # GenerationMixin. The PhiForCausalLM class used in Moondream implements
    # prepare_inputs_for_generation but does not explicitly inherit from
    # GenerationMixin, so .generate() can fail. Patch the runtime class and
    # assign a GenerationConfig so Moondream question answering can generate.
    if isinstance(model.text_model, GenerationMixin):
        return
    model.text_model.__class__ = type(
        "PhiWithGenerate",
        (model.text_model.__class__, GenerationMixin),
        {},
    )
    model.text_model.generation_config = GenerationConfig.from_model_config(
        model.text_model.config
    )


def load_moondream_model(
    adapter_path: str,
    base_model: str = BASE_MODEL,
    revision: str = MODEL_REVISION,
    device: str | None = None,
    cache_dir: str | None = None,
    base_weights_path: str | None = None,
) -> tuple[Any, Any]:
    """Load Moondream2 with a PEFT adapter."""
    if device is None:
        device = get_default_device()
    dtype = torch.float32 if device in {"cpu", "mps"} else torch.bfloat16
    if cache_dir is None:
        cache_dir = os.path.join(os.getcwd(), "temp", "cache")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        revision=revision,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        revision=revision,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if device == "cuda" else None,
        torch_dtype=dtype,
        device_map={"": device},
        cache_dir=cache_dir,
        use_safetensors=True,
    )
    patch_generation_mixin(model)
    if base_weights_path is not None:
        # Optional port of the research notebook's base-weight hot-swap.
        # The paper repository ships the adapter, so the default path stays
        # Hugging Face base model + adapter without a hard-coded checkpoint.
        state = load_file(base_weights_path)  # read .safetensors safely
        missing, unexpected = model.load_state_dict(
            state, strict=False
        )  # ignore LoRA keys
        print("missing:", len(missing), "unexpected:", len(unexpected))
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    model.to(device)
    return model, tokenizer


def answer_sample(model: Any, tokenizer: Any, sample: dict[str, Any]) -> str:
    """Run Moondream2 question answering for one prepared sample."""
    question = sample["qa"][0]["question"]
    return model.answer_question(
        model.encode_image(sample["image"]),
        question,
        tokenizer=tokenizer,
    )
