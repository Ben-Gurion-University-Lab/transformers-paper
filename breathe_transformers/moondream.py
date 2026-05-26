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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    dynamic_module_utils,
)
from transformers.cache_utils import DynamicCache
from transformers.generation.utils import GenerationMixin

from breathe_transformers.ast import resolve_audio_path
from breathe_transformers.audio_utils import (
    get_5sec_clips,
    load_audio_waveform,
    trim_and_norm,
)
from breathe_transformers.torch_utils import get_default_device


BASE_MODEL = "vikhyatk/moondream2"
MODEL_REVISION = "2024-08-26"
WINDOW_WIDTHS = [0.025, 0.1, 0.175]
SAMPLING_RATE = 16000


def _metadata_value(row: pd.Series, column: str) -> Any:
    """Return a pandas metadata value as a plain Python scalar when possible."""
    value = row[column]
    return value.item() if isinstance(value, np.generic) else value  # ty: ignore


def _is_hf_snapshot_cached(cache_dir: str, model_name: str, revision: str) -> bool:
    """Return whether a Hugging Face snapshot ref exists in the local cache."""
    repo_cache = Path(cache_dir) / f"models--{model_name.replace('/', '--')}"
    ref_path = repo_cache / "refs" / revision
    if not ref_path.exists():
        return False
    snapshot_id = ref_path.read_text().strip()
    return (repo_cache / "snapshots" / snapshot_id).exists()


def build_prompt(row: pd.Series) -> dict[str, Any]:
    """Build a prompt from required sex, age_years, and recording_site columns."""
    prompt = {
        "task": (
            "Write a diagnosis for this patient by analyzing the respiratory sound "
            "spectrogram."
        ),
        "patient_sex": _metadata_value(row, "sex"),
        "patient_age": _metadata_value(row, "age_years"),
        "recording_site": str(_metadata_value(row, "recording_site")).strip(),
    }
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


def extract_moondream_audio_samples(
    row: pd.Series,
    metadata_dir: str,
    sampling_rate: int = SAMPLING_RATE,
    overlap_percent: int = 50,
) -> list[dict[str, Any]]:
    """Create timed samples using the row's audio_path relative to metadata_dir."""
    audio_path = Path(str(_metadata_value(row, "audio_path")))
    if not audio_path.is_absolute():
        audio_path = Path(metadata_dir) / audio_path
    waveform, original_sample_rate = load_audio_waveform(audio_path)

    if original_sample_rate != sampling_rate:
        resampler = torchaudio.transforms.Resample(original_sample_rate, sampling_rate)
        waveform = resampler(waveform)

    waveform = waveform.mean(dim=0).unsqueeze(0)  # Convert to mono
    trimmed = trim_and_norm(waveform.numpy()[0], sample_rate=sampling_rate)
    prompt = json.dumps(build_prompt(row))
    sample_id = str(_metadata_value(row, "sample_id"))
    target_label = _metadata_value(row, "target_label")

    samples = []
    for clip_idx, (clip, start_seconds, end_seconds) in enumerate(
        get_5sec_clips(
            trimmed,
            sampling_rate,
            overlap_percent,
            return_segments=True,
        )
    ):
        image = spectrogram_channels_to_image(
            extract_spectrogram_channels(clip, sampling_rate)
        )
        samples.append(
            {
                "sample_id": sample_id,
                "audio_path": str(audio_path),
                "clip_index": clip_idx,
                "start_seconds": start_seconds + 2,
                "end_seconds": end_seconds + 2,
                "prompt": prompt,
                "target_label": target_label,
                "image": image,
            }
        )
    return samples


def patch_generation_mixin(model: Any) -> None:
    """Patch Moondream's text model so generation APIs are available.

    Note: As of Transformers v4.50+, PreTrainedModel no longer includes
    GenerationMixin. The PhiForCausalLM class used in Moondream implements
    prepare_inputs_for_generation but does not explicitly inherit from
    GenerationMixin, so .generate() can fail. Patch the runtime class and
    assign a GenerationConfig so Moondream question answering can generate.
    """
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


def patch_dynamic_cache_compat() -> None:
    """Patch Transformers cache compatibility for Moondream2 remote code.

    Note: The pinned Moondream2 revision imports its own Phi model implementation.
    That code calls DynamicCache.get_usable_length(), which existed in older
    Transformers releases. The project currently uses Transformers 4.57.x, where
    DynamicCache exposes get_seq_length() instead.
    Adding the old method name keeps the pinned model code runnable without
    changing package versions or editing Hugging Face's cached remote module.
    """
    if hasattr(DynamicCache, "get_usable_length"):
        return

    def get_usable_length(self, new_seq_length: int | None = None, layer_idx: int = 0):
        return self.get_seq_length(layer_idx)

    DynamicCache.get_usable_length = get_usable_length  # ty: ignore


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
    modules_cache = os.path.join(cache_dir, "modules")
    os.makedirs(modules_cache, exist_ok=True)
    os.environ["HF_MODULES_CACHE"] = modules_cache
    dynamic_module_utils.HF_MODULES_CACHE = modules_cache
    local_files_only = _is_hf_snapshot_cached(cache_dir, base_model, revision)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        revision=revision,
        trust_remote_code=True,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        revision=revision,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if device == "cuda" else None,
        torch_dtype=dtype,
        cache_dir=cache_dir,
        use_safetensors=True,
        local_files_only=local_files_only,
    )
    patch_generation_mixin(model)
    patch_dynamic_cache_compat()

    if base_weights_path is not None:
        # The repository ships the adapter, so the default behaviour relies on the
        # Hugging Face base model + supplied adapter.
        # This code branch allows base-weight hot-swap.
        state = load_file(base_weights_path)  # read .safetensors safely
        missing, unexpected = model.load_state_dict(
            state, strict=False
        )  # ignore LoRA keys
        print("missing:", len(missing), "unexpected:", len(unexpected))
    model = PeftModel.from_pretrained(
        model,
        adapter_path,
        local_files_only=local_files_only,
    )
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
