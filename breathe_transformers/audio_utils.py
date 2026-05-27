"""Audio analysis utilities."""

import numpy as np
import soundfile as sf # ty: ignore
import torch


def load_audio_waveform(audio_path) -> tuple[torch.Tensor, int]:
    """Load audio into a channel-first tensor without requiring torchcodec."""
    samples, sample_rate = sf.read(audio_path, always_2d=True)
    waveform = torch.from_numpy(samples.T).float()
    return waveform, sample_rate


def trim_and_norm(
    sample_waveform: np.ndarray,
    sample_rate: int = 22050,
    trimmed_seconds: int = 2,
    norm_technique: str = "STD",
    max_deviations: int = 6,
    percentile: int = 95,
):
    """Trim and normalize an audio sample.

    Args:
        sample_waveform (np.ndarray): Input waveform samples.
        sample_rate (int): Sampling rate of the waveform.
        trimmed_seconds (int): Seconds to trim from the start and end.
        norm_technique (str): Normalization technique, either "STD" or "PCT".
        max_deviations (int): Standard deviation multiplier for "STD" normalization.
        percentile (int): Percentile threshold for "PCT" normalization.

    Returns:
        np.ndarray: The trimmed and normalized waveform.
    """
    trim_start = int(
        (len(sample_waveform) * trimmed_seconds) / (len(sample_waveform) / sample_rate)
    )
    trim_end = int(len(sample_waveform) - trim_start)
    trimmed_sample_waveform = sample_waveform[trim_start:trim_end]

    if norm_technique == "STD":
        mean_signal = np.mean(trimmed_sample_waveform)
        standard_deviation = np.std(trimmed_sample_waveform)
        clipped_waveform = np.clip(
            trimmed_sample_waveform,
            a_min=mean_signal - max_deviations * standard_deviation,  # ty: ignore
            a_max=mean_signal + max_deviations * standard_deviation,  # ty: ignore
        )
    elif norm_technique == "PCT":
        threshold = np.percentile(abs(trimmed_sample_waveform), percentile)
        clipped_waveform = np.clip(
            trimmed_sample_waveform,
            a_min=(-threshold),  # ty: ignore
            a_max=threshold,
        )

    if clipped_waveform.size == 0:
        return np.array([])  # Return an empty array if clipped_waveform is empty

    normalized_waveform = np.max(abs(clipped_waveform))
    norm_clipped_waveform = np.divide(clipped_waveform, normalized_waveform)
    return norm_clipped_waveform


def get_Nsec_clips(
    sample_waveform,
    sample_rate,
    N,
    overlap_percent=25,
    return_segments=False,
) -> list:
    """Return full N-second clips from the centered span of a waveform.

    Set return_segments to include start/end seconds relative to the
    original waveform.

    Args:
        sample_waveform (np.ndarray): Input waveform samples.
        sample_rate (int): Sampling rate of the waveform.
        N (int): Clip length in seconds.
        overlap_percent (float): Percentage of overlap between adjacent clips.
        return_segments (bool): Return (clip, start_seconds, end_seconds) tuples instead of clips.

    Returns:
        np.ndarray | list: Audio clips, or timed clip tuples.
    """
    # Calculate how many N-second segments fit in the sample.
    CLIP_LENGTH_SECONDS = N
    clip_length = CLIP_LENGTH_SECONDS * sample_rate

    # Calculate step size based on overlap percentage
    step_size = int(clip_length * (1 - overlap_percent / 100))
    if step_size == 0:  # Prevent division by zero
        step_size = 1

    # Calculate how many full clips we can get
    clip_count = int((len(sample_waveform) - clip_length) / step_size) + 1
    if clip_count <= 0:
        return []  # Return an empty list if the sample is too short

    # Split the centered portion of the sample into clips.
    total_span = (clip_count - 1) * step_size + clip_length
    threshold = len(sample_waveform) - total_span

    lower_threshold_id = int(threshold / 2)
    upper_threshold_id = lower_threshold_id + total_span

    clips = []
    for i in range(clip_count):
        start = lower_threshold_id + (i * step_size)
        end = start + clip_length
        if end <= upper_threshold_id:
            clip = sample_waveform[start:end]
            if return_segments:
                clips.append((clip, start / sample_rate, end / sample_rate))
            else:
                clips.append(clip)

    if return_segments:
        return clips
    return np.array(clips)


def get_5sec_clips(
    sample_waveform,
    sample_rate,
    overlap_percent=25,
    return_segments=False,
) -> list:
    """Return overlapping five-second clips from a waveform."""
    return get_Nsec_clips(
        sample_waveform,
        sample_rate,
        5,
        overlap_percent,
        return_segments=return_segments,
    )


def get_10sec_clips(
    sample_waveform,
    sample_rate,
    overlap_percent=25,
    return_segments=False,
) -> list:
    """Return overlapping ten-second clips from a waveform."""
    return get_Nsec_clips(
        sample_waveform,
        sample_rate,
        10,
        overlap_percent,
        return_segments=return_segments,
    )
