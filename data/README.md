# Data

## Included Files

- `sample_audio/` contains 10 anonymized WAV files for inference smoke checks.
- `sample_metadata.csv` links those files to metadata and labels.
- `dataset_metadata.csv` contains metadata for the study dataset.
- `target_label` contains the bundled asthma model label (`asthma` or `not asthma`).
- `sample_audio/sample_011_figure_source.wav` is used by the figure reproduction notebooks.

## Dataset

The study used an anonymized dataset collected at the Regional Children's Clinical Hospital of Perm Krai, Perm, Russia. The protocol was approved by the Ethics Committee of Perm State Medical University and followed the Declaration of Helsinki. The dataset is institutional property of Perm State Medical University.

The `dataset_metadata.csv` contains the deidentified metadata for the 1,613 recordings summarized in Table 1 of the accompanying paper. Empty values indicate unavailable metadata.

Each audio recording is accompanied by clinical annotations and structured metadata: sex, age, recording location, diagnosis and quality flags.

Respiratory sound recordings were performed during quiet breathing at four anatomical points: mouth, trachea, right second intercostal space, and right paravertebral area.

## Access

Requests for access should be directed to Sergey Malinin at <info@mstek.ru>.
