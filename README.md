# Transformer Architectures for Respiratory Sound Analysis and Multimodal Diagnosis

The repository holds the code accompanying the "Transformer Architectures for Respiratory Sound Analysis and Multimodal Diagnosis" paper.

It contains the shared Python library, sample WAV files, model files, and command-line scripts for running AST and Moondream2 asthma inference and evaluation.

## Prerequisites

- `git-lfs`, required because sample audio and model weights are stored with Git LFS.
- `uv`, used to install the Python environment.

After cloning, make sure LFS files are present:

```bash
git lfs pull
```

Install dependencies:

```bash
uv sync
```

## Repository Contents

```text
├── breathe_transformers/    shared Python library code
├── data/                    sample WAV files and sample metadata
├── hparams/                 sample training configuration artifacts
├── models/                  AST model files, Moondream2 adapter files, and model notes
├── notebooks/               notebook reproducing the training workflow on a public dataset
└── scripts/                 inference, evaluation, dataset preparation and training CLIs
```

## Sample Workflow

Run AST on one WAV file:

```bash
python scripts/ast_inference.py --model_path models/ast_asthma/final_model --audio_path data/sample_audio/sample_001.wav
```

Run AST on the sample WAV directory:

```bash
python scripts/ast_inference.py --model_path models/ast_asthma/final_model --audio_dir data/sample_audio
```

Run Moondream2 on the sample metadata:

```bash
python scripts/moondream_inference.py --adapter_path models/moondream_asthma_adapter --metadata_csv data/sample_metadata.csv --output_mode csv
```

See [models/README.md](models/README.md) for model-specific commands, metadata path handling, cache behavior and examples for saving predictions and metrics.
