# Transformer Architectures for Respiratory Sound Analysis and Multimodal Diagnosis

This repository contains supporting code for the "Transformer Architectures for Respiratory Sound Analysis and Multimodal Diagnosis" paper. It contains the shared Python library code, sample WAV files, model files, and command-line scripts for running AST and Moondream2 asthma inference and evaluation.

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
├── breathe_transformers/    shared Python implementation
├── data/                    sample WAV files and sample metadata
├── models/                  pre-trained AST model files and Moondream2 adapter files
└── scripts/                 inference and evaluation scripts
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

See [models/README.md](models/README.md) for model-specific commands, Moondream2 cache behavior, metadata path handling, and examples for saving predictions and metrics.
