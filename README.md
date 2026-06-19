# Transformer architectures for respiratory sound analysis and multimodal diagnosis

The repository holds the code accompanying the "Transformer architectures for respiratory sound analysis and multimodal diagnosis" paper.

It contains Python code, sample audio data, model artifacts, and notebooks for AST and Moondream2 respiratory sound experiments on the "Asthma vs Not Asthma" classification task.

## Contents

```text
├── breathe_transformers/    shared Python library
├── data/                    deidentified sample WAV files and metadata, dataset metadata
├── hparams/                 training configuration artifacts
├── models/                  AST model files, Moondream2 adapter files, and model notes
├── notebooks/               training workflow and figure reproduction notebooks
├── scripts/                 dataset preparation, training, inference, and evaluation CLIs
├── CITATION.bib             citation metadata
├── LICENSE                  repository license
├── pyproject.toml           project dependencies
└── uv.lock                  locked dependency versions
```

## Data

This repository includes anonymized sample WAV files, sample metadata and dataset metadata for inference and figure reproduction.

See details in [data/README.md](data/README.md).

## Installation

LFS must be installed on the machine to pull the model weights and adapter files.

```bash
git lfs pull
uv sync
```

## Usage

Run AST inference:

```bash
python scripts/ast_inference.py --model_path models/ast_asthma/final_model --audio_path data/sample_audio/sample_001.wav
```

Run Moondream2 inference:

```bash
python scripts/moondream_inference.py --adapter_path models/moondream_asthma_adapter --metadata_csv data/sample_metadata.csv --output_mode csv
```

See [models/README.md](models/README.md) for evaluation commands and model-specific notes.

## Citation

Cite this work with [CITATION.bib](CITATION.bib).

## License

See [LICENSE](LICENSE).
