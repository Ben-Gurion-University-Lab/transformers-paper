# Models

Model artifacts used by the sample inference and evaluation scripts.

## Contents

```text
├── ast_asthma/final_model/       trained 5-second AST classifier
└── moondream_asthma_adapter/     LoRA adapter for Moondream2
```

Training configuration artifacts are in [`../hparams/`](../hparams/).

## Method summary

Input recordings are normalized and split into 5-second clips before model inference.

The AST classifier takes audio-derived spectrograms and predicts `asthma` or `not asthma`.

The Moondream2 adapter takes audio-derived spectrogram images plus structured metadata: sex, age, and recording point. Its output is parsed as JSON and reduced to the same two labels.

The evaluation scripts compare predicted labels with `target_label` and compute accuracy, sensitivity, specificity, F1-score, and Youden index.

## Inference

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

Moondream2 uses `vikhyatk/moondream2` as the base model. The adapter expects the `2024-08-26` revision.

## Evaluation

Save AST predictions and metrics:

```bash
python scripts/ast_inference.py --model_path models/ast_asthma/final_model --audio_dir data/sample_audio --output_mode csv > ast_predictions.csv
python scripts/evaluate_ast.py --predictions_csv ast_predictions.csv --metadata_csv data/sample_metadata.csv > ast_metrics.json
```

Save Moondream2 predictions and metrics:

```bash
python scripts/moondream_inference.py --adapter_path models/moondream_asthma_adapter --metadata_csv data/sample_metadata.csv --output_mode csv > moondream_predictions.csv
python scripts/evaluate_moondream.py --predictions_csv moondream_predictions.csv > moondream_metrics.json
```

## Offline Moondream2

After the base model is cached locally, set:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```
