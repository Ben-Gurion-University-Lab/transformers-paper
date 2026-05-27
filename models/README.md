# Models

This directory contains the model artifacts used by the sample inference scripts.

## AST

`ast_asthma/final_model/` is a trained 5-second AST asthma classifier.

Use it with:

```bash
python scripts/ast_inference.py --model_path models/ast_asthma/final_model --audio_path data/sample_audio/sample_001.wav
python scripts/ast_inference.py --model_path models/ast_asthma/final_model --audio_dir data/sample_audio
```

Inference and evaluation write results to stdout. Use `--output_mode csv` and redirect stdout to save files:

```bash
python scripts/ast_inference.py --model_path models/ast_asthma/final_model --audio_dir data/sample_audio --output_mode csv > ast_predictions.csv
python scripts/evaluate_ast.py --predictions_csv ast_predictions.csv --metadata_csv data/sample_metadata.csv > ast_metrics.json
```

## Moondream2

`moondream_asthma_adapter/` is a LoRA adapter for the Moondream2 asthma classifier.

It requires the Hugging Face base model:

- `vikhyatk/moondream2`
- revision: `2024-08-26`

Use it with:

```bash
python scripts/moondream_inference.py --adapter_path models/moondream_asthma_adapter --metadata_csv data/sample_metadata.csv --output_mode csv
```

For deterministic offline runs after `temp/cache` has been populated, set:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/moondream_inference.py --adapter_path models/moondream_asthma_adapter --metadata_csv data/sample_metadata.csv --cache_dir temp/cache --output_mode csv
```

Without those variables, Hugging Face may still make optional metadata requests even when the model files are already cached which will fail without internet connection.

Moondream2 reads audio files from the metadata CSV's `audio_path` column.
e.g. for `data/sample_metadata.csv`, `sample_audio/sample_001.wav` resolves to `data/sample_audio/sample_001.wav`.

Inference and evaluation write results to stdout. Save predictions and metrics with shell redirection:

```bash
python scripts/moondream_inference.py --adapter_path models/moondream_asthma_adapter --metadata_csv data/sample_metadata.csv --output_mode csv > moondream_predictions.csv
python scripts/evaluate_moondream.py --predictions_csv moondream_predictions.csv > moondream_metrics.json
```
