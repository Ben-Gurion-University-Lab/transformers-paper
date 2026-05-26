# Models

This directory contains the model artifacts used by the sample inference scripts.

## AST

`ast_asthma/final_model/` is a trained 5-second AST asthma classifier.

Use it with:

```bash
python scripts/ast_inference.py --model_path models/ast_asthma/final_model --audio_path data/sample_audio/sample_001.wav
python scripts/ast_inference.py --model_path models/ast_asthma/final_model --audio_dir data/sample_audio
```

## Moondream2

`moondream_asthma_adapter/` is a LoRA adapter for the Moondream2 asthma classifier.

It requires the Hugging Face base model:

- `vikhyatk/moondream2`
- revision: `2024-08-26`

Use it with:

```bash
python scripts/moondream_inference.py --adapter_path models/moondream_asthma_adapter --metadata_csv data/sample_metadata.csv
```

Moondream2 reads audio files from the metadata CSV's `audio_path` column.
e.g. for `data/sample_metadata.csv`, `sample_audio/sample_001.wav` resolves to `data/sample_audio/sample_001.wav`.
