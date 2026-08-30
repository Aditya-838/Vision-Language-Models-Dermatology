# DermaCon-IN vision-language thesis repository

This repository organizes the completed master's-thesis implementation for dermatology vision-language modeling on DermaCon-IN. It preserves the original notebooks as the source of truth and exposes their existing MedGemma-1.5-4B and Qwen2.5-VL-7B-Instruct workflows in a compact research-code layout.

The completed research comprises zero-shot structured and free-text experiments, QLoRA fine-tuning, classification evaluation, clinical/free-text report evaluation, Fitzpatrick skin-type analysis, and a MedGemma Gradio demo. No experiment is run by importing the extracted modules.

## Layout

- `medgemma/` and `qwen/`: model-specific inference, training, and evaluation code extracted from the corresponding notebooks.
- `prompts/`: verbatim prompts used by the completed notebooks.
- `eval_configs/`: recorded model, QLoRA, inference, training, and evaluation values, with configurable external paths.
- `notebooks/`: unchanged archival copies of the completed notebooks.
- `demo/demo.py`: the existing two-mode MedGemma Gradio demo, not launched automatically.
- `data/` and `results/`: documentation only; private data and image-level outputs are not included.

## Completed methodology

Both branches use 4-bit NF4 QLoRA with rank 16, alpha 32, dropout 0.05, a learning rate of `2e-4`, cosine scheduling, warmup ratio `0.03`, batch size 1, gradient accumulation 4, and two epochs. Classification assesses exact disease, main-class, and subclass matching along with weighted metrics and Fitzpatrick-group analyses. Clinical-report evaluation uses a stratified sample of up to ten images per main class, pseudo-reference text metrics, and the existing Claude judging rubric.

## Data and security

DermaCon-IN images and clinical data are external/private and are not redistributed. Model checkpoints and generated image-level outputs are also excluded. New code reads `HF_TOKEN`, `WANDB_API_KEY`, and `ANTHROPIC_API_KEY` from environment variables; never put credentials in source files.

See [data/README.md](data/README.md) and [results/README.md](results/README.md) for constraints and expected artifact categories.
