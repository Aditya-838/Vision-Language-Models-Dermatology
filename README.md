# Vision-Language Models for Multimodal Dermatology

Evaluation of two open-source Vision-Language Models — **MedGemma-1.5-4B** and **Qwen2.5-VL-7B-Instruct** — for dermatological disease classification and clinical report generation on the **DermaCon-IN** dataset, with a focus on darker skin tones (Fitzpatrick Skin Types 3–6).

## Overview

Most dermatology AI benchmarks are dominated by lighter skin tones (FST 1–2), raising concerns about model reliability on underrepresented populations. This project evaluates whether **medical-domain pretraining** in MedGemma transfers better than **general-purpose pretraining** in Qwen2.5-VL for dermatological image understanding.

The two models were evaluated in both **zero-shot** and **QLoRA fine-tuned** settings.

- **Models:** MedGemma-1.5-4B and Qwen2.5-VL-7B-Instruct
- **Tasks:** Structured disease classification and free-text clinical report generation
- **Classification levels:** Main class, sub-class, and disease label
- **Evaluation:** Accuracy, precision, recall, F1, Fitzpatrick skin-tone analysis, ROUGE, METEOR, BLEU, and LLM-as-judge scoring using Claude Haiku

The experiments were completed as part of the associated MSc research. This repository provides the organized implementation corresponding to those experiments.

## Repository Structure

```text
├── medgemma/
│   ├── inference/
│   │   ├── zeroshot.py
│   │   └── finetuned.py
│   ├── training/
│   │   └── finetune.py
│   └── evaluation/
│       ├── classification.py
│       └── clinical_reports.py
│
├── qwen/
│   ├── inference/
│   │   ├── zeroshot.py
│   │   └── finetuned.py
│   ├── training/
│   │   └── finetune.py
│   └── evaluation/
│       ├── classification.py
│       └── clinical_reports.py
│
├── prompts/
│   ├── structured_classification.txt
│   ├── zero_shot_free_text_system.txt
│   ├── fine_tuning_validation.txt
│   ├── clinical_report_system.txt
│   ├── clinical_report_user.txt
│   └── claude_judge_rubric.txt
│
├── eval_configs/
│   ├── medgemma.json
│   └── qwen.json
│
├── demo/
│   └── demo.py
│
├── research_utils.py
├── environment.yml
├── README.md
└── .gitignore
```

The repository is organized by model and research function to keep the MedGemma and Qwen implementations separate while avoiding an unnecessarily large software architecture.

## Key Results

### Zero-shot Classification Accuracy (%)

| Prediction Level | MedGemma-1.5-4B | Qwen2.5-VL-7B |
|---|---:|---:|
| Main Class (7 classes) | 28.61 | 23.83 |
| Sub-Class (18 classes) | 23.35 | 16.27 |
| Disease Label (245 classes) | 16.17 | 11.20 |

### Zero-shot vs. Fine-tuned Accuracy (%)

| Prediction Level | MedGemma ZS → FT | Qwen ZS → FT |
|---|---:|---:|
| Main Class | 28.61 → **68.52** | 23.83 → **57.89** |
| Sub-Class | 23.35 → 58.37 | 16.27 → 49.09 |
| Disease Label | 16.17 → 40.86 | 11.20 → 34.16 |

### Fine-tuned vs. Swin-B Baseline (Main-class Level)

| Model | Accuracy | F1 | Precision | Recall |
|---|---:|---:|---:|---:|
| Swin-B (dataset baseline) | 70.41 | 69.69 | 69.83 | 69.83 |
| MedGemma-1.5-4B (fine-tuned) | 68.52 | 66.84 | 67.51 | 68.52 |
| Qwen2.5-VL-7B (fine-tuned) | 57.89 | 55.35 | 58.89 | 57.89 |

MedGemma's fine-tuned main-class accuracy was within approximately 1.9 percentage points of the Swin-B dataset baseline.

### Fairness Across Fitzpatrick Skin Types

Accuracy (%) for zero-shot → fine-tuned models:

| FST Group | MedGemma ZS → FT | Qwen ZS → FT |
|---|---:|---:|
| FST 3 | 36.67 → 65.56 | 22.59 → 58.15 |
| FST 4 | 24.95 → 71.73 | 21.21 → 59.88 |
| FST 5 | 29.00 → 66.17 | 30.48 → 54.28 |
| FST 6 (darkest) | 8.00 → 64.00 | 16.00 → 56.00 |

FST 6 had the lowest zero-shot accuracy for both models and showed substantial improvement after fine-tuning.

### Free-text Clinical Report Quality

Evaluation was performed on 70 images using an LLM-based judge.

| Metric | MedGemma-1.5-4B | Qwen2.5-VL-7B |
|---|---:|---:|
| Disease Correctness | 21.43 | 27.14 |
| Visual Accuracy | 0.74 | 0.73 |
| Clinical Coherence | 1.13 | 1.20 |

### Free-text Lexical Similarity

| Metric | MedGemma-1.5-4B | Qwen2.5-VL-7B |
|---|---:|---:|
| ROUGE-1 | 11.11 | 8.44 |
| ROUGE-L | 5.87 | 4.32 |
| METEOR | 20.19 | 14.21 |
| BLEU | 0.68 | 0.56 |

Low lexical scores reflect the mismatch between short structured pseudo-references and free-form generated text.

## Training Configuration (QLoRA)

| Parameter | Value |
|---|---|
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| LoRA Dropout | 0.05 |
| Target Modules | all-linear |
| Quantisation | 4-bit NF4 |
| Epochs | 2 |
| Learning Rate | 2e-4 |
| Optimizer | AdamW |
| LR Scheduler | Cosine |
| Framework | SFTTrainer (HuggingFace TRL) |

## Platform

All GPU experiments were run on Modal.com cloud platform

- **MedGemma:** NVIDIA H100 80GB
- **Qwen:** NVIDIA H200 150GB

## Dataset

DermaCon-IN is a **private dataset** from South Indian outpatient clinics — 5,450 clinical images, 245 disease labels, 7 main classes, 18 sub-classes, and Fitzpatrick Skin Types 3–6.

- Request access from the original authors: https://arxiv.org/abs/2506.06099
- NeurIPS 2025 paper: https://neurips.cc/virtual/2025/poster/121561

> Dataset images and VQA JSON files are not included in this repository.

## Setup

Clone the repository and install the required environment:

```bash
git clone <repository-url>
cd Vision-Language-Models-Dermatology
```

The provided `environment.yml` describes the environment used by the implementation.

For workflows requiring external credentials, set them as environment variables rather than hard-coding them:

```bash
export HF_TOKEN=your_huggingface_token
export WANDB_API_KEY=your_wandb_key
export ANTHROPIC_API_KEY=your_claude_key
```

`ANTHROPIC_API_KEY` is only required for the LLM-based clinical report evaluation.

The private DermaCon-IN dataset and model checkpoints are not included in this repository.

## Research Context

This work was developed as part of MSc research at **Hochschule Luzern (HSLU)** under the supervision of **Prof. Javier Montoya**.

The research investigates the evaluation of multimodal foundation models for dermatological image understanding, with particular attention to performance across darker skin tones.

## Acknowledgments

This project builds on the **DermaCon-IN** dataset and the open-source MedGemma and Qwen2.5-VL models.

**Author:** Aditya Penmetsa
