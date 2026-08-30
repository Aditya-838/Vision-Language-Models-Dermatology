# Vision-Language Models for Multimodal Dermatology

Evaluation of two open-source Vision-Language Models — **MedGemma-1.5-4B** and **Qwen2.5-VL-7B-Instruct** — for dermatological disease classification and clinical report generation on the **DermaCon-IN** dataset, with a focus on dermatology across Fitzpatrick Skin Types 3–6.

## Overview

Most dermatology AI benchmarks are dominated by lighter skin tones, raising concerns about model reliability across underrepresented populations. This project evaluates whether medical-domain pretraining in **MedGemma** provides an advantage over the general-purpose **Qwen2.5-VL** model for dermatological image understanding.

Both models were evaluated in:
- Zero-shot settings
- QLoRA fine-tuned settings
- Structured disease classification
- Free-text clinical report generation
- Fitzpatrick skin-type performance analysis

The repository contains the research implementation corresponding to the completed experiments.

## Models

- **MedGemma-1.5-4B**
- **Qwen2.5-VL-7B-Instruct**

## Research Tasks

### Structured Classification

Disease prediction was evaluated at three levels:
- Main class
- Sub-class
- Disease label

Performance was evaluated using accuracy, precision, recall, and F1-based metrics.

### Clinical Report Generation

The models were also evaluated on free-text clinical report generation using:
- ROUGE-1
- ROUGE-L
- METEOR
- BLEU
- LLM-as-judge evaluation using Claude Haiku

### Fairness Analysis

Performance was analysed across Fitzpatrick Skin Types (FST) 3–6 to assess differences in model performance across darker skin tones.

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
├── qwen/
│   ├── inference/
│   │   ├── zeroshot.py
│   │   └── finetuned.py
│   ├── training/
│   │   └── finetune.py
│   └── evaluation/
│       ├── classification.py
│       └── clinical_reports.py
├── prompts/
├── eval_configs/
├── demo/
│   └── demo.py
├── research_utils.py
├── environment.yml
├── README.md
└── .gitignore
```

The repository is intentionally organized around the two model implementations rather than a large shared software framework.

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

### Fine-tuned Models vs. Swin-B Baseline

| Model | Accuracy | F1 | Precision | Recall |
|---|---:|---:|---:|---:|
| Swin-B (dataset baseline) | 70.41 | 69.69 | 69.83 | 69.83 |
| MedGemma-1.5-4B | 68.52 | 66.84 | 67.51 | 68.52 |
| Qwen2.5-VL-7B | 57.89 | 55.35 | 58.89 | 57.89 |

### Fairness Across Fitzpatrick Skin Types

Accuracy (%) for zero-shot → fine-tuned models:

| FST Group | MedGemma ZS → FT | Qwen ZS → FT |
|---|---:|---:|
| FST 3 | 36.67 → 65.56 | 22.59 → 58.15 |
| FST 4 | 24.95 → 71.73 | 21.21 → 59.88 |
| FST 5 | 29.00 → 66.17 | 30.48 → 54.28 |
| FST 6 | 8.00 → 64.00 | 16.00 → 56.00 |

FST 6 showed the lowest zero-shot accuracy for both models and substantial improvement after fine-tuning.

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

The relatively low lexical scores reflect the mismatch between short structured pseudo-references and free-form generated clinical text.

## Training Configuration

Both models were fine-tuned using QLoRA.

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

The completed GPU experiments were run on the **Modal** cloud platform.

- MedGemma: NVIDIA H100 80GB
- Qwen: NVIDIA H200 150GB

## Dataset

**DermaCon-IN** is a private dermatology dataset collected from South Indian outpatient clinics.

The dataset contains:
- 5,450 clinical images
- 245 disease labels
- 7 main classes
- 18 sub-classes
- Fitzpatrick Skin Types 3–6

The dataset and associated clinical images are **not included in this repository**.

Access to the original dataset should be obtained from the dataset authors.

## Repository Usage

This repository provides the organized research implementation corresponding to the completed thesis experiments.

The Python modules under `medgemma/` and `qwen/` contain the extracted model inference, training, and evaluation implementations.

The original experimental notebooks are maintained separately as the archival record of the experiments and are not part of this public code repository.

Credentials should be supplied through environment variables rather than hard-coded in source files.

```bash
export HF_TOKEN=your_huggingface_token
export WANDB_API_KEY=your_wandb_key
export ANTHROPIC_API_KEY=your_claude_key
```

`ANTHROPIC_API_KEY` is only required for the LLM-based clinical report evaluation.

## Research Context

This work was developed as part of MSc research at **Hochschule Luzern (HSLU)** under the supervision of **Prof. Javier Montoya**.

The research investigates the evaluation of multimodal foundation models for dermatological image understanding, with particular attention to performance across darker skin tones.

## Acknowledgments

This project builds on the **DermaCon-IN** dataset and the open-source MedGemma and Qwen2.5-VL models.

**Author:** Aditya Penmetsa
