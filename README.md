# multilang-simtrans

This repository provides the experiment setup to investigate the relation between language similarity and machine translation quality.

## Introduction

In this experiment, we:

1. **Compute language similarity** using multilingual embedding models (mDeBERTa, LaBSE) and more metrics (URIEL, chrF)
2. **Evaluate translation quality** before and after fine-tuning (mBART-50, nllb-200)
3. **Analyze the correlation** between similarity and translation quality

## Directory Structure

```
multilang-simtrans/
├── configs/
│   └── config.yaml              # All experiment settings
├── src/
│   ├── data/                    # Data loading (FLORES, OPUS-100)
│   ├── similarity/              # Similarity metrics (mDeBERTa, LaBSE)
│   ├── translation/             # Translation model, evaluation, training
│   ├── analysis/                # Correlation and plots
│   └── utils/                   # Common utilities
├── scripts/
│   ├── run_similarity.py        # Compute language similarity
│   ├── run_evaluate.py          # Evaluate translation
|   ├── run_finetune.py          # Fine-tune translation model
│   └── run_analysis.py          # Analyze correlations and draw plots
│   └── translate.py             # Interactively translate
├── notebooks/                   # Deprecated exploration notebooks
├── outputs/                     # Results (JSON, figures)
└── checkpoints/                 # Trained model weights (gitignored)
```

## Installation

Requires `Python>=3.10.0`

```bash
git clone git@github.com:snunlp-2025-fall-team17/multilang-simtrans.git
cd multilang-simtrans
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 1. Compute Language Similarity

```bash
python scripts/run_similarity.py --config configs/config.yaml
```

### 2. Run Translation Experiments

```bash
# Finetune the transaltion model
python scripts/run_finetune.py --config configs/config.yaml

# Evaluate the baseline and finetuned models
python scripts/run_evaluate.py --config configs/config.yaml --baseline --finetuned

# You can force retrain even if checkpoints exist
# python scripts/run_finetune.py --config configs/config.yaml --retrain
```

### 3. Analysis

```bash
python scripts/run_analysis.py --config configs/config.yaml
```

This computes Pearson and Spearman correlations between:

- Similarity vs. baseline translation quality
- Similarity vs. finetuned translation quality
- Similarity vs. improvement (delta)

## Configuration

All settings are in `configs/config.yaml`:

- **language_pairs**: Which pairs to evaluate (12 pairs by default)
- **datasets**: Which dataset is used for each case
- **similarity**: Similarity metrics
- **translation**: Model, evaluation metrics, fine-tuning hyperparameters
- **analysis**: How to draw plots
- **paths**: Result outputs and checkpoints paths
