# Word Order in Translation

This repository investigates the correlation between language similarity and machine translation quality.

## Introduction

It is generally accepted that language similarity affects translation quality. However, it is hard to find studies that look closely at the exact relationship between word order similarity and translation performance. In this study, we:

1. **Compute language similarity** using multilingual embedding models (mDeBERTa, LaBSE)
2. **Evaluate translation quality** before and after fine-tuning (mBART-50)
3. **Analyze the correlation** between similarity and translation improvement

## Project Structure

```
word-order-in-translation/
├── configs/
│   └── config.yaml              # All experiment settings
├── src/
│   ├── data/                    # Data loading (FLORES, OPUS-100)
│   ├── similarity/              # Similarity metrics (mDeBERTa, LaBSE)
│   ├── translation/             # Translation model, evaluation, training
│   ├── analysis/                # Correlation analysis
│   └── utils/                   # Common utilities
├── scripts/
│   ├── run_similarity.py        # Compute language similarity
│   ├── run_translation.py       # Evaluate translation
|   ├── run_finetune.py          # Finetune translation model
│   └── run_analysis.py          # Analyze correlations
├── notebooks/                   # Original exploration notebooks
├── outputs/                     # Results (JSON files, figures)
└── checkpoints/                 # Trained model weights (gitignored)
```

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 1. Compute Language Similarity

```bash
python scripts/run_similarity.py --config configs/config.yaml
```

This computes embedding-based similarity scores for all configured language pairs using mDeBERTa with mean pooling.

### 2. Run Translation Experiments

```bash
# Finetune the transaltion model
python scripts/run_finetune.py --config configs/config.yaml

# Evaluate the baseline and finetuned models
python scripts/run_translation.py --config configs/config.yaml --baseline --finetuned

# Force retrain even if checkpoints exist, e.g., after adjusting hyperparameters
python scripts/run_fientune.py --config configs/config.yaml --retrain
```

### 3. Analyze Correlations

```bash
python scripts/run_analysis.py --config configs/config.yaml
```

This computes Pearson and Spearman correlations between:

- Similarity vs. baseline translation quality
- Similarity vs. fine-tuned translation quality
- Similarity vs. improvement (delta)

## Configuration

All settings are in `configs/config.yaml`:

- **language_pairs**: Which pairs to evaluate (12 pairs by default)
- **datasets**: Which dataset is used for each case
- **similarity**: Embedding model, pooling method, similarity function
- **translation**: Model, training hyperparameters, evaluation metrics
- **paths**: Result outputs and checkpoints paths

## Related Work

- Johnson, Melvin, et al. "Google's multilingual neural machine translation system: Enabling zero-shot translation." TACL 2017.
