# Word Order in Translation

This repository investigates the correlation between language similarity and machine translation quality.

## Introduction

It is generally accepted that language similarity affects translation quality. However, it is hard to find studies that look closely at the exact relationship between word order similarity and translation performance. In this study, we:

1. **Compute language similarity** using multilingual embedding models (mDeBERTa)
2. **Evaluate translation quality** before and after fine-tuning (mBART-50)
3. **Analyze the correlation** between similarity and translation improvement

## Project Structure

```
word-order-in-translation/
├── configs/
│   └── config.yaml              # All experiment settings
├── src/
│   ├── data/                    # Data loading (FLORES, OPUS-100)
│   ├── similarity/              # Similarity metrics (mDeBERTa)
│   ├── translation/             # Translation model, evaluation, training
│   ├── analysis/                # Correlation analysis
│   └── utils/                   # Common utilities
├── scripts/
│   ├── run_similarity.py        # Compute language similarity
│   ├── run_translation.py       # Train and evaluate translation
│   └── run_analysis.py          # Analyze correlations
├── notebooks/                   # Original exploration notebooks
├── outputs/                     # Results (JSON files, figures)
└── checkpoints/                 # Trained model weights
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
# Full pipeline: baseline → fine-tuning → evaluation
python scripts/run_translation.py --config configs/config.yaml

# Baseline evaluation only (no fine-tuning)
python scripts/run_translation.py --config configs/config.yaml --baseline-only

# Resume from existing checkpoints
python scripts/run_translation.py --config configs/config.yaml --resume
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

- **languages**: Language codes for mBART and FLORES
- **language_pairs**: Which pairs to evaluate (12 pairs by default)
- **similarity**: Embedding model, pooling method, similarity function
- **translation**: Model, training hyperparameters, evaluation metrics

## Language Pairs

We evaluate 12 language pairs spanning the similarity spectrum:

| Category           | Pairs                             | Expected Similarity |
| ------------------ | --------------------------------- | ------------------- |
| High (same family) | es-fr, de-en                      | High                |
| East Asian         | ko-ja, ja-zh, ko-zh               | Medium-High         |
| English pivot      | en-ko, en-ja, en-zh, en-fr, en-es | Varies              |
| Low similarity     | en-ar, ar-zh                      | Low                 |

## Output Structure

```
outputs/
├── similarity/
│   └── mdeberta_mean.json       # Similarity scores per pair
├── translation/
│   ├── baseline_bleu.json       # Baseline BLEU scores
│   ├── baseline_comet.json      # Baseline COMET scores
│   ├── finetuned_bleu.json      # Fine-tuned BLEU scores
│   └── finetuned_comet.json     # Fine-tuned COMET scores
└── analysis/
    └── correlation.json         # Correlation analysis results
```

## Methods

### Similarity Computation

- **Model**: mDeBERTa-v3-base (microsoft/mdeberta-v3-base)
- **Pooling**: Mean pooling over token embeddings
- **Similarity**: Cosine similarity between sentence embeddings
- **Data**: FLORES-200 dev set (997 parallel sentences)

### Translation Model

- **Model**: mBART-50 many-to-many (facebook/mbart-large-50-many-to-many-mmt)
- **Training**: Fine-tune on OPUS-100 with 10K samples per pair
- **Evaluation**: BLEU and COMET on FLORES-200 devtest

## Related Work

- Johnson, Melvin, et al. "Google's multilingual neural machine translation system: Enabling zero-shot translation." TACL 2017.
