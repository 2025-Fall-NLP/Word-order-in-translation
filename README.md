# word-order-in-translation
This repo is for SNU 2025 Fall NLP term project.

## Introduction

It is generally accepted that word order similarity affects translation quality. However, it is hard to find studies that look closely at the exact relationship between word order similarity and translation performance. In this study, we plan to define a measure of word order similarity between two languages and check the machine translation quality for different language pairs to see if higher similarity really leads to better translation. 


Plus, we plan to find various methods such as data preprocessing and prompt engineering to improve the quality of low-performance language pairs.

## Main Tasks

* Quantitative representation of word order similarity and exploration of its relationship with translation quality
    * (Optional) Analysis of cases where languages have similar word order but show large differences in translation quality

* Seeking improvement methods for language pairs with low translation performance

## Notebooks Overview
1. flores.ipynb — Loading Same-Meaning Sentence Sets

Loads FLORES dataset sentence sets that share the same meaning across multiple languages.
These aligned sentences serve as the foundation for embedding similarity analysis and translation performance evaluation.

2. deberta.ipynb — Embedding Generation & Cosine Similarity

Get same-meaning sentence sets loaded from flores.py and processes them using the mBERTa model to generate multilingual sentence embeddings,
Compute cosine similarity between embeddings.
This helps quantify how closely languages align semantically in embedding space.

3. translation_model.ipynb — Baseline Translation & Fine-Tuning

Evaluates the baseline translation performance of MBART and performs fine-tuning using the TED2020 dataset.
Then, measures improvement using BLEU and COMET scores and analyze correlation between cosine similarity and the improvement across language pairs.


## Related work

Johnson, Melvin, et al. "Google’s multilingual neural machine translation system: Enabling zero-shot translation." Transactions of the Association for Computational Linguistics 5 (2017): 339-351.