"""Fine-tuning trainer for translation models using Seq2SeqTrainer."""

import shutil
from pathlib import Path
from typing import Any, Dict

import evaluate
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from src.data.base import ParallelSentences

from .base import TrainableMixin


def cleanup_checkpoints(output_dir: str) -> None:
    """Remove intermediate checkpoint-* directories to save disk space."""
    output_path = Path(output_dir)
    for ckpt_dir in output_path.glob("checkpoint-*"):
        if ckpt_dir.is_dir():
            shutil.rmtree(ckpt_dir)
            print(f"  Removed {ckpt_dir.name}")


def create_training_dataset(
    data: ParallelSentences, val_fraction: float = 0.1, seed: int = 42
) -> tuple:
    """Create train/val split from parallel sentences."""
    dataset = Dataset.from_dict(
        {"src": data.src_sentences, "tgt": data.tgt_sentences}
    ).shuffle(seed=seed)
    n_val = int(len(dataset) * val_fraction)
    return dataset.select(range(len(dataset) - n_val)), dataset.select(
        range(len(dataset) - n_val, len(dataset))
    )


def finetune_translation_model(
    translator: TrainableMixin,
    train_data: ParallelSentences,
    src_lang: str,
    tgt_lang: str,
    output_dir: str,
    training_cfg: Dict[str, Any],
    val_fraction: float = 0.1,
) -> Dict[str, Any]:
    """Fine-tune a translation model."""
    model = translator.model
    tokenizer = translator.tokenizer

    train_ds, val_ds = create_training_dataset(train_data, val_fraction)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    max_length = training_cfg.get("max_length", 128)
    preprocess = lambda ex: translator.preprocess_batch(
        ex, src_lang, tgt_lang, "src", "tgt", max_length
    )

    train_tok = train_ds.map(
        preprocess, batched=True, remove_columns=train_ds.column_names
    )
    val_tok = val_ds.map(preprocess, batched=True, remove_columns=val_ds.column_names)

    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        learning_rate=training_cfg.get("learning_rate", 3e-5),
        per_device_train_batch_size=training_cfg.get("batch_size", 8),
        per_device_eval_batch_size=training_cfg.get("batch_size", 8),
        num_train_epochs=training_cfg.get("epochs", 3),
        predict_with_generate=True,
        generation_max_length=max_length,
        generation_num_beams=4,
        bf16=training_cfg.get("bf16", True) and torch.cuda.is_available(),
        dataloader_num_workers=0,
        save_total_limit=1,
        save_only_model=True,
        save_safetensors=True,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
    )

    bleu = evaluate.load("sacrebleu")

    def compute_metrics(pred):
        preds, labels = pred
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        return {
            "bleu": bleu.compute(
                predictions=decoded_preds, references=[[l] for l in decoded_labels]
            )["score"]
        }

    callbacks = (
        [EarlyStoppingCallback(training_cfg.get("early_stopping_patience", 2))]
        if training_cfg.get("early_stopping_patience")
        else []
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    print(f"Training {src_lang}->{tgt_lang}...")
    result = trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    cleanup_checkpoints(output_dir)

    return {
        "train_loss": result.training_loss,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
    }


def get_checkpoint_path(
    checkpoint_dir: str, model_type: str, src_lang: str, tgt_lang: str
) -> Path:
    return Path(checkpoint_dir) / model_type / f"{src_lang}-{tgt_lang}"


def checkpoint_exists(
    checkpoint_dir: str, model_type: str, src_lang: str, tgt_lang: str
) -> bool:
    p = get_checkpoint_path(checkpoint_dir, model_type, src_lang, tgt_lang)
    return (
        (p / "config.json").exists()
        or (p / "pytorch_model.bin").exists()
        or (p / "model.safetensors").exists()
    )
