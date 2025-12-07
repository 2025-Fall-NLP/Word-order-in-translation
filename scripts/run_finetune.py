#!/usr/bin/env python
"""Fine-tune translation model for each language pair."""

import argparse
import sys
from pathlib import Path
import os

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import get_dataset
from src.translation import (
    checkpoint_exists,
    finetune_translation_model,
    get_checkpoint_path,
    get_model,
)
from src.utils import Config, get_pair_key


def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    parser = argparse.ArgumentParser(description="Fine-tune translation model")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--retrain", action="store_true", help="Retrain even if checkpoint exists"
    )
    args = parser.parse_args()

    cfg = Config.load(args.config)

    train_loader = get_dataset(cfg.training_dataset["type"])(cfg.training_dataset)
    translator = get_model(cfg.trans_model["type"])(cfg.trans_model)

    print("\n" + "=" * 60 + "\nFINE-TUNING\n" + "=" * 60)
    for src, tgt in cfg.language_pairs:
        key = get_pair_key(src, tgt)
        ckpt = get_checkpoint_path(
            cfg.checkpoint_dir, cfg.trans_model["type"], src, tgt
        )

        if not args.retrain and checkpoint_exists(
            cfg.checkpoint_dir, cfg.trans_model["type"], src, tgt
        ):
            print(f"\n[SKIP] {key}: checkpoint exists (use --retrain to overwrite)")
            continue

        print(f"\n[TRAIN] {key}")
        train_data = train_loader.load(src, tgt)
        translator.reload()
        finetune_translation_model(
            translator, train_data, src, tgt, str(ckpt), cfg.training
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
