#!/usr/bin/env python
"""Fine-tune translation model for each language pair."""

import argparse
import sys
from pathlib import Path

import yaml

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import get_dataset
from src.translation import (
    checkpoint_exists,
    finetune_translation_model,
    get_checkpoint_path,
    get_model,
)
from src.utils import get_pair_key


def main():
    parser = argparse.ArgumentParser(description="Fine-tune translation model")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--retrain", action="store_true", help="Retrain even if checkpoint exists"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    pairs = config["language_pairs"]
    trans_cfg = config["translation"]
    model_cfg, train_cfg = trans_cfg["model"], trans_cfg["training"]
    ckpt_dir = config["paths"]["checkpoint_dir"]

    train_loader = get_dataset(config["datasets"]["training"]["type"])(
        config["datasets"]["training"]
    )
    translator = get_model(model_cfg["type"])(model_cfg)

    print("\n" + "=" * 60 + "\nFINE-TUNING\n" + "=" * 60)
    for src, tgt in pairs:
        key = get_pair_key(src, tgt)
        ckpt = get_checkpoint_path(ckpt_dir, model_cfg["type"], src, tgt)

        if not args.retrain and checkpoint_exists(
            ckpt_dir, model_cfg["type"], src, tgt
        ):
            print(f"\n[SKIP] {key}: checkpoint exists (use --retrain to overwrite)")
            continue

        print(f"\n[TRAIN] {key}")
        train_data = train_loader.load(src, tgt)
        translator.reload()
        finetune_translation_model(
            translator, train_data, src, tgt, str(ckpt), train_cfg
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
