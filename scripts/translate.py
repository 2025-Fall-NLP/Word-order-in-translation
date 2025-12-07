#!/usr/bin/env python
"""Interactive translation script."""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.translation import (
    BaseTranslator,
    checkpoint_exists,
    get_checkpoint_path,
    get_model,
)
from src.utils import Config


def main():
    parser = argparse.ArgumentParser(description="Interactive translation")
    parser.add_argument("--config", required=True, help="Config file path")
    parser.add_argument("--src", required=True, help="Source language code")
    parser.add_argument("--tgt", required=True, help="Target language code")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    translator_baseline = get_model(cfg.trans_model["type"])(cfg.trans_model)
    translator_finetuned = None

    if checkpoint_exists(
        cfg.checkpoint_dir, cfg.trans_model["type"], args.src, args.tgt
    ):
        ckpt = get_checkpoint_path(
            cfg.checkpoint_dir, cfg.trans_model["type"], args.src, args.tgt
        )
        translator_finetuned = get_model(cfg.trans_model["type"])(cfg.trans_model)
        translator_finetuned.load(str(ckpt))
    else:
        print(f"No finetuned checkpoint found for {args.src}-{args.tgt}")

    print(f"Translating: {args.src} -> {args.tgt}")
    print("Enter text to translate (Ctrl+C to exit):\n")

    try:
        while True:
            text = input("> ").strip()
            if not text:
                continue
            result = translator_baseline.translate([text], args.src, args.tgt)[0]
            print(f"[BASELINE]  {result}\n")
            if translator_finetuned:
                result = translator_finetuned.translate([text], args.src, args.tgt)[0]
                print(f"[FINETUNED]  {result}\n")
    except (KeyboardInterrupt, EOFError):
        print("\nBye!")


if __name__ == "__main__":
    main()
