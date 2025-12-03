#!/usr/bin/env python
"""Run translation experiments: baseline eval, fine-tuning, finetuned eval."""

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
    get_eval,
    get_model,
)
from src.utils import get_pair_key, save_results


def evaluate_model(translator, src, tgt, data_loader, eval_cfg):
    data = data_loader.load(src, tgt)
    translations = translator.translate(
        data.src_sentences, src, tgt, show_progress=True
    )

    results = {}
    for mc in eval_cfg["metrics"]:
        metric = get_eval(mc["type"])(mc)
        try:
            score = metric.compute(translations, data.tgt_sentences, data.src_sentences)
            results[metric.name] = score
            print(f"    {metric.name}: {score:.4f}")
        except Exception as e:
            print(f"    {metric.name}: FAILED ({e})")
            results[metric.name] = None
    return results


def main():
    parser = argparse.ArgumentParser(description="Translation experiments")
    parser.add_argument("--config", required=True)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument(
        "--retrain", action="store_true", help="Retrain even if checkpoint exists"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    pairs = config["language_pairs"]
    trans_cfg = config["translation"]
    model_cfg, eval_cfg, train_cfg = (
        trans_cfg["model"],
        trans_cfg["evaluation"],
        trans_cfg["training"],
    )
    output_dir = Path(config["paths"]["output_dir"]) / "translation"
    ckpt_dir = config["paths"]["checkpoint_dir"]

    # Initialize dataset loaders from registry
    eval_data_cfg = config["datasets"]["evaluation"]
    train_data_cfg = config["datasets"]["training"]
    eval_loader = get_dataset(eval_data_cfg["type"])(eval_data_cfg)
    train_loader = get_dataset(train_data_cfg["type"])(train_data_cfg)

    # Initialize translator (mBART has its own DEFAULT_LANG_CODES)
    translator = get_model(model_cfg["type"])(model_cfg)

    # BASELINE
    print("\n" + "=" * 60 + "\nBASELINE EVALUATION\n" + "=" * 60)
    baseline = {
        m["type"]: {"path": output_dir / f"baseline_{m['type']}.json", "results": {}}
        for m in eval_cfg["metrics"]
    }

    for src, tgt in pairs:
        key = get_pair_key(src, tgt)
        print(f"\n[EVAL] {key}")
        scores = evaluate_model(translator, src, tgt, eval_loader, eval_cfg)
        for name, score in scores.items():
            if score is not None:
                baseline[name]["results"][key] = score
                save_results(
                    baseline[name]["path"],
                    {"stage": "baseline", "metric": name, "model": model_cfg["name"]},
                    baseline[name]["results"],
                )

    if args.baseline_only:
        return

    # FINE-TUNING
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

    # FINETUNED EVAL
    print("\n" + "=" * 60 + "\nFINETUNED EVALUATION\n" + "=" * 60)
    finetuned = {
        m["type"]: {"path": output_dir / f"finetuned_{m['type']}.json", "results": {}}
        for m in eval_cfg["metrics"]
    }

    for src, tgt in pairs:
        key = get_pair_key(src, tgt)
        ckpt = get_checkpoint_path(ckpt_dir, model_cfg["type"], src, tgt)
        if not checkpoint_exists(ckpt_dir, model_cfg["type"], src, tgt):
            print(f"\n[SKIP] {key}: no checkpoint")
            continue
        print(f"\n[EVAL] {key}")
        translator.load(str(ckpt))
        scores = evaluate_model(translator, src, tgt, eval_loader, eval_cfg)
        for name, score in scores.items():
            if score is not None:
                finetuned[name]["results"][key] = score
                save_results(
                    finetuned[name]["path"],
                    {"stage": "finetuned", "metric": name, "model": model_cfg["name"]},
                    finetuned[name]["results"],
                )

    # SUMMARY
    print("\n" + "=" * 60 + "\nSUMMARY\n" + "=" * 60)
    for m in eval_cfg["metrics"]:
        name = m["type"]
        print(f"\n{name.upper()}:\n{'Pair':<10} {'Base':>8} {'Fine':>8} {'Δ':>8}")
        for src, tgt in pairs:
            key = get_pair_key(src, tgt)
            b = baseline[name]["results"].get(key)
            f = finetuned[name]["results"].get(key)
            d = f"{f-b:+.2f}" if b and f else "N/A"
            print(f"{key:<10} {b or 'N/A':>8} {f or 'N/A':>8} {d:>8}")

    print("\nDone!")


if __name__ == "__main__":
    main()
