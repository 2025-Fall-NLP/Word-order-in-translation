#!/usr/bin/env python
"""Evaluate translation models (baseline or finetuned)."""

import argparse
import sys
from pathlib import Path

import yaml

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import get_dataset
from src.translation import checkpoint_exists, get_checkpoint_path, get_eval, get_model
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
    parser = argparse.ArgumentParser(description="Evaluate translation models")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--baseline", action="store_true", help="Evaluate baseline model"
    )
    parser.add_argument(
        "--finetuned", action="store_true", help="Evaluate finetuned model"
    )
    args = parser.parse_args()

    if not args.baseline and not args.finetuned:
        parser.error("Specify --baseline and/or --finetuned")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    pairs = config["language_pairs"]
    trans_cfg = config["translation"]
    model_cfg, eval_cfg = trans_cfg["model"], trans_cfg["evaluation"]
    output_dir = Path(config["paths"]["output_dir"]) / "translation"
    ckpt_dir = config["paths"]["checkpoint_dir"]

    eval_loader = get_dataset(config["datasets"]["evaluation"]["type"])(
        config["datasets"]["evaluation"]
    )
    translator = get_model(model_cfg["type"])(model_cfg)

    # BASELINE
    if args.baseline:
        print("\n" + "=" * 60 + "\nBASELINE EVALUATION\n" + "=" * 60)
        results_by_metric = {m["type"]: {} for m in eval_cfg["metrics"]}

        for src, tgt in pairs:
            key = get_pair_key(src, tgt)
            print(f"\n[EVAL] {key}")
            scores = evaluate_model(translator, src, tgt, eval_loader, eval_cfg)
            for name, score in scores.items():
                if score is not None:
                    results_by_metric[name][key] = score
                    save_results(
                        output_dir / f"baseline_{name}.json",
                        {
                            "stage": "baseline",
                            "metric": name,
                            "model": model_cfg["name"],
                        },
                        results_by_metric[name],
                    )

    # FINETUNED
    if args.finetuned:
        print("\n" + "=" * 60 + "\nFINETUNED EVALUATION\n" + "=" * 60)
        results_by_metric = {m["type"]: {} for m in eval_cfg["metrics"]}

        for src, tgt in pairs:
            key = get_pair_key(src, tgt)
            if not checkpoint_exists(ckpt_dir, model_cfg["type"], src, tgt):
                print(f"\n[SKIP] {key}: no checkpoint")
                continue
            print(f"\n[EVAL] {key}")
            ckpt = get_checkpoint_path(ckpt_dir, model_cfg["type"], src, tgt)
            translator.load(str(ckpt))
            scores = evaluate_model(translator, src, tgt, eval_loader, eval_cfg)
            for name, score in scores.items():
                if score is not None:
                    results_by_metric[name][key] = score
                    save_results(
                        output_dir / f"finetuned_{name}.json",
                        {
                            "stage": "finetuned",
                            "metric": name,
                            "model": model_cfg["name"],
                        },
                        results_by_metric[name],
                    )

    print("\nDone!")


if __name__ == "__main__":
    main()
