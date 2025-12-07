#!/usr/bin/env python
"""Evaluate translation models (baseline or finetuned)."""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import get_dataset
from src.translation import (
    checkpoint_exists,
    get_checkpoint_path,
    get_eval_metric,
    get_model,
)
from src.utils import (
    OUTPUT_PREFIX_BASELINE,
    OUTPUT_PREFIX_FINETUNED,
    OUTPUT_DIR_TRANSLATION,
    Config,
    get_pair_key,
    save_results,
)


def evaluate_model(translator, src, tgt, data_loader, eval_metrics):
    data = data_loader.load(src, tgt)
    translations = translator.translate(
        data.src_sentences, src, tgt, show_progress=True
    )

    results = {}
    for mc in eval_metrics:
        metric = get_eval_metric(mc["type"])(mc)
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
    parser.add_argument("--baseline", action="store_true", help="Evaluate baseline")
    parser.add_argument("--finetuned", action="store_true", help="Evaluate finetuned")
    args = parser.parse_args()

    if not args.baseline and not args.finetuned:
        parser.error("Specify --baseline and/or --finetuned")

    cfg = Config.load(args.config)
    output_dir = cfg.output_dir / OUTPUT_DIR_TRANSLATION

    eval_loader = get_dataset(cfg.evaluation_dataset["type"])(cfg.evaluation_dataset)
    translator = get_model(cfg.trans_model["type"])(cfg.trans_model)

    # BASELINE
    if args.baseline:
        print("\n" + "=" * 60 + "\nBASELINE EVALUATION\n" + "=" * 60)
        results_by_metric = {metric["type"]: {} for metric in cfg.eval_metrics}

        for src, tgt in cfg.language_pairs:
            key = get_pair_key(src, tgt)
            print(f"\n[EVAL] {key}")
            scores = evaluate_model(translator, src, tgt, eval_loader, cfg.eval_metrics)
            for name, score in scores.items():
                if score is not None:
                    results_by_metric[name][key] = score
                    save_results(
                        output_dir / f"{OUTPUT_PREFIX_BASELINE}{name}.json",
                        {
                            "stage": "baseline",
                            "metric": name,
                            "model": cfg.trans_model["name"],
                        },
                        results_by_metric[name],
                    )

    # FINETUNED
    if args.finetuned:
        print("\n" + "=" * 60 + "\nFINETUNED EVALUATION\n" + "=" * 60)
        results_by_metric = {metric["type"]: {} for metric in cfg.eval_metrics}

        for src, tgt in cfg.language_pairs:
            key = get_pair_key(src, tgt)
            if not checkpoint_exists(
                cfg.checkpoint_dir, cfg.trans_model["type"], src, tgt
            ):
                print(f"\n[SKIP] {key}: no checkpoint")
                continue
            print(f"\n[EVAL] {key}")
            ckpt = get_checkpoint_path(
                cfg.checkpoint_dir, cfg.trans_model["type"], src, tgt
            )
            translator.load(str(ckpt))
            scores = evaluate_model(translator, src, tgt, eval_loader, cfg.eval_metrics)
            for name, score in scores.items():
                if score is not None:
                    results_by_metric[name][key] = score
                    save_results(
                        output_dir / f"{OUTPUT_PREFIX_FINETUNED}{name}.json",
                        {
                            "stage": "finetuned",
                            "metric": name,
                            "model": cfg.trans_model["name"],
                        },
                        results_by_metric[name],
                    )

    print("\nDone!")


if __name__ == "__main__":
    main()
