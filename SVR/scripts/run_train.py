from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from svr.data import (
    DEFAULT_SELF_RUBRIC_FIELD_CANDIDATES,
    load_preference_examples,
    split_train_dev,
)
from svr.trainer import SVRTrainConfig, SVRTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SVR-lite without using test/reference rubrics."
    )
    parser.add_argument("--train-path", nargs="+", required=True)
    parser.add_argument("--dev-path", nargs="*")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument(
        "--self-rubric-field",
        action="append",
        default=list(DEFAULT_SELF_RUBRIC_FIELD_CANDIDATES),
        help="Repeat to add more train-time self-rubric field names.",
    )
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--num-rounds", type=int, default=3)
    parser.add_argument("--adv-loss-weight", type=float, default=0.5)
    parser.add_argument("--support-margin-threshold", type=float, default=0.15)
    parser.add_argument("--max-support-pairs", type=int, default=256)
    parser.add_argument("--disable-adversarial-probing", action="store_true")
    parser.add_argument("--disable-support-expansion", action="store_true")
    parser.add_argument("--disable-rubric-rewrite", action="store_true")
    parser.add_argument("--llm-model", default="gpt-oss-120b")
    parser.add_argument("--llm-reasoning-effort", default="medium")
    parser.add_argument("--score-max-tokens", type=int, default=8192)
    parser.add_argument("--mine-max-tokens", type=int, default=8192)
    parser.add_argument("--rewrite-max-tokens", type=int, default=8192)
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-max-concurrency", type=int, default=64)
    parser.add_argument("--llm-progress-log-interval", type=int, default=64)
    parser.add_argument("--bank-progress-log-interval", type=int, default=256)
    parser.add_argument(
        "--disable-round-resume",
        action="store_true",
        help="Ignore existing round outputs and restart training rounds from scratch.",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _guard_against_rubricbench(paths: list[str]) -> None:
    for path in paths:
        if "rubricbench" in path.lower():
            raise ValueError(
                "Training paths must not point at RubricBench/test data. "
                f"Found suspicious path: {path}"
            )


def main() -> None:
    args = parse_args()
    train_paths = list(args.train_path)
    dev_paths = list(args.dev_path or [])
    _guard_against_rubricbench(train_paths + dev_paths)
    print(
        f"[SVR] run_train start: train_paths={train_paths} dev_paths={dev_paths} "
        f"output_dir={args.output_dir}",
        flush=True,
    )

    train_examples = load_preference_examples(
        paths=train_paths,
        self_rubric_fields=args.self_rubric_field,
        limit=args.limit,
    )
    print(f"[SVR] loaded train_examples={len(train_examples)}", flush=True)
    if dev_paths:
        dev_examples = load_preference_examples(
            paths=dev_paths,
            self_rubric_fields=args.self_rubric_field,
            limit=args.dev_limit,
        )
    else:
        train_examples, dev_examples = split_train_dev(
            train_examples,
            dev_ratio=args.dev_ratio,
            seed=args.random_seed,
        )
    print(
        f"[SVR] using train_examples={len(train_examples)} dev_examples={len(dev_examples)}",
        flush=True,
    )

    trainer = SVRTrainer(
        SVRTrainConfig(
            random_seed=args.random_seed,
            dev_ratio=args.dev_ratio,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            hidden_dim=args.hidden_dim,
            top_k=args.top_k,
            num_rounds=args.num_rounds,
            adv_loss_weight=args.adv_loss_weight,
            support_pair_margin_threshold=args.support_margin_threshold,
            max_support_pairs=args.max_support_pairs,
            enable_adversarial_probing=not args.disable_adversarial_probing,
            enable_support_expansion=not args.disable_support_expansion,
            rewrite_selected_rubrics=not args.disable_rubric_rewrite,
            llm_model=args.llm_model,
            llm_reasoning_effort=args.llm_reasoning_effort,
            score_max_tokens=args.score_max_tokens,
            mine_max_tokens=args.mine_max_tokens,
            rewrite_max_tokens=args.rewrite_max_tokens,
            llm_base_url=args.llm_base_url,
            llm_max_concurrency=args.llm_max_concurrency,
            llm_progress_log_interval=args.llm_progress_log_interval,
            bank_progress_log_interval=args.bank_progress_log_interval,
            reuse_round_resume_state=not args.disable_round_resume,
            device=args.device,
        )
    )
    summary = trainer.train(
        train_examples=train_examples,
        dev_examples=dev_examples,
        output_dir=args.output_dir,
    )
    print(f"Saved SVR-lite artifacts to {args.output_dir}")
    print(
        "dev_accuracy={:.4f} bank_size={} support_recall@k={:.4f}".format(
            summary["dev_metrics"]["accuracy"],
            summary["bank_size"],
            summary["dev_metrics"]["support_recall_at_k"],
        )
    )


if __name__ == "__main__":
    main()
