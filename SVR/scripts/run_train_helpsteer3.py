from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from svr.data import DEFAULT_SELF_RUBRIC_FIELD_CANDIDATES, split_train_dev, load_preference_examples
from svr.trainer import SVRTrainConfig, SVRTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preset SVR training entrypoint for HelpSteer3 preference data."
    )
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dev-ratio", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--num-rounds", type=int, default=3)
    parser.add_argument("--max-support-pairs", type=int, default=1024)
    parser.add_argument("--support-margin-threshold", type=float, default=0.2)
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
        "--resume-candidate-cache-dir",
        default=None,
        help=(
            "Existing experiment output dir, candidate_stage_cache dir, or "
            "candidate_examples.pkl path to load initial candidate cache from."
        ),
    )
    parser.add_argument(
        "--require-initial-train-cache",
        action="store_true",
        help="Fail instead of rerunning initial train candidate mining when cache is unavailable.",
    )
    parser.add_argument(
        "--disable-round-resume",
        action="store_true",
        help="Ignore existing round outputs and restart training rounds from scratch.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        "[SVR] HelpSteer3 run: "
        f"train_path={args.train_path} output_dir={args.output_dir}",
        flush=True,
    )
    examples = load_preference_examples(
        paths=[args.train_path],
        self_rubric_fields=DEFAULT_SELF_RUBRIC_FIELD_CANDIDATES,
        limit=args.limit,
    )
    print(f"[SVR] loaded examples={len(examples)}", flush=True)
    train_examples, dev_examples = split_train_dev(
        examples,
        dev_ratio=args.dev_ratio,
        seed=args.random_seed,
    )
    print(
        f"[SVR] split train={len(train_examples)} dev={len(dev_examples)}",
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
            candidate_stage_cache_source_dir=args.resume_candidate_cache_dir,
            require_initial_train_candidate_cache=args.require_initial_train_cache,
            device=args.device,
        )
    )
    summary = trainer.train(
        train_examples=train_examples,
        dev_examples=dev_examples,
        output_dir=args.output_dir,
    )
    print(f"Saved HelpSteer3 SVR artifacts to {args.output_dir}")
    print(
        "best_round={} dev_accuracy={:.4f} bank_size={} support_recall@k={:.4f}".format(
            summary["best_round_name"],
            summary["dev_metrics"]["accuracy"],
            summary["bank_size"],
            summary["dev_metrics"]["support_recall_at_k"],
        )
    )


if __name__ == "__main__":
    main()
