from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from svr.data import (
    DEFAULT_REFERENCE_RUBRIC_FIELD_CANDIDATES,
    DEFAULT_SELF_RUBRIC_FIELD_CANDIDATES,
    load_preference_examples,
)
from svr.eval_gap import EvaluationConfig, SVRGapEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SVR-lite against self and optional reference rubrics."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--test-path", nargs="+", required=True)
    parser.add_argument(
        "--self-rubric-field",
        action="append",
        default=list(DEFAULT_SELF_RUBRIC_FIELD_CANDIDATES),
    )
    parser.add_argument(
        "--reference-rubric-field",
        action="append",
        default=list(DEFAULT_REFERENCE_RUBRIC_FIELD_CANDIDATES),
        help="Fields used only for evaluation metrics, never for training.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--output-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_preference_examples(
        paths=args.test_path,
        self_rubric_fields=args.self_rubric_field,
        reference_rubric_fields=args.reference_rubric_field,
        limit=args.limit,
    )
    output_path = args.output_path
    if output_path is None:
        output_path = os.path.join(args.model_dir, "eval_gap.json")

    evaluator = SVRGapEvaluator(EvaluationConfig(top_k=args.top_k))
    result = evaluator.evaluate(
        examples=examples,
        model_dir=args.model_dir,
        output_path=output_path,
    )
    print(f"Saved evaluation summary to {output_path}")
    print(
        "self_accuracy={:.4f} svr_accuracy={:.4f} reference_accuracy={}".format(
            result["self"]["accuracy"],
            result["svr"]["accuracy"],
            (
                f"{result['reference']['accuracy']:.4f}"
                if result["reference"] is not None
                else "N/A"
            ),
        )
    )
    if result["gap_closed"] is not None:
        print(f"gap_closed={result['gap_closed']:.4f}")


if __name__ == "__main__":
    main()
