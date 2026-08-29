from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from svr.inference import SVRInferenceEngine
from svr.schema import PreferenceExample
from svr.utils import (
    dump_json,
    gold_aligned_margin,
    importance_weight,
    orient_preference_pair,
    preferred_position,
    semantic_prediction_payload,
)


@dataclass
class EvaluationConfig:
    top_k: int = 6


class SVRGapEvaluator:
    def __init__(self, config: EvaluationConfig | None = None):
        self.config = config or EvaluationConfig()

    def evaluate(
        self,
        *,
        examples: Sequence[PreferenceExample],
        model_dir: str,
        output_path: str | None = None,
    ) -> dict[str, Any]:
        engine = SVRInferenceEngine(model_dir=model_dir)
        self_summary = self._evaluate_static_rubrics(
            examples=examples,
            mode_name="self",
            rubric_getter=lambda example: example.self_rubrics,
            scorer=engine.scorer,
        )
        svr_summary = self._evaluate_svr(examples=examples, engine=engine)

        has_reference = any(example.reference_rubrics for example in examples)
        reference_summary = None
        if has_reference:
            reference_summary = self._evaluate_static_rubrics(
                examples=examples,
                mode_name="reference",
                rubric_getter=lambda example: example.reference_rubrics,
                scorer=engine.scorer,
            )

        result = {
            "self": self_summary,
            "svr": svr_summary,
            "reference": reference_summary,
            "gap_closed": None,
        }
        if reference_summary is not None:
            numerator = svr_summary["accuracy"] - self_summary["accuracy"]
            denominator = reference_summary["accuracy"] - self_summary["accuracy"]
            if denominator > 1e-6:
                result["gap_closed"] = numerator / denominator

        if output_path is not None:
            dump_json(output_path, result)
        return result

    def _evaluate_svr(
        self,
        *,
        examples: Sequence[PreferenceExample],
        engine: SVRInferenceEngine,
    ) -> dict[str, Any]:
        correct = 0
        ties = 0
        margins = []
        item_details = []
        for example in examples:
            oriented = orient_preference_pair(
                example_id=example.example_id,
                positive_response=example.chosen_response,
                negative_response=example.rejected_response,
                salt="eval_gap:svr",
            )
            prediction = engine.score_pair(
                prompt_text=example.prompt_text,
                prompt_messages=example.prompt_messages,
                response_a=oriented.response_a,
                response_b=oriented.response_b,
                top_k=self.config.top_k,
            )
            is_correct = prediction.preferred_side == oriented.gold_side
            correct += int(is_correct)
            ties += int(prediction.preferred_side == "tie")
            margins.append(gold_aligned_margin(prediction.weighted_margin, oriented.gold_side))
            item_details.append(
                {
                    "example_id": example.example_id,
                    "preferred_position": preferred_position(
                        prediction.preferred_side,
                        gold_side=oriented.gold_side,
                    ),
                    "margin_vs_chosen": gold_aligned_margin(
                        prediction.weighted_margin,
                        oriented.gold_side,
                    ),
                    "prediction": semantic_prediction_payload(
                        prediction,
                        gold_side=oriented.gold_side,
                    ),
                    "correct": is_correct,
                }
            )

        total = len(examples)
        return {
            "mode": "svr",
            "total_items": total,
            "accuracy": correct / total if total else 0.0,
            "tie_rate": ties / total if total else 0.0,
            "avg_margin": sum(margins) / total if total else 0.0,
            "results": item_details,
        }

    def _evaluate_static_rubrics(
        self,
        *,
        examples: Sequence[PreferenceExample],
        mode_name: str,
        rubric_getter,
        scorer,
    ) -> dict[str, Any]:
        correct = 0
        ties = 0
        usable = 0
        margins = []
        item_details = []

        for example in examples:
            rubrics = list(rubric_getter(example))
            if not rubrics:
                item_details.append(
                    {
                        "example_id": example.example_id,
                        "skipped": True,
                        "reason": f"missing {mode_name} rubrics",
                    }
                )
                continue

            selected = []
            for idx, rubric in enumerate(rubrics):
                selected.append(
                    {
                        "bank_id": idx,
                        "text": rubric.text,
                        "facet": rubric.facet,
                        "importance": rubric.importance,
                        "source": rubric.source,
                        "grounding": rubric.grounding,
                        "selection_weight": importance_weight(rubric.importance),
                    }
                )
            oriented = orient_preference_pair(
                example_id=example.example_id,
                positive_response=example.chosen_response,
                negative_response=example.rejected_response,
                salt=f"eval_gap:{mode_name}",
            )
            prediction = scorer.compare_responses(
                prompt_messages=example.prompt_messages,
                prompt_text=example.prompt_text,
                response_a=oriented.response_a,
                response_b=oriented.response_b,
                selected_rubrics=selected,
                difficulty_analysis=example.difficulty_analysis,
            )
            usable += 1
            is_correct = prediction.preferred_side == oriented.gold_side
            correct += int(is_correct)
            ties += int(prediction.preferred_side == "tie")
            margins.append(gold_aligned_margin(prediction.weighted_margin, oriented.gold_side))
            item_details.append(
                {
                    "example_id": example.example_id,
                    "preferred_position": preferred_position(
                        prediction.preferred_side,
                        gold_side=oriented.gold_side,
                    ),
                    "margin_vs_chosen": gold_aligned_margin(
                        prediction.weighted_margin,
                        oriented.gold_side,
                    ),
                    "prediction": semantic_prediction_payload(
                        prediction,
                        gold_side=oriented.gold_side,
                    ),
                    "correct": is_correct,
                }
            )

        return {
            "mode": mode_name,
            "total_items": len(examples),
            "usable_items": usable,
            "accuracy": correct / usable if usable else 0.0,
            "tie_rate": ties / usable if usable else 0.0,
            "avg_margin": sum(margins) / usable if usable else 0.0,
            "results": item_details,
        }
