from __future__ import annotations

import unittest

from rrd_rubricbench.models import BenchmarkExample, RubricCandidate, SampledResponse
from rrd_rubricbench.rrd import PAPER_SAMPLE_MODELS, RRDPipeline


class DummyRunner:
    model = "dummy"
    reasoning_effort = "medium"
    max_concurrency = 4
    max_tokens = 16

    def complete_text(self, **kwargs: object) -> tuple[str, dict[str, object]]:
        return "{}", {}


def make_example() -> BenchmarkExample:
    return BenchmarkExample(
        case_id="case",
        domain="test",
        prompt_text="prompt",
        prompt_messages=[{"role": "user", "content": "prompt"}],
        response_a="A",
        response_b="B",
        gold_candidate="b",
    )


class RRDCoreTest(unittest.TestCase):
    def test_wu_weights_use_sample_major_vote_matrix(self) -> None:
        rubric_votes = [
            [False, False, False, False, False, False, False, True],
            [False, False, False, False, False, False, False, True],
            [False, False, False, False, False, False, False, True],
            [False, False, False, False, False, False, True, False],
            [False, False, False, True, False, False, False, False],
            [False, False, False, False, False, False, True, False],
            [False, False, False, False, False, False, False, True],
            [False, False, False, False, False, False, False, True],
            [False, False, False, False, False, False, False, True],
        ]
        accepted = [
            RubricCandidate(text=f"rubric {idx}", sample_votes=votes)
            for idx, votes in enumerate(rubric_votes)
        ]

        votes_matrix = RRDPipeline._sample_major_votes_matrix(accepted, sample_count=8)

        self.assertEqual(len(votes_matrix), 8)
        self.assertTrue(all(len(row) == len(accepted) for row in votes_matrix))

        pipeline = RRDPipeline(runner=DummyRunner())
        weights = pipeline._compute_weights(
            [rubric.text for rubric in accepted],
            votes_matrix=votes_matrix,
        )

        self.assertEqual(len(weights), len(accepted))
        self.assertNotEqual(weights, [1.0] * len(accepted))
        self.assertAlmostEqual(sum(weights), len(accepted))

    def test_evaluate_example_writes_computed_weights_to_accepted_rubrics(self) -> None:
        accepted = [
            RubricCandidate(text=f"rubric {idx}", sample_votes=votes)
            for idx, votes in enumerate(
                [
                    [False, False, False, False, False, False, False, True],
                    [False, False, False, False, False, False, True, False],
                    [False, False, False, True, False, False, False, False],
                ]
            )
        ]
        sampled = [
            SampledResponse(
                index=idx,
                group_index=0,
                group_tag="group",
                model="dummy",
                text=f"sample {idx}",
                temperature=0.0,
                top_p=1.0,
            )
            for idx in range(8)
        ]

        pipeline = RRDPipeline(runner=DummyRunner())
        pipeline._aggregated_judge = lambda example, accepted, weights: (  # type: ignore[method-assign]
            "a",
            "rrd_margin",
            0.0,
            [],
        )

        result = pipeline.evaluate_example(make_example(), sampled, accepted)

        self.assertEqual([rubric.weight for rubric in accepted], result["weights"])
        self.assertNotEqual([rubric.weight for rubric in accepted], [1.0] * len(accepted))

    def test_sample_example_uses_paper_models_four_each(self) -> None:
        class SamplingRunner(DummyRunner):
            def complete_text(self, **kwargs: object) -> tuple[str, dict[str, object]]:
                return f"sample from {kwargs['model']}", {}

        pipeline = RRDPipeline(runner=SamplingRunner())

        sampled = pipeline.sample_example(make_example())

        self.assertEqual(len(sampled), 8)
        self.assertEqual([sample.model for sample in sampled[:4]], [PAPER_SAMPLE_MODELS[0]] * 4)
        self.assertEqual([sample.model for sample in sampled[4:]], [PAPER_SAMPLE_MODELS[1]] * 4)
        self.assertEqual([sample.index for sample in sampled], list(range(8)))

    def test_directionality_guardrail_rejects_rubric_that_favors_weak_reference(self) -> None:
        class GuardrailRunner(DummyRunner):
            def complete_text(self, **kwargs: object) -> tuple[str, dict[str, object]]:
                prompt = str(kwargs.get("prompt", ""))
                if "<RESPONSE>" not in prompt:
                    return "reference response", {}
                if "strong response" in prompt:
                    return "<EVALUATION> NO </EVALUATION>", {}
                if "weak response" in prompt:
                    return "<EVALUATION> YES </EVALUATION>", {}
                return "reference response", {}

        pipeline = RRDPipeline(runner=GuardrailRunner())
        pipeline._reference_response_cache[make_example().case_id] = (
            SampledResponse(0, 0, "strong", "strong-model", "strong response", 0.7, 0.95),
            SampledResponse(1, 1, "weak", "weak-model", "weak response", 0.7, 0.95),
        )

        self.assertFalse(pipeline._passes_directionality_guardrail(make_example(), "rubric"))

    def test_directionality_guardrail_allows_rubric_that_does_not_favor_weak_reference(self) -> None:
        class GuardrailRunner(DummyRunner):
            def complete_text(self, **kwargs: object) -> tuple[str, dict[str, object]]:
                return "<EVALUATION> YES </EVALUATION>", {}

        pipeline = RRDPipeline(runner=GuardrailRunner())
        pipeline._reference_response_cache[make_example().case_id] = (
            SampledResponse(0, 0, "strong", "strong-model", "strong response", 0.7, 0.95),
            SampledResponse(1, 1, "weak", "weak-model", "weak response", 0.7, 0.95),
        )

        self.assertTrue(pipeline._passes_directionality_guardrail(make_example(), "rubric"))

    def test_final_rubric_judge_scores_each_response_separately(self) -> None:
        class FinalJudgeRunner(DummyRunner):
            def __init__(self) -> None:
                self.prompts: list[str] = []

            def complete_text(self, **kwargs: object) -> tuple[str, dict[str, object]]:
                prompt = str(kwargs.get("prompt", ""))
                self.prompts.append(prompt)
                if "<RESPONSE>\nA\n</RESPONSE>" in prompt:
                    return "<EVALUATION> YES </EVALUATION>", {}
                if "<RESPONSE>\nB\n</RESPONSE>" in prompt:
                    return "<EVALUATION> NO </EVALUATION>", {}
                return "<EVALUATION> NO </EVALUATION>", {}

        runner = FinalJudgeRunner()
        pipeline = RRDPipeline(runner=runner)

        result = pipeline._judge_rubric_pairwise(make_example(), "rubric", 0)

        self.assertEqual(result, {"response_a": True, "response_b": False})
        self.assertEqual(len(runner.prompts), 2)
        self.assertTrue(all("<RESPONSE>" in prompt for prompt in runner.prompts))
        self.assertTrue(all("RESPONSE_A:" not in prompt for prompt in runner.prompts))

    def test_successful_decomposition_does_not_count_as_rejection(self) -> None:
        class InitialRubricRunner(DummyRunner):
            def complete_text(self, **kwargs: object) -> tuple[str, dict[str, object]]:
                return "<RUBRIC>coarse rubric</RUBRIC>", {}

        pipeline = RRDPipeline(runner=InitialRubricRunner())
        pipeline.termination_threshold = 1

        calls = {"judge": 0}

        def judge_samples(
            example: BenchmarkExample,
            rubric: str,
            sampled: list[SampledResponse],
        ) -> dict[str, object]:
            calls["judge"] += 1
            if calls["judge"] == 1:
                votes = [True, True, True, False]
            else:
                votes = [True, False, False, False]
            return {
                "votes": votes,
                "yes_count": sum(votes),
                "vote_rate": sum(votes) / len(votes),
                "group_vote_rates": {"group": sum(votes) / len(votes)},
                "judge_calls": len(votes),
            }

        pipeline._judge_rubric_on_samples = judge_samples  # type: ignore[method-assign]
        pipeline._decompose_rubric = lambda **kwargs: ["leaf rubric"]  # type: ignore[method-assign]
        pipeline._passes_directionality_guardrail = lambda example, rubric: True  # type: ignore[method-assign]
        pipeline._overlap_or_conflict_reason = lambda rubric, existing_rubrics, **kwargs: None  # type: ignore[method-assign]

        sampled = [
            SampledResponse(
                index=idx,
                group_index=0,
                group_tag="group",
                model="dummy",
                text=f"sample {idx}",
                temperature=0.0,
                top_p=1.0,
            )
            for idx in range(4)
        ]
        result = pipeline.build_and_iterate_rubrics(make_example(), sampled)

        self.assertEqual(result["stats"]["rejected_count"], 0)
        self.assertEqual([r.text for r in result["accepted_rubrics"]], ["leaf rubric"])
        self.assertEqual(
            [step["decision"] for step in result["trace"]["steps"]],
            ["decompose", "accept"],
        )


if __name__ == "__main__":
    unittest.main()
