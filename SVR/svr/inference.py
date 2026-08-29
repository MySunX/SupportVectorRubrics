from __future__ import annotations

import os
from typing import Any

import torch

from svr.llm_ops import (
    LLMRubricRewriter,
    LLMPairwiseRubricScorer,
    OpenAICompatibleCompletionRunner,
    OpenAICompatibleLLMConfig,
    RealRewriteConfig,
    RealScorerConfig,
)
from svr.model import PromptSelectorModel, PromptVectorizer
from svr.schema import BankEntry
from svr.utils import load_json


class NoOpRubricRewriter:
    def rewrite(
        self,
        *,
        prompt_text: str,
        selected_rubrics: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        del prompt_text
        return [dict(item) for item in selected_rubrics]

    async def arewrite(
        self,
        *,
        prompt_text: str,
        selected_rubrics: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return self.rewrite(
            prompt_text=prompt_text,
            selected_rubrics=selected_rubrics,
        )


class SVRInferenceEngine:
    def __init__(
        self,
        *,
        model_dir: str,
        device: str = "cpu",
    ):
        self.model_dir = model_dir
        self.device = torch.device(device)
        self.vectorizer = PromptVectorizer.load(os.path.join(model_dir, "vectorizer.pkl"))
        self.bank = [
            BankEntry(**item) for item in load_json(os.path.join(model_dir, "bank.json"))
        ]
        inference_config_path = os.path.join(model_dir, "inference_config.json")
        if os.path.isfile(inference_config_path):
            inference_config = load_json(inference_config_path)
        else:
            inference_config = {
                "rewrite_selected_rubrics": False,
                "llm_model": "gpt-oss-120b",
            }
        self.inference_config = inference_config
        rewrite_selected_rubrics = bool(
            inference_config.get("rewrite_selected_rubrics", False)
        )
        llm_runner = OpenAICompatibleCompletionRunner(
            OpenAICompatibleLLMConfig(
                model=str(inference_config.get("llm_model", "gpt-oss-120b")),
                reasoning_effort=str(
                    inference_config.get("llm_reasoning_effort", "high")
                ),
                temperature=float(inference_config.get("llm_temperature", 0.0)),
                top_p=float(inference_config.get("llm_top_p", 1.0)),
                request_timeout_sec=int(
                    inference_config.get("llm_request_timeout_sec", 900)
                ),
                retry_times=int(inference_config.get("llm_retry_times", 8)),
                retry_backoff_seconds=float(
                    inference_config.get("llm_retry_backoff_seconds", 1.0)
                ),
                llm_base_url=inference_config.get("llm_base_url"),
                cache_dir=os.path.join(model_dir, "llm_cache"),
            )
        )
        if rewrite_selected_rubrics:
            self.rewriter = LLMRubricRewriter(
                RealRewriteConfig(
                    runner=llm_runner,
                    max_tokens=int(inference_config.get("rewrite_max_tokens", 4096)),
                )
            )
        else:
            self.rewriter = NoOpRubricRewriter()
        self.scorer = LLMPairwiseRubricScorer(
            RealScorerConfig(
                runner=llm_runner,
                chunk_size=int(inference_config.get("judge_rubric_chunk_size", 8)),
                judge_max_tokens=int(inference_config.get("score_max_tokens", 8192)),
            )
        )

        checkpoint = torch.load(
            os.path.join(model_dir, "model.pt"),
            map_location=self.device,
        )
        self.model = PromptSelectorModel(
            input_dim=int(checkpoint["input_dim"]),
            bank_size=int(checkpoint["bank_size"]),
            hidden_dim=int(checkpoint["hidden_dim"]),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def select_rubrics(self, prompt_text: str, top_k: int = 6) -> list[dict[str, Any]]:
        matrix = self.vectorizer.transform([prompt_text])
        tensor = torch.tensor(matrix.toarray(), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            bank_scores = self.model.score_bank(tensor)[0]
        k = min(top_k, bank_scores.numel())
        values, indices = torch.topk(bank_scores, k=k)

        selected = []
        for score, index in zip(values.cpu().tolist(), indices.cpu().tolist()):
            entry = self.bank[int(index)]
            selected.append(
                {
                    "bank_id": entry.bank_id,
                    "text": entry.text,
                    "facet": entry.facet,
                    "importance": entry.importance,
                    "source": entry.source,
                    "grounding": entry.grounding,
                    "selection_weight": float(score),
                }
            )
        return self.rewriter.rewrite(
            prompt_text=prompt_text,
            selected_rubrics=selected,
        )

    async def aselect_rubrics(
        self,
        prompt_text: str,
        top_k: int = 6,
    ) -> list[dict[str, Any]]:
        matrix = self.vectorizer.transform([prompt_text])
        tensor = torch.tensor(matrix.toarray(), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            bank_scores = self.model.score_bank(tensor)[0]
        k = min(top_k, bank_scores.numel())
        values, indices = torch.topk(bank_scores, k=k)

        selected = []
        for score, index in zip(values.cpu().tolist(), indices.cpu().tolist()):
            entry = self.bank[int(index)]
            selected.append(
                {
                    "bank_id": entry.bank_id,
                    "text": entry.text,
                    "facet": entry.facet,
                    "importance": entry.importance,
                    "source": entry.source,
                    "grounding": entry.grounding,
                    "selection_weight": float(score),
                }
            )
        if hasattr(self.rewriter, "arewrite"):
            return await self.rewriter.arewrite(
                prompt_text=prompt_text,
                selected_rubrics=selected,
            )
        return self.rewriter.rewrite(
            prompt_text=prompt_text,
            selected_rubrics=selected,
        )

    def score_pair(
        self,
        *,
        prompt_text: str,
        prompt_messages: list[dict[str, str]] | None = None,
        response_a: str,
        response_b: str,
        top_k: int = 6,
    ):
        selected = self.select_rubrics(prompt_text, top_k=top_k)
        return self.scorer.compare_responses(
            prompt_messages=prompt_messages,
            prompt_text=prompt_text,
            response_a=response_a,
            response_b=response_b,
            selected_rubrics=selected,
        )

    async def ascore_pair(
        self,
        *,
        prompt_text: str,
        prompt_messages: list[dict[str, str]] | None = None,
        response_a: str,
        response_b: str,
        top_k: int = 6,
    ):
        selected = await self.aselect_rubrics(prompt_text, top_k=top_k)
        return await self.scorer.acompare_responses(
            prompt_messages=prompt_messages,
            prompt_text=prompt_text,
            response_a=response_a,
            response_b=response_b,
            selected_rubrics=selected,
        )
