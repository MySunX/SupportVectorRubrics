from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import pickle
import shutil
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch

from svr.bank import RubricBankBuilder, RubricBankBuilderConfig
from svr.bank import rubric_similarity
from svr.llm_ops import (
    LLMAdversarialProbe,
    LLMCandidateMiner,
    LLMPairwiseRubricScorer,
    OpenAICompatibleCompletionRunner,
    OpenAICompatibleLLMConfig,
    RealAdversarialConfig,
    RealMinerConfig,
    RealRewriteConfig,
    RealScorerConfig,
    LLMRubricRewriter,
)
from svr.model import (
    PromptSelectorModel,
    PromptVectorizer,
    VectorizerConfig,
)
from svr.pruning import (
    RubricBankPruner,
    RubricBankPrunerConfig,
)
from svr.schema import (
    BankEntry,
    HardNegativeResult,
    PreferenceExample,
    RubricItem,
    SupportPairRecord,
)
from svr.utils import (
    canonicalize_cache_model_name,
    cache_compatible_model_names,
    dump_json,
    ensure_dir,
    normalize_text_signature,
    stable_hash,
)

RUBRIC_TEXT_VALIDATION_VERSION = 1


@dataclass
class SVRTrainConfig:
    random_seed: int = 42
    dev_ratio: float = 0.1
    epochs: int = 30
    calibration_epochs: int = 3
    batch_size: int = 64
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 256
    support_positive_delta: float = 0.05
    support_loss_weight: float = 1.0
    preference_loss_weight: float = 1.0
    adv_loss_weight: float = 0.5
    lambda_w: float = 1e-3
    lambda_alpha: float = 1e-4
    positive_support_weight: float = 6.0
    negative_support_weight: float = 1.0
    top_k: int = 6
    num_rounds: int = 3
    support_pair_margin_threshold: float = 0.15
    max_support_pairs: int = 256
    min_support_pairs: int = 8
    weight_prune_threshold: float = 0.08
    min_activation_rate: float = 0.01
    prune_redundancy_threshold: float = 0.92
    min_bank_observed_count: int = 1
    max_bank_size: int = 1024
    bank_similarity_threshold: float = 0.88
    vectorizer_max_features: int = 4096
    vectorizer_min_df: int = 1
    hard_negative_max_candidates: int = 12
    hard_negative_dataset_candidates: int = 4
    enable_adversarial_probing: bool = True
    enable_support_expansion: bool = True
    rewrite_selected_rubrics: bool = True
    rewriter_add_prompt_hint: bool = True
    llm_model: str = "gpt-oss-120b"
    llm_reasoning_effort: str = "medium"
    llm_temperature: float = 0.0
    llm_top_p: float = 1.0
    llm_request_timeout_sec: int = 900
    llm_retry_times: int = 25
    llm_retry_backoff_seconds: float = 1.0
    llm_base_url: str | None = None
    judge_rubric_chunk_size: int = 8
    score_max_tokens: int = 8192
    mine_max_tokens: int = 8192
    rewrite_max_tokens: int = 8192
    llm_max_concurrency: int = 64
    llm_progress_log_interval: int = 64
    bank_progress_log_interval: int = 256
    reuse_candidate_stage_cache: bool = True
    reuse_bank_stage_cache: bool = True
    reuse_training_payload_stage_cache: bool = True
    reuse_round_resume_state: bool = True
    candidate_stage_cache_source_dir: str | None = None
    require_initial_train_candidate_cache: bool = False
    device: str = "cpu"


@dataclass
class _RoundResumePreparation:
    next_round_idx: int
    round_summaries: list[dict[str, Any]]
    best_round_dir: str | None
    best_dev_accuracy: float
    training_complete: bool
    train_candidate_cache: dict[str, list[RubricItem]] | None = None
    dev_candidate_cache: dict[str, list[RubricItem]] | None = None
    excluded_signatures: set[str] = field(default_factory=set)
    hard_negative_map: dict[str, HardNegativeResult] = field(default_factory=dict)


class SVRTrainer:
    def __init__(self, config: SVRTrainConfig | None = None):
        self.config = config or SVRTrainConfig()
        self.llm_runner = None
        self.miner = None
        self.scorer = None
        self.rewriter = None
        self.adversarial_probe = None
        self.bank_builder = RubricBankBuilder(
            RubricBankBuilderConfig(
                similarity_threshold=self.config.bank_similarity_threshold,
                min_observed_count=self.config.min_bank_observed_count,
                max_bank_size=self.config.max_bank_size,
            )
        )
        self.pruner = RubricBankPruner(
            RubricBankPrunerConfig(
                min_weight=self.config.weight_prune_threshold,
                min_activation_rate=self.config.min_activation_rate,
                redundancy_threshold=self.config.prune_redundancy_threshold,
                min_keep=max(self.config.top_k, self.config.min_support_pairs),
            )
        )

    def train(
        self,
        *,
        train_examples: list[PreferenceExample],
        dev_examples: list[PreferenceExample],
        output_dir: str,
    ) -> dict[str, Any]:
        if not train_examples:
            raise ValueError("train_examples is empty")
        if not dev_examples:
            raise ValueError("dev_examples is empty")

        ensure_dir(output_dir)
        self._set_seed(self.config.random_seed)
        self._configure_runtime(output_dir)
        self._progress(
            "training start: "
            f"train_examples={len(train_examples)} dev_examples={len(dev_examples)} "
            f"llm_model={self.config.llm_model}"
        )

        vectorizer = PromptVectorizer(
            VectorizerConfig(
                max_features=self.config.vectorizer_max_features,
                min_df=self.config.vectorizer_min_df,
            )
        )
        train_x = vectorizer.fit_transform([example.prompt_text for example in train_examples])
        dev_x = vectorizer.transform([example.prompt_text for example in dev_examples])
        self._progress(
            "vectorizer ready: "
            f"input_dim={train_x.shape[1]} output_dir={output_dir}"
        )

        device = torch.device(self.config.device)
        train_x_tensor = torch.tensor(train_x.toarray(), dtype=torch.float32, device=device)
        dev_x_tensor = torch.tensor(dev_x.toarray(), dtype=torch.float32, device=device)
        vectorizer.save(os.path.join(output_dir, "vectorizer.pkl"))

        resume = self._prepare_round_resume(
            train_examples=train_examples,
            dev_examples=dev_examples,
            output_dir=output_dir,
        )
        if resume.training_complete:
            return self._finalize_training(
                output_dir=output_dir,
                round_summaries=resume.round_summaries,
                best_round_dir=resume.best_round_dir,
                best_dev_accuracy=resume.best_dev_accuracy,
            )

        train_candidate_cache = resume.train_candidate_cache
        dev_candidate_cache = resume.dev_candidate_cache
        if train_candidate_cache is None or dev_candidate_cache is None:
            raise RuntimeError("resume preparation did not provide candidate caches")

        excluded_signatures = set(resume.excluded_signatures)
        hard_negative_map = dict(resume.hard_negative_map)
        round_summaries = list(resume.round_summaries)
        best_round_dir = resume.best_round_dir
        best_dev_accuracy = resume.best_dev_accuracy

        for round_idx in range(resume.next_round_idx, self.config.num_rounds + 1):
            round_dir = os.path.join(output_dir, f"round_{round_idx:02d}")
            ensure_dir(round_dir)
            round_result = self._load_partial_round_fit_result(
                round_idx=round_idx,
                round_dir=round_dir,
                train_examples=train_examples,
                train_candidate_cache=train_candidate_cache,
                hard_negative_map=hard_negative_map,
                train_x_tensor=train_x_tensor,
                device=device,
            )
            if round_result is None:
                bank = self._build_bank_from_candidate_cache(
                    train_candidate_cache,
                    excluded_signatures=excluded_signatures,
                    description=f"round_{round_idx:02d}_bank_build",
                    output_dir=round_dir,
                )
                if not bank:
                    if self.llm_runner is not None:
                        self.llm_runner.log_cache_summary()
                    raise ValueError(
                        "Rubric bank is empty after candidate-cache filtering. "
                        "No SVR round can be trained."
                    )
                self._progress(
                    f"round {round_idx} start: bank_size={len(bank)} "
                    f"hard_negatives={len(hard_negative_map)}"
                )

                round_result = self._fit_round(
                    round_idx=round_idx,
                    round_dir=round_dir,
                    bank=bank,
                    train_examples=train_examples,
                    dev_examples=dev_examples,
                    train_candidate_cache=train_candidate_cache,
                    dev_candidate_cache=dev_candidate_cache,
                    hard_negative_map=hard_negative_map,
                    train_x_tensor=train_x_tensor,
                    dev_x_tensor=dev_x_tensor,
                    device=device,
                )
            round_summaries.append(round_result["summary"])
            self._progress(
                f"round {round_idx} done: "
                f"train_acc={round_result['summary']['train_metrics']['accuracy']:.4f} "
                f"dev_acc={round_result['summary']['dev_metrics']['accuracy']:.4f} "
                f"bank_size={round_result['summary']['bank_size']}"
            )

            dev_accuracy = round_result["summary"]["dev_metrics"]["accuracy"]
            if dev_accuracy > best_dev_accuracy:
                best_dev_accuracy = dev_accuracy
                best_round_dir = round_dir

            if round_idx >= self.config.num_rounds:
                break

            support_pairs = self._select_support_pairs(
                examples=train_examples,
                bank=round_result["bank"],
                model=round_result["model"],
                x_tensor=train_x_tensor,
                clean_z=round_result["train_payload"]["clean_z"],
                adv_z=round_result["train_payload"]["adv_z"],
                adv_mask=round_result["train_payload"]["adv_mask"],
            )

            if self.config.enable_adversarial_probing:
                hard_negative_map = self._mine_hard_negatives(
                    examples=train_examples,
                    support_pairs=support_pairs,
                    bank=round_result["bank"],
                    model=round_result["model"],
                    x_tensor=train_x_tensor,
                )
            else:
                hard_negative_map = {}

            expansion_count = 0
            if self.config.enable_support_expansion and support_pairs:
                expansion_count = self._expand_candidate_cache(
                    train_examples=train_examples,
                    support_pairs=support_pairs,
                    hard_negative_map=hard_negative_map,
                    candidate_cache=train_candidate_cache,
                )

            excluded_signatures.update(round_result["summary"]["prune_summary"]["removed_signatures"])
            round_summaries[-1]["support_pairs"] = [item.to_dict() for item in support_pairs]
            round_summaries[-1]["hard_negatives"] = {
                key: value.to_dict() for key, value in hard_negative_map.items()
            }
            round_summaries[-1]["expansion_count"] = int(expansion_count)

            if not hard_negative_map and expansion_count == 0:
                round_summaries[-1]["early_stop_reason"] = (
                    "no_new_hard_negatives_or_expansions"
                )
                dump_json(
                    os.path.join(round_dir, "round_summary.json"),
                    round_summaries[-1],
                )
                self._save_round_resume_state(
                    output_dir=output_dir,
                    round_idx=round_idx,
                    train_examples=train_examples,
                    dev_examples=dev_examples,
                    candidate_cache=train_candidate_cache,
                    excluded_signatures=excluded_signatures,
                    hard_negative_map=hard_negative_map,
                )
                self._progress(
                    f"early stop after round {round_idx}: no new hard negatives or expansions"
                )
                break

            dump_json(
                os.path.join(round_dir, "round_summary.json"),
                round_summaries[-1],
            )
            self._save_round_resume_state(
                output_dir=output_dir,
                round_idx=round_idx,
                train_examples=train_examples,
                dev_examples=dev_examples,
                candidate_cache=train_candidate_cache,
                excluded_signatures=excluded_signatures,
                hard_negative_map=hard_negative_map,
            )

        return self._finalize_training(
            output_dir=output_dir,
            round_summaries=round_summaries,
            best_round_dir=best_round_dir,
            best_dev_accuracy=best_dev_accuracy,
        )

    def _prepare_round_resume(
        self,
        *,
        train_examples: list[PreferenceExample],
        dev_examples: list[PreferenceExample],
        output_dir: str,
    ) -> _RoundResumePreparation:
        if not self.config.reuse_round_resume_state:
            train_candidate_cache = self._build_candidate_cache(
                examples=train_examples,
                description="initial_train_candidate_mining",
                output_dir=output_dir,
            )
            dev_candidate_cache = self._build_candidate_cache(
                examples=dev_examples,
                description="initial_dev_candidate_mining",
                output_dir=output_dir,
            )
            return _RoundResumePreparation(
                next_round_idx=1,
                round_summaries=[],
                best_round_dir=None,
                best_dev_accuracy=float("-inf"),
                training_complete=False,
                train_candidate_cache=train_candidate_cache,
                dev_candidate_cache=dev_candidate_cache,
            )

        completed_round_summaries, training_complete = self._load_completed_round_summaries(
            output_dir=output_dir
        )
        best_round_dir, best_dev_accuracy = self._best_round_from_summaries(
            round_summaries=completed_round_summaries,
            output_dir=output_dir,
        )
        next_round_idx = len(completed_round_summaries) + 1
        if training_complete:
            self._progress(
                "round resume: detected complete training state "
                f"completed_rounds={len(completed_round_summaries)}"
            )
            return _RoundResumePreparation(
                next_round_idx=next_round_idx,
                round_summaries=completed_round_summaries,
                best_round_dir=best_round_dir,
                best_dev_accuracy=best_dev_accuracy,
                training_complete=True,
            )

        train_candidate_cache = self._build_candidate_cache(
            examples=train_examples,
            description="initial_train_candidate_mining",
            output_dir=output_dir,
        )
        dev_candidate_cache = self._build_candidate_cache(
            examples=dev_examples,
            description="initial_dev_candidate_mining",
            output_dir=output_dir,
        )

        excluded_signatures: set[str] = set()
        hard_negative_map: dict[str, HardNegativeResult] = {}
        if completed_round_summaries:
            (
                train_candidate_cache,
                excluded_signatures,
                hard_negative_map,
            ) = self._restore_round_state_from_completed_rounds(
                output_dir=output_dir,
                train_examples=train_examples,
                dev_examples=dev_examples,
                candidate_cache=train_candidate_cache,
                completed_round_summaries=completed_round_summaries,
            )
            self._progress(
                "round resume: restored training state "
                f"completed_rounds={len(completed_round_summaries)} next_round={next_round_idx}"
            )
        else:
            self._progress("round resume: no completed rounds detected")

        return _RoundResumePreparation(
            next_round_idx=next_round_idx,
            round_summaries=completed_round_summaries,
            best_round_dir=best_round_dir,
            best_dev_accuracy=best_dev_accuracy,
            training_complete=False,
            train_candidate_cache=train_candidate_cache,
            dev_candidate_cache=dev_candidate_cache,
            excluded_signatures=excluded_signatures,
            hard_negative_map=hard_negative_map,
        )

    def _load_completed_round_summaries(
        self,
        *,
        output_dir: str,
    ) -> tuple[list[dict[str, Any]], bool]:
        completed: list[dict[str, Any]] = []
        training_complete = False
        for round_idx in range(1, self.config.num_rounds + 1):
            round_dir = os.path.join(output_dir, f"round_{round_idx:02d}")
            summary = self._load_round_summary(round_dir=round_dir)
            if summary is None or not self._round_has_materialized_outputs(round_dir=round_dir):
                break
            status = self._round_summary_status(summary=summary, round_idx=round_idx)
            if status is None:
                break
            completed.append(summary)
            if status == "training_complete":
                training_complete = True
                break
        return completed, training_complete

    @staticmethod
    def _load_round_summary(*, round_dir: str) -> dict[str, Any] | None:
        summary_path = os.path.join(round_dir, "round_summary.json")
        if not os.path.isfile(summary_path):
            return None
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _round_has_materialized_outputs(*, round_dir: str) -> bool:
        for name in ("model.pt", "bank.json", "round_summary.json"):
            if not os.path.isfile(os.path.join(round_dir, name)):
                return False
        return True

    @staticmethod
    def _round_summary_has_fit_outputs(
        *,
        summary: dict[str, Any],
        round_idx: int,
    ) -> bool:
        try:
            summary_round_idx = int(summary.get("round_idx", -1))
        except (TypeError, ValueError):
            return False
        if summary_round_idx != round_idx:
            return False
        return all(
            key in summary
            for key in (
                "bank_size",
                "train_metrics",
                "dev_metrics",
                "history",
                "prune_summary",
                "num_hard_negatives",
            )
        )

    def _round_summary_status(
        self,
        *,
        summary: dict[str, Any],
        round_idx: int,
    ) -> str | None:
        try:
            summary_round_idx = int(summary.get("round_idx", -1))
        except (TypeError, ValueError):
            return None
        if summary_round_idx != round_idx:
            return None
        for key in ("bank_size", "train_metrics", "dev_metrics", "prune_summary"):
            if key not in summary:
                return None
        if round_idx >= self.config.num_rounds:
            return "training_complete"
        if summary.get("early_stop_reason"):
            return "training_complete"
        if all(
            key in summary
            for key in ("support_pairs", "hard_negatives", "expansion_count")
        ):
            return "transition_complete"
        return None

    @staticmethod
    def _best_round_from_summaries(
        *,
        round_summaries: list[dict[str, Any]],
        output_dir: str,
    ) -> tuple[str | None, float]:
        best_round_dir: str | None = None
        best_dev_accuracy = float("-inf")
        for summary in round_summaries:
            metrics = summary.get("dev_metrics")
            if not isinstance(metrics, dict):
                continue
            try:
                dev_accuracy = float(metrics.get("accuracy", float("-inf")))
                round_idx = int(summary.get("round_idx", -1))
            except (TypeError, ValueError):
                continue
            if dev_accuracy > best_dev_accuracy and round_idx > 0:
                best_dev_accuracy = dev_accuracy
                best_round_dir = os.path.join(output_dir, f"round_{round_idx:02d}")
        return best_round_dir, best_dev_accuracy

    def _finalize_training(
        self,
        *,
        output_dir: str,
        round_summaries: list[dict[str, Any]],
        best_round_dir: str | None,
        best_dev_accuracy: float,
    ) -> dict[str, Any]:
        if best_round_dir is None:
            if self.llm_runner is not None:
                self.llm_runner.log_cache_summary()
            raise RuntimeError("No SVR training round completed successfully")

        self._publish_best_round(best_round_dir=best_round_dir, output_dir=output_dir)
        self._write_inference_config(output_dir=output_dir)

        best_round_name = os.path.basename(best_round_dir)
        best_round_summary = next(
            summary
            for summary in round_summaries
            if f"round_{summary['round_idx']:02d}" == best_round_name
        )
        result = {
            "config": asdict(self.config),
            "best_round_dir": best_round_dir,
            "best_round_name": best_round_name,
            "best_dev_accuracy": best_dev_accuracy,
            "round_summaries": round_summaries,
        }
        dump_json(os.path.join(output_dir, "train_summary.json"), result)
        self._progress(
            "training finished: "
            f"best_round={best_round_name} best_dev_accuracy={best_dev_accuracy:.4f}"
        )
        if self.llm_runner is not None:
            self.llm_runner.log_cache_summary()
        return {
            "bank_size": best_round_summary["bank_size"],
            "train_metrics": best_round_summary["train_metrics"],
            "dev_metrics": best_round_summary["dev_metrics"],
            "history": round_summaries,
            "best_round_name": best_round_name,
            "best_dev_accuracy": best_dev_accuracy,
        }

    def _write_inference_config(self, *, output_dir: str) -> None:
        dump_json(
            os.path.join(output_dir, "inference_config.json"),
            {
                "rewrite_selected_rubrics": self.config.rewrite_selected_rubrics,
                "rewriter_add_prompt_hint": self.config.rewriter_add_prompt_hint,
                "llm_model": self.config.llm_model,
                "llm_reasoning_effort": self.config.llm_reasoning_effort,
                "llm_temperature": self.config.llm_temperature,
                "llm_top_p": self.config.llm_top_p,
                "llm_request_timeout_sec": self.config.llm_request_timeout_sec,
                "llm_retry_times": self.config.llm_retry_times,
                "llm_retry_backoff_seconds": self.config.llm_retry_backoff_seconds,
                "llm_base_url": self.config.llm_base_url,
                "judge_rubric_chunk_size": self.config.judge_rubric_chunk_size,
                "score_max_tokens": self.config.score_max_tokens,
                "rewrite_max_tokens": self.config.rewrite_max_tokens,
            },
        )

    def _restore_round_state_from_completed_rounds(
        self,
        *,
        output_dir: str,
        train_examples: list[PreferenceExample],
        dev_examples: list[PreferenceExample],
        candidate_cache: dict[str, list[RubricItem]],
        completed_round_summaries: list[dict[str, Any]],
    ) -> tuple[
        dict[str, list[RubricItem]],
        set[str],
        dict[str, HardNegativeResult],
    ]:
        restored_cache = candidate_cache
        excluded_signatures: set[str] = set()
        hard_negative_map: dict[str, HardNegativeResult] = {}
        latest_completed_round = int(completed_round_summaries[-1]["round_idx"])

        restored_from_round = 0
        for round_idx in range(latest_completed_round, 0, -1):
            state = self._load_round_resume_state(
                output_dir=output_dir,
                round_idx=round_idx,
                train_examples=train_examples,
                dev_examples=dev_examples,
            )
            if state is None:
                continue
            restored_cache, excluded_signatures, hard_negative_map = state
            restored_from_round = round_idx
            self._progress(
                "round resume snapshot hit: "
                f"round={round_idx} path={self._round_resume_state_path(output_dir=output_dir, round_idx=round_idx)}"
            )
            break

        if restored_from_round == 0:
            self._progress(
                "round resume legacy replay: "
                f"target_round={latest_completed_round}"
            )
        elif restored_from_round < latest_completed_round:
            self._progress(
                "round resume replay from snapshot: "
                f"snapshot_round={restored_from_round} target_round={latest_completed_round}"
            )

        for summary in completed_round_summaries:
            round_idx = int(summary["round_idx"])
            if round_idx <= restored_from_round:
                continue
            hard_negative_map = self._deserialize_hard_negative_map(
                summary.get("hard_negatives", {})
            )
            added = self._replay_round_transition(
                train_examples=train_examples,
                candidate_cache=restored_cache,
                summary=summary,
                hard_negative_map=hard_negative_map,
            )
            excluded_signatures.update(
                str(signature)
                for signature in summary.get("prune_summary", {}).get(
                    "removed_signatures", []
                )
                if str(signature)
            )
            expected_added = summary.get("expansion_count")
            if isinstance(expected_added, int) and expected_added != added:
                self._progress(
                    "round resume replay expansion_count_mismatch: "
                    f"round={round_idx} expected={expected_added} restored={added}"
                )

        self._save_round_resume_state(
            output_dir=output_dir,
            round_idx=latest_completed_round,
            train_examples=train_examples,
            dev_examples=dev_examples,
            candidate_cache=restored_cache,
            excluded_signatures=excluded_signatures,
            hard_negative_map=hard_negative_map,
        )
        return restored_cache, excluded_signatures, hard_negative_map

    def _replay_round_transition(
        self,
        *,
        train_examples: list[PreferenceExample],
        candidate_cache: dict[str, list[RubricItem]],
        summary: dict[str, Any],
        hard_negative_map: dict[str, HardNegativeResult],
    ) -> int:
        support_pairs = self._deserialize_support_pairs(summary.get("support_pairs", []))
        if not support_pairs:
            return 0
        return self._expand_candidate_cache(
            train_examples=train_examples,
            support_pairs=support_pairs,
            hard_negative_map=hard_negative_map,
            candidate_cache=candidate_cache,
        )

    @staticmethod
    def _deserialize_support_pairs(payload: Any) -> list[SupportPairRecord]:
        if not isinstance(payload, list):
            return []
        results: list[SupportPairRecord] = []
        for item in payload:
            if isinstance(item, SupportPairRecord):
                if item.example_id:
                    results.append(item)
                continue
            if not isinstance(item, dict):
                continue
            example_id = str(item.get("example_id", "")).strip()
            if not example_id:
                continue
            selected_bank_ids = item.get("selected_bank_ids", [])
            if not isinstance(selected_bank_ids, list):
                selected_bank_ids = []
            adv_margin = item.get("adv_margin")
            try:
                record = SupportPairRecord(
                    example_id=example_id,
                    clean_margin=float(item.get("clean_margin", 0.0)),
                    is_misclassified=bool(item.get("is_misclassified", False)),
                    has_hard_negative=bool(item.get("has_hard_negative", False)),
                    adv_margin=float(adv_margin) if adv_margin is not None else None,
                    selected_bank_ids=[int(bank_id) for bank_id in selected_bank_ids],
                )
            except (TypeError, ValueError):
                continue
            results.append(record)
        return results

    @staticmethod
    def _deserialize_hard_negative_map(payload: Any) -> dict[str, HardNegativeResult]:
        if not isinstance(payload, dict):
            return {}
        results: dict[str, HardNegativeResult] = {}
        for example_id, item in payload.items():
            if isinstance(item, HardNegativeResult):
                if item.example_id:
                    results[item.example_id] = item
                continue
            if not isinstance(item, dict):
                continue
            resolved_example_id = str(item.get("example_id", example_id)).strip()
            if not resolved_example_id:
                continue
            try:
                results[resolved_example_id] = HardNegativeResult(
                    example_id=resolved_example_id,
                    response_text=str(item.get("response_text", "")),
                    source=str(item.get("source", "")),
                    weighted_margin_vs_chosen=float(
                        item.get("weighted_margin_vs_chosen", 0.0)
                    ),
                    candidate_count=int(item.get("candidate_count", 0)),
                    selected_rubrics=list(item.get("selected_rubrics", [])),
                )
            except (TypeError, ValueError):
                continue
        return results

    def _round_resume_state_dir(self, *, output_dir: str) -> str:
        return os.path.join(os.path.abspath(output_dir), "round_resume_state")

    def _round_resume_state_path(self, *, output_dir: str, round_idx: int) -> str:
        return os.path.join(
            self._round_resume_state_dir(output_dir=output_dir),
            f"round_{round_idx:02d}.pkl",
        )

    def _round_resume_state_config_payload(self) -> dict[str, Any]:
        payload = asdict(self.config)
        for key in (
            "num_rounds",
            "device",
            "llm_max_concurrency",
            "llm_progress_log_interval",
            "bank_progress_log_interval",
            "candidate_stage_cache_source_dir",
            "require_initial_train_candidate_cache",
            "reuse_candidate_stage_cache",
            "reuse_bank_stage_cache",
            "reuse_training_payload_stage_cache",
            "reuse_round_resume_state",
            "llm_base_url",
        ):
            payload.pop(key, None)
        payload["rubric_text_validation_version"] = RUBRIC_TEXT_VALIDATION_VERSION
        return payload

    def _round_resume_state_config_signature(self) -> str:
        payload = self._round_resume_state_config_payload()
        return stable_hash(json.dumps(payload, ensure_ascii=False, sort_keys=True))

    def _examples_signature(self, *, examples: list[PreferenceExample]) -> str:
        payload = [
            {
                "example_id": example.example_id,
                "fingerprint": self._candidate_stage_cache_example_fingerprint(example),
            }
            for example in examples
        ]
        return stable_hash(
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            prefix="round_resume_examples_",
        )

    def _load_round_resume_state(
        self,
        *,
        output_dir: str,
        round_idx: int,
        train_examples: list[PreferenceExample],
        dev_examples: list[PreferenceExample],
    ) -> tuple[
        dict[str, list[RubricItem]],
        set[str],
        dict[str, HardNegativeResult],
    ] | None:
        cache_path = self._round_resume_state_path(
            output_dir=output_dir,
            round_idx=round_idx,
        )
        if not os.path.isfile(cache_path):
            return None
        try:
            with open(cache_path, "rb") as f:
                payload = pickle.load(f)
        except Exception as exc:  # noqa: BLE001
            self._progress(
                f"round_resume_state load_failed: path={cache_path} error={exc}"
            )
            return None

        if payload.get("config_signature") != self._round_resume_state_config_signature():
            self._progress(
                f"round_resume_state config_mismatch: path={cache_path}"
            )
            return None
        if payload.get("train_examples_signature") != self._examples_signature(
            examples=train_examples
        ):
            self._progress(
                f"round_resume_state train_examples_mismatch: path={cache_path}"
            )
            return None
        if payload.get("dev_examples_signature") != self._examples_signature(
            examples=dev_examples
        ):
            self._progress(
                f"round_resume_state dev_examples_mismatch: path={cache_path}"
            )
            return None

        candidate_cache = payload.get("candidate_cache")
        excluded_signatures = payload.get("excluded_signatures")
        hard_negative_payload = payload.get("hard_negative_map")
        if (
            not isinstance(candidate_cache, dict)
            or not isinstance(excluded_signatures, list)
            or not isinstance(hard_negative_payload, dict)
        ):
            self._progress(
                f"round_resume_state invalid_payload: path={cache_path}"
            )
            return None
        expected_example_ids = {example.example_id for example in train_examples}
        if not expected_example_ids.issubset(candidate_cache.keys()):
            self._progress(
                f"round_resume_state candidate_cache_mismatch: path={cache_path}"
            )
            return None

        return (
            {
                example.example_id: candidate_cache[example.example_id]
                for example in train_examples
            },
            {str(item) for item in excluded_signatures if str(item)},
            self._deserialize_hard_negative_map(hard_negative_payload),
        )

    def _save_round_resume_state(
        self,
        *,
        output_dir: str,
        round_idx: int,
        train_examples: list[PreferenceExample],
        dev_examples: list[PreferenceExample],
        candidate_cache: dict[str, list[RubricItem]],
        excluded_signatures: set[str],
        hard_negative_map: dict[str, HardNegativeResult],
    ) -> None:
        if not self.config.reuse_round_resume_state:
            return
        cache_path = self._round_resume_state_path(
            output_dir=output_dir,
            round_idx=round_idx,
        )
        ensure_dir(os.path.dirname(cache_path))
        payload = {
            "cache_schema_version": 1,
            "round_idx": round_idx,
            "config_signature": self._round_resume_state_config_signature(),
            "config_payload": self._round_resume_state_config_payload(),
            "train_examples_signature": self._examples_signature(
                examples=train_examples
            ),
            "dev_examples_signature": self._examples_signature(examples=dev_examples),
            "candidate_cache": {
                example.example_id: candidate_cache[example.example_id]
                for example in train_examples
            },
            "excluded_signatures": sorted(excluded_signatures),
            "hard_negative_map": {
                example_id: result.to_dict()
                for example_id, result in hard_negative_map.items()
            },
        }
        with open(cache_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        self._progress(
            "round_resume_state saved: "
            f"path={cache_path} round={round_idx} examples={len(train_examples)}"
        )

    def _configure_runtime(self, output_dir: str) -> None:
        llm_cache_dir = os.path.join(output_dir, "llm_cache")
        llm_runner = OpenAICompatibleCompletionRunner(
            OpenAICompatibleLLMConfig(
                model=self.config.llm_model,
                reasoning_effort=self.config.llm_reasoning_effort,
                temperature=self.config.llm_temperature,
                top_p=self.config.llm_top_p,
                request_timeout_sec=self.config.llm_request_timeout_sec,
                retry_times=self.config.llm_retry_times,
                retry_backoff_seconds=self.config.llm_retry_backoff_seconds,
                llm_base_url=self.config.llm_base_url,
                cache_dir=llm_cache_dir,
                max_concurrency=self.config.llm_max_concurrency,
                progress_log_interval=self.config.llm_progress_log_interval,
            )
        )
        self.llm_runner = llm_runner
        self.miner = LLMCandidateMiner(
            RealMinerConfig(
                runner=llm_runner,
                prompt_only_max_tokens=self.config.score_max_tokens,
                contrastive_max_tokens=self.config.mine_max_tokens,
            )
        )
        self.scorer = LLMPairwiseRubricScorer(
            RealScorerConfig(
                runner=llm_runner,
                chunk_size=self.config.judge_rubric_chunk_size,
                judge_max_tokens=self.config.score_max_tokens,
            )
        )
        self.adversarial_probe = LLMAdversarialProbe(
            RealAdversarialConfig(
                runner=llm_runner,
                scorer=self.scorer,
                max_candidates=self.config.hard_negative_max_candidates,
                max_tokens=self.config.mine_max_tokens,
            )
        )
        self.rewriter = LLMRubricRewriter(
            RealRewriteConfig(
                runner=llm_runner,
                max_tokens=self.config.rewrite_max_tokens,
            )
        )
        self._progress(
            "runtime configured: real llm path "
            f"base_url={self.config.llm_base_url or os.getenv('LLM_BASE_URL') or os.getenv('OPENAI_BASE_URL') or os.getenv('OPENAI_API_BASE')}"
        )

    def _fit_round(
        self,
        *,
        round_idx: int,
        round_dir: str,
        bank: list[BankEntry],
        train_examples: list[PreferenceExample],
        dev_examples: list[PreferenceExample],
        train_candidate_cache: dict[str, list[RubricItem]],
        dev_candidate_cache: dict[str, list[RubricItem]],
        hard_negative_map: dict[str, HardNegativeResult],
        train_x_tensor: torch.Tensor,
        dev_x_tensor: torch.Tensor,
        device: torch.device,
    ) -> dict[str, Any]:
        train_payload = self._build_training_payload(
            examples=train_examples,
            candidate_cache=train_candidate_cache,
            bank=bank,
            hard_negative_map=hard_negative_map,
            description=f"round_{round_idx:02d}_train_payload_initial",
            output_dir=round_dir,
        )
        dev_payload = self._build_training_payload(
            examples=dev_examples,
            candidate_cache=dev_candidate_cache,
            bank=bank,
            hard_negative_map={},
            description=f"round_{round_idx:02d}_dev_payload_initial",
            output_dir=round_dir,
        )

        model = PromptSelectorModel(
            input_dim=train_x_tensor.shape[1],
            bank_size=len(bank),
            hidden_dim=self.config.hidden_dim,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        train_tensors = self._payload_to_tensors(train_payload, device=device)
        dev_tensors = self._payload_to_tensors(dev_payload, device=device)
        diversity_matrix = self._build_bank_diversity_matrix(
            bank=bank,
            device=device,
        )
        history = self._optimize_model(
            model=model,
            optimizer=optimizer,
            train_x_tensor=train_x_tensor,
            train_tensors=train_tensors,
            dev_x_tensor=dev_x_tensor,
            dev_tensors=dev_tensors,
            diversity_matrix=diversity_matrix,
            epochs=self.config.epochs,
        )

        activation_rates, activation_counts, global_weights = self._compute_bank_usage(
            model=model,
            x_tensor=train_x_tensor,
        )
        self._apply_bank_statistics(
            bank=bank,
            support_counts=train_payload["support_counts"],
            activation_counts=activation_counts,
        )
        pruned_bank, prune_summary = self.pruner.prune(
            bank,
            global_weights=global_weights,
            activation_rates=activation_rates,
        )

        if len(pruned_bank) < len(bank):
            kept_original_ids = prune_summary["kept_bank_ids"]
            keep_indices = sorted(int(item) for item in kept_original_ids)
            model.prune_output(keep_indices)
            pruned_bank = [bank[idx] for idx in keep_indices]
            for new_bank_id, entry in enumerate(pruned_bank):
                entry.bank_id = new_bank_id

            train_payload = self._build_training_payload(
                examples=train_examples,
                candidate_cache=train_candidate_cache,
                bank=pruned_bank,
                hard_negative_map=hard_negative_map,
                description=f"round_{round_idx:02d}_train_payload_pruned",
                output_dir=round_dir,
            )
            dev_payload = self._build_training_payload(
                examples=dev_examples,
                candidate_cache=dev_candidate_cache,
                bank=pruned_bank,
                hard_negative_map={},
                description=f"round_{round_idx:02d}_dev_payload_pruned",
                output_dir=round_dir,
            )
            train_tensors = self._payload_to_tensors(train_payload, device=device)
            dev_tensors = self._payload_to_tensors(dev_payload, device=device)
            diversity_matrix = self._build_bank_diversity_matrix(
                bank=pruned_bank,
                device=device,
            )
            if self.config.calibration_epochs > 0:
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=self.config.learning_rate * 0.5,
                    weight_decay=self.config.weight_decay,
                )
                calibration_history = self._optimize_model(
                    model=model,
                    optimizer=optimizer,
                    train_x_tensor=train_x_tensor,
                    train_tensors=train_tensors,
                    dev_x_tensor=dev_x_tensor,
                    dev_tensors=dev_tensors,
                    diversity_matrix=diversity_matrix,
                    epochs=self.config.calibration_epochs,
                )
                history.extend(calibration_history)

            activation_rates, activation_counts, global_weights = self._compute_bank_usage(
                model=model,
                x_tensor=train_x_tensor,
            )
            self._apply_bank_statistics(
                bank=pruned_bank,
                support_counts=train_payload["support_counts"],
                activation_counts=activation_counts,
            )

        train_metrics = self._evaluate_tensors(
            model=model,
            x_tensor=train_x_tensor,
            tensors=train_tensors,
        )
        dev_metrics = self._evaluate_tensors(
            model=model,
            x_tensor=dev_x_tensor,
            tensors=dev_tensors,
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_dim": model.input_dim,
                "hidden_dim": model.hidden_dim,
                "bank_size": model.bank_size,
            },
            os.path.join(round_dir, "model.pt"),
        )
        dump_json(
            os.path.join(round_dir, "bank.json"),
            [entry.to_dict() for entry in pruned_bank],
        )

        removed_signatures = [
            normalize_text_signature(bank[item].text)
            for item in prune_summary["removed_bank_ids"]
            if 0 <= int(item) < len(bank)
        ]
        round_summary = {
            "round_idx": round_idx,
            "bank_size": len(pruned_bank),
            "train_metrics": train_metrics,
            "dev_metrics": dev_metrics,
            "history": history,
            "prune_summary": {
                **prune_summary,
                "removed_signatures": removed_signatures,
            },
            "num_hard_negatives": len(hard_negative_map),
        }
        dump_json(os.path.join(round_dir, "round_summary.json"), round_summary)

        return {
            "model": model,
            "bank": pruned_bank,
            "train_payload": train_payload,
            "summary": round_summary,
        }

    def _load_partial_round_fit_result(
        self,
        *,
        round_idx: int,
        round_dir: str,
        train_examples: list[PreferenceExample],
        train_candidate_cache: dict[str, list[RubricItem]],
        hard_negative_map: dict[str, HardNegativeResult],
        train_x_tensor: torch.Tensor,
        device: torch.device,
    ) -> dict[str, Any] | None:
        summary = self._load_round_summary(round_dir=round_dir)
        if summary is None or not self._round_has_materialized_outputs(round_dir=round_dir):
            return None
        if self._round_summary_status(summary=summary, round_idx=round_idx) is not None:
            return None
        if not self._round_summary_has_fit_outputs(summary=summary, round_idx=round_idx):
            return None

        summary_hard_negatives = summary.get("num_hard_negatives")
        if (
            isinstance(summary_hard_negatives, int)
            and summary_hard_negatives != len(hard_negative_map)
        ):
            self._progress(
                "partial round resume skipped: "
                f"round={round_idx} hard_negative_mismatch "
                f"summary={summary_hard_negatives} restored={len(hard_negative_map)}"
            )
            return None

        bank = self._load_round_bank(round_dir=round_dir)
        if bank is None:
            return None
        try:
            summary_bank_size = int(summary.get("bank_size", -1))
        except (TypeError, ValueError):
            self._progress(
                f"partial round resume skipped: round={round_idx} invalid_bank_size"
            )
            return None
        if len(bank) != summary_bank_size:
            self._progress(
                "partial round resume skipped: "
                f"round={round_idx} bank_size_mismatch "
                f"summary={summary_bank_size} actual={len(bank)}"
            )
            return None

        model = self._load_round_model(
            round_dir=round_dir,
            expected_input_dim=int(train_x_tensor.shape[1]),
            expected_bank_size=len(bank),
            device=device,
        )
        if model is None:
            return None

        payload_description = self._resolve_round_train_payload_description(
            round_idx=round_idx,
            round_dir=round_dir,
            summary=summary,
        )
        train_payload = self._build_training_payload(
            examples=train_examples,
            candidate_cache=train_candidate_cache,
            bank=bank,
            hard_negative_map=hard_negative_map,
            description=payload_description,
            output_dir=round_dir,
        )
        self._progress(
            "partial round resume: "
            f"round={round_idx} reused_saved_fit_outputs "
            f"payload={payload_description} bank_size={len(bank)}"
        )
        return {
            "model": model,
            "bank": bank,
            "train_payload": train_payload,
            "summary": summary,
        }

    def _resolve_round_train_payload_description(
        self,
        *,
        round_idx: int,
        round_dir: str,
        summary: dict[str, Any],
    ) -> str:
        pruned_description = f"round_{round_idx:02d}_train_payload_pruned"
        initial_description = f"round_{round_idx:02d}_train_payload_initial"
        for description in (pruned_description, initial_description):
            cache_dir = self._training_payload_stage_cache_dir(
                output_dir=round_dir,
                description=description,
            )
            if os.path.isdir(cache_dir):
                return description

        prune_summary = summary.get("prune_summary")
        if isinstance(prune_summary, dict) and prune_summary.get("removed_bank_ids"):
            return pruned_description
        return initial_description

    def _load_round_model(
        self,
        *,
        round_dir: str,
        expected_input_dim: int,
        expected_bank_size: int,
        device: torch.device,
    ) -> PromptSelectorModel | None:
        model_path = os.path.join(round_dir, "model.pt")
        if not os.path.isfile(model_path):
            return None
        try:
            checkpoint = torch.load(model_path, map_location=device)
        except Exception as exc:  # noqa: BLE001
            self._progress(
                f"partial round resume skipped: model_load_failed path={model_path} error={exc}"
            )
            return None
        if not isinstance(checkpoint, dict):
            self._progress(
                f"partial round resume skipped: invalid_model_payload path={model_path}"
            )
            return None
        state_dict = checkpoint.get("model_state_dict")
        if not isinstance(state_dict, dict):
            self._progress(
                f"partial round resume skipped: missing_model_state path={model_path}"
            )
            return None
        try:
            input_dim = int(checkpoint.get("input_dim", expected_input_dim))
            hidden_dim = int(checkpoint.get("hidden_dim", self.config.hidden_dim))
            bank_size = int(checkpoint.get("bank_size", expected_bank_size))
        except (TypeError, ValueError):
            self._progress(
                f"partial round resume skipped: invalid_model_metadata path={model_path}"
            )
            return None
        if input_dim != expected_input_dim or bank_size != expected_bank_size:
            self._progress(
                "partial round resume skipped: "
                f"model_shape_mismatch path={model_path} "
                f"expected_input_dim={expected_input_dim} actual_input_dim={input_dim} "
                f"expected_bank_size={expected_bank_size} actual_bank_size={bank_size}"
            )
            return None
        model = PromptSelectorModel(
            input_dim=input_dim,
            bank_size=bank_size,
            hidden_dim=hidden_dim,
        ).to(device)
        try:
            model.load_state_dict(state_dict)
        except Exception as exc:  # noqa: BLE001
            self._progress(
                f"partial round resume skipped: model_state_mismatch path={model_path} error={exc}"
            )
            return None
        model.eval()
        return model

    def _load_round_bank(
        self,
        *,
        round_dir: str,
    ) -> list[BankEntry] | None:
        bank_path = os.path.join(round_dir, "bank.json")
        if not os.path.isfile(bank_path):
            return None
        try:
            with open(bank_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:  # noqa: BLE001
            self._progress(
                f"partial round resume skipped: bank_load_failed path={bank_path} error={exc}"
            )
            return None
        if not isinstance(payload, list):
            self._progress(
                f"partial round resume skipped: invalid_bank_payload path={bank_path}"
            )
            return None

        bank: list[BankEntry] = []
        try:
            for idx, item in enumerate(payload):
                if not isinstance(item, dict):
                    raise TypeError(f"bank entry {idx} is not a dict")
                bank.append(
                    BankEntry(
                        bank_id=int(item.get("bank_id", idx)),
                        text=str(item.get("text", "")),
                        facet=str(item.get("facet", "correctness")),
                        importance=str(item.get("importance", "major")),
                        source=str(item.get("source", "unknown")),
                        grounding=str(item.get("grounding", "")),
                        aliases=[
                            str(alias)
                            for alias in item.get("aliases", [])
                            if str(alias)
                        ],
                        observed_count=int(item.get("observed_count", 0)),
                        support_count=int(item.get("support_count", 0)),
                        activation_count=int(item.get("activation_count", 0)),
                        metadata=(
                            dict(item.get("metadata", {}))
                            if isinstance(item.get("metadata"), dict)
                            else {}
                        ),
                    )
                )
        except Exception as exc:  # noqa: BLE001
            self._progress(
                f"partial round resume skipped: invalid_bank_entry path={bank_path} error={exc}"
            )
            return None
        return bank

    def _build_candidate_cache(
        self,
        *,
        examples: list[PreferenceExample],
        description: str,
        output_dir: str,
    ) -> dict[str, list[RubricItem]]:
        assert self.llm_runner is not None
        assert self.miner is not None
        stage_cache, stage_cache_diagnostics = self._load_candidate_stage_cache(
            examples=examples,
            description=description,
            output_dir=output_dir,
        )
        if stage_cache is not None:
            return stage_cache
        if (
            description == "initial_train_candidate_mining"
            and self.config.require_initial_train_candidate_cache
        ):
            raise RuntimeError(
                "initial_train_candidate_mining cache is required but unavailable. "
                f"details={self._format_candidate_stage_cache_diagnostics(stage_cache_diagnostics)}"
            )

        async def _run() -> dict[str, list[RubricItem]]:
            async def _mine_one(example: PreferenceExample, _: int) -> tuple[str, list[RubricItem]]:
                return example.example_id, await self.miner.amine(example)

            results = await self.llm_runner.parallel_map(
                examples,
                _mine_one,
                description=description,
                max_concurrency=self.config.llm_max_concurrency,
                progress_log_interval=self.config.llm_progress_log_interval,
            )
            return {example_id: rubrics for example_id, rubrics in results}

        candidate_cache = asyncio.run(_run())
        self._save_candidate_stage_cache(
            examples=examples,
            description=description,
            output_dir=output_dir,
            candidate_cache=candidate_cache,
        )
        return candidate_cache

    def _candidate_stage_cache_path(self, *, output_dir: str, description: str) -> str:
        del description
        normalized = os.path.abspath(output_dir)
        if normalized.endswith(".pkl"):
            return normalized
        if os.path.basename(normalized) == "candidate_stage_cache":
            return os.path.join(normalized, "candidate_examples.pkl")
        return os.path.join(normalized, "candidate_stage_cache", "candidate_examples.pkl")

    def _candidate_stage_cache_search_roots(self, *, output_dir: str) -> list[str]:
        roots = [output_dir]
        source_dir = self.config.candidate_stage_cache_source_dir
        if source_dir is not None:
            normalized_source = os.path.abspath(source_dir)
            normalized_output = os.path.abspath(output_dir)
            if normalized_source != normalized_output:
                roots.append(normalized_source)
        return roots

    def _candidate_stage_cache_config_payload(
        self,
        *,
        include_model: bool,
        model_override: str | None = None,
        normalize_model: bool = True,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "llm_reasoning_effort": self.config.llm_reasoning_effort,
            "llm_temperature": self.config.llm_temperature,
            "llm_top_p": self.config.llm_top_p,
            "score_max_tokens": self.config.score_max_tokens,
            "mine_max_tokens": self.config.mine_max_tokens,
            "rubric_text_validation_version": RUBRIC_TEXT_VALIDATION_VERSION,
        }
        if include_model:
            model_name = self.config.llm_model if model_override is None else model_override
            payload["llm_model"] = (
                canonicalize_cache_model_name(model_name)
                if normalize_model
                else str(model_name)
            )
        return payload

    def _candidate_stage_cache_config_signature(self) -> str:
        payload = self._candidate_stage_cache_config_payload(include_model=False)
        return stable_hash(json.dumps(payload, ensure_ascii=False, sort_keys=True))

    def _candidate_stage_cache_legacy_compatible_signatures(self) -> set[str]:
        signatures: set[str] = set()
        model_names = set(cache_compatible_model_names(self.config.llm_model))
        for model_name in model_names:
            payload = self._candidate_stage_cache_config_payload(
                include_model=True,
                model_override=model_name,
                normalize_model=False,
            )
            signatures.add(
                stable_hash(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            )
        return signatures

    @staticmethod
    def _candidate_stage_cache_content_example_id(example: PreferenceExample) -> str:
        return stable_hash(
            (
                f"{example.prompt_text}\n<chosen>\n{example.chosen_response}\n"
                f"<rejected>\n{example.rejected_response}"
            ),
            prefix="ex_",
        )

    @classmethod
    def _candidate_stage_cache_legacy_example_fingerprint(
        cls,
        example: PreferenceExample,
    ) -> str:
        payload = {
            "example_id": example.example_id,
            "prompt_text": example.prompt_text,
            "prompt_messages": example.prompt_messages,
            "chosen_response": example.chosen_response,
            "rejected_response": example.rejected_response,
            "meta": example.meta,
            "raw_record": example.raw_record,
        }
        return stable_hash(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
        )

    @staticmethod
    def _candidate_stage_cache_example_fingerprint(example: PreferenceExample) -> str:
        payload = {
            "example_id": example.example_id,
            "prompt_text": example.prompt_text,
            "prompt_messages": example.prompt_messages,
            "chosen_response": example.chosen_response,
            "rejected_response": example.rejected_response,
            "self_rubrics": [item.to_dict() for item in example.self_rubrics],
            "reference_rubrics": [item.to_dict() for item in example.reference_rubrics],
            "candidate_responses": example.candidate_responses,
        }
        return stable_hash(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
        )

    @staticmethod
    def _format_candidate_stage_cache_diagnostic(diagnostic: dict[str, Any]) -> str:
        parts = [
            f"path={diagnostic.get('path')}",
            f"reason={diagnostic.get('reason', 'unknown')}",
        ]
        requested_examples = diagnostic.get("requested_examples")
        if requested_examples is not None:
            parts.append(f"requested_examples={requested_examples}")
        cached_examples = diagnostic.get("cached_examples")
        if cached_examples is not None:
            parts.append(f"cached_examples={cached_examples}")
        mismatch_type = diagnostic.get("mismatch_type")
        if mismatch_type is not None:
            parts.append(f"mismatch_type={mismatch_type}")
        example_id = diagnostic.get("example_id")
        if example_id is not None:
            parts.append(f"example_id={example_id}")
        stored_signature = diagnostic.get("stored_config_signature")
        if stored_signature is not None:
            parts.append(f"stored_config_signature={stored_signature}")
        expected_signature = diagnostic.get("expected_config_signature")
        if expected_signature is not None:
            parts.append(f"expected_config_signature={expected_signature}")
        error = diagnostic.get("error")
        if error is not None:
            parts.append(f"error={error}")
        return " ".join(parts)

    def _format_candidate_stage_cache_diagnostics(
        self,
        diagnostics: list[dict[str, Any]],
    ) -> str:
        if not diagnostics:
            return "no_candidate_cache_paths_checked"
        return "; ".join(
            self._format_candidate_stage_cache_diagnostic(diagnostic)
            for diagnostic in diagnostics
        )

    def _load_candidate_stage_cache(
        self,
        *,
        examples: list[PreferenceExample],
        description: str,
        output_dir: str,
    ) -> tuple[dict[str, list[RubricItem]] | None, list[dict[str, Any]]]:
        if not self.config.reuse_candidate_stage_cache:
            return None, []
        diagnostics: list[dict[str, Any]] = []
        for cache_root in self._candidate_stage_cache_search_roots(output_dir=output_dir):
            cache_path = self._candidate_stage_cache_path(
                output_dir=cache_root,
                description=description,
            )
            stage_cache, diagnostic = self._load_candidate_stage_cache_from_path(
                examples=examples,
                description=description,
                cache_path=cache_path,
            )
            diagnostics.append(diagnostic)
            if stage_cache is not None:
                return stage_cache, diagnostics
        return None, diagnostics

    def _load_candidate_stage_cache_from_path(
        self,
        *,
        examples: list[PreferenceExample],
        description: str,
        cache_path: str,
    ) -> tuple[dict[str, list[RubricItem]] | None, dict[str, Any]]:
        diagnostic: dict[str, Any] = {
            "path": cache_path,
            "requested_examples": len(examples),
        }
        if not os.path.isfile(cache_path):
            diagnostic["reason"] = "missing"
            return None, diagnostic
        try:
            with open(cache_path, "rb") as f:
                payload = pickle.load(f)
        except Exception as exc:  # noqa: BLE001
            diagnostic["reason"] = "load_failed"
            diagnostic["error"] = str(exc)
            self._progress(
                f"{description} stage_cache load_failed: path={cache_path} error={exc}"
            )
            return None, diagnostic

        fingerprint_map = payload.get("example_fingerprints")
        candidate_cache = payload.get("candidate_cache")
        if isinstance(fingerprint_map, dict):
            diagnostic["cached_examples"] = len(fingerprint_map)

        expected_signature = self._candidate_stage_cache_config_signature()
        stored_signature = payload.get("config_signature")
        expected_payload = self._candidate_stage_cache_config_payload(include_model=False)
        stored_payload = payload.get("config_payload")
        diagnostic["stored_config_signature"] = stored_signature
        diagnostic["expected_config_signature"] = expected_signature
        if isinstance(stored_payload, dict):
            if stored_payload != expected_payload:
                diagnostic["reason"] = "config_mismatch"
                self._progress(
                    f"{description} stage_cache config_mismatch: "
                    f"path={cache_path} stored={stored_signature} expected={expected_signature} "
                    f"requested_examples={len(examples)} cached_examples={diagnostic.get('cached_examples', 'unknown')}"
                )
                return None, diagnostic
        elif stored_signature not in {
            expected_signature,
            *self._candidate_stage_cache_legacy_compatible_signatures(),
        }:
            diagnostic["reason"] = "config_mismatch"
            self._progress(
                f"{description} stage_cache config_mismatch: "
                f"path={cache_path} stored={stored_signature} expected={expected_signature} "
                f"requested_examples={len(examples)} cached_examples={diagnostic.get('cached_examples', 'unknown')}"
            )
            return None, diagnostic
        elif stored_signature != expected_signature:
            self._progress(
                f"{description} stage_cache legacy_model_signature_accepted: "
                f"path={cache_path} stored={stored_signature} expected={expected_signature}"
            )

        difficulty_map = payload.get("difficulty_analysis", {})
        if not isinstance(fingerprint_map, dict) or not isinstance(candidate_cache, dict):
            diagnostic["reason"] = "invalid_payload"
            self._progress(
                f"{description} stage_cache invalid_payload: path={cache_path}"
            )
            return None, diagnostic

        for example in examples:
            cached_fingerprint = fingerprint_map.get(example.example_id)
            if example.example_id not in candidate_cache:
                diagnostic["reason"] = "example_mismatch"
                diagnostic["mismatch_type"] = "missing_candidate_entry"
                diagnostic["example_id"] = example.example_id
                self._progress(
                    f"{description} stage_cache example_mismatch: "
                    f"path={cache_path} example_id={example.example_id}"
                )
                return None, diagnostic
            fingerprint = self._candidate_stage_cache_example_fingerprint(example)
            if cached_fingerprint == fingerprint:
                continue
            legacy_fingerprint = self._candidate_stage_cache_legacy_example_fingerprint(
                example
            )
            if cached_fingerprint == legacy_fingerprint:
                continue
            if (
                example.example_id
                == self._candidate_stage_cache_content_example_id(example)
            ):
                self._progress(
                    f"{description} stage_cache example_mismatch_ignored: "
                    f"path={cache_path} example_id={example.example_id} "
                    "reason=content_hashed_example_id"
                )
                continue
            diagnostic["reason"] = "example_mismatch"
            diagnostic["mismatch_type"] = "fingerprint_mismatch"
            diagnostic["example_id"] = example.example_id
            self._progress(
                f"{description} stage_cache example_mismatch: "
                f"path={cache_path} example_id={example.example_id}"
            )
            return None, diagnostic

        for example in examples:
            example.difficulty_analysis = difficulty_map.get(example.example_id)
        diagnostic["reason"] = "hit"
        self._progress(
            f"{description} stage_cache hit: path={cache_path} "
            f"examples={len(examples)} cached_examples={len(fingerprint_map)}"
        )
        return (
            {
                example.example_id: candidate_cache[example.example_id]
                for example in examples
            },
            diagnostic,
        )

    def _save_candidate_stage_cache(
        self,
        *,
        examples: list[PreferenceExample],
        description: str,
        output_dir: str,
        candidate_cache: dict[str, list[RubricItem]],
    ) -> None:
        if not self.config.reuse_candidate_stage_cache:
            return
        cache_path = self._candidate_stage_cache_path(
            output_dir=output_dir,
            description=description,
        )
        ensure_dir(os.path.dirname(cache_path))
        payload: dict[str, Any]
        if os.path.isfile(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    payload = pickle.load(f)
            except Exception:
                payload = {}
        else:
            payload = {}

        if payload.get("config_signature") != self._candidate_stage_cache_config_signature():
            payload = {
                "config_signature": self._candidate_stage_cache_config_signature(),
                "config_payload": self._candidate_stage_cache_config_payload(
                    include_model=False
                ),
                "config_payload_with_model": self._candidate_stage_cache_config_payload(
                    include_model=True
                ),
                "example_fingerprints": {},
                "candidate_cache": {},
                "difficulty_analysis": {},
            }
        else:
            payload["config_payload"] = self._candidate_stage_cache_config_payload(
                include_model=False
            )
            payload["config_payload_with_model"] = (
                self._candidate_stage_cache_config_payload(include_model=True)
            )

        fingerprint_map = payload.setdefault("example_fingerprints", {})
        cached_candidate_cache = payload.setdefault("candidate_cache", {})
        difficulty_map = payload.setdefault("difficulty_analysis", {})
        for example in examples:
            fingerprint_map[example.example_id] = self._candidate_stage_cache_example_fingerprint(example)
            cached_candidate_cache[example.example_id] = candidate_cache[example.example_id]
            if example.difficulty_analysis is not None:
                difficulty_map[example.example_id] = example.difficulty_analysis

        with open(cache_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        self._progress(
            f"{description} stage_cache saved: path={cache_path} "
            f"examples={len(examples)} cached_examples={len(payload['example_fingerprints'])}"
        )

    def _build_training_payload(
        self,
        *,
        examples: list[PreferenceExample],
        candidate_cache: dict[str, list[RubricItem]],
        bank: list[BankEntry],
        hard_negative_map: dict[str, HardNegativeResult],
        description: str,
        output_dir: str,
    ) -> dict[str, Any]:
        cache_context = self._prepare_training_payload_stage_cache_context(
            examples=examples,
            candidate_cache=candidate_cache,
            bank=bank,
            hard_negative_map=hard_negative_map,
        )
        stage_cache = self._load_training_payload_stage_cache(
            description=description,
            output_dir=output_dir,
            expected_config_signatures=cache_context["compatible_config_signatures"],
            expected_input_signatures=cache_context["compatible_input_signatures"],
            expected_example_count=len(examples),
            bank_size=len(bank),
        )
        if stage_cache is not None:
            return stage_cache
        assert self.llm_runner is not None
        assert self.scorer is not None
        return asyncio.run(
            self._build_training_payload_async(
                examples=examples,
                candidate_cache=candidate_cache,
                bank=bank,
                hard_negative_map=hard_negative_map,
                description=description,
                output_dir=output_dir,
                cache_context=cache_context,
            )
        )

    async def _build_training_payload_async(
        self,
        *,
        examples: list[PreferenceExample],
        candidate_cache: dict[str, list[RubricItem]],
        bank: list[BankEntry],
        hard_negative_map: dict[str, HardNegativeResult],
        description: str,
        output_dir: str,
        cache_context: dict[str, Any],
    ) -> dict[str, Any]:
        assert self.llm_runner is not None
        bank_size = len(bank)
        rows: list[dict[str, Any] | None] = [None] * len(examples)
        uncached_items: list[tuple[int, PreferenceExample]] = []
        row_cache_hits = 0

        for idx, example in enumerate(examples):
            row = self._load_training_payload_row_cache(
                description=description,
                output_dir=output_dir,
                row_idx=idx,
                example_id=example.example_id,
                expected_input_signatures=cache_context["compatible_row_input_signatures"][idx],
                bank_size=bank_size,
            )
            if row is None:
                uncached_items.append((idx, example))
                continue
            rows[idx] = row
            row_cache_hits += 1

        if row_cache_hits > 0:
            self._progress(
                f"{description} training_payload_row_cache hit: "
                f"{row_cache_hits}/{len(examples)} "
                f"path={self._training_payload_row_cache_dir(output_dir=output_dir, description=description)}"
            )

        async def _process_one(
            item: tuple[int, PreferenceExample],
            _: int,
        ) -> tuple[int, dict[str, Any]]:
            idx, example = item
            try:
                matched_bank_ids = self._match_candidates_to_bank(
                    candidate_cache.get(example.example_id, []),
                    bank,
                )
                candidate_bank_ids = self._select_candidate_bank_ids(
                    example=example,
                    bank=bank,
                    matched_bank_ids=matched_bank_ids,
                )
                clean_z = await self.scorer.ascore_pairwise_features(
                    prompt_messages=example.prompt_messages,
                    prompt_text=example.prompt_text,
                    chosen_response=example.chosen_response,
                    rejected_response=example.rejected_response,
                    bank=bank,
                    bank_ids=candidate_bank_ids,
                    difficulty_analysis=example.difficulty_analysis,
                )

                hard_negative = hard_negative_map.get(example.example_id)
                if hard_negative is not None:
                    adv_z = await self.scorer.ascore_pairwise_features(
                        prompt_messages=example.prompt_messages,
                        prompt_text=example.prompt_text,
                        chosen_response=example.chosen_response,
                        rejected_response=hard_negative.response_text,
                        bank=bank,
                        bank_ids=candidate_bank_ids,
                        difficulty_analysis=example.difficulty_analysis,
                    )
                    adv_flag = 1.0
                else:
                    adv_z = np.zeros(len(bank), dtype=np.float32)
                    adv_flag = 0.0

                target = np.zeros(len(bank), dtype=np.float32)
                positive_ids = []
                for bank_id in matched_bank_ids:
                    positive = clean_z[bank_id] > self.config.support_positive_delta
                    if hard_negative is not None:
                        positive = (
                            positive
                            or adv_z[bank_id] > self.config.support_positive_delta
                        )
                    if positive:
                        target[bank_id] = 1.0
                        positive_ids.append(bank_id)

                if not positive_ids and matched_bank_ids:
                    best_local_id = max(
                        matched_bank_ids,
                        key=lambda item: float(clean_z[item]),
                    )
                    if clean_z[best_local_id] > 0:
                        target[best_local_id] = 1.0
                        positive_ids.append(best_local_id)

                if not positive_ids:
                    best_global_id = int(np.argmax(clean_z))
                    if clean_z[best_global_id] > self.config.support_positive_delta:
                        target[best_global_id] = 1.0
                        positive_ids.append(best_global_id)

                row = {
                    "clean_z": clean_z,
                    "adv_z": adv_z,
                    "adv_mask": adv_flag,
                    "support_target": target,
                    "positive_ids": positive_ids,
                }
                self._save_training_payload_row_cache(
                    description=description,
                    output_dir=output_dir,
                    row_idx=idx,
                    example_id=example.example_id,
                    input_signature=cache_context["row_input_signatures"][idx],
                    row=row,
                )
                return idx, row
            except Exception as exc:  # noqa: BLE001
                error_message = f"{type(exc).__name__}: {exc}"
                self._progress(
                    f"{description} training_payload_row_failed: "
                    f"row_idx={idx} example_id={example.example_id} error={error_message}"
                )
                self.llm_runner.record_error(
                    namespace=description,
                    error_kind="training_payload_row_failure",
                    message=error_message,
                    prompt=example.prompt_text,
                    extra={
                        "row_idx": idx,
                        "example_id": example.example_id,
                    },
                )
                raise

        if uncached_items:
            computed_rows = await self.llm_runner.parallel_map(
                uncached_items,
                _process_one,
                description=description,
                max_concurrency=self.config.llm_max_concurrency,
                progress_log_interval=self.config.llm_progress_log_interval,
            )
            for row_idx, row in computed_rows:
                rows[row_idx] = row

        normalized_rows: list[dict[str, Any]] = []
        for row in rows:
            if row is None:
                raise RuntimeError(
                    f"{description} produced incomplete training payload rows"
                )
            normalized_rows.append(row)

        payload = self._assemble_training_payload_from_rows(
            rows=normalized_rows,
            bank_size=bank_size,
        )
        self._save_training_payload_stage_cache(
            description=description,
            output_dir=output_dir,
            config_signature=cache_context["config_signature"],
            input_signature=cache_context["input_signature"],
            payload=payload,
        )
        return payload

    def _prepare_training_payload_stage_cache_context(
        self,
        *,
        examples: list[PreferenceExample],
        candidate_cache: dict[str, list[RubricItem]],
        bank: list[BankEntry],
        hard_negative_map: dict[str, HardNegativeResult],
    ) -> dict[str, Any]:
        config_signature = self._training_payload_stage_cache_config_signature()
        compatible_config_signatures = self._training_payload_stage_cache_compatible_config_signatures()
        compatible_config_signatures.add(config_signature)
        legacy_config_signatures = sorted(
            signature
            for signature in compatible_config_signatures
            if signature != config_signature
        )
        bank_signature = self._training_payload_bank_signature(bank)
        row_input_signatures: list[str] = []
        compatible_row_input_signatures: list[tuple[str, ...]] = []
        legacy_stage_row_signatures: dict[str, list[str]] = {
            signature: [] for signature in legacy_config_signatures
        }
        for example in examples:
            current_signature = self._training_payload_row_input_signature(
                example=example,
                candidate_rubrics=candidate_cache.get(example.example_id, []),
                hard_negative=hard_negative_map.get(example.example_id),
                bank_signature=bank_signature,
                config_signature=config_signature,
            )
            row_input_signatures.append(current_signature)
            compatible_signatures = [current_signature]
            for legacy_signature in legacy_config_signatures:
                legacy_row_signature = self._training_payload_row_input_signature(
                    example=example,
                    candidate_rubrics=candidate_cache.get(example.example_id, []),
                    hard_negative=hard_negative_map.get(example.example_id),
                    bank_signature=bank_signature,
                    config_signature=legacy_signature,
                )
                compatible_signatures.append(legacy_row_signature)
                legacy_stage_row_signatures[legacy_signature].append(legacy_row_signature)
            compatible_row_input_signatures.append(tuple(compatible_signatures))

        compatible_input_signatures = {
            self._training_payload_stage_input_signature(
                row_input_signatures=row_input_signatures
            )
        }
        for legacy_signature in legacy_config_signatures:
            compatible_input_signatures.add(
                self._training_payload_stage_input_signature(
                    row_input_signatures=legacy_stage_row_signatures[legacy_signature]
                )
            )
        return {
            "config_signature": config_signature,
            "compatible_config_signatures": tuple(
                [config_signature, *legacy_config_signatures]
            ),
            "bank_signature": bank_signature,
            "row_input_signatures": row_input_signatures,
            "compatible_row_input_signatures": compatible_row_input_signatures,
            "input_signature": self._training_payload_stage_input_signature(
                row_input_signatures=row_input_signatures
            ),
            "compatible_input_signatures": tuple(sorted(compatible_input_signatures)),
        }

    def _training_payload_stage_cache_dir(
        self,
        *,
        output_dir: str,
        description: str,
    ) -> str:
        normalized = os.path.abspath(output_dir)
        safe_description = description.replace(os.sep, "__")
        return os.path.join(
            normalized,
            "training_payload_stage_cache",
            safe_description,
        )

    def _training_payload_stage_cache_path(
        self,
        *,
        output_dir: str,
        description: str,
    ) -> str:
        return os.path.join(
            self._training_payload_stage_cache_dir(
                output_dir=output_dir,
                description=description,
            ),
            "payload.pkl",
        )

    def _training_payload_row_cache_dir(
        self,
        *,
        output_dir: str,
        description: str,
    ) -> str:
        return os.path.join(
            self._training_payload_stage_cache_dir(
                output_dir=output_dir,
                description=description,
            ),
            "rows",
        )

    def _training_payload_row_cache_path(
        self,
        *,
        output_dir: str,
        description: str,
        row_idx: int,
        example_id: str,
    ) -> str:
        file_name = f"{row_idx:06d}_{stable_hash(example_id)}.pkl"
        return os.path.join(
            self._training_payload_row_cache_dir(
                output_dir=output_dir,
                description=description,
            ),
            file_name,
        )

    def _training_payload_stage_cache_config_signature(self) -> str:
        payload = self._training_payload_stage_cache_config_payload()
        return stable_hash(json.dumps(payload, ensure_ascii=False, sort_keys=True))

    def _training_payload_stage_cache_config_payload(
        self,
        *,
        model_override: str | None = None,
        normalize_model: bool = True,
    ) -> dict[str, Any]:
        model_name = self.config.llm_model if model_override is None else model_override
        return {
            "cache_schema_version": 1,
            "support_positive_delta": self.config.support_positive_delta,
            "top_k": self.config.top_k,
            "llm_model": (
                canonicalize_cache_model_name(model_name)
                if normalize_model
                else str(model_name)
            ),
            "llm_reasoning_effort": self.config.llm_reasoning_effort,
            "llm_temperature": self.config.llm_temperature,
            "llm_top_p": self.config.llm_top_p,
            "judge_rubric_chunk_size": self.config.judge_rubric_chunk_size,
            "score_max_tokens": self.config.score_max_tokens,
            "rubric_text_validation_version": RUBRIC_TEXT_VALIDATION_VERSION,
        }

    def _training_payload_stage_cache_compatible_config_signatures(self) -> set[str]:
        signatures: set[str] = set()
        for model_name in cache_compatible_model_names(self.config.llm_model):
            payload = self._training_payload_stage_cache_config_payload(
                model_override=model_name,
                normalize_model=False,
            )
            signatures.add(
                stable_hash(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            )
        return signatures

    @staticmethod
    def _training_payload_bank_signature(bank: list[BankEntry]) -> str:
        digest = hashlib.sha256()
        for entry in bank:
            payload = {
                "bank_id": entry.bank_id,
                "text": entry.text,
                "facet": entry.facet,
                "importance": entry.importance,
                "source": entry.source,
                "grounding": entry.grounding,
            }
            digest.update(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                ).encode("utf-8")
            )
            digest.update(b"\0")
        return f"payload_bank_{digest.hexdigest()}"

    def _training_payload_row_input_signature(
        self,
        *,
        example: PreferenceExample,
        candidate_rubrics: list[RubricItem],
        hard_negative: HardNegativeResult | None,
        bank_signature: str,
        config_signature: str,
    ) -> str:
        payload = {
            "config_signature": config_signature,
            "example_fingerprint": self._candidate_stage_cache_example_fingerprint(example),
            "difficulty_analysis": example.difficulty_analysis,
            "candidate_rubrics": [item.to_dict() for item in candidate_rubrics],
            "hard_negative": hard_negative.to_dict() if hard_negative is not None else None,
            "bank_signature": bank_signature,
        }
        return stable_hash(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ),
            prefix="payload_row_",
        )

    @staticmethod
    def _training_payload_stage_input_signature(
        *,
        row_input_signatures: list[str],
    ) -> str:
        digest = hashlib.sha256()
        for item in row_input_signatures:
            digest.update(item.encode("utf-8"))
            digest.update(b"\0")
        return f"payload_stage_{digest.hexdigest()}"

    def _load_training_payload_stage_cache(
        self,
        *,
        description: str,
        output_dir: str,
        expected_config_signatures: tuple[str, ...],
        expected_input_signatures: tuple[str, ...],
        expected_example_count: int,
        bank_size: int,
    ) -> dict[str, Any] | None:
        if not self.config.reuse_training_payload_stage_cache:
            return None
        cache_path = self._training_payload_stage_cache_path(
            output_dir=output_dir,
            description=description,
        )
        if not os.path.isfile(cache_path):
            return None
        try:
            with open(cache_path, "rb") as f:
                payload = pickle.load(f)
        except Exception as exc:  # noqa: BLE001
            self._progress(
                f"{description} training_payload_stage_cache load_failed: "
                f"path={cache_path} error={exc}"
            )
            return None

        if payload.get("config_signature") not in expected_config_signatures:
            self._progress(
                f"{description} training_payload_stage_cache config_mismatch: "
                f"path={cache_path}"
            )
            return None
        if payload.get("input_signature") not in expected_input_signatures:
            self._progress(
                f"{description} training_payload_stage_cache input_mismatch: "
                f"path={cache_path}"
            )
            return None

        normalized_payload = self._normalize_training_payload_payload(
            payload.get("payload"),
            expected_example_count=expected_example_count,
            bank_size=bank_size,
        )
        if normalized_payload is None:
            self._progress(
                f"{description} training_payload_stage_cache invalid_payload: "
                f"path={cache_path}"
            )
            return None
        self._progress(
            f"{description} training_payload_stage_cache hit: "
            f"path={cache_path} examples={expected_example_count}"
        )
        return normalized_payload

    def _save_training_payload_stage_cache(
        self,
        *,
        description: str,
        output_dir: str,
        config_signature: str,
        input_signature: str,
        payload: dict[str, Any],
    ) -> None:
        if not self.config.reuse_training_payload_stage_cache:
            return
        cache_path = self._training_payload_stage_cache_path(
            output_dir=output_dir,
            description=description,
        )
        ensure_dir(os.path.dirname(cache_path))
        with open(cache_path, "wb") as f:
            pickle.dump(
                {
                    "config_signature": config_signature,
                    "input_signature": input_signature,
                    "payload": payload,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        self._progress(
            f"{description} training_payload_stage_cache saved: "
            f"path={cache_path} examples={len(payload['adv_mask'])}"
        )

    def _load_training_payload_row_cache(
        self,
        *,
        description: str,
        output_dir: str,
        row_idx: int,
        example_id: str,
        expected_input_signatures: tuple[str, ...],
        bank_size: int,
    ) -> dict[str, Any] | None:
        if not self.config.reuse_training_payload_stage_cache:
            return None
        cache_path = self._training_payload_row_cache_path(
            output_dir=output_dir,
            description=description,
            row_idx=row_idx,
            example_id=example_id,
        )
        if not os.path.isfile(cache_path):
            return None
        try:
            with open(cache_path, "rb") as f:
                payload = pickle.load(f)
        except Exception:
            return None
        if payload.get("input_signature") not in expected_input_signatures:
            return None
        return self._normalize_training_payload_row(
            payload.get("row"),
            bank_size=bank_size,
        )

    def _save_training_payload_row_cache(
        self,
        *,
        description: str,
        output_dir: str,
        row_idx: int,
        example_id: str,
        input_signature: str,
        row: dict[str, Any],
    ) -> None:
        if not self.config.reuse_training_payload_stage_cache:
            return
        cache_path = self._training_payload_row_cache_path(
            output_dir=output_dir,
            description=description,
            row_idx=row_idx,
            example_id=example_id,
        )
        ensure_dir(os.path.dirname(cache_path))
        with open(cache_path, "wb") as f:
            pickle.dump(
                {
                    "input_signature": input_signature,
                    "row": row,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @staticmethod
    def _normalize_training_payload_row(
        row: Any,
        *,
        bank_size: int,
    ) -> dict[str, Any] | None:
        if not isinstance(row, dict):
            return None
        try:
            clean_z = np.asarray(row["clean_z"], dtype=np.float32)
            adv_z = np.asarray(row["adv_z"], dtype=np.float32)
            support_target = np.asarray(row["support_target"], dtype=np.float32)
            adv_mask = float(row["adv_mask"])
            positive_ids = [int(item) for item in row["positive_ids"]]
        except Exception:
            return None
        if (
            clean_z.shape != (bank_size,)
            or adv_z.shape != (bank_size,)
            or support_target.shape != (bank_size,)
        ):
            return None
        return {
            "clean_z": clean_z,
            "adv_z": adv_z,
            "adv_mask": adv_mask,
            "support_target": support_target,
            "positive_ids": positive_ids,
        }

    def _normalize_training_payload_payload(
        self,
        payload: Any,
        *,
        expected_example_count: int,
        bank_size: int,
    ) -> dict[str, Any] | None:
        if not isinstance(payload, dict):
            return None
        try:
            clean_z = np.asarray(payload["clean_z"], dtype=np.float32)
            adv_z = np.asarray(payload["adv_z"], dtype=np.float32)
            adv_mask = np.asarray(payload["adv_mask"], dtype=np.float32)
            support_targets = np.asarray(payload["support_targets"], dtype=np.float32)
            support_counts = [int(item) for item in payload["support_counts"]]
        except Exception:
            return None
        if clean_z.shape != (expected_example_count, bank_size):
            return None
        if adv_z.shape != (expected_example_count, bank_size):
            return None
        if adv_mask.shape != (expected_example_count,):
            return None
        if support_targets.shape != (expected_example_count, bank_size):
            return None
        if len(support_counts) != bank_size:
            return None
        return {
            "clean_z": clean_z,
            "adv_z": adv_z,
            "adv_mask": adv_mask,
            "support_targets": support_targets,
            "support_counts": support_counts,
        }

    def _assemble_training_payload_from_rows(
        self,
        *,
        rows: list[dict[str, Any]],
        bank_size: int,
    ) -> dict[str, Any]:
        support_counts = np.zeros(bank_size, dtype=np.int32)
        normalized_rows: list[dict[str, Any]] = []
        for row in rows:
            normalized_row = self._normalize_training_payload_row(
                row,
                bank_size=bank_size,
            )
            if normalized_row is None:
                raise ValueError("invalid training payload row")
            normalized_rows.append(normalized_row)
            for bank_id in normalized_row["positive_ids"]:
                support_counts[bank_id] += 1
        return {
            "clean_z": np.stack(
                [row["clean_z"] for row in normalized_rows],
                axis=0,
            ).astype(np.float32),
            "adv_z": np.stack(
                [row["adv_z"] for row in normalized_rows],
                axis=0,
            ).astype(np.float32),
            "adv_mask": np.asarray(
                [row["adv_mask"] for row in normalized_rows],
                dtype=np.float32,
            ),
            "support_targets": np.stack(
                [row["support_target"] for row in normalized_rows],
                axis=0,
            ).astype(np.float32),
            "support_counts": support_counts.tolist(),
        }

    def _select_candidate_bank_ids(
        self,
        *,
        example: PreferenceExample,
        bank: list[BankEntry],
        matched_bank_ids: list[int],
    ) -> list[int]:
        if matched_bank_ids:
            return matched_bank_ids[: max(self.config.top_k * 2, self.config.top_k)]

        prompt_signature = normalize_text_signature(example.prompt_text)
        prompt_tokens = set(prompt_signature.split())
        scored: list[tuple[float, int]] = []
        for entry in bank:
            rubric_tokens = set(normalize_text_signature(entry.text).split())
            if not rubric_tokens:
                continue
            overlap = len(prompt_tokens & rubric_tokens) / max(1, len(rubric_tokens))
            if overlap > 0:
                scored.append((overlap, entry.bank_id))
        scored.sort(reverse=True)
        selected_ids = [bank_id for _, bank_id in scored[: max(self.config.top_k, 4)]]
        if selected_ids:
            return selected_ids
        return list(range(min(len(bank), max(self.config.top_k, 4))))

    def _payload_to_tensors(self, payload: dict[str, Any], *, device: torch.device) -> dict[str, torch.Tensor]:
        return {
            "clean_z": torch.tensor(payload["clean_z"], dtype=torch.float32, device=device),
            "adv_z": torch.tensor(payload["adv_z"], dtype=torch.float32, device=device),
            "adv_mask": torch.tensor(payload["adv_mask"], dtype=torch.float32, device=device),
            "support_targets": torch.tensor(payload["support_targets"], dtype=torch.float32, device=device),
        }

    def _optimize_model(
        self,
        *,
        model: PromptSelectorModel,
        optimizer: torch.optim.Optimizer,
        train_x_tensor: torch.Tensor,
        train_tensors: dict[str, torch.Tensor],
        dev_x_tensor: torch.Tensor,
        dev_tensors: dict[str, torch.Tensor],
        diversity_matrix: torch.Tensor,
        epochs: int,
    ) -> list[dict[str, Any]]:
        best_state = copy.deepcopy(model.state_dict())
        best_dev_score = float("-inf")
        history: list[dict[str, Any]] = []

        for epoch_idx in range(epochs):
            model.train()
            permutation = torch.randperm(train_x_tensor.size(0), device=train_x_tensor.device)
            epoch_losses: list[float] = []

            for batch_start in range(0, train_x_tensor.size(0), self.config.batch_size):
                batch_indices = permutation[batch_start : batch_start + self.config.batch_size]
                xb = train_x_tensor[batch_indices]
                clean_z = train_tensors["clean_z"][batch_indices]
                adv_z = train_tensors["adv_z"][batch_indices]
                adv_mask = train_tensors["adv_mask"][batch_indices]
                yb = train_tensors["support_targets"][batch_indices]

                alpha = model(xb)
                global_weights = model.global_weights().unsqueeze(0)
                clean_margin = ((alpha * global_weights) * clean_z).sum(dim=1)
                clean_loss = torch.nn.functional.softplus(-clean_margin).mean()

                if torch.any(adv_mask > 0):
                    adv_margin = ((alpha * global_weights) * adv_z).sum(dim=1)
                    adv_loss = (
                        torch.nn.functional.softplus(-adv_margin) * adv_mask
                    ).sum() / adv_mask.sum().clamp_min(1.0)
                else:
                    adv_loss = torch.zeros((), dtype=torch.float32, device=xb.device)

                if torch.any(adv_mask > 0):
                    preference_loss = 0.5 * (clean_loss + adv_loss)
                else:
                    preference_loss = clean_loss

                # Penalize spreading mass over highly similar rubrics.
                diversity_loss = (
                    torch.einsum("bi,ij,bj->b", alpha, diversity_matrix, alpha).mean()
                )

                total_loss = preference_loss + diversity_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                epoch_losses.append(float(total_loss.detach().cpu().item()))

            train_metrics = self._evaluate_tensors(
                model=model,
                x_tensor=train_x_tensor,
                tensors=train_tensors,
            )
            dev_metrics = self._evaluate_tensors(
                model=model,
                x_tensor=dev_x_tensor,
                tensors=dev_tensors,
            )
            history.append(
                {
                    "epoch": epoch_idx + 1,
                    "loss": float(np.mean(epoch_losses)) if epoch_losses else None,
                    "train": train_metrics,
                    "dev": dev_metrics,
                }
            )
            current_dev_score = (
                dev_metrics["accuracy"]
                + 0.25 * dev_metrics["adv_accuracy"]
                + 0.05 * dev_metrics["support_recall_at_k"]
            )
            if current_dev_score > best_dev_score:
                best_dev_score = current_dev_score
                best_state = copy.deepcopy(model.state_dict())

        model.load_state_dict(best_state)
        return history

    def _evaluate_tensors(
        self,
        *,
        model: PromptSelectorModel,
        x_tensor: torch.Tensor,
        tensors: dict[str, torch.Tensor],
    ) -> dict[str, Any]:
        model.eval()
        with torch.no_grad():
            alpha = model(x_tensor)
            global_weights = model.global_weights().unsqueeze(0)
            bank_scores = alpha * global_weights

            clean_margin = (bank_scores * tensors["clean_z"]).sum(dim=1)
            clean_predictions = clean_margin > 0
            clean_accuracy = float(clean_predictions.float().mean().cpu().item())

            adv_margin = (bank_scores * tensors["adv_z"]).sum(dim=1)
            adv_mask = tensors["adv_mask"]
            if torch.any(adv_mask > 0):
                adv_predictions = adv_margin > 0
                adv_accuracy = float(
                    (adv_predictions.float() * adv_mask).sum().cpu().item()
                    / adv_mask.sum().cpu().item()
                )
                adv_margin_mean = float(
                    (adv_margin * adv_mask).sum().cpu().item()
                    / adv_mask.sum().cpu().item()
                )
            else:
                adv_accuracy = 0.0
                adv_margin_mean = 0.0

            top_k = min(self.config.top_k, bank_scores.shape[1])
            top_ids = torch.topk(bank_scores, k=top_k, dim=1).indices
            positive_targets = tensors["support_targets"] > 0
            recalls = []
            for row_idx in range(top_ids.shape[0]):
                target_ids = torch.nonzero(positive_targets[row_idx], as_tuple=False).flatten()
                if target_ids.numel() == 0:
                    continue
                hit_count = sum(
                    int(target_id.item() in top_ids[row_idx].tolist())
                    for target_id in target_ids
                )
                recalls.append(hit_count / max(1, target_ids.numel()))

            support_recall = float(np.mean(recalls)) if recalls else 0.0
            activation_rate = float((alpha > 0).float().mean().cpu().item())
            avg_margin = float(clean_margin.mean().cpu().item())
            avg_weight = float(model.global_weights().mean().cpu().item())

        return {
            "accuracy": clean_accuracy,
            "adv_accuracy": adv_accuracy,
            "support_recall_at_k": support_recall,
            "activation_rate": activation_rate,
            "avg_margin": avg_margin,
            "adv_margin": adv_margin_mean,
            "avg_weight": avg_weight,
        }

    def _select_support_pairs(
        self,
        *,
        examples: list[PreferenceExample],
        bank: list[BankEntry],
        model: PromptSelectorModel,
        x_tensor: torch.Tensor,
        clean_z: np.ndarray,
        adv_z: np.ndarray,
        adv_mask: np.ndarray,
    ) -> list[SupportPairRecord]:
        model.eval()
        with torch.no_grad():
            alpha = model(x_tensor)
            global_weights = model.global_weights().unsqueeze(0)
            bank_scores = alpha * global_weights
            clean_margin_tensor = (
                bank_scores * torch.tensor(clean_z, dtype=torch.float32, device=x_tensor.device)
            ).sum(dim=1)
            top_ids = torch.topk(
                bank_scores,
                k=min(self.config.top_k, bank_scores.shape[1]),
                dim=1,
            ).indices
            if np.any(adv_mask > 0):
                adv_margin_tensor = (
                    bank_scores * torch.tensor(adv_z, dtype=torch.float32, device=x_tensor.device)
                ).sum(dim=1)
            else:
                adv_margin_tensor = torch.zeros_like(clean_margin_tensor)

        support_records: list[SupportPairRecord] = []
        for idx, example in enumerate(examples):
            clean_margin = float(clean_margin_tensor[idx].cpu().item())
            has_hard_negative = bool(adv_mask[idx] > 0)
            adv_margin = (
                float(adv_margin_tensor[idx].cpu().item()) if has_hard_negative else None
            )
            is_misclassified = clean_margin <= 0
            score = clean_margin
            if adv_margin is not None:
                score = min(score, adv_margin)
            if (
                is_misclassified
                or score <= self.config.support_pair_margin_threshold
            ):
                support_records.append(
                    SupportPairRecord(
                        example_id=example.example_id,
                        clean_margin=clean_margin,
                        is_misclassified=is_misclassified,
                        has_hard_negative=has_hard_negative,
                        adv_margin=adv_margin,
                        selected_bank_ids=[
                            int(item) for item in top_ids[idx].cpu().tolist()
                        ],
                    )
                )

        if not support_records:
            ranked = sorted(
                [
                    (
                        float(clean_margin_tensor[idx].cpu().item()),
                        idx,
                    )
                    for idx in range(len(examples))
                ],
                key=lambda item: item[0],
            )
            for _, idx in ranked[: min(self.config.min_support_pairs, len(ranked))]:
                support_records.append(
                    SupportPairRecord(
                        example_id=examples[idx].example_id,
                        clean_margin=float(clean_margin_tensor[idx].cpu().item()),
                        is_misclassified=float(clean_margin_tensor[idx].cpu().item()) <= 0,
                        has_hard_negative=False,
                        adv_margin=None,
                        selected_bank_ids=[
                            int(item) for item in top_ids[idx].cpu().tolist()
                        ],
                    )
                )

        support_records.sort(
            key=lambda item: (
                min(item.clean_margin, item.adv_margin)
                if item.adv_margin is not None
                else item.clean_margin
            )
        )
        return support_records[: self.config.max_support_pairs]

    def _mine_hard_negatives(
        self,
        *,
        examples: list[PreferenceExample],
        support_pairs: list[SupportPairRecord],
        bank: list[BankEntry],
        model: PromptSelectorModel,
        x_tensor: torch.Tensor,
    ) -> dict[str, HardNegativeResult]:
        assert self.llm_runner is not None
        assert self.adversarial_probe is not None
        return asyncio.run(
            self._mine_hard_negatives_async(
                examples=examples,
                support_pairs=support_pairs,
                bank=bank,
                model=model,
                x_tensor=x_tensor,
            )
        )

    async def _mine_hard_negatives_async(
        self,
        *,
        examples: list[PreferenceExample],
        support_pairs: list[SupportPairRecord],
        bank: list[BankEntry],
        model: PromptSelectorModel,
        x_tensor: torch.Tensor,
    ) -> dict[str, HardNegativeResult]:
        assert self.llm_runner is not None
        support_by_id = {record.example_id: record for record in support_pairs}

        model.eval()
        with torch.no_grad():
            bank_scores = model.score_bank(x_tensor)

        support_examples = [
            (row_idx, example)
            for row_idx, example in enumerate(examples)
            if example.example_id in support_by_id
        ]

        async def _mine_one(
            item: tuple[int, PreferenceExample],
            _: int,
        ) -> tuple[str, HardNegativeResult | None]:
            row_idx, example = item
            selected_rubrics = self._select_rubrics_from_row_scores(
                bank=bank,
                bank_scores=bank_scores[row_idx].cpu().tolist(),
            )
            result = await self.adversarial_probe.amine_hard_negative(
                example=example,
                selected_rubrics=selected_rubrics,
            )
            return example.example_id, result

        results = await self.llm_runner.parallel_map(
            support_examples,
            _mine_one,
            description="mine_hard_negatives",
            max_concurrency=self.config.llm_max_concurrency,
            progress_log_interval=self.config.llm_progress_log_interval,
        )
        return {
            example_id: result
            for example_id, result in results
            if result is not None
        }

    def _expand_candidate_cache(
        self,
        *,
        train_examples: list[PreferenceExample],
        support_pairs: list[SupportPairRecord],
        hard_negative_map: dict[str, HardNegativeResult],
        candidate_cache: dict[str, list[RubricItem]],
    ) -> int:
        assert self.llm_runner is not None
        assert self.miner is not None
        return asyncio.run(
            self._expand_candidate_cache_async(
                train_examples=train_examples,
                support_pairs=support_pairs,
                hard_negative_map=hard_negative_map,
                candidate_cache=candidate_cache,
            )
        )

    async def _expand_candidate_cache_async(
        self,
        *,
        train_examples: list[PreferenceExample],
        support_pairs: list[SupportPairRecord],
        hard_negative_map: dict[str, HardNegativeResult],
        candidate_cache: dict[str, list[RubricItem]],
    ) -> int:
        assert self.llm_runner is not None
        example_by_id = {example.example_id: example for example in train_examples}

        async def _expand_one(
            record: SupportPairRecord,
            _: int,
        ) -> tuple[str, list[RubricItem], list[RubricItem]]:
            example = example_by_id[record.example_id]
            pair_candidates = await self.miner.amine_from_pair(
                example=example,
                positive_response=example.chosen_response,
                negative_response=example.rejected_response,
                source="support_pair_expansion",
            )
            hard_negative = hard_negative_map.get(record.example_id)
            if hard_negative is None:
                return record.example_id, pair_candidates, []
            adv_candidates = await self.miner.amine_from_pair(
                example=example,
                positive_response=example.chosen_response,
                negative_response=hard_negative.response_text,
                source=f"adversarial_expansion::{hard_negative.source}",
            )
            return record.example_id, pair_candidates, adv_candidates

        results = await self.llm_runner.parallel_map(
            support_pairs,
            _expand_one,
            description="expand_candidate_cache",
            max_concurrency=self.config.llm_max_concurrency,
            progress_log_interval=self.config.llm_progress_log_interval,
        )
        added = 0
        for example_id, pair_candidates, adv_candidates in results:
            added += self._merge_candidates(candidate_cache, example_id, pair_candidates)
            added += self._merge_candidates(candidate_cache, example_id, adv_candidates)
        return added

    def _merge_candidates(
        self,
        candidate_cache: dict[str, list[RubricItem]],
        example_id: str,
        new_candidates: list[RubricItem],
    ) -> int:
        if not new_candidates:
            return 0
        existing = candidate_cache.setdefault(example_id, [])
        existing_signatures = {
            normalize_text_signature(item.text) for item in existing
        }
        added = 0
        for item in new_candidates:
            signature = normalize_text_signature(item.text)
            if not signature or signature in existing_signatures:
                continue
            existing.append(item)
            existing_signatures.add(signature)
            added += 1
        return added

    def _build_bank_from_candidate_cache(
        self,
        candidate_cache: dict[str, list[RubricItem]],
        *,
        excluded_signatures: set[str],
        description: str,
        output_dir: str,
    ) -> list[BankEntry]:
        self._progress(
            f"{description} prep: "
            f"example_groups={len(candidate_cache)} "
            f"excluded_signatures={len(excluded_signatures)}"
        )
        stage_cache = self._load_bank_stage_cache(
            candidate_cache=candidate_cache,
            excluded_signatures=excluded_signatures,
            description=description,
            output_dir=output_dir,
        )
        if stage_cache is not None:
            return stage_cache
        filtered_groups = []
        for rubrics in candidate_cache.values():
            filtered_groups.append(
                [
                    item
                    for item in rubrics
                    if normalize_text_signature(item.text) not in excluded_signatures
                ]
            )
        bank = self.bank_builder.build(
            filtered_groups,
            description=description,
            progress_callback=self._progress,
            progress_log_interval=self.config.bank_progress_log_interval,
        )
        self._save_bank_stage_cache(
            candidate_cache=candidate_cache,
            excluded_signatures=excluded_signatures,
            description=description,
            output_dir=output_dir,
            bank=bank,
        )
        return bank

    def _bank_stage_cache_path(self, *, output_dir: str, description: str) -> str:
        normalized = os.path.abspath(output_dir)
        safe_description = description.replace(os.sep, "__")
        return os.path.join(
            normalized,
            "bank_stage_cache",
            f"{safe_description}.pkl",
        )

    def _bank_stage_cache_config_signature(self) -> str:
        payload = {
            "cache_schema_version": 1,
            "bank_similarity_threshold": self.config.bank_similarity_threshold,
            "min_bank_observed_count": self.config.min_bank_observed_count,
            "max_bank_size": self.config.max_bank_size,
            "rubric_text_validation_version": RUBRIC_TEXT_VALIDATION_VERSION,
        }
        return stable_hash(json.dumps(payload, ensure_ascii=False, sort_keys=True))

    @staticmethod
    def _bank_stage_cache_input_signature(
        *,
        candidate_cache: dict[str, list[RubricItem]],
        excluded_signatures: set[str],
    ) -> str:
        digest = hashlib.sha256()
        for example_id, rubrics in candidate_cache.items():
            digest.update(example_id.encode("utf-8"))
            digest.update(b"\0")
            for rubric in rubrics:
                payload = json.dumps(
                    rubric.to_dict(),
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
                digest.update(payload.encode("utf-8"))
                digest.update(b"\0")
            digest.update(b"\1")
        digest.update(b"<excluded>\0")
        for signature in sorted(excluded_signatures):
            digest.update(signature.encode("utf-8"))
            digest.update(b"\0")
        return f"bank_{digest.hexdigest()}"

    def _load_bank_stage_cache(
        self,
        *,
        candidate_cache: dict[str, list[RubricItem]],
        excluded_signatures: set[str],
        description: str,
        output_dir: str,
    ) -> list[BankEntry] | None:
        if not self.config.reuse_bank_stage_cache:
            return None
        cache_path = self._bank_stage_cache_path(
            output_dir=output_dir,
            description=description,
        )
        if not os.path.isfile(cache_path):
            return None
        try:
            with open(cache_path, "rb") as f:
                payload = pickle.load(f)
        except Exception as exc:  # noqa: BLE001
            self._progress(
                f"{description} bank_stage_cache load_failed: path={cache_path} error={exc}"
            )
            return None

        if payload.get("config_signature") != self._bank_stage_cache_config_signature():
            self._progress(
                f"{description} bank_stage_cache config_mismatch: path={cache_path}"
            )
            return None

        expected_input_signature = self._bank_stage_cache_input_signature(
            candidate_cache=candidate_cache,
            excluded_signatures=excluded_signatures,
        )
        if payload.get("input_signature") != expected_input_signature:
            self._progress(
                f"{description} bank_stage_cache input_mismatch: path={cache_path}"
            )
            return None

        bank_payload = payload.get("bank")
        if not isinstance(bank_payload, list):
            self._progress(
                f"{description} bank_stage_cache invalid_payload: path={cache_path}"
            )
            return None
        try:
            bank = [BankEntry(**item) for item in bank_payload]
        except Exception as exc:  # noqa: BLE001
            self._progress(
                f"{description} bank_stage_cache invalid_bank: path={cache_path} error={exc}"
            )
            return None
        self._progress(
            f"{description} bank_stage_cache hit: path={cache_path} bank_size={len(bank)}"
        )
        return bank

    def _save_bank_stage_cache(
        self,
        *,
        candidate_cache: dict[str, list[RubricItem]],
        excluded_signatures: set[str],
        description: str,
        output_dir: str,
        bank: list[BankEntry],
    ) -> None:
        if not self.config.reuse_bank_stage_cache:
            return
        cache_path = self._bank_stage_cache_path(
            output_dir=output_dir,
            description=description,
        )
        ensure_dir(os.path.dirname(cache_path))
        payload = {
            "config_signature": self._bank_stage_cache_config_signature(),
            "input_signature": self._bank_stage_cache_input_signature(
                candidate_cache=candidate_cache,
                excluded_signatures=excluded_signatures,
            ),
            "bank": [entry.to_dict() for entry in bank],
        }
        with open(cache_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        self._progress(
            f"{description} bank_stage_cache saved: path={cache_path} bank_size={len(bank)}"
        )

    def _match_candidates_to_bank(
        self,
        candidates: list[RubricItem],
        bank: list[BankEntry],
    ) -> list[int]:
        matched_ids: list[int] = []
        for candidate in candidates:
            match = self.bank_builder.match(candidate, bank)
            if match is None:
                continue
            bank_id, _ = match
            if bank_id not in matched_ids:
                matched_ids.append(bank_id)
        return matched_ids

    def _compute_bank_usage(
        self,
        *,
        model: PromptSelectorModel,
        x_tensor: torch.Tensor,
    ) -> tuple[list[float], list[int], list[float]]:
        model.eval()
        with torch.no_grad():
            alpha = model(x_tensor)
            activation_counts = (alpha > 0).sum(dim=0).cpu().tolist()
            activation_rates = (alpha > 0).float().mean(dim=0).cpu().tolist()
            global_weights = model.global_weights().cpu().tolist()
        return (
            [float(item) for item in activation_rates],
            [int(item) for item in activation_counts],
            [float(item) for item in global_weights],
        )

    def _build_bank_diversity_matrix(
        self,
        *,
        bank: list[BankEntry],
        device: torch.device,
    ) -> torch.Tensor:
        bank_size = len(bank)
        matrix = torch.zeros((bank_size, bank_size), dtype=torch.float32, device=device)
        threshold = float(self.config.prune_redundancy_threshold)
        if bank_size <= 1:
            return matrix
        for left_idx in range(bank_size):
            left_text = bank[left_idx].text
            for right_idx in range(left_idx + 1, bank_size):
                score = rubric_similarity(left_text, bank[right_idx].text)
                if score < threshold:
                    continue
                normalized = (float(score) - threshold) / max(1e-6, 1.0 - threshold)
                matrix[left_idx, right_idx] = normalized
                matrix[right_idx, left_idx] = normalized
        return matrix

    @staticmethod
    def _apply_bank_statistics(
        *,
        bank: list[BankEntry],
        support_counts: list[int],
        activation_counts: list[int],
    ) -> None:
        for idx, entry in enumerate(bank):
            entry.support_count = int(support_counts[idx]) if idx < len(support_counts) else 0
            entry.activation_count = (
                int(activation_counts[idx]) if idx < len(activation_counts) else 0
            )

    def _select_rubrics_from_row_scores(
        self,
        *,
        bank: list[BankEntry],
        bank_scores: list[float],
    ) -> list[dict[str, Any]]:
        ranked = sorted(
            zip(bank, bank_scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )[: self.config.top_k]
        selected = []
        for entry, score in ranked:
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
        return selected

    def _publish_best_round(self, *, best_round_dir: str, output_dir: str) -> None:
        for name in ("model.pt", "bank.json", "round_summary.json"):
            src = os.path.join(best_round_dir, name)
            dst = os.path.join(output_dir, name)
            shutil.copy2(src, dst)
        dump_json(
            os.path.join(output_dir, "best_round.json"),
            {
                "best_round_dir": best_round_dir,
                "best_round_name": os.path.basename(best_round_dir),
            },
        )

    @staticmethod
    def _set_seed(seed: int) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    @staticmethod
    def _progress(message: str) -> None:
        print(f"[SVR] {message}", flush=True)
