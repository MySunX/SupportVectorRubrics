from __future__ import annotations

from dataclasses import dataclass

from svr.bank import rubric_similarity
from svr.schema import BankEntry


@dataclass
class RubricBankPrunerConfig:
    min_weight: float = 0.08
    min_activation_rate: float = 0.01
    redundancy_threshold: float = 0.92
    min_keep: int = 8


class RubricBankPruner:
    def __init__(self, config: RubricBankPrunerConfig | None = None):
        self.config = config or RubricBankPrunerConfig()

    def prune(
        self,
        bank: list[BankEntry],
        *,
        global_weights: list[float],
        activation_rates: list[float],
    ) -> tuple[list[BankEntry], dict]:
        if not bank:
            return [], {"removed_bank_ids": [], "kept_bank_ids": []}

        scored_items = []
        for entry, weight, activation in zip(bank, global_weights, activation_rates):
            entry.support_count = int(entry.support_count)
            entry.activation_count = int(entry.activation_count)
            score = (
                4.0 * float(weight)
                + 2.0 * float(activation)
                + 0.2 * float(entry.support_count)
                + 0.1 * float(entry.observed_count)
            )
            scored_items.append((entry, float(weight), float(activation), score))

        mandatory = [
            item
            for item in scored_items
            if item[1] >= self.config.min_weight
            or item[2] >= self.config.min_activation_rate
            or item[0].support_count > 0
        ]
        if len(mandatory) < min(self.config.min_keep, len(scored_items)):
            scored_items.sort(key=lambda item: item[3], reverse=True)
            mandatory_ids = {item[0].bank_id for item in mandatory}
            for item in scored_items:
                if item[0].bank_id in mandatory_ids:
                    continue
                mandatory.append(item)
                mandatory_ids.add(item[0].bank_id)
                if len(mandatory) >= min(self.config.min_keep, len(scored_items)):
                    break

        mandatory.sort(key=lambda item: item[3], reverse=True)
        kept: list[BankEntry] = []
        removed_bank_ids: list[int] = []
        for entry, weight, activation, score in mandatory:
            is_redundant = False
            for kept_entry in kept:
                if rubric_similarity(entry.text, kept_entry.text) >= self.config.redundancy_threshold:
                    is_redundant = True
                    break
            if is_redundant:
                removed_bank_ids.append(entry.bank_id)
                continue
            kept.append(entry)

        kept.sort(key=lambda entry: entry.bank_id)
        kept_bank_ids = [entry.bank_id for entry in kept]
        for new_bank_id, entry in enumerate(kept):
            entry.bank_id = new_bank_id

        return kept, {
            "kept_bank_ids": kept_bank_ids,
            "removed_bank_ids": removed_bank_ids,
            "num_before": len(bank),
            "num_after": len(kept),
        }
