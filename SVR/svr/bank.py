from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from typing import Callable, Iterable

from svr.rubric_quality import (
    normalize_candidate_rubric_text,
    rubric_text_quality_issue,
)
from svr.schema import BankEntry, RubricItem
from svr.utils import (
    jaccard_similarity,
    normalize_text_signature,
    token_set,
)


@dataclass
class RubricBankBuilderConfig:
    similarity_threshold: float = 0.88
    min_observed_count: int = 1
    max_bank_size: int | None = None
    enforce_self_contained_text: bool = True


def rubric_similarity(left: str, right: str) -> float:
    token_sim = jaccard_similarity(left, right)
    char_sim = SequenceMatcher(
        None,
        normalize_text_signature(left),
        normalize_text_signature(right),
    ).ratio()
    return max(token_sim, char_sim)


@dataclass(frozen=True)
class _TextFeatures:
    signature: str
    tokens: frozenset[str]
    prefix: str
    length: int


@dataclass
class _BankIndex:
    entry_features: list[_TextFeatures]
    signature_to_id: dict[str, int]
    facet_to_ids: dict[str, set[int]]
    token_to_ids: dict[str, set[int]]
    prefix_to_ids: dict[str, set[int]]


class RubricBankBuilder:
    def __init__(self, config: RubricBankBuilderConfig | None = None):
        self.config = config or RubricBankBuilderConfig()
        self._bank_index_cache: dict[int, tuple[tuple[str, ...], _BankIndex]] = {}

    def build(
        self,
        rubric_groups: Iterable[Iterable[RubricItem]],
        *,
        description: str = "build_bank",
        progress_callback: Callable[[str], None] | None = None,
        progress_log_interval: int = 256,
    ) -> list[BankEntry]:
        exact_entries, raw_count, dropped_invalid = self._collapse_exact_duplicates(
            rubric_groups
        )
        exact_entries.sort(key=self._entry_sort_key)
        total_exact = len(exact_entries)
        if progress_callback is not None:
            progress_callback(
                f"{description} start: "
                f"raw_candidates={raw_count} "
                f"dropped_invalid={dropped_invalid} "
                f"deduped_candidates={total_exact}"
            )
        bank: list[BankEntry] = []
        index = _BankIndex(
            entry_features=[],
            signature_to_id={},
            facet_to_ids=defaultdict(set),
            token_to_ids=defaultdict(set),
            prefix_to_ids=defaultdict(set),
        )
        interval = max(1, int(progress_log_interval))
        for entry_idx, entry in enumerate(exact_entries, start=1):
            self._merge_entry(bank, index, entry)
            if progress_callback is not None and (
                entry_idx == 1
                or entry_idx == total_exact
                or entry_idx % interval == 0
            ):
                progress_callback(
                    f"{description} progress: "
                    f"{entry_idx}/{total_exact} bank_size={len(bank)}"
                )

        filtered = [
            entry
            for entry in bank
            if entry.observed_count >= self.config.min_observed_count
        ]
        filtered.sort(
            key=lambda item: (
                -item.observed_count,
                {"critical": 0, "major": 1, "minor": 2}.get(item.importance, 3),
                item.facet,
                item.text,
            )
        )
        if self.config.max_bank_size is not None:
            filtered = filtered[: self.config.max_bank_size]

        for bank_id, entry in enumerate(filtered):
            entry.bank_id = bank_id
        self._remember_bank_index(filtered)
        if progress_callback is not None:
            progress_callback(
                f"{description} done: "
                f"raw_candidates={raw_count} "
                f"dropped_invalid={dropped_invalid} "
                f"deduped_candidates={total_exact} "
                f"bank_size={len(filtered)}"
            )
        return filtered

    def match(self, rubric: RubricItem, bank: list[BankEntry]) -> tuple[int, float] | None:
        if not bank:
            return None
        features = self._text_features(rubric.text)
        index = self._get_bank_index(bank)
        best_idx, best_score = self._find_best_match(
            features=features,
            facet=rubric.facet,
            bank=bank,
            index=index,
        )
        if best_idx is None or best_score < self.config.similarity_threshold:
            return None
        return best_idx, min(best_score, 1.0)

    def _collapse_exact_duplicates(
        self,
        rubric_groups: Iterable[Iterable[RubricItem]],
    ) -> tuple[list[BankEntry], int, int]:
        collapsed: list[BankEntry] = []
        signature_to_entry: dict[str, BankEntry] = {}
        raw_count = 0
        dropped_invalid = 0
        for rubric_group in rubric_groups:
            for rubric in rubric_group:
                raw_count += 1
                normalized_text = normalize_candidate_rubric_text(rubric.text)
                if self.config.enforce_self_contained_text:
                    quality_issue = rubric_text_quality_issue(normalized_text)
                    if quality_issue is not None:
                        dropped_invalid += 1
                        continue
                entry = self._entry_from_rubric(
                    replace(
                        rubric,
                        text=normalized_text,
                    )
                )
                signature = self._text_features(entry.text).signature
                if signature and signature in signature_to_entry:
                    self._merge_entry_payload(signature_to_entry[signature], entry)
                    continue
                collapsed.append(entry)
                if signature:
                    signature_to_entry[signature] = entry
        return collapsed, raw_count, dropped_invalid

    @staticmethod
    def _entry_from_rubric(rubric: RubricItem) -> BankEntry:
        return BankEntry(
            bank_id=-1,
            text=rubric.text,
            facet=rubric.facet,
            importance=rubric.importance,
            source=rubric.source,
            grounding=rubric.grounding,
            aliases=[],
            observed_count=1,
            metadata=dict(rubric.metadata),
        )

    @staticmethod
    def _text_features(text: str) -> _TextFeatures:
        signature = normalize_text_signature(text)
        return _TextFeatures(
            signature=signature,
            tokens=frozenset(token_set(signature)),
            prefix=signature[:32],
            length=len(signature),
        )

    def _merge_entry(
        self,
        bank: list[BankEntry],
        index: _BankIndex,
        incoming: BankEntry,
    ) -> None:
        features = self._text_features(incoming.text)
        best_idx, best_score = self._find_best_match(
            features=features,
            facet=incoming.facet,
            bank=bank,
            index=index,
        )
        if best_idx is None or best_score < self.config.similarity_threshold:
            incoming.bank_id = len(bank)
            bank.append(incoming)
            self._add_to_index(index, incoming)
            return

        target = bank[best_idx]
        old_features = index.entry_features[best_idx]
        self._merge_entry_payload(target, incoming)
        self._refresh_index_entry(
            index=index,
            bank_id=best_idx,
            entry=target,
            old_features=old_features,
        )

    def _find_best_match(
        self,
        *,
        features: _TextFeatures,
        facet: str,
        bank: list[BankEntry],
        index: _BankIndex,
    ) -> tuple[int | None, float]:
        if features.signature:
            exact_match = index.signature_to_id.get(features.signature)
            if exact_match is not None:
                return exact_match, 1.0

        candidate_ids = self._candidate_ids(
            features=features,
            facet=facet,
            bank=bank,
            index=index,
        )

        best_idx = None
        best_score = -1.0
        for candidate_id in candidate_ids:
            entry = bank[candidate_id]
            score = self._candidate_similarity(
                left=features,
                right=index.entry_features[candidate_id],
                same_facet=(facet == entry.facet),
            )
            if score > best_score:
                best_idx = candidate_id
                best_score = score
        return best_idx, best_score

    def _candidate_ids(
        self,
        *,
        features: _TextFeatures,
        facet: str,
        bank: list[BankEntry],
        index: _BankIndex,
    ) -> set[int]:
        if not bank:
            return set()

        posting_limit = self._posting_limit(len(bank))
        token_postings: list[tuple[int, str, set[int]]] = []
        for token in features.tokens:
            token_ids = index.token_to_ids.get(token)
            if token_ids:
                token_postings.append((len(token_ids), token, token_ids))
        token_postings.sort(key=lambda item: (item[0], -len(item[1]), item[1]))

        candidate_ids: set[int] = set()
        small_postings = [
            item for item in token_postings if item[0] <= posting_limit
        ]
        for _, _, token_ids in small_postings[: self._max_token_postings()]:
            candidate_ids.update(token_ids)

        if candidate_ids:
            return candidate_ids

        if token_postings:
            return set(token_postings[0][2])

        if features.prefix:
            prefix_ids = index.prefix_to_ids.get(features.prefix, set())
            if prefix_ids and len(prefix_ids) <= posting_limit:
                return set(prefix_ids)

        facet_ids = index.facet_to_ids.get(facet, set())
        if facet_ids and len(facet_ids) <= posting_limit:
            return set(facet_ids)
        if len(bank) <= 128:
            return set(range(len(bank)))
        return set()

    def _candidate_similarity(
        self,
        *,
        left: _TextFeatures,
        right: _TextFeatures,
        same_facet: bool,
    ) -> float:
        facet_bonus = 0.05 if same_facet else 0.0
        token_score = self._token_similarity(left.tokens, right.tokens)
        if token_score + facet_bonus >= self.config.similarity_threshold:
            return token_score + facet_bonus

        char_upper = self._signature_similarity_upper_bound(left.length, right.length)
        if char_upper + facet_bonus < self.config.similarity_threshold:
            return token_score + facet_bonus
        if not left.signature or not right.signature:
            return token_score + facet_bonus

        char_score = SequenceMatcher(None, left.signature, right.signature).ratio()
        return max(token_score, char_score) + facet_bonus

    @staticmethod
    def _token_similarity(left: frozenset[str], right: frozenset[str]) -> float:
        if not left or not right:
            return 0.0
        union = left | right
        if not union:
            return 0.0
        return len(left & right) / len(union)

    @staticmethod
    def _signature_similarity_upper_bound(left_len: int, right_len: int) -> float:
        if left_len <= 0 or right_len <= 0:
            return 0.0
        return (2.0 * min(left_len, right_len)) / (left_len + right_len)

    @staticmethod
    def _posting_limit(bank_size: int) -> int:
        if bank_size <= 256:
            return bank_size
        return min(2048, max(256, bank_size // 16))

    @staticmethod
    def _max_token_postings() -> int:
        return 3

    @staticmethod
    def _entry_sort_key(entry: BankEntry) -> tuple[int, int, str, str]:
        return (
            -entry.observed_count,
            {"critical": 0, "major": 1, "minor": 2}.get(entry.importance, 3),
            entry.facet,
            entry.text,
        )

    def _add_to_index(self, index: _BankIndex, entry: BankEntry) -> None:
        features = self._text_features(entry.text)
        bank_id = entry.bank_id
        if bank_id != len(index.entry_features):
            raise ValueError("bank index is out of sync with bank entries")
        index.entry_features.append(features)
        if features.signature and features.signature not in index.signature_to_id:
            index.signature_to_id[features.signature] = bank_id
        index.facet_to_ids[entry.facet].add(bank_id)
        for token in features.tokens:
            index.token_to_ids[token].add(bank_id)
        if features.prefix:
            index.prefix_to_ids[features.prefix].add(bank_id)

    def _refresh_index_entry(
        self,
        *,
        index: _BankIndex,
        bank_id: int,
        entry: BankEntry,
        old_features: _TextFeatures,
    ) -> None:
        new_features = self._text_features(entry.text)
        if new_features == old_features:
            return
        if (
            old_features.signature
            and index.signature_to_id.get(old_features.signature) == bank_id
        ):
            del index.signature_to_id[old_features.signature]
        for token in old_features.tokens:
            token_ids = index.token_to_ids.get(token)
            if token_ids is None:
                continue
            token_ids.discard(bank_id)
            if not token_ids:
                del index.token_to_ids[token]
        if old_features.prefix:
            prefix_ids = index.prefix_to_ids.get(old_features.prefix)
            if prefix_ids is not None:
                prefix_ids.discard(bank_id)
                if not prefix_ids:
                    del index.prefix_to_ids[old_features.prefix]

        index.entry_features[bank_id] = new_features
        if new_features.signature and new_features.signature not in index.signature_to_id:
            index.signature_to_id[new_features.signature] = bank_id
        for token in new_features.tokens:
            index.token_to_ids[token].add(bank_id)
        if new_features.prefix:
            index.prefix_to_ids[new_features.prefix].add(bank_id)

    @staticmethod
    def _merge_entry_payload(target: BankEntry, incoming: BankEntry) -> None:
        target.observed_count += incoming.observed_count
        if incoming.text != target.text and incoming.text not in target.aliases:
            target.aliases.append(incoming.text)
        for alias in incoming.aliases:
            if alias != target.text and alias not in target.aliases:
                target.aliases.append(alias)
        if (
            len(incoming.text) > len(target.text)
            and incoming.importance == target.importance
        ):
            target.text = incoming.text
        if not target.grounding and incoming.grounding:
            target.grounding = incoming.grounding
        if incoming.importance == "critical":
            target.importance = "critical"
        elif incoming.importance == "major" and target.importance == "minor":
            target.importance = "major"

    def _remember_bank_index(self, bank: list[BankEntry]) -> None:
        index = self._build_bank_index(bank)
        self._bank_index_cache[id(bank)] = (self._bank_fingerprint(bank), index)

    def _get_bank_index(self, bank: list[BankEntry]) -> _BankIndex:
        fingerprint = self._bank_fingerprint(bank)
        cached = self._bank_index_cache.get(id(bank))
        if cached is not None and cached[0] == fingerprint:
            return cached[1]
        index = self._build_bank_index(bank)
        self._bank_index_cache[id(bank)] = (fingerprint, index)
        return index

    @staticmethod
    def _bank_fingerprint(bank: list[BankEntry]) -> tuple[str, ...]:
        return tuple(entry.text for entry in bank)

    def _build_bank_index(self, bank: list[BankEntry]) -> _BankIndex:
        index = _BankIndex(
            entry_features=[],
            signature_to_id={},
            facet_to_ids=defaultdict(set),
            token_to_ids=defaultdict(set),
            prefix_to_ids=defaultdict(set),
        )
        for idx, entry in enumerate(bank):
            entry.bank_id = idx
            self._add_to_index(index, entry)
        return index
