from __future__ import annotations

import re

from svr.utils import normalize_whitespace

_FIELD_LABELS = {
    "anchor",
    "anchor_id",
    "facet",
    "grounding",
    "importance",
    "point_id",
    "rubric",
    "source",
}

_GENERIC_SUBJECTS = {
    "the response",
    "the answer",
    "the reply",
    "the code",
    "the solution",
    "the output",
    "response",
    "answer",
    "reply",
    "code",
    "solution",
    "output",
}

_DANGLING_SUFFIXES = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "to",
    "of",
    "for",
    "with",
    "in",
    "on",
    "at",
    "by",
    "from",
    "as",
    "that",
    "which",
    "whose",
    "than",
    "into",
    "onto",
    "about",
    "around",
    "through",
    "over",
    "under",
    "between",
    "within",
    "without",
}

_PREDICATE_HINTS = {
    "acknowledge",
    "acknowledges",
    "answer",
    "answers",
    "avoid",
    "avoids",
    "be",
    "cite",
    "cites",
    "clarify",
    "clarifies",
    "compile",
    "compiles",
    "contain",
    "contains",
    "cover",
    "covers",
    "describe",
    "describes",
    "exceed",
    "exceeds",
    "explain",
    "explains",
    "follow",
    "follows",
    "give",
    "gives",
    "include",
    "includes",
    "keep",
    "keeps",
    "match",
    "matches",
    "mention",
    "mentions",
    "provide",
    "provides",
    "refuse",
    "refuses",
    "remain",
    "remains",
    "respond",
    "responds",
    "return",
    "returns",
    "run",
    "runs",
    "show",
    "shows",
    "solve",
    "solves",
    "start",
    "starts",
    "state",
    "states",
    "stay",
    "stays",
    "summarize",
    "summarizes",
    "support",
    "supports",
    "translate",
    "translates",
    "use",
    "uses",
    "verify",
    "verifies",
    "write",
    "writes",
}

_AUXILIARY_QUESTION_PATTERN = re.compile(
    r"^(?:does|do|is|are|should|must|can|could|would|will|has|have|had)\b",
    flags=re.IGNORECASE,
)
_BARE_QUESTION_STEM_PATTERN = re.compile(
    r"^(?:does|do|is|are|should|must|can|could|would|will|has|have|had)\s+"
    r"(?:(?:the\s+)?(?:response|answer|reply|code|solution|output)|the|this|that|all)?"
    r"\s*\?$",
    flags=re.IGNORECASE,
)
_GENERIC_SUBJECT_PREFIX_PATTERN = re.compile(
    r"^(?:the\s+)?(?:response|answer|reply|code|solution|output)\b",
    flags=re.IGNORECASE,
)
_WORD_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?", flags=re.IGNORECASE)


def normalize_candidate_rubric_text(text: str) -> str:
    normalized = normalize_whitespace(str(text or ""))
    if not normalized:
        return ""
    return re.sub(r"\s+([?!.,;:])", r"\1", normalized)


def rubric_text_quality_issue(text: str) -> str | None:
    normalized = normalize_candidate_rubric_text(text)
    if not normalized:
        return "empty_text"

    lowered = normalized.lower()
    if lowered in _FIELD_LABELS:
        return "field_label_only"
    if lowered in _GENERIC_SUBJECTS:
        return "generic_subject_only"
    if _BARE_QUESTION_STEM_PATTERN.match(normalized):
        return "incomplete_question_stem"

    word_tokens = [token.lower() for token in _WORD_PATTERN.findall(lowered)]
    if not word_tokens:
        return "missing_word_tokens"
    if len(word_tokens) == 1:
        return "single_token_fragment"
    if normalized.endswith(("(", "[", "{", ":", "/", "-", "\"", "'")):
        return "dangling_terminal"
    if any(
        normalized.count(opening) > normalized.count(closing)
        for opening, closing in (("(", ")"), ("[", "]"), ("{", "}"))
    ):
        return "unbalanced_brackets"
    if word_tokens[-1] in _DANGLING_SUFFIXES:
        return "dangling_suffix"
    if _looks_like_bare_subject_fragment(normalized, word_tokens):
        return "bare_subject_fragment"
    return None


def is_self_contained_rubric_text(text: str) -> bool:
    return rubric_text_quality_issue(text) is None


def _looks_like_bare_subject_fragment(
    normalized: str,
    word_tokens: list[str],
) -> bool:
    lowered = normalized.lower().rstrip(".")
    if lowered.endswith("?"):
        if not _AUXILIARY_QUESTION_PATTERN.match(lowered):
            return False
        remainder = _question_remainder(lowered)
        if not remainder:
            return True
        remainder_tokens = [token.lower() for token in _WORD_PATTERN.findall(remainder)]
        if not remainder_tokens:
            return True
        return remainder_tokens[-1] in _DANGLING_SUFFIXES

    if word_tokens[0] == "no":
        return False
    if (
        len(word_tokens) <= 3
        and word_tokens[0] in {"the", "this", "that", "these", "those", "all"}
        and not any(token in _PREDICATE_HINTS for token in word_tokens)
    ):
        return True

    subject_match = _GENERIC_SUBJECT_PREFIX_PATTERN.match(lowered)
    if subject_match is None:
        return False
    remainder = lowered[subject_match.end() :].strip()
    if not remainder:
        return True
    remainder_tokens = [token.lower() for token in _WORD_PATTERN.findall(remainder)]
    if not remainder_tokens:
        return True
    if remainder_tokens[-1] in _DANGLING_SUFFIXES:
        return True
    if len(remainder_tokens) <= 2 and not any(
        token in _PREDICATE_HINTS for token in remainder_tokens
    ):
        return True
    return False


def _question_remainder(lowered: str) -> str:
    text = lowered[:-1].strip() if lowered.endswith("?") else lowered
    aux_match = _AUXILIARY_QUESTION_PATTERN.match(text)
    if aux_match is None:
        return text
    remainder = text[aux_match.end() :].strip()
    for prefix in (
        "the response",
        "the answer",
        "the reply",
        "the code",
        "the solution",
        "the output",
        "response",
        "answer",
        "reply",
        "code",
        "solution",
        "output",
        "the",
        "this",
        "that",
        "all",
    ):
        if remainder.startswith(prefix):
            remainder = remainder[len(prefix) :].strip()
            break
    return remainder
