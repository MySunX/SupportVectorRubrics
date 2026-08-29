from __future__ import annotations

from collections.abc import Sequence


def _block(title: str, content: str) -> str:
    return f"<{title}>\n{content.strip()}\n</{title}>"


def _format_messages(messages: Sequence[dict[str, str]]) -> str:
    lines = []
    for i, msg in enumerate(messages, 1):
        role = str(msg.get("role", "message"))
        content = str(msg.get("content", "")).strip()
        lines.append(f"[{i}] role={role}\n{content}")
    return "\n\n".join(lines)


def _format_responses(responses: Sequence[str]) -> str:
    blocks = []
    for idx, resp in enumerate(responses, 1):
        blocks.append(f"[{idx}]\n{resp.strip()}")
    return "\n\n".join(blocks)


def build_sampling_prompt(*, prompt_messages: Sequence[dict[str, str]]) -> str:
    return f"""
You are a helpful assistant.

Write a high-quality response to the user prompt below.

TASK PROMPT:
{_format_messages(prompt_messages)}

Answer the user directly and naturally.
""".strip()


def build_initial_rubric_prompt(*, prompt_messages: Sequence[dict[str, str]], responses: Sequence[str]) -> str:
    return f"""
Role: You are a rubric designer for an LLM-as-judge system.

Inputs you will receive:
- Prompt: the task/question the response must answer.
- Responses: a set of responses to be evaluated against rubrics.

Goal: Design a comprehensive set of rubrics for evaluating responses to the given prompt. Only write rubrics you are confident about. Only propose the best new rubrics.

Requirements:
- Propose rubrics that collectively cover the most important dimensions needed to judge whether a response correctly and helpfully satisfies the prompt.
- Each rubric must be consistently judgeable across many responses (avoid vague wording like "good", "nice", "high-quality").
- Each rubric must be prompt-specific (tied to what the user asked), not generic writing advice.
- Each rubric should be written as a single criterion with clear, binary pass/fail boundaries. Prefer objective checks.
- New rubric MUST NOT answer the question directly.
- New rubric MUST NOT repeat any of the responses provided.

Tips for writing good rubrics:
- MECE: Mutually Exclusive, Collectively Exhaustive
- Completeness: cover all important aspects of an ideal response
- No overlapping: do not punish the same error multiple times
- Diversity: do not make all rubrics simple "mentions A/B" items
- Atomicity: one criterion per rubric
- Specificity: binary and objective
- Self-contained: enough information to evaluate without extra context
- No external search: the criterion must be verifiable from the prompt and response

Below are the inputs:
{_block("PROMPT", _format_messages(prompt_messages))}

{_block("RESPONSES", _format_responses(responses))}

Output STRICTLY in below format. No other text is allowed:
<RUBRIC> Rubric 1 </RUBRIC>
<RUBRIC> Rubric 2 </RUBRIC>
...
""".strip()


def build_decomposition_prompt(
    *,
    prompt_messages: Sequence[dict[str, str]],
    responses: Sequence[str],
    current_rubric: str,
    other_rubrics: Sequence[str],
) -> str:
    other_block = "\n".join(f"- {r}" for r in other_rubrics) if other_rubrics else "- (none)"
    return f"""
Role: You are a rubric designer for an LLM-as-judge system.

Inputs you will receive:
- Prompt: the task/question the response must answer.
- Responses: the subset of sampled responses that satisfied the current rubric.
- Current rubric: criterion currently used by a judge. This rubric has already been satisfied by multiple responses, so it is too coarse and fails to distinguish response quality.
- Other rubrics: other rubrics that the new rubric must NOT overlap with.

Goal: Propose exactly TWO new rubrics that are more granular than the existing ones and can better differentiate candidate responses. Only write rubrics you are confident about. Only propose the best new rubrics.

What "more granular" means:
- Each new rubric must target a specific, discriminative dimension of quality that is not sufficiently captured by the existing rubrics.
- New rubrics should NOT miss critical information contained in the existing rubric.
- Each rubric must be consistently judgeable across many responses.
- Each rubric must be prompt-specific.
- Each rubric should be written as a single criterion with clear, binary pass/fail boundaries.
- New rubric MUST NOT repeat any of the responses provided.
- New rubric MUST NOT answer the question directly.

Below are the inputs:
{_block("PROMPT", _format_messages(prompt_messages))}

{_block("RESPONSES", _format_responses(responses))}

{_block("CURRENT_RUBRIC", current_rubric)}

<OTHER_RUBRICS>
{other_block}
</OTHER_RUBRICS>

Output STRICTLY in below format. No other text is allowed:
<RUBRIC> New rubric 1 </RUBRIC>
<RUBRIC> New rubric 2 </RUBRIC>
""".strip()


def build_overlap_prompt(*, existing_rubrics: Sequence[str], new_rubric: str) -> str:
    existing = "\n".join(f"- {r}" for r in existing_rubrics) if existing_rubrics else "- (none)"
    return f"""
You are checking whether a new rubric substantially overlaps with ANY of the existing rubrics. If ANY overlap is found, output YES; otherwise output NO.

Definition of substantial overlapping:
- The new rubric has the same intent as an existing rubric, or is a strict subset/superset of it, or >= 70% of its meaning is covered by the existing rubric so that applying both would not materially change scoring outcomes.
- Match on meaning, not wording. Treat synonyms, paraphrases, and inverses with the same effect as overlapping.
- Ignore trivial phrasing, tone, and example differences unless they change the requirement.

Edge cases:
- If scopes are disjoint (different capability/goal) -> NO.
- If the new rubric adds only minor qualifiers without changing evaluation -> YES.
- If the new rubric merely narrows the context while keeping the same criterion (subset) or broadens it (superset) -> YES.

EXISTING_RUBRICS:
{existing}

NEW_RUBRIC:
{new_rubric}

Output STRICTLY in below format. No other text is allowed:
<EVALUATION> YES/NO </EVALUATION>
""".strip()


def build_conflict_prompt(*, existing_rubrics: Sequence[str], new_rubric: str) -> str:
    existing = "\n".join(f"- {r}" for r in existing_rubrics) if existing_rubrics else "- (none)"
    return f"""
You are checking whether a new rubric expresses opposite meaning of ANY of the existing rubrics. If ANY opposition is found, output YES; otherwise output NO.

Definition of opposition:
- Opposite means the new rubric asserts the negation or reverse polarity of the same requirement, property, or direction as an existing rubric.
- Examples:
  - require X <-> forbid/avoid X
  - must include X <-> must NOT include X
  - prefer more of X <-> prefer less of X
  - answer should be optimistic <-> answer should be pessimistic
- Do NOT flag different targets or contexts.
- Do NOT flag orthogonal dimensions.
- Do NOT flag mere differences in emphasis, strength, scope, or style.
- Do NOT flag stricter/looser thresholds unless they clearly reverse direction on the same axis.

EXISTING_RUBRICS:
{existing}

NEW_RUBRIC:
{new_rubric}

Output STRICTLY in below format. No other text is allowed:
<EVALUATION> YES/NO </EVALUATION>
""".strip()


def build_rubric_judge_prompt(*, response: str, rubric: str) -> str:
    return f"""
You are a judge, evaluating whether a response satisfies the given rubric. If the response satisfies the criterion of the rubric, output YES; otherwise output NO.

Requirement:
- You must follow the rubric strictly, and only consider the criteria listed in the rubric.
- You must NOT consider any other factors, such as your own opinions or external knowledge.

Below between <RESPONSE> and </RESPONSE> is the response to evaluate on
<RESPONSE>
{response.strip()}
</RESPONSE>

Below between <RUBRIC> and </RUBRIC> is the rubric to evaluate on
<RUBRIC>
{rubric.strip()}
</RUBRIC>

Output STRICTLY in below format. No other text is allowed:
<EVALUATION> YES/NO </EVALUATION>
""".strip()


def build_direct_pairwise_prompt(*, prompt_messages: Sequence[dict[str, str]], response_a: str, response_b: str) -> str:
    return f"""
You are a judge comparing two responses to the same task prompt.
Choose the response that better satisfies the user's request.

TASK PROMPT:
{_format_messages(prompt_messages)}

RESPONSE_A:
{response_a.strip()}

RESPONSE_B:
{response_b.strip()}

Return JSON only with the following schema:
{{"preferred_candidate":"A|B","rationale":"short"}}
""".strip()
