# RRD RubricBench Reproduction

This directory contains a clean-room reproduction of Recursive Rubric Decomposition (RRD) for RubricBench.

Reference:

[1] Shen W F, Qiu X, Whitehouse C, et al. Rethinking rubric generation for improving llm judge and reward modeling for open-ended tasks[J]. arXiv preprint arXiv:2602.05125, 2026.

## Pipeline

For each RubricBench example, the script runs four stages:

1. Sample four responses from GPT-4o and four responses from Gemini 2.5 Pro.
2. Propose rubrics and recursively decompose broad rubrics.
3. Filter directionality, overlap, and conflicts, then score accepted rubrics on both responses.
4. Aggregate rubric votes with the WU weighting rule and write the final prediction.

The sample models and RRD thresholds are defined in `src/rrd_rubricbench/rrd.py`. The main proposer and final judge are controlled by `--llm-model`.

## Install

Python 3.10 or newer is required.

```bash
cd rrd_rubricbench_repro
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Run

The runner uses an OpenAI-compatible chat-completions endpoint. Set the endpoint and key, then run:

```bash
export OPENAI_BASE_URL=http://your-llm-host:8000/v1
export OPENAI_API_KEY=your_api_key

python scripts/run_rrd_rubricbench.py \
  --benchmark-path ./data/rubricbench.json \
  --output-dir ./outputs/run_001 \
  --llm-model gpt-oss-120b \
  --llm-reasoning-effort high \
  --limit 20
```

`--base-url` and `--api-key` can be used instead of environment variables. Use `--start-index` and `--limit` for partial runs. Reduce `--llm-max-concurrency` or `--example-max-concurrency` when the endpoint is rate-limited.

## Outputs

The output directory contains:

```text
outputs/<run>/
|-- predictions.jsonl       Per-example predictions and rubric traces
|-- summary.json            Overall and per-domain accuracy
`-- checkpoints/            Resumable sampling, build, eval, and result files
```

Completed checkpoint stages are loaded automatically on a later run with the same configuration. The default LLM cache is `.cache`.

## Data format

The benchmark file can be JSON or JSONL. Each record should provide a prompt, `response_a`, `response_b`, and a preferred candidate. The loader also accepts a numeric `label` field and maps `0` to candidate A and `1` to candidate B.
