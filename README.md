# LLM Unit Testing: Writing Tests for Your Prompts

A runnable pytest project for unit testing LLM prompts. Every assertion type from the
article exists here as real code: a versioned prompt fixture, deterministic schema and
string assertions, a semantic-similarity test whose threshold is calibrated from data,
an LLM-as-judge test with an explicit rubric at temperature 0, a fast mocked logic tier,
side-by-side Promptfoo and DeepEval configs, and a GitHub Actions workflow that tiers
mocked, live, and nightly runs.

> Companion code for the Autonoma blog post: **[LLM Unit Testing: Writing Tests for Your Prompts](https://getautonoma.com/blog/llm-unit-testing)**

## The suite runs with no API key

That is the design, not a limitation. Model and embedding calls are stubbed with
recorded responses so the tests that check *your code* stay free and instant. Only the
live tiers need credentials.

```
52 passed, 1 skipped in 0.2s
```

## Requirements

Python 3.10 or newer. Node 20 or newer if you want to run the Promptfoo config.

## Quickstart

```bash
git clone https://github.com/Autonoma-Tools/llm-unit-testing.git
cd llm-unit-testing
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest
```

No key, no network, no config. The one skipped test is the DeepEval tier, which needs
the extra dependencies below.

## What is in here

```
prompts/summarize_ticket/v3.yaml        versioned prompt fixture with golden cases
prompts/summarize_ticket/promptfoo_prompt.yaml   the same prompt in Promptfoo chat format
summarizer/prompt_runtime.py           render, budget, call, parse. The model call is injected
tests/test_structured_output.py         Pydantic schema assertions on JSON output
tests/test_string_assertions.py         must-contain and must-not-contain on raw text
tests/test_semantic_similarity.py       cosine similarity with an empirically calibrated threshold
tests/test_llm_judge.py                 LLM-as-judge with an explicit rubric, pinned model, temperature 0
tests/test_prompt_logic_mocked.py       tier 1: template, budget, parsing, error paths, zero network
tests/test_deepeval_metrics.py          the same idea in DeepEval, pytest-native
promptfooconfig.yaml                    the same idea in Promptfoo, config-first
.github/workflows/test.yml              the three-tier CI split
```

## Running each tier

### Tier 1: mocked and replayed, every commit

Free, instant, no credentials. This is what runs on every push.

```bash
pytest -m "not live" \
  tests/test_prompt_logic_mocked.py \
  tests/test_structured_output.py \
  tests/test_string_assertions.py \
  tests/test_semantic_similarity.py \
  tests/test_llm_judge.py
```

`tests/test_prompt_logic_mocked.py` blocks socket creation in an autouse fixture, so
"this tier makes no API calls" is a test failure rather than a promise.

To watch the similarity thresholds being calibrated instead of guessed:

```bash
pytest tests/test_semantic_similarity.py -s -k calibration
```

### Tier 2: live golden subset, every pull request

Real model calls on a handful of golden cases. Bounded cost.

```bash
cp .env.example .env        # then put a real key in it
pip install -r requirements.txt -r requirements-eval.txt
export OPENAI_API_KEY=...
pytest -m live tests/test_deepeval_metrics.py
npx promptfoo@latest eval --filter-first-n 2
```

### Tier 3: full eval suite, nightly

```bash
pytest tests/
npx promptfoo@latest eval
npx promptfoo@latest view     # browse the result grid
```

## Swapping the stubs for a real model

Three stubs exist, each marked in place with what to replace it with:

| Stub | File | Replace with |
|---|---|---|
| `generate()` | `tests/test_structured_output.py`, `tests/test_string_assertions.py` | your provider's chat completion, using `model` and `temperature` from the fixture |
| `get_embedding()` | `tests/test_semantic_similarity.py` | a real embeddings call. The calibration routine works unchanged |
| `call_judge()` | `tests/test_llm_judge.py` | a chat completion at temperature 0 against the pinned judge model |

For application code, `summarizer/prompt_runtime.py` takes the backend by injection:
`set_model_backend(fn)` where `fn(system, user) -> str`. A working OpenAI
implementation is in `tests/test_deepeval_metrics.py`.

## Adapting this to your own prompt

1. Copy `prompts/summarize_ticket/v3.yaml` to `prompts/<your_prompt>/v1.yaml` and rewrite
   `system_prompt`, `user_template`, and `golden_cases`.
2. Rewrite the `TicketSummary` model in `tests/test_structured_output.py` to your output
   contract. Keep `extra="forbid"`: it is what catches a renamed field.
3. Put your required patterns and forbidden boilerplate in each golden case's
   `must_contain`, `must_contain_patterns`, and `must_not_contain`.
4. For anything with no fixed correct string, add `reference_summary` plus at least three
   `accepted_summaries` and let `calibrate_threshold()` pick the floor.
5. Reach for the judge last, and only for properties the four steps above cannot express.

## Security

No keys are committed here, and none are needed for the default test run. `.env` is
gitignored; only `.env.example` is tracked.

## About

This repository is maintained by [Autonoma](https://getautonoma.com) as reference material
for the linked blog post. Every assertion in here checks what a model *said*. Autonoma
tests the layer above: its Planner reads your codebase to plan behavioral test cases, and
its Executor drives the real UI to confirm the response actually produced the right
outcome in the running app.

If something here is wrong, out of date, or unclear, please
[open an issue](https://github.com/Autonoma-Tools/llm-unit-testing/issues/new).

## License

Released under the [MIT License](./LICENSE) 2026 Autonoma Labs.
