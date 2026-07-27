"""The same idea as the rest of this suite, expressed in DeepEval.

Contrast with promptfooconfig.yaml. That file declares prompts, cases, and
graders as data and runs them with `promptfoo eval`. This file is an ordinary
pytest module: DeepEval metrics are constructed in Python and handed to
assert_test, so prompt tests live next to every other test you already have.

Neither is a better tool. Pick the one that matches where your team already
works. Both are ways to run the assertion types in this repo with less
boilerplate, not different quality tiers.

This module is marked `live`: DeepEval's metrics are themselves model calls, and
the summarizer output being graded comes from a real model. It is skipped when
DeepEval is not installed or no key is set, which is why the every-commit tier
stays free.

Run it:  pip install -r requirements-eval.txt && OPENAI_API_KEY=... pytest -m live tests/test_deepeval_metrics.py
"""

from __future__ import annotations

import os

import pytest

deepeval = pytest.importorskip(
    "deepeval", reason="pip install -r requirements-eval.txt to run this tier"
)

from deepeval import assert_test  # noqa: E402
from deepeval.metrics import AnswerRelevancyMetric  # noqa: E402
from deepeval.test_case import LLMTestCase  # noqa: E402

from summarizer.prompt_runtime import (  # noqa: E402
    load_prompt,
    set_model_backend,
    summarize_ticket,
)

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="live tier needs OPENAI_API_KEY",
    ),
]

# The grader model, pinned for the same reason the judge model is pinned in
# tests/test_llm_judge.py: a metric whose model drifts is a metric whose history
# is not comparable.
GRADER_MODEL = "gpt-4o-mini-2024-07-18"

RELEVANCY_THRESHOLD = 0.7

PROMPT = load_prompt()
GOLDEN_CASES = PROMPT["golden_cases"]
CASE_IDS = [case["id"] for case in GOLDEN_CASES]


def openai_backend(system: str, user: str) -> str:
    """A real model call, using the model and temperature from the fixture."""
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=PROMPT["model"],
        temperature=PROMPT["temperature"],
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content or ""


@pytest.fixture(autouse=True)
def _use_a_real_model() -> None:
    set_model_backend(openai_backend)


@pytest.mark.parametrize("case", GOLDEN_CASES, ids=CASE_IDS)
def test_summary_is_relevant_to_the_ticket(case: dict) -> None:
    result = summarize_ticket(case["variables"])

    test_case = LLMTestCase(
        input=case["variables"]["ticket_body"],
        actual_output=result["summary"],
        expected_output=case["reference_summary"],
    )
    relevancy = AnswerRelevancyMetric(
        threshold=RELEVANCY_THRESHOLD,
        model=GRADER_MODEL,
        include_reason=True,
    )

    # assert_test raises with the metric's reason attached on failure, so a red
    # CI run tells you why the summary was judged irrelevant.
    assert_test(test_case, [relevancy])


@pytest.mark.parametrize("case", GOLDEN_CASES, ids=CASE_IDS)
def test_deterministic_fields_still_hold_against_a_live_model(case: dict) -> None:
    """The cheap assertions do not stop mattering once a metric is involved.

    Same expectations as tests/test_structured_output.py, run against a real
    response. This is the pull-request tier: a handful of golden cases, real cost,
    bounded.
    """
    result = summarize_ticket(case["variables"])

    for field, expected_value in case["expected"].items():
        assert result[field] == expected_value, (
            f"{case['id']}: expected {field}={expected_value!r}, got {result[field]!r}"
        )
