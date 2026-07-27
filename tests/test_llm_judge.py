"""LLM-as-judge, for the residual nothing cheaper can express.

Three rules make the difference between a judge and a random number generator:

1. An explicit rubric. Named criteria with a definition each, not "rate this
   response 1 to 5", which asks for a plausible number rather than a measurement.
2. A pinned judge model. Not "the latest", or your judge drifts underneath you at
   the same time as the model you are testing, and you cannot tell which moved.
3. Temperature 0. It does not make a judge deterministic, it stops it from
   volunteering extra variance on top of what is already there.

This is the most expensive and least stable assertion type in the suite. Reach
for it last, on the small set of properties a schema and an embedding score
genuinely cannot express.

Run it:  pytest tests/test_llm_judge.py -v
"""

from __future__ import annotations

import json
import re

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

# Pinned to a dated snapshot on purpose. "gpt-4o" is a moving target: the weights
# behind that alias change, and when they do, every judged score in your history
# shifts with no commit to blame. A dated snapshot makes a judge upgrade an
# explicit, reviewable, one-line diff.
JUDGE_MODEL = "gpt-4o-2024-08-06"

# Not negotiable for a grader.
JUDGE_TEMPERATURE = 0.0

# The rubric. Each criterion is a name plus a definition precise enough that two
# people reading the same response would agree on pass or fail.
RUBRIC: tuple[tuple[str, str], ...] = (
    (
        "addresses_the_specific_problem",
        "The reply restates the customer's actual problem, with its specifics, "
        "rather than a generic category of problem.",
    ),
    (
        "states_a_concrete_next_step",
        "The reply names one concrete next step and who takes it, rather than "
        "saying the issue is being looked into.",
    ),
    (
        "makes_no_unfounded_promise",
        "The reply promises no refund, deadline, credit, or outcome that the "
        "provided context does not already authorize.",
    ),
    (
        "tone_is_calm_and_non_defensive",
        "The reply neither blames the customer nor over-apologizes, and does not "
        "argue with the customer's account of events.",
    ),
)

RUBRIC_NAMES = tuple(name for name, _ in RUBRIC)


class CriterionResult(BaseModel):
    """One rubric line, graded."""

    model_config = ConfigDict(extra="forbid")

    criterion: str
    passed: bool
    evidence: str = Field(min_length=1)  # a verdict with no quote is not a verdict


class JudgeVerdict(BaseModel):
    """The judge's structured output, validated like any other model response."""

    model_config = ConfigDict(extra="forbid")

    results: list[CriterionResult]

    @model_validator(mode="after")
    def _must_cover_the_whole_rubric(self) -> "JudgeVerdict":
        graded = [result.criterion for result in self.results]
        if sorted(graded) != sorted(RUBRIC_NAMES):
            missing = sorted(set(RUBRIC_NAMES) - set(graded))
            unexpected = sorted(set(graded) - set(RUBRIC_NAMES))
            raise ValueError(
                f"verdict does not cover the rubric; missing={missing} "
                f"unexpected={unexpected}"
            )
        return self

    @property
    def failures(self) -> list[str]:
        return [result.criterion for result in self.results if not result.passed]

    @property
    def passed(self) -> bool:
        return not self.failures


def build_judge_prompt(customer_message: str, reply: str) -> str:
    """Render the rubric into a grading prompt.

    Everything the judge is allowed to consider is in this string. No score
    scale, no "use your judgment", no room to average four things into one number.
    """
    criteria_block = "\n".join(
        f"- {name}: {definition}" for name, definition in RUBRIC
    )
    return (
        "You are grading a support reply against a fixed rubric.\n"
        "Grade each criterion independently as pass or fail. Do not average them "
        "and do not produce an overall score.\n"
        "For each criterion, quote the span of the reply that decided it.\n\n"
        f"Criteria:\n{criteria_block}\n\n"
        f"Customer message:\n{customer_message}\n\n"
        f"Reply under test:\n{reply}\n\n"
        'Return only JSON of the form {"results": [{"criterion": "<name>", '
        '"passed": true, "evidence": "<quote>"}]} with one entry per criterion, '
        "using the criterion names exactly as written above."
    )


# Every judge call this module makes, recorded so the tests can assert on how the
# judge was invoked and not just on what it returned.
JUDGE_CALLS: list[dict] = []

# Recorded judge verdicts, so this file runs with no credentials and no network.
RECORDED_VERDICTS = {
    "good_reply": {
        "results": [
            {
                "criterion": "addresses_the_specific_problem",
                "passed": True,
                "evidence": "two charges of 49 dollars on ACCT-482913",
            },
            {
                "criterion": "states_a_concrete_next_step",
                "passed": True,
                "evidence": "a billing specialist will review both charges today",
            },
            {
                "criterion": "makes_no_unfounded_promise",
                "passed": True,
                "evidence": "if the second charge is confirmed as duplicate",
            },
            {
                "criterion": "tone_is_calm_and_non_defensive",
                "passed": True,
                "evidence": "Thanks for flagging this.",
            },
        ]
    },
    "evasive_reply": {
        "results": [
            {
                "criterion": "addresses_the_specific_problem",
                "passed": False,
                "evidence": "we have received your billing inquiry",
            },
            {
                "criterion": "states_a_concrete_next_step",
                "passed": False,
                "evidence": "our team is looking into it",
            },
            {
                "criterion": "makes_no_unfounded_promise",
                "passed": True,
                "evidence": "no promise is made",
            },
            {
                "criterion": "tone_is_calm_and_non_defensive",
                "passed": True,
                "evidence": "we appreciate your patience",
            },
        ]
    },
    "overpromising_reply": {
        "results": [
            {
                "criterion": "addresses_the_specific_problem",
                "passed": True,
                "evidence": "the duplicate 49 dollar charge",
            },
            {
                "criterion": "states_a_concrete_next_step",
                "passed": True,
                "evidence": "I am refunding it now",
            },
            {
                "criterion": "makes_no_unfounded_promise",
                "passed": False,
                "evidence": "you will see the money back within one hour, guaranteed",
            },
            {
                "criterion": "tone_is_calm_and_non_defensive",
                "passed": True,
                "evidence": "Sorry about that.",
            },
        ]
    },
}


def call_judge(
    prompt: str,
    model: str = JUDGE_MODEL,
    temperature: float = JUDGE_TEMPERATURE,
    case_id: str | None = None,
) -> str:
    """Stand-in for a real judge call.

    Replace the body with your provider's chat completion, passing model and
    temperature straight through. The case_id argument exists only so the offline
    stub can replay a recorded verdict. A real implementation ignores it.
    """
    JUDGE_CALLS.append({"prompt": prompt, "model": model, "temperature": temperature})
    if case_id is None or case_id not in RECORDED_VERDICTS:
        raise AssertionError(f"no recorded verdict for case {case_id!r}")
    return json.dumps(RECORDED_VERDICTS[case_id])


def judge_reply(customer_message: str, reply: str, case_id: str) -> JudgeVerdict:
    prompt = build_judge_prompt(customer_message, reply)
    raw = call_judge(
        prompt,
        model=JUDGE_MODEL,
        temperature=JUDGE_TEMPERATURE,
        case_id=case_id,
    )
    return JudgeVerdict.model_validate_json(raw)


CUSTOMER_MESSAGE = (
    "I was charged twice for my subscription this month on ACCT-482913, two "
    "charges of 49 dollars on the same day. Please refund one of them."
)

GOOD_REPLY = (
    "Thanks for flagging this. I can see two charges of 49 dollars on "
    "ACCT-482913 dated the same day. A billing specialist will review both "
    "charges today, and if the second charge is confirmed as duplicate it will "
    "be reversed to your original payment method."
)

EVASIVE_REPLY = (
    "Hello, we have received your billing inquiry and our team is looking into "
    "it. We appreciate your patience and will be in touch."
)

OVERPROMISING_REPLY = (
    "Sorry about that. I am refunding the duplicate 49 dollar charge now and you "
    "will see the money back within one hour, guaranteed."
)


@pytest.fixture(autouse=True)
def _clear_recorded_calls():
    JUDGE_CALLS.clear()
    yield
    JUDGE_CALLS.clear()


def test_rubric_is_written_into_the_prompt() -> None:
    prompt = build_judge_prompt(CUSTOMER_MESSAGE, GOOD_REPLY)

    for name, definition in RUBRIC:
        assert name in prompt
        assert definition in prompt


def test_prompt_asks_for_named_criteria_not_a_vague_score() -> None:
    prompt = build_judge_prompt(CUSTOMER_MESSAGE, GOOD_REPLY).lower()

    for banned in ("1 to 5", "1-5", "out of 10", "rate this response", "score from"):
        assert banned not in prompt, f"judge prompt fell back to {banned!r}"

    assert "pass or fail" in prompt


def test_judge_model_is_pinned_to_a_dated_snapshot() -> None:
    # A floating alias would let the grader change without a commit.
    assert re.search(r"-\d{4}-\d{2}-\d{2}$", JUDGE_MODEL), (
        f"judge model {JUDGE_MODEL!r} is not pinned to a dated snapshot"
    )


def test_judge_is_called_at_temperature_zero_with_the_pinned_model() -> None:
    judge_reply(CUSTOMER_MESSAGE, GOOD_REPLY, case_id="good_reply")

    assert len(JUDGE_CALLS) == 1
    assert JUDGE_CALLS[0]["temperature"] == 0.0
    assert JUDGE_CALLS[0]["model"] == JUDGE_MODEL


def test_a_good_reply_passes_every_criterion() -> None:
    verdict = judge_reply(CUSTOMER_MESSAGE, GOOD_REPLY, case_id="good_reply")

    assert verdict.passed
    assert verdict.failures == []


def test_an_evasive_reply_fails_the_criteria_it_should() -> None:
    verdict = judge_reply(CUSTOMER_MESSAGE, EVASIVE_REPLY, case_id="evasive_reply")

    assert not verdict.passed
    # Named failures, so a red test says which property broke.
    assert verdict.failures == [
        "addresses_the_specific_problem",
        "states_a_concrete_next_step",
    ]


def test_an_overpromising_reply_fails_only_the_promise_criterion() -> None:
    verdict = judge_reply(
        CUSTOMER_MESSAGE, OVERPROMISING_REPLY, case_id="overpromising_reply"
    )

    assert verdict.failures == ["makes_no_unfounded_promise"]


def test_a_verdict_missing_a_criterion_is_rejected() -> None:
    partial = {"results": RECORDED_VERDICTS["good_reply"]["results"][:2]}

    with pytest.raises(ValidationError) as excinfo:
        JudgeVerdict.model_validate(partial)

    assert "does not cover the rubric" in str(excinfo.value)


def test_a_verdict_with_an_invented_criterion_is_rejected() -> None:
    invented = {
        "results": [
            *RECORDED_VERDICTS["good_reply"]["results"][:3],
            {"criterion": "vibes", "passed": True, "evidence": "felt fine"},
        ]
    }

    with pytest.raises(ValidationError):
        JudgeVerdict.model_validate(invented)


def test_a_verdict_without_evidence_is_rejected() -> None:
    unsupported = {
        "results": [
            {**result, "evidence": ""}
            for result in RECORDED_VERDICTS["good_reply"]["results"]
        ]
    }

    with pytest.raises(ValidationError):
        JudgeVerdict.model_validate(unsupported)
