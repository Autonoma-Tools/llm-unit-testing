"""Must-contain and must-not-contain assertions on raw model output.

This is the test that would have caught the regression in the article's opening
story. A teammate tightened the system prompt to stop the model appending a
sign-off, and the model quietly stopped extracting the account ID along with it.
The JSON stayed valid. The schema stayed satisfied. The one field the pipeline
depended on went missing.

A must-contain assertion on the account-id pattern fails on the first commit
instead of six days later.

Run it:  pytest tests/test_string_assertions.py -v
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

PROMPT_PATH = (
    Path(__file__).resolve().parents[1] / "prompts" / "summarize_ticket" / "v3.yaml"
)

# Recorded model responses, so this file runs with no credentials and no network.
# Replace generate() with a real call to run it live.
RECORDED_RESPONSES = {
    "billing_duplicate_charge": (
        '{"account_id": "ACCT-482913", "category": "billing", "urgency": 4, '
        '"summary": "The customer was billed twice on the same day for one '
        'monthly subscription. They are asking for a refund of the duplicate '
        'charge.", "needs_human": true}'
    ),
    "bug_login_redirect_loop": (
        '{"account_id": "ACCT-771204", "category": "bug", "urgency": 4, '
        '"summary": "Signing in on Chrome loops the customer back to the login '
        'screen and blocks the invoices page. Safari is unaffected.", '
        '"needs_human": true}'
    ),
    "how_to_change_billing_email": (
        '{"account_id": "unknown", "category": "how_to", "urgency": 2, '
        '"summary": "The customer wants to change the email address that '
        'receives the monthly invoice. Profile settings only showed the login '
        'email.", "needs_human": false}'
    ),
}

# The actual output from the incident: valid JSON, correct shape, account_id
# silently dropped to the fallback on a ticket that plainly contained one.
REGRESSED_RESPONSE = (
    '{"account_id": "unknown", "category": "billing", "urgency": 4, '
    '"summary": "The customer was billed twice on the same day for one monthly '
    'subscription and wants a refund.", "needs_human": true}'
)

# The other half of the same incident: the sign-off the prompt edit was meant to
# remove, reintroduced by a later edit that reverted one sentence too many.
BOILERPLATE_RESPONSE = (
    '{"account_id": "ACCT-482913", "category": "billing", "urgency": 4, '
    '"summary": "The customer was billed twice and wants a refund. Thanks for '
    'reaching out, we are happy to help.", "needs_human": true}'
)


def load_prompt(path: Path = PROMPT_PATH) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def generate(system: str, user: str, case_id: str | None = None) -> str:
    """Stand-in for a real model call. Swap the body for your provider's client."""
    if case_id is None or case_id not in RECORDED_RESPONSES:
        raise AssertionError(f"no recorded response for case {case_id!r}")
    return RECORDED_RESPONSES[case_id]


def assert_string_contract(
    output: str,
    must_contain: list[str] | None = None,
    must_contain_patterns: list[str] | None = None,
    must_not_contain: list[str] | None = None,
) -> None:
    """Assert presence and absence markers against raw model text.

    Kept as one function so the same contract can be applied to a recorded
    response, a live response, or a regression sample.
    """
    for needle in must_contain or []:
        assert needle in output, f"required substring {needle!r} missing from output"

    for pattern in must_contain_patterns or []:
        assert re.search(pattern, output), (
            f"required pattern {pattern!r} did not match output"
        )

    for needle in must_not_contain or []:
        assert needle.lower() not in output.lower(), (
            f"forbidden substring {needle!r} present in output"
        )


PROMPT = load_prompt()
GOLDEN_CASES = PROMPT["golden_cases"]
CASE_IDS = [case["id"] for case in GOLDEN_CASES]


def render_user_message(case: dict) -> str:
    template = PROMPT["user_template"]
    for name, value in case["variables"].items():
        template = template.replace("{" + name + "}", str(value))
    return template


@pytest.mark.parametrize("case", GOLDEN_CASES, ids=CASE_IDS)
def test_output_meets_its_string_contract(case: dict) -> None:
    output = generate(
        PROMPT["system_prompt"], render_user_message(case), case_id=case["id"]
    )

    assert_string_contract(
        output,
        must_contain=case.get("must_contain"),
        must_contain_patterns=case.get("must_contain_patterns"),
        must_not_contain=case.get("must_not_contain"),
    )


def test_the_dropped_account_id_regression_is_caught() -> None:
    """Proof the assertion above is load-bearing and not decorative.

    Feed it the exact output from the incident and it fails. This is the six-days
    of-silence bug, turned into a red test.
    """
    case = next(c for c in GOLDEN_CASES if c["id"] == "billing_duplicate_charge")

    with pytest.raises(AssertionError) as excinfo:
        assert_string_contract(
            REGRESSED_RESPONSE,
            must_contain=case["must_contain"],
            must_contain_patterns=case["must_contain_patterns"],
            must_not_contain=case["must_not_contain"],
        )

    assert "ACCT-482913" in str(excinfo.value)


def test_reintroduced_boilerplate_is_caught() -> None:
    case = next(c for c in GOLDEN_CASES if c["id"] == "billing_duplicate_charge")

    with pytest.raises(AssertionError) as excinfo:
        assert_string_contract(
            BOILERPLATE_RESPONSE,
            must_contain=case["must_contain"],
            must_not_contain=case["must_not_contain"],
        )

    assert "Thanks for reaching out" in str(excinfo.value)


def test_forbidden_markers_are_matched_case_insensitively() -> None:
    # Models reword their own boilerplate. "BEST REGARDS" is the same bug.
    with pytest.raises(AssertionError):
        assert_string_contract(
            '{"summary": "Refund issued. BEST REGARDS, support"}',
            must_not_contain=["Best regards"],
        )


def test_every_golden_case_declares_at_least_one_marker() -> None:
    # A golden case with no assertions is a golden case that cannot fail.
    for case in GOLDEN_CASES:
        markers = (
            case.get("must_contain", [])
            + case.get("must_contain_patterns", [])
            + case.get("must_not_contain", [])
        )
        assert markers, f"{case['id']} declares no string markers"
