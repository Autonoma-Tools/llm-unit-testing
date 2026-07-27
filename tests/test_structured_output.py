"""Deterministic schema assertions for a structured-output prompt.

The contract here is hard: the model either returns an object with these five
fields, at these types, and nothing else, or the test fails. Nothing about this
is fuzzier than testing any other JSON API response.

Run it:  pytest tests/test_structured_output.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pytest
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

PROMPT_PATH = (
    Path(__file__).resolve().parents[1] / "prompts" / "summarize_ticket" / "v3.yaml"
)

ACCOUNT_ID_PATTERN = r"^(ACCT-\d{6}|unknown)$"


class TicketSummary(BaseModel):
    """The response contract, as a schema instead of a hope.

    extra="forbid" is the part teams skip. Without it, a model that renames a
    field to something plausible still passes, because the field you asked for
    is quietly absent and the one it invented is quietly ignored.
    """

    model_config = ConfigDict(extra="forbid")

    account_id: str = Field(pattern=ACCOUNT_ID_PATTERN)
    category: Literal["billing", "bug", "how_to", "other"]
    urgency: int = Field(ge=1, le=5)
    summary: str = Field(min_length=10)
    needs_human: bool


# Recorded model responses, so this file runs in CI with no credentials and no
# network. Delete this dict and point generate() at your provider to run live.
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


def load_prompt(path: Path = PROMPT_PATH) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def generate(system: str, user: str, case_id: str | None = None) -> str:
    """Stand-in for a real model call.

    Replace the body with your provider's chat completion, using the model and
    temperature from the fixture. The case_id argument exists only so the offline
    stub can replay a recorded response. A real implementation ignores it.
    """
    if case_id is None or case_id not in RECORDED_RESPONSES:
        raise AssertionError(f"no recorded response for case {case_id!r}")
    return RECORDED_RESPONSES[case_id]


PROMPT = load_prompt()
GOLDEN_CASES = PROMPT["golden_cases"]
CASE_IDS = [case["id"] for case in GOLDEN_CASES]


def render_user_message(case: dict) -> str:
    template = PROMPT["user_template"]
    variables = case["variables"]
    for name, value in variables.items():
        template = template.replace("{" + name + "}", str(value))
    return template


@pytest.mark.parametrize("case", GOLDEN_CASES, ids=CASE_IDS)
def test_response_satisfies_the_schema(case: dict) -> None:
    raw = generate(
        PROMPT["system_prompt"], render_user_message(case), case_id=case["id"]
    )

    # Parsing and validation are one step. If either fails, the test fails.
    summary = TicketSummary.model_validate_json(raw)

    for field, expected_value in case["expected"].items():
        assert getattr(summary, field) == expected_value, (
            f"{case['id']}: expected {field}={expected_value!r}, "
            f"got {getattr(summary, field)!r}"
        )


def test_the_fixture_pins_deterministic_decoding() -> None:
    # A structured-output prompt that is graded on exact fields has no business
    # running at a nonzero temperature.
    assert PROMPT["temperature"] == 0
    assert PROMPT["version"] == 3


def test_a_missing_required_field_fails() -> None:
    payload = json.loads(RECORDED_RESPONSES["billing_duplicate_charge"])
    del payload["account_id"]

    with pytest.raises(ValidationError) as excinfo:
        TicketSummary.model_validate(payload)

    assert "account_id" in str(excinfo.value)


def test_a_wrongly_typed_field_fails() -> None:
    payload = json.loads(RECORDED_RESPONSES["billing_duplicate_charge"])
    payload["urgency"] = "high"  # the model wrote a word where an integer belongs

    with pytest.raises(ValidationError):
        TicketSummary.model_validate(payload)


def test_an_out_of_range_enum_value_fails() -> None:
    payload = json.loads(RECORDED_RESPONSES["billing_duplicate_charge"])
    payload["category"] = "refund_request"  # plausible, and not in the contract

    with pytest.raises(ValidationError):
        TicketSummary.model_validate(payload)


def test_a_renamed_field_fails_instead_of_passing_silently() -> None:
    payload = json.loads(RECORDED_RESPONSES["billing_duplicate_charge"])
    payload["accountId"] = payload.pop("account_id")  # camelCase drift

    with pytest.raises(ValidationError) as excinfo:
        TicketSummary.model_validate(payload)

    message = str(excinfo.value)
    assert "account_id" in message  # required field absent
    assert "accountId" in message  # unexpected field present


def test_an_empty_account_id_fails() -> None:
    # The field is present and correctly typed, and still useless downstream.
    # This is the case a key-presence check waves through.
    payload = json.loads(RECORDED_RESPONSES["billing_duplicate_charge"])
    payload["account_id"] = ""

    with pytest.raises(ValidationError):
        TicketSummary.model_validate(payload)
