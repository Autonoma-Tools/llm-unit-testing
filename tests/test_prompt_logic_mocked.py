"""Tier 1: everything that tests your code rather than the model.

Template rendering, variable interpolation, the token budget guard, and what
happens when the model returns garbage. None of it needs a live API call, all of
it runs in milliseconds, and the _no_network fixture below makes that a checked
property rather than a claim: any attempt to open a socket in this file fails the
test.

This is the tier that runs on every commit. There is no cost argument against it.

Run it:  pytest tests/test_prompt_logic_mocked.py -v
"""

from __future__ import annotations

import socket
from unittest.mock import Mock

import pytest

from summarizer.prompt_runtime import (
    CHARS_PER_TOKEN,
    MalformedModelResponse,
    MissingPromptVariable,
    ModelBackendNotConfigured,
    PromptTooLarge,
    build_request,
    check_token_budget,
    estimate_tokens,
    generate,
    load_prompt,
    parse_model_json,
    render_prompt,
    reset_model_backend,
    set_model_backend,
    summarize_ticket,
    template_variables,
)

VALID_RESPONSE = (
    '{"account_id": "ACCT-482913", "category": "billing", "urgency": 4, '
    '"summary": "Two charges for one subscription. The customer wants one '
    'refunded.", "needs_human": true}'
)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Turn "this tier makes no API calls" into an assertion."""

    def _blocked(*args, **kwargs):
        raise AssertionError(
            "tier 1 opened a network connection; it is supposed to be mocked"
        )

    monkeypatch.setattr(socket, "socket", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)


@pytest.fixture(autouse=True)
def _isolated_backend():
    """No test in this file inherits another test's model backend."""
    reset_model_backend()
    yield
    reset_model_backend()


@pytest.fixture
def prompt() -> dict:
    return load_prompt()


@pytest.fixture
def variables(prompt: dict) -> dict:
    return dict(prompt["golden_cases"][0]["variables"])


def test_template_declares_the_variables_we_think_it_does(prompt: dict) -> None:
    assert template_variables(prompt["user_template"]) == {
        "ticket_id",
        "channel",
        "ticket_body",
    }


def test_render_interpolates_every_variable(prompt: dict, variables: dict) -> None:
    rendered = render_prompt(prompt["user_template"], variables)

    assert "T-1041" in rendered
    assert "ACCT-482913" in rendered
    assert "{" not in rendered  # no placeholder survived


def test_render_fails_loudly_on_a_missing_variable(prompt: dict) -> None:
    with pytest.raises(MissingPromptVariable) as excinfo:
        render_prompt(prompt["user_template"], {"ticket_id": "T-1"})

    # Silently rendering "None" into a prompt is how a prompt bug becomes a
    # data bug three services downstream.
    assert excinfo.value.missing == ["channel", "ticket_body"]


def test_render_leaves_literal_json_braces_alone() -> None:
    template = 'Return {"ok": true} for ticket {ticket_id}.'

    assert render_prompt(template, {"ticket_id": "T-9"}) == (
        'Return {"ok": true} for ticket T-9.'
    )


def test_render_ignores_extra_variables(prompt: dict, variables: dict) -> None:
    rendered = render_prompt(prompt["user_template"], {**variables, "unused": "x"})

    assert "unused" not in rendered
    assert "T-1041" in rendered


def test_token_estimate_tracks_length() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("a" * (CHARS_PER_TOKEN * 10)) == 10
    assert estimate_tokens("a" * (CHARS_PER_TOKEN * 10 + 1)) == 11


def test_token_budget_passes_under_the_limit() -> None:
    assert check_token_budget("a" * 40, max_tokens=100) == 10


def test_token_budget_fails_over_the_limit() -> None:
    with pytest.raises(PromptTooLarge) as excinfo:
        check_token_budget("a" * 40_000, max_tokens=1_200)

    assert excinfo.value.max_tokens == 1_200
    assert excinfo.value.estimated_tokens == 10_000


def test_a_pasted_log_dump_is_rejected_before_it_costs_anything(
    prompt: dict, variables: dict
) -> None:
    # The realistic failure: a customer pastes a 50k character stack trace.
    variables["ticket_body"] = "ERROR at line 1\n" * 4_000

    with pytest.raises(PromptTooLarge):
        build_request(variables, prompt=prompt)


def test_build_request_reports_a_token_estimate(prompt: dict, variables: dict) -> None:
    request = build_request(variables, prompt=prompt)

    assert request.system == prompt["system_prompt"]
    assert "T-1041" in request.user
    assert 0 < request.estimated_tokens <= prompt["max_input_tokens"]


def test_summarize_ticket_parses_a_mocked_response(
    prompt: dict, variables: dict
) -> None:
    backend = Mock(return_value=VALID_RESPONSE)
    set_model_backend(backend)

    result = summarize_ticket(variables, prompt=prompt)

    assert result["account_id"] == "ACCT-482913"
    assert result["needs_human"] is True

    # The mock also lets us assert on what we sent, which a live call makes awkward.
    backend.assert_called_once()
    sent_system, sent_user = backend.call_args.args
    assert "Return a single JSON object" in sent_system
    assert "ACCT-482913" in sent_user


def test_malformed_json_raises_and_keeps_the_raw_text(
    prompt: dict, variables: dict
) -> None:
    # The classic: a chatty preamble in front of otherwise valid JSON.
    set_model_backend(lambda system, user: 'Sure! Here you go:\n{"account_id": "A"}')

    with pytest.raises(MalformedModelResponse) as excinfo:
        summarize_ticket(variables, prompt=prompt)

    # Keeping the raw response on the exception is what makes the CI log useful.
    assert excinfo.value.raw.startswith("Sure!")


def test_truncated_json_raises(prompt: dict, variables: dict) -> None:
    set_model_backend(lambda system, user: '{"account_id": "ACCT-482913", "cat')

    with pytest.raises(MalformedModelResponse):
        summarize_ticket(variables, prompt=prompt)


def test_a_json_array_is_not_an_acceptable_object() -> None:
    with pytest.raises(MalformedModelResponse) as excinfo:
        parse_model_json('[{"account_id": "ACCT-482913"}]')

    assert "expected a JSON object" in str(excinfo.value)


def test_an_empty_response_raises() -> None:
    with pytest.raises(MalformedModelResponse):
        parse_model_json("")


def test_generate_refuses_to_run_without_a_configured_backend() -> None:
    # The default state of this module is "no network", which is why the tier is
    # safe to run anywhere.
    with pytest.raises(ModelBackendNotConfigured):
        generate("system", "user")


def test_the_mocked_backend_is_the_only_thing_called(
    prompt: dict, variables: dict
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_model(system: str, user: str) -> str:
        calls.append((system, user))
        return VALID_RESPONSE

    set_model_backend(fake_model)
    summarize_ticket(variables, prompt=prompt)
    summarize_ticket(variables, prompt=prompt)

    assert len(calls) == 2
