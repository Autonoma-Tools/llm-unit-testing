"""Runtime around the summarize_ticket prompt.

Everything here is ordinary Python with no network calls of its own. The model
call is injected, which is what lets the every-commit test tier run in
milliseconds with no credentials (see tests/test_prompt_logic_mocked.py).

To run against a real model, call set_model_backend() with a function that takes
(system, user) and returns the raw model text. tests/test_deepeval_metrics.py
contains a working OpenAI backend you can copy.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import yaml

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"

# Rough characters-per-token ratio for English prose. Deliberately conservative:
# it is a budget guard, not an accounting tool. Swap in your provider's real
# tokenizer if you need exact counts.
CHARS_PER_TOKEN = 4

_PLACEHOLDER = re.compile(r"\{([a-z_][a-z0-9_]*)\}")


class PromptError(Exception):
    """Base class for every failure this module raises."""


class MissingPromptVariable(PromptError):
    """A template placeholder had no matching value."""

    def __init__(self, missing: list[str]) -> None:
        self.missing = sorted(missing)
        super().__init__(
            "missing template variables: " + ", ".join(self.missing)
        )


class PromptTooLarge(PromptError):
    """The rendered prompt exceeded the fixture's declared token budget."""

    def __init__(self, estimated_tokens: int, max_tokens: int) -> None:
        self.estimated_tokens = estimated_tokens
        self.max_tokens = max_tokens
        super().__init__(
            f"rendered prompt is about {estimated_tokens} tokens, "
            f"budget is {max_tokens}"
        )


class MalformedModelResponse(PromptError):
    """The model returned something that was not the JSON object we asked for."""

    def __init__(self, message: str, raw: str = "") -> None:
        self.raw = raw
        super().__init__(message)


class ModelBackendNotConfigured(PromptError):
    """generate() was called before a backend was injected."""


@dataclass(frozen=True)
class RenderedPrompt:
    """The two message bodies a chat model call needs."""

    system: str
    user: str
    estimated_tokens: int


def load_prompt(name: str = "summarize_ticket", version: str = "v3") -> dict:
    """Load a versioned prompt fixture as plain data."""
    path = PROMPTS_DIR / name / f"{version}.yaml"
    with path.open(encoding="utf-8") as handle:
        prompt = yaml.safe_load(handle)

    for key in ("version", "model", "temperature", "system_prompt", "user_template"):
        if key not in prompt:
            raise PromptError(f"prompt fixture {path} is missing required key '{key}'")
    return prompt


def template_variables(template: str) -> set[str]:
    """Every placeholder name a template expects."""
    return set(_PLACEHOLDER.findall(template))


def render_prompt(template: str, variables: dict) -> str:
    """Interpolate a template, failing loudly on a missing variable.

    Uses an explicit placeholder regex rather than str.format so that literal
    JSON braces inside a prompt body are left alone.
    """
    missing = template_variables(template) - set(variables)
    if missing:
        raise MissingPromptVariable(list(missing))
    return _PLACEHOLDER.sub(lambda match: str(variables[match.group(1)]), template)


def estimate_tokens(text: str) -> int:
    """Cheap upper-ish bound on the token count of a string."""
    if not text:
        return 0
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def check_token_budget(text: str, max_tokens: int) -> int:
    """Raise PromptTooLarge if text blows the budget, otherwise return the estimate."""
    estimated = estimate_tokens(text)
    if estimated > max_tokens:
        raise PromptTooLarge(estimated, max_tokens)
    return estimated


def build_request(
    variables: dict,
    prompt: Optional[dict] = None,
    max_input_tokens: Optional[int] = None,
) -> RenderedPrompt:
    """Render both message bodies and enforce the token budget."""
    prompt = prompt or load_prompt()
    system = prompt["system_prompt"]
    user = render_prompt(prompt["user_template"], variables)
    budget = max_input_tokens or prompt.get("max_input_tokens", 1200)
    estimated = check_token_budget(system + user, budget)
    return RenderedPrompt(system=system, user=user, estimated_tokens=estimated)


_backend: Optional[Callable[[str, str], str]] = None


def set_model_backend(backend: Callable[[str, str], str]) -> None:
    """Inject the function that actually calls a model."""
    global _backend
    _backend = backend


def reset_model_backend() -> None:
    """Remove any injected backend. Tests use this to guarantee isolation."""
    global _backend
    _backend = None


def generate(system: str, user: str) -> str:
    """Call the injected backend and return the raw model text."""
    if _backend is None:
        raise ModelBackendNotConfigured(
            "No model backend is configured. Call "
            "summarizer.prompt_runtime.set_model_backend(fn) with a function "
            "that takes (system, user) and returns the raw model text."
        )
    return _backend(system, user)


def parse_model_json(raw: str) -> dict:
    """Parse a model response that is supposed to be a single JSON object."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MalformedModelResponse(
            f"model did not return valid JSON: {exc.msg} at position {exc.pos}",
            raw=raw,
        ) from exc
    if not isinstance(data, dict):
        raise MalformedModelResponse(
            f"model returned a {type(data).__name__}, expected a JSON object",
            raw=raw,
        )
    return data


def summarize_ticket(
    variables: dict,
    prompt: Optional[dict] = None,
    max_input_tokens: Optional[int] = None,
) -> dict:
    """Render, call the backend, and parse. The whole path under test."""
    prompt = prompt or load_prompt()
    request = build_request(variables, prompt=prompt, max_input_tokens=max_input_tokens)
    raw = generate(request.system, request.user)
    return parse_model_json(raw)
