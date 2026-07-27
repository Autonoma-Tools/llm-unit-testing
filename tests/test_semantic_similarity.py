"""Semantic similarity assertions, with the threshold chosen empirically.

Use this when a correct answer has no fixed wording. The interesting part is not
the cosine formula, it is where the threshold comes from: calibrate_threshold()
scores a set of known-good answers against the reference and puts the floor just
below the worst true positive. Picking 0.8 because it looks like a reasonable
number is how semantic tests become the flakiest tests in a suite.

Run it:            pytest tests/test_semantic_similarity.py -v
See the numbers:   pytest tests/test_semantic_similarity.py -s -k calibration
"""

from __future__ import annotations

import hashlib
import re
import statistics
from pathlib import Path

import numpy as np
import pytest
import yaml

PROMPT_PATH = (
    Path(__file__).resolve().parents[1] / "prompts" / "summarize_ticket" / "v3.yaml"
)

# Margin below the worst known-good score. Small enough that a real regression
# still fails, wide enough to absorb ordinary embedding noise.
CALIBRATION_MARGIN = 0.05

EMBEDDING_DIM = 512

_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    """
    a an and are as at be been but by can for from had has have how in into is it
    its of on only or that the their them there they this to was were what when
    which who will with would you your
    """.split()
)


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in _WORD.findall(text.lower())
        if len(token) > 2 and token not in _STOPWORDS
    ]


def get_embedding(text: str) -> np.ndarray:
    """Offline stand-in for an embeddings API.

    Signed hashing of content words into a fixed-width vector. It is not a
    language model, but it is deterministic, needs no credentials, and behaves
    like an embedding for the purpose of this test: paraphrases of the same
    answer land near each other, unrelated text lands near zero.

    Replace the body with a real call, for example:

        from openai import OpenAI
        client = OpenAI()
        vector = client.embeddings.create(
            model="text-embedding-3-small", input=text
        ).data[0].embedding
        return np.asarray(vector, dtype=np.float64)

    The rest of this file, including the calibration routine, works unchanged.
    """
    vector = np.zeros(EMBEDDING_DIM, dtype=np.float64)
    for token in _tokens(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % EMBEDDING_DIM
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign

    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(left, right) / denominator)


def similarity(candidate: str, reference: str) -> float:
    return cosine_similarity(get_embedding(candidate), get_embedding(reference))


def calibrate_threshold(
    reference: str,
    known_good: list[str],
    margin: float = CALIBRATION_MARGIN,
    label: str = "",
) -> float:
    """Derive a threshold from data instead of guessing one.

    Scores every known-good answer against the reference, prints the distribution
    so a human can look at it, and returns a floor just below the worst true
    positive. If that floor lands uncomfortably low, the honest read is that this
    check does not discriminate on this case and you need a different assertion,
    not a nudged number.
    """
    scores = sorted(similarity(answer, reference) for answer in known_good)

    print(f"\ncalibration {label or 'case'}: n={len(scores)}")
    for score, answer in zip(scores, sorted(known_good, key=lambda a: similarity(a, reference))):
        print(f"  {score:.3f}  {answer[:72]}")
    print(
        f"  min={scores[0]:.3f} median={statistics.median(scores):.3f} "
        f"max={scores[-1]:.3f}"
    )

    threshold = round(scores[0] - margin, 3)
    print(f"  threshold = worst true positive - {margin} = {threshold:.3f}")
    return threshold


# Recorded model output, so this file runs with no credentials and no network.
# Each summary is worded differently from every calibration answer on purpose:
# the point is to score meaning, not overlap with a memorized string.
RECORDED_SUMMARIES = {
    "billing_duplicate_charge": (
        "The customer received two identical subscription charges on one day and "
        "is requesting a refund for the duplicate charge."
    ),
    "bug_login_redirect_loop": (
        "A login redirect loop on Chrome blocks the customer from reaching the "
        "invoices page, while Safari signs in normally."
    ),
    "how_to_change_billing_email": (
        "The customer is asking where to change the email address that receives "
        "the monthly invoice, having found only the login email in settings."
    ),
}

OFF_TOPIC_SUMMARY = (
    "The office cafeteria will be closed on Friday for scheduled maintenance."
)


def load_prompt(path: Path = PROMPT_PATH) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


PROMPT = load_prompt()
GOLDEN_CASES = PROMPT["golden_cases"]
CASE_IDS = [case["id"] for case in GOLDEN_CASES]


@pytest.mark.parametrize("case", GOLDEN_CASES, ids=CASE_IDS)
def test_summary_is_semantically_close_to_the_reference(case: dict) -> None:
    reference = case["reference_summary"]
    threshold = calibrate_threshold(
        reference, case["accepted_summaries"], label=case["id"]
    )

    score = similarity(RECORDED_SUMMARIES[case["id"]], reference)

    assert score >= threshold, (
        f"{case['id']}: similarity {score:.3f} below calibrated threshold "
        f"{threshold:.3f}"
    )


@pytest.mark.parametrize("case", GOLDEN_CASES, ids=CASE_IDS)
def test_an_off_topic_answer_falls_below_the_threshold(case: dict) -> None:
    """A threshold that nothing can fail is not a threshold."""
    reference = case["reference_summary"]
    threshold = calibrate_threshold(
        reference, case["accepted_summaries"], label=case["id"]
    )

    score = similarity(OFF_TOPIC_SUMMARY, reference)

    assert score < threshold, (
        f"{case['id']}: off-topic answer scored {score:.3f}, at or above the "
        f"threshold {threshold:.3f}. The threshold is too low to be useful."
    )


def test_the_threshold_is_derived_from_the_worst_true_positive() -> None:
    case = GOLDEN_CASES[0]
    scores = [
        similarity(answer, case["reference_summary"])
        for answer in case["accepted_summaries"]
    ]
    threshold = calibrate_threshold(
        case["reference_summary"], case["accepted_summaries"], label=case["id"]
    )

    assert threshold == pytest.approx(min(scores) - CALIBRATION_MARGIN, abs=1e-3)
    assert threshold < min(scores)  # every known-good answer still passes
    assert threshold > 0.0  # a non-positive floor would accept anything


def test_identical_text_scores_one() -> None:
    reference = GOLDEN_CASES[0]["reference_summary"]
    assert similarity(reference, reference) == pytest.approx(1.0, abs=1e-9)


def test_similarity_cannot_catch_a_factually_wrong_answer() -> None:
    """The documented ceiling of this assertion type, asserted rather than assumed.

    Same wording, wrong number. Cosine similarity measures closeness in meaning
    space, not truth, so this scores like a correct answer and passes the
    threshold. If a wrong number is the failure you care about, this is the wrong
    test, and no amount of threshold tuning fixes it.
    """
    case = next(c for c in GOLDEN_CASES if c["id"] == "billing_duplicate_charge")
    threshold = calibrate_threshold(
        case["reference_summary"], case["accepted_summaries"], label=case["id"]
    )

    confidently_wrong = (
        "The customer was charged seventeen times for the same monthly "
        "subscription on the same day and is asking for one of the duplicate "
        "charges to be refunded."
    )
    score = similarity(confidently_wrong, case["reference_summary"])

    assert score >= threshold
