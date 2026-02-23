"""
Refusal detection for negative evaluation samples.

Deterministic pattern-matching check to detect whether a model answer
correctly refuses to answer an unanswerable question. Ported from
tmp/evaluate_multi_lora.py.
"""

REFUSAL_PATTERNS = [
    "cannot answer",
    "can't answer",
    "unable to answer",
    "not able to answer",
    "does not contain",
    "does not provide",
    "insufficient information",
    "not enough information",
    "not provided",
    "not mentioned",
]


def is_refusal(answer: str) -> bool:
    """
    Check if an answer is a refusal (model correctly declines to answer).

    Looks for common refusal patterns in the answer text. Used for
    negative evaluation samples where the model should refuse because
    the question is unanswerable from the provided context.

    Args:
        answer: Model-generated answer text

    Returns:
        True if the answer contains a refusal pattern
    """
    answer_lower = answer.lower()
    return any(pattern in answer_lower for pattern in REFUSAL_PATTERNS)
