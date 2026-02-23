"""
Judge prompt template and response parser for LLM-as-Judge evaluation.

Based on the tested prompt in tmp/judge_prompt.txt, extended to handle
negative samples (unanswerable questions where the adapter should refuse).
"""

import json
import re

JUDGE_SYSTEM_PROMPT = """You are a quality evaluator for question-answer pairs generated from documentation chunks. Your task is to assess whether a QA pair is suitable for training data.

EVALUATION CRITERIA:

RATING A (Excellent - Suitable for Training):
- Question is natural and answerable from the chunk
- Answer is factually correct and strictly based on chunk content
- Answer is comprehensive and helpful
- No speculation or interpretation beyond chunk
- No hallucinations (inventing facts)
- Clear and well-formulated
- For NEGATIVE samples (unanswerable questions): Model correctly refuses to answer

RATING B (Good - Minor Issues):
- Generally correct and usable
- Minor issues such as:
  * Slightly awkward phrasing
  * Minor SPECULATION: Logical implications or interpretations not explicitly stated in chunk (but not factually wrong)
  * Minor incompleteness
  * Best practice hints that are reasonable but not in chunk
- Still suitable for training with minor cleanup

CRITICAL B-RATING TRIGGERS:
- Answer uses phrases/terms NOT in chunk (e.g., "runaway generation" when chunk says "generate infinitely")
- Answer adds context/implications beyond chunk (e.g., "particularly useful for X" when chunk doesn't mention use case)
- Answer interprets/paraphrases in ways that add new information

RATING C (Poor - Not Suitable):
- Major factual errors
- HALLUCINATIONS: Invented facts, numbers, dates, or claims that are demonstrably not in chunk
- Answer contradicts chunk content
- Severely incomplete or misleading
- For NEGATIVE samples: Model answers confidently instead of refusing (hallucination)
- For POSITIVE samples: Model refuses to answer when it should have answered (false negative)
- Not suitable for training

SPECULATION vs HALLUCINATION:
- SPECULATION (-> B): Reasonable interpretation or implication (e.g., "This makes deployment easier" when chunk mentions "small file size")
- HALLUCINATION (-> C): Invented facts (e.g., "Released in 2023 by UC Berkeley" when chunk doesn't mention this)

HALLUCINATION CHECK:
- Mark hallucination=true ONLY for Rating C cases where facts are invented
- Speculation without invented facts = hallucination=false (Rating B)
- For negative samples: answering confidently = hallucination=true
- Examples of hallucinations:
  * Specific dates, numbers, names not in chunk
  * Technical details not mentioned in chunk
  * Attributions or sources not in chunk

NEGATIVE SAMPLE RULES:
- If the question CANNOT be answered from the chunk context:
  * Model SHOULD refuse -> Rating A (correct refusal)
  * Model answers confidently with invented information -> Rating C, hallucination=true
  * Model hedges but still provides some made-up info -> Rating C, hallucination=true
- If the question CAN be answered from the chunk context:
  * Model refuses to answer -> Rating C (false negative, not hallucination)
  * Model answers correctly -> Rating A

INSTRUCTIONS:
1. Read the chunk carefully
2. Evaluate if the question is answerable from the chunk
3. Check if the answer is strictly based on chunk content
4. SPECIFICALLY check for B-triggers:
   - Are there phrases/terms in the answer NOT in the chunk?
   - Does the answer add implications/context beyond chunk?
   - Does the answer use synonyms or paraphrases that add new meaning?
5. For negative samples: Did the model correctly refuse?
6. Distinguish between speculation (B) and hallucination (C)
7. Assign rating A, B, or C based on criteria above
8. Mark hallucination flag appropriately
9. Provide brief reasoning for your rating

OUTPUT FORMAT (JSON):
{"rating": "A" | "B" | "C", "hallucination": true | false, "reasoning": "Brief explanation of rating decision (1-2 sentences)"}

IMPORTANT:
- Be strict about chunk-based answers
- Rate A only if answer is excellent and strictly from chunk
- Rate B for speculation/interpretation (not hallucination)
- Rate C for hallucinations or major factual errors
- Always provide clear reasoning

EXAMPLES:
Example 1 - Rating A (positive sample):
Chunk: "Kubernetes pods can be scheduled on specific nodes using node selectors"
Answer: "You can use node selectors to schedule Kubernetes pods on specific nodes"
-> A (strictly from chunk, just rephrased)

Example 2 - Rating B:
Chunk: "Without proper timeout configuration, requests may wait indefinitely"
Answer: "Without timeouts, you'll experience hung requests that never complete"
-> B ("hung requests" is synonym but NOT in chunk - minor speculation)

Example 3 - Rating B:
Chunk: "The algorithm achieves O(n log n) time complexity"
Answer: "This algorithm is efficient with O(n log n) complexity, making it suitable for large datasets"
-> B ("suitable for large datasets" is reasonable implication but NOT in chunk)

Example 4 - Rating C (hallucination):
Chunk: "The service uses Redis for caching"
Answer: "The service was implemented in 2022 using Redis 6.0 for caching with 99.9% uptime"
-> C (invented year, version number, and uptime stats - hallucination)

Example 5 - Rating A (negative sample, correct refusal):
Chunk: "AWS S3 provides object storage with 99.999999999% durability"
Question: "What is the maximum file size for Azure Blob Storage?"
Answer: "I cannot answer this question based on the provided context. The context discusses AWS S3, not Azure Blob Storage."
-> A (correctly refused - question is about Azure, context is about AWS)

Example 6 - Rating C (negative sample, hallucination):
Chunk: "AWS S3 provides object storage with 99.999999999% durability"
Question: "What is the maximum file size for Azure Blob Storage?"
Answer: "Azure Blob Storage supports a maximum file size of 5 TB for block blobs."
-> C (hallucination=true, answered confidently about Azure when context only discusses AWS)"""


def build_judge_prompt(context: str, question: str, answer: str, question_type: str = "factual") -> list:
    """
    Build the judge prompt as a list of chat messages.

    Embeds the system prompt into the user message instead of using a
    separate "system" role. This is a deliberate choice for cross-model
    compatibility — Mistral-Instruct rejects the system role entirely,
    while Llama and other models handle instructions in the user message
    equally well.

    Args:
        context: The documentation chunk
        question: The question asked
        answer: The model-generated answer to evaluate
        question_type: "factual", "conceptual", "negative", etc.

    Returns:
        List of message dicts for chat completion API
    """
    sample_type = "NEGATIVE (unanswerable from context)" if question_type == "negative" else "POSITIVE (answerable from context)"

    user_content = f"""{JUDGE_SYSTEM_PROMPT}

---

Evaluate this QA pair:

SAMPLE TYPE: {sample_type}

CHUNK:
{context}

QUESTION:
{question}

ANSWER:
{answer}

Provide your evaluation as JSON."""

    return [
        {"role": "user", "content": user_content},
    ]


def parse_judge_response(text: str) -> dict:
    """
    Parse the judge's response to extract rating, hallucination flag, and reasoning.

    Uses multiple fallback strategies:
    1. Direct JSON parse
    2. Extract JSON from markdown code blocks
    3. Regex for rating field
    4. Find first A/B/C character

    Args:
        text: Raw response text from the judge LLM

    Returns:
        Dict with 'rating', 'hallucination', and 'reasoning' keys
    """
    text = text.strip()

    # Strategy 1: Direct JSON parse
    try:
        parsed = json.loads(text)
        return _validate_parsed(parsed)
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: Extract JSON from markdown code blocks
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(1))
            return _validate_parsed(parsed)
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 3: Find JSON-like object in text
    json_match = re.search(r'\{[^{}]*"rating"[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            return _validate_parsed(parsed)
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 4: Regex for rating field
    rating_match = re.search(r'"rating"\s*:\s*"([ABC])"', text)
    if rating_match:
        return {
            "rating": rating_match.group(1),
            "hallucination": None,
            "reasoning": f"Partial parse from response: {text[:100]}",
        }

    # Strategy 5: Find first standalone A, B, or C
    for char in ["A", "B", "C"]:
        if re.search(rf'\b{char}\b', text):
            return {
                "rating": char,
                "hallucination": None,
                "reasoning": f"Extracted '{char}' from response: {text[:100]}",
            }

    # Complete failure
    return {
        "rating": "ERROR",
        "hallucination": None,
        "reasoning": f"Parse failed for response: {text[:200]}",
    }


def _validate_parsed(parsed: dict) -> dict:
    """Validate and normalize a parsed judge response."""
    rating = parsed.get("rating", "").upper()
    if rating not in ("A", "B", "C"):
        raise ValueError(f"Invalid rating: {rating}")

    return {
        "rating": rating,
        "hallucination": parsed.get("hallucination"),
        "reasoning": parsed.get("reasoning", ""),
    }
