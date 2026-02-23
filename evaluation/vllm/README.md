# vLLM Evaluation Scripts

Offline evaluation scripts for LoRA adapters using an LLM-as-Judge approach. These scripts run as steps in the [Argo Evaluation Workflow](../../workflows/vllm/README.md) but also work standalone.

## Architecture

The evaluation is a batch process on a fixed dataset — no live traffic involved. A separate Judge LLM grades the adapter's answers, and a deterministic refusal check validates behavior on unanswerable questions.

```
Eval Data (S3)       vLLM Server              Judge vLLM            MLflow
+-----------+     +----------------+       +----------------+    +----------+
| eval.jsonl| --> | live adapter   |--ans->|                |    |          |
| 1200 sam  |     | cand adapter   |--ans->| Judge Instruct |--> |eval-pass |
+-----------+     +----------------+       | (A/B/C grades) |    |eval-fail |
                                           +----------------+    +----------+
```

## Structure

```
evaluation/vllm/
  steps/
    run_inference.py       Step 1: Send eval samples to adapter, collect answers
    run_judge.py           Step 2: Send answers to Judge LLM for A/B/C grading
    compare_decision.py    Step 3: Aggregate scores, compare, set MLflow alias
  eval_utils/
    judge_prompt.py        Judge prompt template + JSON response parser
    refusal.py             Deterministic refusal detection (pattern matching)
```

## How the Steps Connect

```
run_inference.py          run_judge.py            compare_decision.py
(called 2x in parallel)
----------------          ---------------          -------------------
Send eval.jsonl to        Send each answer to      Aggregate A/B/C grades
one adapter via API       Judge LLM for grading    Compare candidate vs live
Detect refusals           Parse JSON ratings       Run 3-check decision
Compute refusal rate      Compute A/B/C rates      Log to MLflow experiment
                                                   Set eval-passed/failed alias
```

## Decision Logic

Three checks must ALL pass for the candidate to be approved:

| # | Check | Comparative (live exists) | Absolute (first deployment) |
|---|-------|---------------------------|----------------------------|
| 1 | **A-Rate** | Candidate >= Live | >= 70% |
| 2 | **C-Rate** | Candidate <= Live | <= 10% |
| 3 | **Refusal Rate** | Candidate >= 90% AND >= Live | >= 90% |

Checks 1 and 2 use **LLM Judge grades** across all samples.
Check 3 uses **deterministic pattern matching** (`is_refusal()`) on negative samples only.

The absolute thresholds serve as a safety net for first deployments when no live adapter exists for comparison.

## Judge Design

### Grading Criteria

The Judge evaluates each answer on a 3-point scale:

- **A (Excellent):** Correct, complete, strictly from context. For negative samples: correctly refuses.
- **B (Minor Issues):** Mostly correct, minor speculation/incompleteness. Not hallucination.
- **C (Poor):** Factual errors, hallucinations, or wrong behavior (answers when should refuse, refuses when should answer).

### Cross-Model Compatibility

The judge prompt is embedded in the `user` message (not sent as a separate `system` role). This is a deliberate design choice — Mistral-Instruct rejects the system role entirely, while Llama and other models handle instructions in the user message equally well. This makes the evaluation workflow model-agnostic.

### Response Parsing

`parse_judge_response()` uses 5 fallback strategies to extract the rating from the judge's response:

1. Direct JSON parse
2. JSON from markdown code blocks
3. Regex for JSON object with rating field
4. Regex for `"rating": "A"` pattern
5. First standalone A/B/C character

Parse failures default to rating C (conservative).

## Refusal Detection

`is_refusal(answer)` uses deterministic pattern matching to detect when the model correctly refuses to answer unanswerable questions. This runs during inference (not during judging) and is independent of the Judge LLM.

Patterns detected: "cannot answer", "can't answer", "unable to answer", "does not contain", "does not provide", "insufficient information", "not enough information", "not provided", "not mentioned".

This matters because smaller Judge models (e.g. Mistral-7B-Instruct) may not reliably detect hallucinations on negative samples. The deterministic refusal check serves as a hard gate regardless of judge quality.

## Steps

### Step 1: run_inference.py

Sends evaluation samples to a single LoRA adapter via the vLLM `/v1/completions` API. Called twice in parallel — once per adapter (candidate and live).

```bash
python steps/run_inference.py \
  --vllm-url=http://vllm-service.ml-models:8000 \
  --adapter-name=aws-rag-qa-candidate \
  --eval-data-path=/data/eval.jsonl \
  --max-tokens=512 \
  --temperature=0.0 \
  --max-concurrent=15 \
  --output-dir=/output
```

**Output:** `inference_results.json` (summary + per-sample results with `is_refusal` flag), `inference_status.txt`

### Step 2: run_judge.py

Takes inference results from both adapters and sends each answer to the Judge LLM for A/B/C grading via `/v1/chat/completions`.

```bash
python steps/run_judge.py \
  --judge-url=http://judge-vllm-service.ml-models:8000 \
  --candidate-results=/input/candidate_results.json \
  --live-results=/input/live_results.json \
  --max-concurrent=15 \
  --max-tokens=128 \
  --output-dir=/output
```

**Output:** `judge_results.json` (per-adapter grades + summaries), `judge_status.txt`

### Step 3: compare_decision.py

Aggregates judge grades and inference metrics, runs the 3-check decision, logs to MLflow, and sets the eval alias.

```bash
MLFLOW_TRACKING_URI=http://mlflow.ai-platform:80 \
python steps/compare_decision.py \
  --judge-results=/input/judge_results.json \
  --candidate-inference=/input/candidate_inference.json \
  --live-inference=/input/live_inference.json \
  --model-name=mistral-7b-lora \
  --min-a-rate=0.70 \
  --max-c-rate=0.10 \
  --min-refusal-rate=0.90 \
  --mlflow-experiment=vllm-lora-evaluation \
  --output-dir=/output
```

**Output:** `eval_decision.json` (decision + all metrics + checks), `eval_status.txt`

## MLflow Experiment Tracking

Each evaluation run creates an MLflow experiment run in `vllm-lora-evaluation`:

| Category | Key | Example |
|----------|-----|---------|
| **Params** | candidate_adapter, live_adapter, judge_model, eval_samples_total, min_a_rate, max_c_rate, min_refusal_rate, mode, refusal_patterns | `aws-rag-qa-candidate`, `comparative` |
| **Metrics** | candidate_a_rate, candidate_c_rate, candidate_refusal_rate, live_a_rate, live_c_rate, live_refusal_rate | `0.8592`, `0.0208`, `0.9750` |
| **Tags** | eval_decision, judge_prompt.hash | `passed`, `36aee9fa` |
| **Dataset** | eval-v{N} (context: evaluation) | digest `d9a73a27`, source `s3://.../eval.jsonl` |
| **Artifacts** | judge_results.json, inference results, judge_prompt.txt | Full results for debugging |

## MLflow Alias Lifecycle

```
Training      Deployment      Gate 1 (Eval)        Gate 2 (Perf)        Promote
========      ==========      =============        =============        =======

candidate --> staged -------> eval-passed -------> perf-passed -------> live
                         \--> eval-failed     \--> perf-failed          previous
```

| Alias | Set By | Meaning |
|-------|--------|---------|
| `candidate` | quality_gate.py | Passed training quality gate |
| `staged` | set_staged_alias.py | Deployed, smoke test passed |
| `eval-passed` | compare_decision.py | Passed all 3 evaluation checks (Gate 1) |
| `eval-failed` | compare_decision.py | Failed at least one evaluation check |
| `perf-passed` | perf_decision.py | Passed all 4 performance checks (Gate 2) |
| `perf-failed` | perf_decision.py | Failed at least one performance check |
| `live` | promote_adapter.py | Serving production traffic |
| `previous` | promote_adapter.py | Rollback target |

Only one of `eval-passed` / `eval-failed` exists at a time. Only one of `perf-passed` / `perf-failed` exists at a time. All intermediate aliases are cleaned up after promotion.

## Judge Model Options

The evaluation workflow is model-agnostic. The judge model is determined by whatever is deployed on the judge vLLM instance.

| Model | GPUs | Strengths | Limitations |
|-------|------|-----------|-------------|
| Mistral-7B-Instruct-AWQ | 1x L4 | Fast, low cost, good JSON output | Weak at detecting hallucinations on negative samples |
| Llama 3.1 70B Instruct AWQ | 4x L4 | Strong reasoning, reliable hallucination detection | Requires GPU quota |

Switch between them by deploying the appropriate ArgoCD Application (or Kustomize overlay):
- `infra/k8s/vllm/overlays/judge/` — Mistral-7B (testing)
- `infra/k8s/vllm/overlays/judge-large/` — Llama 70B (production)

## Local Development

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000

# Inference (requires running vLLM)
python evaluation/vllm/steps/run_inference.py \
  --vllm-url=http://localhost:8000 \
  --adapter-name=aws-rag-qa-candidate \
  --eval-data-path=test_data/eval.jsonl \
  --max-concurrent=5 --output-dir=/tmp/eval

# Judge (requires running judge vLLM)
python evaluation/vllm/steps/run_judge.py \
  --judge-url=http://localhost:8001 \
  --candidate-results=/tmp/eval/inference_results.json \
  --output-dir=/tmp/judge

# Compare
python evaluation/vllm/steps/compare_decision.py \
  --judge-results=/tmp/judge/judge_results.json \
  --candidate-inference=/tmp/eval/inference_results.json \
  --output-dir=/tmp/decision
```
