# vLLM Deployment Scripts

Deployment scripts for loading LoRA adapters on vLLM with runtime hot-loading. These scripts run as steps in the [Argo Deployment Workflow](../../workflows/vllm/README.md) but also work standalone.

## Architecture

vLLM serves a base model (Mistral-7B-v0.1-AWQ) with two fixed LoRA adapter slots for A/B testing and blue/green deployments:

```
MLflow Model Registry                     vLLM Pod
┌──────────────────────┐          ┌──────────────────────────┐
│ mistral-7b-lora      │          │ /mnt/adapters/           │
│   alias: live    ────┼─────────>│   aws-rag-qa-live/       │
│   alias: staged  ────┼─────────>│   aws-rag-qa-candidate/  │
│   alias: candidate   │          │                          │
│   alias: previous    │          │ vLLM API:                │
└──────────────────────┘          │   /v1/load_lora_adapter  │
                                  │   /v1/unload_lora_adapter│
                                  └──────────────────────────┘
```

**Key design:** MLflow is the single source of truth for adapter artifacts. No separate staging bucket. Both runtime loading and cold starts pull directly from MLflow.

## Structure

```
deployment/vllm/
  steps/
    load_adapter.py           Step 1: MLflow → kubectl cp → API load
    smoke_test.py             Step 2: Inference validation
    set_staged_alias.py       Step 3: Set 'staged' alias in MLflow
    init_load_adapters.py     Init-container: restore adapters on cold start
    promote_adapter.py        Standalone: promote candidate → live
    restart_deployment.py     Fallback: rolling restart (legacy)
  utils/
    vllm_client.py            vLLM OpenAI-compatible API client
    mlflow_helpers.py         MLflow Model Registry operations
```

## How the Steps Connect

```
load_adapter.py           smoke_test.py           set_staged_alias.py
---------------           ---------------          -------------------
Download from MLflow      Send test prompts        Set 'staged' alias
kubectl cp to vLLM pod    Check responses          in MLflow after
Unload old adapter        Validate latency         smoke test passes
Load new via API
Verify in /v1/models
```

## Steps

### Step 1: load_adapter.py

Downloads a LoRA adapter from MLflow and loads it into the running vLLM server via the runtime API. No pod restart needed.

**Flow:**
1. Download adapter from MLflow (by alias or version)
2. Find the running vLLM pod via `kubectl get pods`
3. `kubectl cp` adapter files to `/mnt/adapters/<slot-name>/` on the pod
4. `POST /v1/unload_lora_adapter` — remove old version from GPU memory
5. `POST /v1/load_lora_adapter` — load new version
6. Verify adapter appears in `GET /v1/models`

**Requires:** `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` on the vLLM server.

```bash
python steps/load_adapter.py \
  --model-name=mistral-7b-lora \
  --source=alias:candidate \
  --slot-name=aws-rag-qa-candidate \
  --vllm-url=http://vllm-service.ml-models:8000 \
  --output-dir=/output
```

### Step 2: smoke_test.py

Sends test inference requests to the deployed adapter via the vLLM OpenAI-compatible API.

**Checks:**
- Response has choices with non-empty text
- Latency within threshold (default: 30s)
- Uses `/v1/completions` (not `/v1/chat/completions`) because the base model has no chat template

```bash
python steps/smoke_test.py \
  --vllm-url=http://vllm-service.ml-models:8000 \
  --adapter-name=aws-rag-qa-candidate \
  --num-requests=3 \
  --max-latency=30 \
  --output-dir=/output
```

### Step 3: set_staged_alias.py

Sets the `staged` alias on the model version in MLflow after the smoke test passes. This signals that the adapter is deployed and testable.

```bash
python steps/set_staged_alias.py \
  --model-name=mistral-7b-lora \
  --source=alias:candidate \
  --output-dir=/output
```

### Init Container: init_load_adapters.py

Runs as a Kubernetes init-container before vLLM starts. Restores adapters from MLflow on cold starts (pod restart, scaling, node failure).

**Flow:**
1. Query MLflow for `live` alias → download to `/mnt/adapters/aws-rag-qa-live/`
2. Query MLflow for `staged` alias → download to `/mnt/adapters/aws-rag-qa-candidate/`
3. Skip gracefully if no aliases exist (first deployment)

### Standalone: promote_adapter.py

Promotes an adapter to the live slot. Used by the [Promote Workflow](../../workflows/vllm/README.md) after Gate 1 (Evaluation) and Gate 2 (Performance) pass. Supports rollback via `--source-alias=previous`.

**Flow:**
1. Resolve adapter version from MLflow alias (`--source-alias`, default: `staged`)
2. Download adapter from MLflow
3. `kubectl cp` to the live slot on the vLLM pod
4. Unload old live adapter, load new one via runtime API
5. Update MLflow aliases: source version → `live`, old live → `previous`
6. Clean up intermediate aliases (`staged`, `eval-passed`, `perf-passed`, `eval-failed`, `perf-failed`)

```bash
# After full pipeline (Gate 1 + Gate 2)
python steps/promote_adapter.py \
  --model-name=mistral-7b-lora \
  --source-alias=perf-passed \
  --vllm-url=http://vllm-service.ml-models:8000

# Rollback to previous version
python steps/promote_adapter.py \
  --model-name=mistral-7b-lora \
  --source-alias=previous
```

## Utility Modules

### vllm_client.py

`VllmClient` class for the vLLM OpenAI-compatible API:
- `health_check()` — `GET /health`
- `list_models()` — `GET /v1/models`
- `completion(model, prompt)` — `POST /v1/completions` (base model / LoRA inference)
- `chat_completion(model, messages)` — `POST /v1/chat/completions` (instruct/chat models, used by Judge)
- `load_lora_adapter(name, path)` — `POST /v1/load_lora_adapter`
- `unload_lora_adapter(name)` — `POST /v1/unload_lora_adapter`
- `wait_until_healthy(timeout)` — polling with timeout
- `wait_until_model_available(name, timeout)` — polling with timeout

### mlflow_helpers.py

`MLflowHelper` class for MLflow Model Registry operations:
- `download_adapter(name, source, output_dir)` — Downloads adapter artifacts
- `parse_source(name, source)` — Resolves `alias:candidate` or `version:N` to a model URI
- `update_deployment_aliases(name, version)` — Manages live/previous alias rotation
- `set_alias(name, alias, version)` / `delete_alias(name, alias)` — Direct alias management

## MLflow Alias Lifecycle

```
Training       Deployment       Gate 1 (Eval)         Gate 2 (Perf)       Promote
========       ==========       =============         =============       =======

train_lora.py
    |
    v
quality_gate.py
    |
    v
  candidate ──> load_adapter.py
                     |
                     v
                smoke_test.py
                     |
                     v
                set_staged_alias.py
                     |
                     v
               cand + staged ──> compare_decision.py
                                      |           |
                                      v           v
                                 eval-passed  eval-failed
                                      |
                                      v
                                 perf_decision.py
                                      |           |
                                      v           v
                                 perf-passed  perf-failed
                                      |
                                      v
                                 promote_adapter.py
                                      |
                                      v
                                live ──> (serving traffic)
                             previous ──> (rollback target)
```

| Alias | Set By | Meaning |
|-------|--------|---------|
| `candidate` | quality_gate.py | Passed training quality gate, ready for deployment |
| `staged` | set_staged_alias.py | Deployed to candidate slot, smoke test passed |
| `eval-passed` | compare_decision.py | Passed all 3 evaluation checks (Gate 1) |
| `eval-failed` | compare_decision.py | Failed at least one evaluation check |
| `perf-passed` | perf_decision.py | Passed all 4 performance checks (Gate 2) |
| `perf-failed` | perf_decision.py | Failed at least one performance check |
| `live` | promote_adapter.py | Promoted to live slot, serving production traffic |
| `previous` | promote_adapter.py | Previous live version, rollback target |
