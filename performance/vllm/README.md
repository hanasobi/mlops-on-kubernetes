# vLLM Performance Load Test Scripts

Performance load test scripts for validating LoRA adapter latency, throughput, and reliability under concurrent load. These scripts run as steps in the [Argo Performance Workflow](../../workflows/vllm/README.md) but also work standalone.

## Architecture

The performance gate generates concurrent load against the candidate adapter and collects **server-side metrics from Prometheus** (not client-side timing). This eliminates test-system noise and gives accurate measurements of adapter performance as seen by vLLM.

```
Eval Data (S3)       vLLM Server              Prometheus           MLflow
+-----------+     +----------------+       +----------------+    +----------+
| eval.jsonl| --> | candidate      |------>| vLLM metrics   |    |          |
| (prompts) |     | adapter        |       | P95 latency    |--> |perf-pass |
+-----------+     +----------------+       | TTFT, RPS      |    |perf-fail |
                   10 concurrent reqs      +----------------+    +----------+
```

## Structure

```
performance/vllm/
  steps/
    run_load_test.py       Step 1: Generate load, query Prometheus for metrics
    perf_decision.py       Step 2: Evaluate metrics, set MLflow alias
```

## How the Steps Connect

```
run_load_test.py              perf_decision.py
----------------              -------------------
Fire concurrent requests      Load test results JSON
Wait for Prometheus scrape    Fetch baseline from MLflow
Query PromQL for metrics      Run 4-check decision
Write results JSON            Log to MLflow experiment
                              Set perf-passed/failed alias
```

## Decision Logic

Four checks must ALL pass for the candidate to be approved:

| # | Check | Absolute Floor | Relative (vs. Baseline) |
|---|-------|----------------|------------------------|
| 1 | **P95 E2E Latency** | < 8.0s | <= baseline * 1.15 |
| 2 | **P95 TTFT** | < 2.0s | <= baseline * 1.15 |
| 3 | **Throughput** | > 2.0 req/s | >= baseline * 0.85 |
| 4 | **Success Rate** | >= 99% | >= baseline |

In **absolute mode** (first deployment, no baseline): only absolute floors apply.
In **comparative mode** (baseline exists): both absolute AND relative must pass.

The baseline is the most recent MLflow run tagged `perf_decision=passed` in the `vllm-lora-performance` experiment.

## Prometheus Metrics

All performance metrics come from Prometheus via PromQL queries. vLLM exposes metrics via a ServiceMonitor, and Prometheus scrapes them every 30 seconds.

| Metric | PromQL |
|--------|--------|
| P50/P95/P99 E2E Latency | `histogram_quantile(0.xx, rate(vllm:e2e_request_latency_seconds_bucket{job="vllm-service"}[WINDOW]))` |
| P50/P95/P99 TTFT | `histogram_quantile(0.xx, rate(vllm:time_to_first_token_seconds_bucket{job="vllm-service"}[WINDOW]))` |
| Throughput | `sum(rate(vllm:request_success_total{job="vllm-service"}[WINDOW]))` |
| KV-Cache Usage | `vllm:kv_cache_usage_perc{job="vllm-service"}` |
| GPU Utilization | `DCGM_FI_DEV_GPU_UTIL{gpu="0"}` |

The `WINDOW` is computed from the actual test duration (rounded up to the nearest minute, minimum 1m). After load generation completes, the script waits 20 seconds for the Prometheus scrape cycle before querying.

**Note:** vLLM tracks metrics under the base model name, not per LoRA adapter. Since the load test controls the test window and sends all requests to the candidate adapter, the metrics accurately reflect candidate performance.

## Steps

### Step 1: run_load_test.py

Generates concurrent load against the candidate adapter and collects server-side metrics from Prometheus.

**Flow:**
1. Health check vLLM, verify adapter in `/v1/models`
2. Load prompts from eval.jsonl (`prompt_inference` field)
3. Fire all requests via `ThreadPoolExecutor` (default: 10 concurrent)
4. Wait 20s for Prometheus scrape cycle
5. Query Prometheus with computed time window
6. Write results JSON + status file

```bash
python steps/run_load_test.py \
  --vllm-url=http://vllm-service.ml-models:8000 \
  --adapter-name=aws-rag-qa-candidate \
  --eval-data-path=/data/eval.jsonl \
  --concurrency=10 \
  --max-tokens=256 \
  --prometheus-url=http://monitoring-kube-prometheus-prometheus.ai-platform:9090 \
  --output-dir=/output
```

**Output:** `load_test_results.json` (Prometheus metrics + client success/failure counts), `load_test_status.txt`

### Step 2: perf_decision.py

Evaluates load test results against absolute thresholds and relative baselines from MLflow.

**Flow:**
1. Load test results JSON
2. Query MLflow for most recent passed baseline
3. Run 4-check decision (absolute + relative)
4. Log all metrics to MLflow experiment `vllm-lora-performance`
5. Set `perf-passed` or `perf-failed` alias on candidate version

```bash
MLFLOW_TRACKING_URI=http://mlflow.ai-platform:80 \
python steps/perf_decision.py \
  --load-test-results=/input/load_test_results.json \
  --model-name=mistral-7b-lora \
  --max-p95-latency=8.0 \
  --max-p95-ttft=2.0 \
  --min-throughput=2.0 \
  --min-success-rate=0.99 \
  --output-dir=/output
```

**Output:** `perf_decision.json` (decision + all metrics + checks), `perf_status.txt`

## MLflow Experiment Tracking

Each performance test creates an MLflow experiment run in `vllm-lora-performance`:

| Category | Key | Example |
|----------|-----|---------|
| **Params** | model_name, candidate_adapter, candidate_version, concurrency, total_requests, mode, all thresholds, baseline_run_id | `mistral-7b-lora`, `comparative` |
| **Metrics** | p50/p95/p99_latency_s, p50/p95/p99_ttft_s, throughput_rps, client_success_rate, kv_cache_usage, gpu_utilization, wall_clock_duration_s | `4.8913`, `3.6364` |
| **Baseline Metrics** | baseline_p95_latency_s, baseline_p95_ttft_s, baseline_throughput_rps, baseline_client_success_rate | (from previous passed run) |
| **Tag** | perf_decision | `passed` or `failed` |
| **Dataset** | eval-v{N} (context: performance-test) | digest `d9a73a27`, source `s3://.../eval.jsonl` |
| **Artifacts** | load_test_results.json | Full Prometheus metrics |

## MLflow Alias Lifecycle

```
Evaluation           Performance          Promotion
==========           ===========          =========

eval-passed -------> perf-passed -------> live
                \--> perf-failed          previous
```

| Alias | Set By | Meaning |
|-------|--------|---------|
| `eval-passed` | compare_decision.py | Passed evaluation (Gate 1) |
| `perf-passed` | perf_decision.py | Passed performance load test (Gate 2) |
| `perf-failed` | perf_decision.py | Failed at least one performance check |
| `live` | promote_adapter.py | Serving production traffic |
| `previous` | promote_adapter.py | Rollback target |

Only one of `perf-passed` / `perf-failed` exists at a time.
