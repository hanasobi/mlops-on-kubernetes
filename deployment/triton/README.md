# Triton Deployment Scripts

Deployment scripts for serving ONNX models on Triton Inference Server. These scripts run as steps in the [Argo Deployment Workflow](../../workflows/triton/README.md) but also work standalone.

## Structure

```
deployment/triton/
  steps/
    fetch_model.py           Step 1: Download model from MLflow
    determine_version.py     Step 2: Calculate next Triton version
    prepare_artifacts.py     Step 3: Create deployment metadata
    upload_to_s3.py          Step 4: Upload to S3 (atomic staging)
    reload_triton.py         Step 5: Load model into Triton
    update_aliases.py        Step 6: Update MLflow aliases
    verify_deployment.py     Step 7: Functional test
  utils/
    mlflow_helpers.py        MLflow Model Registry operations
    s3_helpers.py            S3 bucket operations
    triton_client.py         Triton HTTP API client
```

## How the Steps Connect

Steps 1-5 run sequentially. Steps 6 and 7 run in parallel after Step 5.

```
fetch_model.py           determine_version.py
  |                         |
  | model.onnx              | triton_version.txt
  | mlflow_metadata.json    |
  v                         v
prepare_artifacts.py  <-----+
  |
  | model.onnx + metadata.json
  v
upload_to_s3.py
  |
  | (model now in S3)
  v
reload_triton.py
  |
  | (model loaded in Triton)
  +------------------+
  |                  |
  v                  v
update_aliases.py    verify_deployment.py
(live/previous)      (inference test)
```

## Steps

### Step 1: fetch_model.py

Downloads the ONNX model and metadata from MLflow Model Registry.

- Loads model by alias (`alias:deploy`) or version (`version:5`)
- Handles both single-file and external data ONNX format
- Extracts MLflow metadata (version, run_id, metrics, params)

```bash
python steps/fetch_model.py --model-name=resnet18-imagenette-onnx --source=alias:deploy --output-dir=/output
```

### Step 2: determine_version.py

Scans S3 for existing model versions and calculates the next one.

- Default: `max_existing_version + 1` (starts at 1 for first deployment)
- Supports `--force-version` to override (re-deploy or skip ahead)

```bash
python steps/determine_version.py --model-name=resnet18-imagenette-onnx --bucket=my-triton-models --output-dir=/output
```

### Step 3: prepare_artifacts.py

Creates `metadata.json` — the deployment audit trail connecting MLflow, Triton, and S3.

- Extracts input/output shapes from the ONNX model programmatically
- Includes MLflow version, Triton version, deployment timestamp, and validation metrics
- Copies `model.onnx` to the output directory so all deployment artifacts are in one place

```bash
python steps/prepare_artifacts.py --mlflow-metadata=/input/mlflow_metadata.json --triton-version=/input/triton_version.txt --model-file=/input/model.onnx --output-dir=/output
```

### Step 4: upload_to_s3.py

Uploads model and metadata to S3 using an atomic staging pattern.

The staging pattern prevents partial deployments:
1. Upload to `model_name/version.uploading/` (staging prefix)
2. Copy to `model_name/version/` (final prefix)
3. Delete staging prefix

Verifies upload by checking file count and sizes match.

```bash
python steps/upload_to_s3.py --model-name=resnet18-imagenette-onnx --triton-version=3 --bucket=my-triton-models --artifacts-dir=/input/artifacts --output-dir=/output
```

### Step 5: reload_triton.py

Triggers Triton to load the new model version from S3.

- Refreshes repository index so Triton sees the new version
- Polls readiness every 2s until the model is ready (default timeout: 60s)
- Verifies the expected version was actually loaded

```bash
python steps/reload_triton.py --model-name=resnet18-imagenette-onnx --triton-url=http://triton:8000 --expected-version=3 --output-dir=/output
```

### Step 6: update_aliases.py

Updates MLflow aliases after successful deployment.

- Moves current `live` alias to `previous` (rollback target)
- Sets `live` alias to the newly deployed version
- Supports `--dry-run` to preview changes without making them

```bash
python steps/update_aliases.py --model-name=resnet18-imagenette-onnx --mlflow-version=4 --mlflow-tracking-uri=http://mlflow:80 --output-dir=/output
```

### Step 7: verify_deployment.py

Performs functional testing of the deployed model.

- Generates synthetic input data (random normal distribution, no dataset needed)
- Validates output shape, datatype, and value range (softmax probabilities)
- Reports inference latency in milliseconds

```bash
python steps/verify_deployment.py --model-name=resnet18-imagenette-onnx --triton-url=http://triton:8000 --expected-input-shape="[1,3,224,224]" --expected-output-shape="[1,10]" --output-dir=/output
```

## Utility Modules

### mlflow_helpers.py

`MLflowHelper` class for MLflow Model Registry operations:
- `load_model(name, source)` — Downloads model and metadata, handles external data format
- `update_deployment_aliases(name, version)` — Manages live/previous alias rotation
- `get_model_version_by_alias(name, alias)` — Lookup version by alias

### s3_helpers.py

`S3Client` class for S3 bucket operations:
- `list_model_versions(name)` — Scans for existing version folders
- `upload_file(local, key)` / `download_file(key, local)` — File transfer
- `copy_directory(src, dst)` / `delete_prefix(prefix)` — Staging pattern support

### triton_client.py

`TritonClient` class wrapping the KServe v2 HTTP API:
- `refresh_repository_index()` — Triggers S3 rescan
- `load_model(name)` / `unload_model(name, version)` — Model lifecycle
- `wait_until_ready(name, timeout)` — Polling with timeout
- `infer(name, inputs)` — Inference request

## S3 Model Repository Layout

```
s3://my-triton-models/
  resnet18-imagenette-onnx/
    1/
      model.onnx         ONNX model (version 1)
      metadata.json       Deployment metadata
    2/
      model.onnx
      metadata.json
    3/
      model.onnx
      metadata.json
```

Triton reads this structure directly. Each numbered folder is a model version. The `metadata.json` is not used by Triton but provides traceability for operations.
