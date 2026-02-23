# mlops-on-kubernetes

End-to-end ML pipelines on Kubernetes — from training to production deployment. This repository contains working implementations across two serving platforms (Triton Inference Server, vLLM), covering deep learning and LLM fine-tuning workflows.

This is not a reference architecture. It's a practitioner's collection of working code, Argo Workflows, ArgoCD configurations, and Kubernetes manifests — built and tested on a real cluster. Each component has its own README documenting what it does, how to run it, and where it fits in the bigger picture.

## Tech Stack

- **Orchestration:** Argo Workflows (pipeline automation), ArgoCD (GitOps deployments)
- **Experiment Tracking:** MLflow (self-hosted, with Model Registry for alias-based promotion)
- **Serving:** Triton Inference Server (ONNX), vLLM (LLMs with Multi-LoRA)
- **Infrastructure:** Kubernetes (EKS), GPU nodes (NVIDIA L4/T4), S3 (model artifacts)
- **Training:** PyTorch, HuggingFace Transformers, QLoRA, ONNX export
- **CI/CD:** GitHub Actions (container image builds), GitHub OIDC → AWS (no static credentials)

## How the Pieces Fit Together

There are two pipelines in this repo, each targeting a different serving platform and ML paradigm. They share the same infrastructure patterns (Kubernetes, Argo, MLflow) but differ in model types, deployment strategies, and levels of automation.

Both pipelines are **fully automated** and use **MLflow Model Registry as a promotion gate** between stages. No model reaches production without passing quality checks — the alias strategy ensures that training, deployment, and evaluation are loosely coupled but connected through a shared contract.

### Pipeline 1: Triton — Deep Learning with ONNX

**What it does:** Trains an image classification model (ResNet-18 on Imagenette) in PyTorch, exports it to ONNX, validates the export, and deploys it to Triton Inference Server.

**Automation level:** Fully automated. Training and deployment run as separate Argo Workflows, connected through MLflow aliases.

**Training Workflow** (4 steps, ~24 min on T4 GPU):
1. Train ResNet-18 on ImageNette with MLflow tracking (`train.py`)
2. **Quality Gate 1:** Compare candidate vs. current champion — must beat `best_val_accuracy >= 0.90` AND show minimum improvement. If passed: `champion` alias set (`promote.py`)
3. Export champion to ONNX (`export_to_onnx.py`)
4. **Quality Gate 2:** Numerical validation — run 100 samples through both PyTorch and ONNX, outputs must match within tolerance (`1e-5`). If passed: register in ONNX registry with `deploy` alias (`validate_onnx.py`)

**Deployment Workflow** (7 steps, ~5 min):
1. Fetch ONNX model from MLflow (`alias:deploy`)
2. Scan S3 for existing versions, calculate next version number
3. Create deployment metadata (MLflow/Triton traceability)
4. Upload to S3 model repository using staging pattern (atomic)
5. Reload Triton — refresh repo index, load model, poll until ready
6. Update MLflow aliases — new version gets `live`, previous gets `previous` (rollback target)
7. Verify deployment with functional test (synthetic input, validate output shape + probabilities)

**MLflow aliases:** `champion` → `deploy` → `live` / `previous`

### Pipeline 2: vLLM — LLM Fine-Tuning with LoRA

**What it does:** Fine-tunes a base LLM (Mistral-7B) using QLoRA for domain-specific tasks (RAG-based Q&A on AWS documentation). Deploys LoRA adapters on vLLM with runtime hot-loading and a 2-slot architecture for zero-downtime deployments. Evaluates adapter quality using an automated LLM-as-Judge pipeline.

**Automation level:** Fully automated. Five separate Argo Workflows handle training, deployment, evaluation, performance testing, and promotion — connected through MLflow aliases.

**Training Workflow** (3 steps, ~4h on T4 GPU):
1. Download training data from S3 (versioned)
2. QLoRA fine-tuning on GPU with MLflow tracking (`train_lora.py`)
3. **Quality Gate:** Check `eval_loss <= 1.0` and `perplexity <= 2.0`. If passed: register adapter in MLflow Model Registry with `candidate` alias (`quality_gate.py`)

**Deployment Workflow** (3 steps, ~2 min):
1. Download adapter from MLflow (`alias:candidate`) → `kubectl cp` to vLLM pod
2. Load adapter via vLLM runtime API (seconds, no pod restart) → smoke test
3. Set `staged` alias in MLflow — adapter is deployed and verified, ready for evaluation

**Evaluation Workflow — Gate 1** (5 steps, ~50 min with Llama-70B judge):
1. Download evaluation dataset from S3 (1200 samples, unseen during training)
2. Run parallel inference: candidate adapter + live adapter on same eval set
3. Send all answers to Judge LLM for A/B/C grading
4. **3-check decision:** A-Rate (candidate >= live), C-Rate (candidate <= live), Refusal Rate (>= 90% on negative samples). All three must pass.
5. Set `eval-passed` or `eval-failed` alias in MLflow

**Performance Workflow — Gate 2** (3 steps, ~6 min):
1. Download evaluation dataset from S3 (same data as Gate 1)
2. Run concurrent load test (10 parallel requests), collect server-side metrics from Prometheus (P50/P95/P99 latency, TTFT, throughput, KV-cache, GPU utilization)
3. **4-check decision:** P95 Latency (< 8s), P95 TTFT (< 2s), Throughput (> 2 req/s), Success Rate (> 99%). Each check has absolute floors + relative comparison against the last passed baseline from MLflow. Set `perf-passed` or `perf-failed` alias.

**Promote Workflow** (1 step, ~1 min):
1. Download adapter from MLflow (`alias:perf-passed`), copy to live slot, reload via runtime API, update MLflow aliases (`live` + `previous`), clean up all intermediate aliases

**MLflow aliases:** `candidate` → `staged` → `eval-passed` → `perf-passed` → `live` / `previous`

**Deep documentation:** This pipeline is covered in depth in the companion blog series [Self-Hosted LLMs für Datensouveränität](https://hanasobi.github.io/self-hosted-llms-tutorial/) (German, 10 posts).

## Design Decisions

**MLflow as single source of truth.** Both pipelines use MLflow Model Registry not just for experiment tracking, but as the coordination layer between workflows. Aliases (`champion`, `deploy`, `candidate`, `staged`, `live`) act as contracts: the deployment workflow doesn't need to know how the training workflow works — it just reads the model with `alias:deploy` (Triton) or `alias:candidate` (vLLM). This decouples the workflows while keeping them connected through a shared promotion lifecycle.

**Training and deployment images are separate.** Training images include GPU-enabled PyTorch and are 3-4 GB. Deployment images contain only lightweight HTTP clients, MLflow SDK, and deployment logic — under 1 GB. This separation avoids bloated images, allows independent versioning, and reflects the reality that training and deployment have fundamentally different dependency profiles.

**Runtime LoRA loading instead of pod restarts.** The vLLM pipeline loads and unloads LoRA adapters via the vLLM runtime API — taking seconds instead of the 8+ minutes a full pod restart requires (including GPU node scaling and model loading). This enables the 2-slot architecture (`aws-rag-qa-live` + `aws-rag-qa-candidate`) where a new adapter can be loaded, tested, and evaluated without disrupting the live adapter.

**Quality gates stop bad models early.** The Triton pipeline has two gates (training metrics AND numerical ONNX validation). The vLLM pipeline has four stages of validation (training metrics, deployment smoke test, LLM-as-Judge evaluation, performance load test). A model that fails any gate doesn't advance — no manual intervention, no "let's deploy and see".

**GitHub OIDC for AWS authentication.** GitHub Actions authenticate to AWS via OpenID Connect — no static credentials stored anywhere. The OIDC token is exchanged for short-lived AWS credentials at runtime. Same principle on the cluster side: Argo Workflows use IRSA (IAM Roles for Service Accounts) for S3 and ECR access.

**Kustomize base + overlays for environment variants.** Triton uses a shared base with CPU and GPU overlays — same deployment, different resource profiles. This avoids manifest duplication and makes it easy to add new variants (e.g., a high-memory GPU overlay for larger models).

## Project Structure

```
mlops-on-kubernetes/
├── .github/workflows/          # GitHub Actions: container image builds
│   ├── triton-build-training-image.yaml
│   ├── triton-build-deployment-image.yaml
│   ├── vllm-build-training-image.yaml
│   └── vllm-build-deployment-image.yaml
│
├── training/                   # Training scripts and configs
│   ├── triton/                 #   Deep learning models (ONNX)
│   │   └── image-classification/
│   │       ├── train.py
│   │       ├── export_to_onnx.py
│   │       ├── validate_onnx.py
│   │       ├── promote.py
│   │       └── config.yaml
│   └── vllm/                   #   LLM fine-tuning (QLoRA)
│       └── lora/
│           ├── train_lora.py
│           ├── quality_gate.py
│           └── config.py
│
├── deployment/                 # Deployment scripts (run inside Argo Workflows)
│   ├── triton/                 #   Triton model delivery + smoke tests
│   └── vllm/                   #   vLLM adapter loading + smoke tests
│       ├── steps/
│       │   ├── load_adapter.py
│       │   ├── smoke_test.py
│       │   └── set_staged_alias.py
│       └── utils/
│
├── evaluation/                 # Evaluation scripts (LLM-as-Judge)
│   └── vllm/
│       ├── steps/
│       │   ├── run_inference.py
│       │   ├── run_judge.py
│       │   └── compare_decision.py
│       └── eval_utils/
│
├── performance/                # Performance load test scripts
│   └── vllm/
│       └── steps/
│           ├── run_load_test.py
│           └── perf_decision.py
│
├── workflows/                  # Argo Workflow definitions
│   ├── triton/
│   │   ├── train-model-image-classification.yaml
│   │   └── deploy-model-image-classification.yaml
│   └── vllm/
│       ├── train-validate-lora.yaml
│       ├── deploy-lora-adapter.yaml
│       ├── evaluate-lora-adapter.yaml
│       ├── performance-test-lora.yaml
│       └── promote-lora-adapter.yaml
│
├── docker/                     # Dockerfiles for container images
│   ├── triton/
│   └── vllm/
│
└── infra/                      # Kubernetes manifests and GitOps configs
    ├── argocd/
    │   ├── triton/
    │   └── vllm/
    └── k8s/
        ├── triton/             #   Kustomize base + overlays (cpu/gpu)
        ├── vllm/               #   Deployment, init-container, ConfigMap
        └── shared/             #   ServiceAccount, RBAC for Argo Workflows
```

### Why this structure?

The top-level directories represent **functional responsibilities** (training, deployment, evaluation, workflows, infra). Within each, subdirectories are organized by **serving platform** (triton, vllm). This means you can trace a complete pipeline by following one platform across the directories:

```
vLLM pipeline:
  training/vllm/ → workflows/vllm/ → deployment/vllm/ → evaluation/vllm/ → infra/k8s/vllm/
```

Training and deployment scripts are separated into different directories (and different container images) intentionally. Training images need GPU-enabled PyTorch and are several GB in size. Deployment images only need lightweight HTTP clients and model delivery logic. Keeping them separate avoids bloated images and allows independent versioning.

## Container Images

GitHub Actions build container images for each pipeline, pushed to ECR (`123456789012.dkr.ecr.eu-central-1.amazonaws.com`):

| Image | Contents | Used by |
|-------|----------|---------|
| `triton-training-tools` | PyTorch, torchvision, ONNX, MLflow | Triton training workflows |
| `mlops-deployment-tools` | Deployment scripts, ONNX, MLflow, boto3 | Triton deployment workflows |
| `vllm-training-tools` | PyTorch, transformers, peft, bitsandbytes, MLflow | vLLM training workflows |
| `vllm-deployment-tools` | MLflow, boto3, requests, kubectl | vLLM deployment, evaluation, performance, and promote workflows, init-container |

## Roadmap

- **End-to-end workflow:** Chain training, deployment, evaluation, performance, and promote into a single Argo workflow with conditional gates
- **Grafana alerts:** Production alerting for performance regression, model drift, and resource saturation
- **Canary deployment:** Gradual traffic shifting with Argo Rollouts instead of instant adapter swap

## Related

- [Self-Hosted LLMs für Datensouveränität](https://hanasobi.github.io/self-hosted-llms-tutorial/) — A 10-part German blog series documenting the complete LLM pipeline in depth: from first deployment to data-sovereign fine-tuning, evaluation, and Multi-LoRA serving. The vLLM pipeline in this repo is the code companion to that series.