# Infrastructure

Kubernetes manifests and ArgoCD application definitions for deploying ML inference services on EKS.

## Directory Structure

```
infra/
├── k8s/                          Kubernetes manifests
│   ├── triton/                   Triton Inference Server
│   │   ├── base/                 Shared resources (Kustomize base)
│   │   │   ├── deployment.yaml
│   │   │   ├── service.yaml
│   │   │   ├── serviceaccount.yaml
│   │   │   └── kustomization.yaml
│   │   └── overlays/
│   │       ├── cpu/              CPU-only variant (2 replicas)
│   │       └── gpu/              GPU variant (1 replica, nvidia.com/gpu)
│   ├── vllm/                     vLLM Inference Server
│   │   ├── base/                 Core vLLM (always deployed)
│   │   │   ├── deployment.yaml                Deployment with init-container + LoRA support
│   │   │   ├── service.yaml
│   │   │   ├── serviceaccount.yaml
│   │   │   ├── configmap.yaml                 Startup script with conditional --lora-modules
│   │   │   ├── networkpolicy.yaml
│   │   │   ├── servicemonitor.yaml
│   │   │   └── kustomization.yaml
│   │   └── overlays/
│   │       ├── judge/            Base + small Judge (Mistral-7B, 1x T4)
│   │       │   ├── deployment-judge.yaml
│   │       │   ├── service-judge.yaml
│   │       │   └── kustomization.yaml
│   │       └── judge-large/      Base + large Judge (Llama-70B, 4x A10G)
│   │           ├── deployment-judge-large.yaml
│   │           ├── service-judge.yaml
│   │           └── kustomization.yaml
│   └── shared/                   Shared resources (cross-stack)
│       ├── workflow-sa.yaml      Argo workflow ServiceAccount (IRSA)
│       └── workflow-sa-rbac.yaml RBAC for pod exec, deployments
│
└── argocd/                       ArgoCD Application definitions
    ├── triton/
    │   └── triton-inference-cpu.yaml
    └── vllm/
        ├── vllm-inference.yaml               without judge
        ├── vllm-inference-judge.yaml         with small judge (T4 - Mistral-7B)
        └── vllm-inference-judge-large.yaml   with large judge (4xA10 - Llama-70B)
```

## Triton Inference Server

### Kustomize Pattern

The Triton deployment uses a **base + overlays** pattern. The base defines the core resources, overlays patch hardware-specific settings:

| Component | Base | CPU Overlay | GPU Overlay |
|-----------|------|-------------|-------------|
| Replicas | 2 | 2 | 1 |
| CPU requests/limits | 2/4 | 2/4 | 4/8 |
| Memory requests/limits | 4Gi/8Gi | 4Gi/8Gi | 8Gi/16Gi |
| GPU | — | — | 1x `nvidia.com/gpu` |
| Node selector | — | `workload: cpu-compute` | `workload: gpu` |
| Tolerations | — | `cpu-compute:NoSchedule` | — |

### Base Resources

**Deployment** (`base/deployment.yaml`)
- Image: `nvcr.io/nvidia/tritonserver:25.11-py3`
- Model repository: `s3://my-triton-models`
- Model control mode: `explicit` (no auto-loading, models loaded via API)
- Loaded model: `resnet18_imagenette`
- Ports: HTTP (8000), gRPC (8001), Metrics (8002)
- Health checks: Triton v2 Health API (`/v2/health/live`, `/v2/health/ready`)

**Service** (`base/service.yaml`)
- Type: ClusterIP (cluster-internal, use Ingress for external access)
- Exposes all three Triton ports

**ServiceAccount** (`base/serviceaccount.yaml`)
- Annotated with `eks.amazonaws.com/role-arn` for EKS IRSA (IAM Roles for Service Accounts)
- Grants the Triton pod read access to the S3 model repository — no static credentials needed

### Deploying with Kustomize

```bash
# Preview the CPU variant
kubectl kustomize infra/k8s/triton/overlays/cpu

# Apply the CPU variant
kubectl apply -k infra/k8s/triton/overlays/cpu

# Apply the GPU variant
kubectl apply -k infra/k8s/triton/overlays/gpu
```

## ArgoCD GitOps

### How It Works

ArgoCD watches the Git repository and automatically syncs Kubernetes resources when manifests change on `main`:

```
Push to main  →  ArgoCD detects change  →  kubectl apply -k overlays/cpu  →  Pods updated
```

### Triton Application

`argocd/triton/triton-inference-cpu.yaml` defines the ArgoCD Application:

| Setting | Value |
|---------|-------|
| Source path | `infra/k8s/triton/overlays/cpu` |
| Target branch | `main` |
| Destination namespace | `ml-models` |
| Auto-sync | enabled |
| Prune | enabled (removes orphaned resources) |
| Self-heal | enabled (reverts manual drift) |

This means: any change to the Triton Kustomize manifests on `main` is automatically applied to the cluster. Manual `kubectl` edits are reverted.

### Adding a GPU Application

To deploy the GPU variant alongside CPU, create a second ArgoCD Application pointing to `overlays/gpu`:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: triton-inference-gpu
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/your-org/mlops-on-kubernetes
    targetRevision: main
    path: infra/k8s/triton/overlays/gpu
  destination:
    server: https://kubernetes.default.svc
    namespace: ml-models
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
```

## vLLM Inference Server

### Base Resources

**Deployment** (`base/deployment.yaml`)
- Image: `vllm/vllm-openai:v0.14.1-cu130`
- Base model: `TheBloke/Mistral-7B-v0.1-AWQ` (4-bit quantized)
- LoRA support: `--enable-lora --max-loras=2 --max-lora-rank=32`
- Runtime LoRA updating: `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`
- Init-container: loads adapters from MLflow on cold start (uses `vllm-deployment-tools` image)
- Startup script: ConfigMap with conditional `--lora-modules` based on which adapter files exist
- Port: HTTP (8000)
- Health checks: `httpGet /health`
- Node selector: `workload: gpu-vllm`

**ConfigMap** (`base/configmap.yaml`)
- `start.sh` startup script that conditionally registers LoRA adapter slots
- Only adds `--lora-modules` for slots where `adapter_config.json` exists

**ServiceAccount** (`base/serviceaccount.yaml`)
- `vllm-sa` with IRSA annotation for S3 read-only access (for MLflow artifact downloads via boto3)

### Kustomize Pattern

The vLLM deployment uses a **base + overlays** pattern. The base deploys only the vLLM inference server (sufficient for performance testing and promotion). The judge overlays add a separate vLLM instance for LLM-as-Judge evaluation.

```bash
# Base only (vLLM + LoRA, no judge) — for perf testing / promotion
kubectl apply -k infra/k8s/vllm/base/

# Base + small judge (Mistral-7B)
kubectl apply -k infra/k8s/vllm/overlays/judge/

# Base + large judge (Llama-70B)
kubectl apply -k infra/k8s/vllm/overlays/judge-large/
```

**Cost optimization:** The judge is only needed during Gate 1 (Evaluation). For Gate 2 (Performance Load Test) and Promote, the base deployment is sufficient. By separating the judge into overlays, the judge GPU nodes can be scaled down when not in use.

### LLM-as-Judge Deployment

The evaluation pipeline uses a separate vLLM instance as an LLM judge to grade model answers. Two overlay variants are available, reflecting the evolution of this setup.

**Why two variants?**

We started with Mistral-7B (`overlays/judge/`) because our AWS service quotas for `g5.12xlarge` instances had not yet been approved. This allowed us to develop and test the full evaluation workflow end-to-end. However, the quality of Mistral-7B as a judge was insufficient — its grading was inconsistent and unreliable for production use.

Once the quotas were approved, we switched to Llama-3.1-70B-AWQ (`overlays/judge-large/`).

**Lessons learned with the large judge (Llama-70B)**

The transition to Llama-70B was not straightforward. The 70B model with tensor parallelism across 4 GPUs required several iterations to run stable:

- **OOM errors:** The model repeatedly ran out of memory during initial attempts. We had to reduce `--max-model-len` to `4096` and lower `--gpu-memory-utilization` to `0.80` (from the typical `0.90`).
- **`--enforce-eager` considered but rejected:** Disabling CUDA graphs would have freed GPU memory, but at the cost of significant inference performance. We avoided this trade-off.
- **Startup time:** The 70B model takes considerably longer to load. The startup probe allows up to 16 minutes (`failureThreshold: 90 * periodSeconds: 10`) compared to 5 minutes for the 7B variant.

**Observations**

1. **Quality:** The large judge produced significantly better and more consistent grading results. It was able to follow the structured judge prompt reliably, which the smaller model could not.
2. **Performance:** Inference was noticeably slower due to the model size and the synchronization overhead across 4 GPUs (tensor parallelism).

| | Small Judge (Mistral-7B) | Large Judge (Llama-3.1-70B) |
|---|---|---|
| Overlay | `overlays/judge/` | `overlays/judge-large/` |
| Model | `TheBloke/Mistral-7B-Instruct-v0.2-AWQ` | `hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4` |
| GPUs | 1 | 4 (tensor parallelism) |
| Memory | 14 Gi | 100 Gi |
| `--gpu-memory-utilization` | 0.88 | 0.80 |
| `--max-model-len` | 4096 | 4096 |
| Node selector | `gpu-vllm` | `gpu-vllm-large` |
| Startup time (max) | ~5 min | ~16 min |
| Instance type | `g5.xlarge` (1x A10G) | `g5.12xlarge` (4x A10G) |

## Shared Resources

Resources used by both Triton and vLLM pipelines.

**ServiceAccount** (`shared/workflow-sa.yaml`)
- `workflow-sa` used by Argo Workflows
- IRSA annotation for S3 read/write access

**RBAC** (`shared/workflow-sa-rbac.yaml`)
- Allows managing pods (get, list, patch, update, watch)
- Allows exec into pods (for `kubectl cp` during adapter loading)
- Allows reading pod logs
- Allows restarting deployments (get, list, watch, patch)

## AWS Integration

| Resource | Purpose |
|----------|---------|
| S3 bucket `my-triton-models` | Model repository for Triton (ONNX files + config) |
| IAM role `my-triton-s3-read` | Read-only S3 access for Triton pods |
| IAM role `my-vllm-s3-read` | Read-only S3 access for vLLM pods (MLflow artifacts) |
| IAM role `my-ml-models-argo-workflows` | S3 read/write for Argo Workflow steps |
| EKS OIDC provider | Federates Kubernetes ServiceAccounts to IAM roles |

The authentication chain:

```
Pod (ServiceAccount)  →  EKS OIDC  →  STS AssumeRoleWithWebIdentity  →  S3 access
```

No AWS credentials are stored in the cluster. The ServiceAccount annotation is the only configuration needed.

## Namespaces

| Namespace | Purpose |
|-----------|---------|
| `ml-models` | Inference services (Triton, vLLM, MLServer) |
| `argocd` | ArgoCD controller and Application definitions |
| `ai-platform` | MLflow, Argo Workflows, supporting services |
