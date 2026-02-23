# vLLM Training Scripts

Training scripts for LoRA fine-tuning of Mistral-7B using QLoRA. These scripts run as steps in the [Argo Training Workflow](../../workflows/vllm/README.md) but also work standalone.

## Structure

```
lora/
  config.py              LoRA configurations + training defaults
  train_lora.py          Step 1: QLoRA fine-tuning with MLflow tracking
  quality_gate.py        Step 2: Metric validation + registry promotion
  mlflow_callback.py     Custom MLflow callback for HuggingFace Trainer
  utils.py               Data loading and formatting utilities
```

## How the Scripts Connect

Each script reads from and writes to MLflow. Only small identifiers are passed between Argo steps:

```
train_lora.py          quality_gate.py
-------------          ----------------
Trains LoRA adapter    Checks eval_loss + perplexity
Logs to MLflow         Registers in Model Registry
Outputs: run-id -----> Sets 'candidate' alias
                       Outputs: result (passed/failed)
```

## config.py

Central configuration with predefined LoRA experiment profiles and training defaults. All values can be overridden via CLI arguments in the Argo Workflow.

### LoRA Configurations

| Config | Rank | Alpha | Target Modules | Use Case |
|--------|------|-------|----------------|----------|
| `minimal` | 4 | 8 | q_proj, v_proj | Parameter-efficient baseline |
| `standard` | 8 | 16 | q_proj, k_proj, v_proj, o_proj | Balanced capacity vs efficiency |
| `aggressive` | 16 | 32 | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | High adaptation capacity |
| `high_capacity` | 32 | 64 | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | Maximum capacity (dropout 0.1) |

### Training Defaults

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_model` | `mistralai/Mistral-7B-v0.1` | Base model from HuggingFace |
| `num_epochs` | `1` | Training epochs |
| `learning_rate` | `2e-4` | Optimizer learning rate |
| `batch_size` | `4` | Per-device batch size |
| `gradient_accumulation_steps` | `4` | Effective batch size = batch_size * accumulation |
| `max_seq_length` | `1024` | Maximum sequence length |
| `max_eval_loss` | `1.0` | Quality gate threshold |
| `max_perplexity` | `2.0` | Quality gate threshold |

## train_lora.py

Fine-tunes Mistral-7B using QLoRA (4-bit quantized LoRA) on domain-specific Q&A data.

**Key behaviors:**
- Loads base model with 4-bit NF4 quantization (BitsAndBytes) — fits on a single GPU
- Applies LoRA adapters to specified transformer layers
- Trains only LoRA parameters (< 1% of total model parameters)
- Logs per-epoch metrics to MLflow: `train_loss`, `eval_loss`, `final_perplexity`
- Saves LoRA adapter artifacts (adapter_config.json + adapter_model.safetensors) to MLflow
- Writes the MLflow Run ID to `/tmp/mlflow_run_id` for the next Argo step

**CLI arguments:**
```bash
python train_lora.py \
  --train-file=/data/train.jsonl \
  --val-file=/data/val.jsonl \
  --data-version=1 \
  --lora-config=standard \
  --log-adapter-to-mlflow \
  [--learning-rate=2e-4] \
  [--num-epochs=1] \
  [--batch-size=4] \
  [--max-seq-length=1024]
```

## quality_gate.py

Checks training metrics against thresholds and registers the adapter in the MLflow Model Registry.

**Gate logic:**
1. Reads `final_eval_loss` and `final_perplexity` from the MLflow run
2. Checks both against configurable thresholds (default: eval_loss <= 1.0, perplexity <= 2.0)
3. If passed: registers the adapter as a new version in the `mistral-7b-lora` registry with the `candidate` alias
4. If failed: outputs `RESULT=failed`, workflow stops

**CLI arguments:**
```bash
python quality_gate.py \
  --run-id=<mlflow-run-id> \
  [--max-eval-loss=1.0] \
  [--max-perplexity=2.0]
```

## MLflow Data Layout

After a successful training run:

```
Experiment: vllm-lora-training
  Run: mistral-7b-lora-standard-v1             (train_lora.py)
    Artifacts (S3):  adapter/                    LoRA adapter files
                       adapter_config.json       LoRA configuration
                       adapter_model.safetensors  Adapter weights (~26 MB)
    Metrics (DB):    train_loss, eval_loss, final_perplexity
    Params (DB):     lora_rank, lora_alpha, learning_rate, ...

Model Registry: mistral-7b-lora
  Version N:  alias=candidate                   (set by quality_gate.py)
```

## Local Development

All scripts work locally with a running MLflow server and GPU:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000

# Train (requires GPU)
python lora/train_lora.py \
  --train-file=test_data/train.jsonl \
  --val-file=test_data/val.jsonl \
  --lora-config=minimal \
  --num-epochs=1

# Quality gate
python lora/quality_gate.py --run-id=<id-from-training>
```
