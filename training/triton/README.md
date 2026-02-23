# Triton Training Scripts

Training scripts for image classification models (ResNet18 on ImageNette). These scripts run as steps in the [Argo Training Workflow](../../workflows/triton/README.md) but also work standalone for local development.

## Scripts

```
image-classification/
  config.yaml          Central configuration (hyperparameters, promotion criteria)
  train.py             Step 1: Train ResNet18 on ImageNette
  promote.py           Step 2: Quality Gate 1 - Champion comparison
  export_to_onnx.py    Step 3: Convert PyTorch to ONNX
  validate_onnx.py     Step 4: Quality Gate 2 - Numerical validation + registration
```

## How the Scripts Connect

Each script reads from and writes to MLflow. The only data passed between Argo steps are small identifiers (Run IDs, result strings):

```
train.py                promote.py              export_to_onnx.py         validate_onnx.py
--------                ----------              -----------------         ----------------
Trains model            Compares candidate      Loads champion from       Loads ONNX + PyTorch
Logs to MLflow          with current champion   PyTorch Registry          from MLflow
Outputs: run-id ----->  Sets 'champion' alias   Exports to ONNX           Compares 100 samples
                        Outputs: result ------> Logs to new MLflow Run    Registers in ONNX Registry
                        (passed/failed)         Outputs: export-run-id -> Sets 'deploy' alias
```

## config.yaml

Central configuration shared by `train.py` and `promote.py`. All values can be overridden via CLI arguments in the Argo Workflow.

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `model` | `name` | `resnet18-imagenette` | Model name in MLflow Registry |
| `model` | `architecture` | `resnet18` | PyTorch model class |
| `model` | `approach` | `baseline` | Training strategy: `baseline`, `feature_extraction`, `fine_tuning` |
| `model` | `pretrained` | `false` | Load ImageNet pretrained weights |
| `model` | `frozen_layers` | `null` | Which layers to freeze: `null`, `backbone` |
| `training` | `epochs` | `10` | Number of training epochs |
| `training` | `learning_rate` | `0.001` | Optimizer learning rate |
| `training` | `batch_size` | `64` | Images per training step |
| `training` | `optimizer` | `adam` | Optimizer: `adam` or `sgd` |
| `data` | `s3_path` | `s3://my-dvc-storage/...` | S3 path to ImageNette dataset |
| `data` | `num_workers` | `4` | DataLoader worker threads |
| `promotion_criteria` | `threshold` | `0.90` | Absolute minimum accuracy |
| `promotion_criteria` | `min_improvement` | `0.005` | Minimum improvement over champion |

## train.py

Trains a ResNet18 model on ImageNette with GPU support and MLflow tracking.

**Key behaviors:**
- Downloads dataset from S3 using `aws s3 sync` (skips if already present)
- Supports three training approaches: baseline (from scratch), feature extraction (frozen backbone), fine-tuning (pretrained, all layers trainable)
- Logs per-epoch metrics to MLflow: `train_loss`, `val_loss`, `val_accuracy`, `best_val_accuracy`, `epoch_time`
- Tracks the best model by validation accuracy, saves it as the `model` artifact
- Writes the MLflow Run ID to `/tmp/mlflow_run_id` for the next Argo step

**CLI arguments:**
```bash
python train.py --config=config.yaml [--epochs=20] [--learning-rate=0.0005] [--batch-size=32]
```

## promote.py

Implements Quality Gate 1: compares the candidate model against the current champion.

**Promotion logic:**
1. If no champion exists (first run): candidate must meet the absolute threshold (default: `best_val_accuracy >= 0.90`)
2. If champion exists: candidate must beat the champion by `min_improvement` (default: `+0.5%`)

**On success:** Registers the model in the PyTorch Registry and sets the `champion` alias. Outputs `RESULT=passed`.

**On failure:** Does nothing to the registry. The current champion remains. Outputs `RESULT=failed`, which causes the Argo Workflow to skip the remaining steps (export + validate).

**CLI arguments:**
```bash
python promote.py --run-id=<mlflow-run-id>
```

## export_to_onnx.py

Converts the champion PyTorch model to ONNX format.

**Key behaviors:**
- Loads the model via `models:/<name>@champion` (alias-based, version-independent)
- Exports with dynamic batch axes for Triton's dynamic batching
- If PyTorch produces external data format (model.onnx + model.onnx.data), converts back to single-file for models under 2GB
- Logs the ONNX artifact in a new MLflow Run (separate from training)
- Creates traceability metadata linking back to the training run
- Does NOT register the model — registration only happens after validation

**CLI arguments:**
```bash
python export_to_onnx.py --model-name=resnet18-imagenette [--alias=champion] [--input-shape=1,3,224,224]
```

## validate_onnx.py

Implements Quality Gate 2: numerical validation of the ONNX export.

**Validation methodology:**
- Generates 100 random input samples (no dataset download needed)
- Runs inference through both PyTorch and ONNX Runtime
- Compares outputs with tolerance `1e-5` (max absolute difference)
- Measures ONNX Runtime speedup vs PyTorch

**On success:** Registers the ONNX model in a separate registry (`<model-name>-onnx`) and sets the `deploy` alias. Adds traceability tags linking back to the parent PyTorch model and validation results.

**On failure:** Does not register. The current `deploy` version remains unchanged.

**CLI arguments:**
```bash
python validate_onnx.py --export-run-id=<mlflow-run-id> [--num-samples=100]
```

## MLflow Data Layout

After a successful training pipeline run, MLflow stores data across two backends:
- **Artifacts** (model files, metadata) → S3 Artifact Store
- **Metrics, Params, Tags** → PostgreSQL Tracking Backend

```
Experiment: imagenette-classification
  Run: resnet18-imagenette                  (train.py)
    Artifacts (S3):  model/                  PyTorch model (best epoch)
    Metrics (DB):    train_loss, val_accuracy, best_val_accuracy, epoch_time
    Params (DB):     training.*, model.*, data.*

Experiment: model-exports
  Run: export-resnet18-imagenette-v9        (export_to_onnx.py)
    Artifacts (S3):  model/model.onnx        ONNX model
    Artifacts (S3):  metadata.json           Traceability metadata
    Metrics (DB):    export_duration_seconds, onnx_file_size_mb
    Metrics (DB):    validation_*            (added by validate_onnx.py)
    Tags (DB):       parent_run_id, parent_model_name, ...
```

## Local Development

All scripts work locally with a running MLflow server:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000

# Train (CPU, uses S3 for data)
python image-classification/train.py --config=image-classification/config.yaml --epochs=2

# Promote (uses last run if no --run-id)
python image-classification/promote.py

# Export
python image-classification/export_to_onnx.py --model-name=resnet18-imagenette

# Validate
python image-classification/validate_onnx.py --export-run-id=<id-from-export>
```
