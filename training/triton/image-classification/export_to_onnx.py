# scripts/export_to_onnx.py
"""
ONNX Export Script for champion models.

This script is the bridge between Gate 1 (Training Performance) and
Gate 2 (Technical Correctness). It takes the PyTorch model with the 'champion'
alias and converts it to ONNX format.

Important: The ONNX model is ONLY logged here, NOT registered!
Registration in the ONNX Registry only happens after successful validation
in the next step. This implements the principle "Quality Gate before Registration".

Workflow:
1. Load champion model from PyTorch Registry
2. Convert to ONNX with dynamic batch axes
3. Log ONNX model as artifact in new MLflow Run
4. Create metadata for traceability
5. Output Export Run ID for the subsequent validation step

This script is designed to run in Argo Workflows, but also works
standalone for local development.
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
import mlflow.onnx
from mlflow.tracking import MlflowClient


def load_champion_from_registry(model_name, alias="champion"):
    """
    Loads the champion model from the MLflow PyTorch Model Registry.

    This function uses the 'champion' alias to identify the best trained
    model. The alias acts like a pointer - it always points to the version
    with the best training performance, regardless of the specific version number.

    This makes the export process robust: No matter which version is currently
    the champion (could be version 3, 7, or 15), we always load the right one.

    Args:
        model_name: Name of the PyTorch Registered Model (e.g. "resnet18-imagenette")
        alias: Alias of the version to load (default: "champion")

    Returns:
        Tuple of (pytorch_model, model_version_info)
        - pytorch_model: The loaded PyTorch model (on CPU)
        - model_version_info: MLflow ModelVersion object with all metadata

    Raises:
        SystemExit(1): If no model with this alias exists
    """
    client = MlflowClient()
    
    print(f"Loading PyTorch model '{model_name}' with alias '{alias}'")
    
    try:
        # Get model version info via the alias
        # This gives us access to all metadata: version number, Run ID,
        # creation timestamp, tags, etc.
        model_version = client.get_model_version_by_alias(model_name, alias)
        
        print(f"✅ Found champion model:")
        print(f"   Version: {model_version.version}")
        print(f"   Source Run ID: {model_version.run_id[:8]}...")
        print(f"   Created: {datetime.fromtimestamp(model_version.creation_timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load the actual PyTorch model
        # The URI format is: models:/<name>@<alias>
        # This is the modern MLflow syntax (since MLflow 2.0)
        model_uri = f"models:/{model_name}@{alias}"
        
        # We explicitly load on CPU because:
        # 1. ONNX export works best on CPU
        # 2. We don't want exports to require GPU nodes
        # 3. GPU memory is valuable and should be reserved for training
        pytorch_model = mlflow.pytorch.load_model(
            model_uri,
            map_location=torch.device('cpu')
        )
        
        # Set model to eval mode
        # This is important for layers like BatchNorm and Dropout that
        # behave differently in training vs inference
        pytorch_model.eval()
        
        print(f"✅ Model loaded successfully and set to eval mode")
        
        return pytorch_model, model_version
        
    except Exception as e:
        print(f"❌ ERROR: Failed to load model from registry")
        print(f"   Model name: {model_name}")
        print(f"   Alias: {alias}")
        print(f"   Error: {e}")
        print("")
        print(f"   Make sure:")
        print(f"   1. Training completed successfully")
        print(f"   2. promote.py ran successfully and set '{alias}' alias")
        print(f"   3. Model name matches config.yaml")
        sys.exit(1)


def export_to_onnx(model, output_path, input_shape=(1, 3, 224, 224)):
    """
    Exports a PyTorch model to ONNX format with dynamic axes.

    We use dynamic axes for the batch dimension, which means the exported
    ONNX model can be used with different batch sizes. This is critical for
    Triton's Dynamic Batching feature, which automatically combines multiple
    requests into a single batch.

    Without dynamic axes, the model would only work with the export batch size
    (here 1). With dynamic axes, it works with batch size 1, 8, 16, 32, etc. -
    whatever Triton needs at runtime.

    Args:
        model: The PyTorch model (should already be in eval() mode)
        output_path: File path where the ONNX model should be saved
        input_shape: Shape of the dummy input for tracing (default: [1, 3, 224, 224])

    Returns:
        output_path: The same path (for convenience)

    Side Effects:
        - Creates model.onnx file
        - May create model.onnx.data file (for large models >2GB)

    Technical Notes:
        - Uses torch.onnx.export with tracing (not scripting)
        - Opset version 14 for broad compatibility
        - Constant folding enabled for optimization
    """
    print(f"Exporting PyTorch model to ONNX format")
    print(f"  Input shape: {input_shape}")
    print(f"  Output path: {output_path}")
    
    # Model must explicitly be on CPU for export
    device = torch.device('cpu')
    model = model.to(device)
    
    # Create dummy input for tracing
    # ONNX export works through "tracing" - PyTorch performs a forward
    # pass with this dummy input and records all operations.
    # These recorded operations are then converted to ONNX ops.
    dummy_input = torch.randn(*input_shape, device=device)
    
    # Input and output names for Triton
    # These names appear in the Triton config.pbtxt and in client code
    input_names = ['input']
    output_names = ['output']
    
    # Dynamic axes: batch size can vary
    # The format is: {'tensor_name': {dimension_index: 'symbolic_name'}}
    # Index 0 is the batch dimension in PyTorch (convention)
    dynamic_axes = {
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
    
    try:
        # The critical export step
        # This can take a few seconds, especially for large models
        start_time = time.time()
        
        torch.onnx.export(
            model,                      # The model to export
            dummy_input,                # Dummy input for tracing
            output_path,                # Where to save
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=14,           # ONNX Opset Version (14 is broadly compatible)
            do_constant_folding=True,   # Optimization: compute constants at export time
            export_params=True,         # Include model weights (yes!)
        )
        
        export_duration = time.time() - start_time
        
        print(f"✅ ONNX export successful")
        print(f"   Duration: {export_duration:.2f} seconds")
        
        # Show file sizes
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"   Model size: {file_size_mb:.1f} MB")
        
        # Check if external data file exists
        # Newer PyTorch versions may produce external data format even for
        # small models. For models under 2GB (the protobuf limit), we convert
        # back to single-file format to simplify the deployment pipeline.
        data_file = output_path + ".data"
        if os.path.exists(data_file):
            total_size = os.path.getsize(output_path) + os.path.getsize(data_file)
            if total_size < 2 * 1024 * 1024 * 1024:  # Under 2GB protobuf limit
                print(f"   Converting external data format to single file...")
                import onnx as onnx_lib
                model_proto = onnx_lib.load(output_path, load_external_data=True)
                onnx_lib.save(model_proto, output_path)
                os.remove(data_file)
                file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                print(f"   Model saved as single file ({file_size_mb:.1f} MB)")
            else:
                data_size_mb = os.path.getsize(data_file) / (1024 * 1024)
                print(f"   External data: {data_size_mb:.1f} MB")
                print(f"   (Total: {file_size_mb + data_size_mb:.1f} MB)")
        
        return output_path, export_duration
        
    except Exception as e:
        print(f"❌ ERROR: ONNX export failed")
        print(f"   Error: {e}")
        print("")
        print(f"   Common causes:")
        print(f"   - Model contains operations not supported by ONNX")
        print(f"   - Dynamic control flow (if/else based on tensor values)")
        print(f"   - Custom PyTorch operations without ONNX mapping")
        sys.exit(1)


def create_metadata(pytorch_model_version, export_run_id):
    """
    Creates comprehensive metadata for traceability and debugging.

    This metadata is critical for production operations. If a problem with
    the deployed model arises six months later, you can trace back through
    this metadata:

    - Which training produced this model? (parent_run_id)
    - With which hyperparameters? (training_params)
    - What was the performance during training? (training_metrics)
    - When was it exported? (exported_at)
    - How large is the ONNX model? (onnx_file_size_mb)

    This is like a "birth certificate" for the ONNX model.

    Args:
        pytorch_model_version: MLflow ModelVersion of the PyTorch model
        export_run_id: MLflow Run ID of the current export run

    Returns:
        Dict with all metadata
    """
    client = MlflowClient()
    
    print(f"Creating traceability metadata")
    
    # Get the training run from which the PyTorch model originated
    training_run = client.get_run(pytorch_model_version.run_id)
    training_experiment = client.get_experiment(training_run.info.experiment_id)
    
    # Collect all relevant information
    metadata = {
        # ===== PyTorch Model Info =====
        "pytorch_model_name": pytorch_model_version.name,
        "pytorch_model_version": pytorch_model_version.version,
        "pytorch_model_alias": "champion",
        
        # ===== Training Run Info =====
        "training_run_id": pytorch_model_version.run_id,
        "training_experiment_id": training_run.info.experiment_id,
        "training_experiment_name": training_experiment.name,
        
        # ===== Export Run Info =====
        "export_run_id": export_run_id,
        
        # ===== Timestamps =====
        "model_trained_at": datetime.fromtimestamp(
            training_run.info.start_time / 1000
        ).isoformat(),
        "model_registered_at": datetime.fromtimestamp(
            pytorch_model_version.creation_timestamp / 1000
        ).isoformat(),
        "exported_at": datetime.now().isoformat(),
        
        # ===== Training Performance =====
        # These metrics show how well the model performed during training
        # Important for later analysis: "Does the model have the same
        # performance in production as during training?"
        "training_metrics": dict(training_run.data.metrics),
        
        # ===== Training Configuration =====
        # All hyperparameters that influenced the training
        # Useful for reproducibility and debugging
        "training_params": dict(training_run.data.params),
        
        # ===== Training Tags =====
        # Contains info like git commit, user, etc. (if set)
        "training_tags": dict(training_run.data.tags),
        
        # ===== Export Configuration =====
        "export_format": "onnx",
        "onnx_opset_version": 14,
        "onnx_dynamic_axes": True,
        "onnx_input_shape": [1, 3, 224, 224],
    }
    
    print(f"✅ Metadata created with {len(metadata)} fields")
    
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Export champion model from PyTorch to ONNX format",
        epilog="This script runs between Gate 1 and Gate 2"
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Name of the PyTorch registered model (e.g. 'resnet18-imagenette')"
    )
    parser.add_argument(
        "--alias",
        default="champion",
        help="Model alias to export (default: 'champion')"
    )
    parser.add_argument(
        "--input-shape",
        default="1,3,224,224",
        help="Input shape for ONNX export, comma-separated (default: 1,3,224,224)"
    )
    
    args = parser.parse_args()
    
    # Parse input shape from string to tuple
    input_shape = tuple(map(int, args.input_shape.split(',')))
    
    # MLflow Tracking URI Setup
    # In Argo this comes from the environment variable
    # Locally it can also come from ~/.mlflow or the environment
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment("model-exports")
    
    print("=" * 70)
    print("ONNX EXPORT")
    print("=" * 70)
    print(f"PyTorch Model: {args.model_name}")
    print(f"Alias: {args.alias}")
    print(f"Input Shape: {input_shape}")
    print(f"MLflow Tracking URI: {tracking_uri}")
    print("-" * 70)
    
    # ===== STEP 1: LOAD CHAMPION =====
    pytorch_model, pytorch_model_version = load_champion_from_registry(
        args.model_name, 
        args.alias
    )
    
    print("-" * 70)
    
    # ===== STEP 2: START EXPORT RUN =====
    # The ONNX model and all metadata are stored in a NEW MLflow run
    # This is conceptually clean: export is a separate process from training
    
    export_run_name = f"export-{args.model_name}-v{pytorch_model_version.version}"
    
    print(f"Starting new MLflow run for export: {export_run_name}")
    
    with mlflow.start_run(run_name=export_run_name) as export_run:
        
        # Set tags for parent-child relationship
        # These tags make it possible to find the training run later
        mlflow.set_tag("parent_run_id", pytorch_model_version.run_id)
        mlflow.set_tag("parent_model_name", args.model_name)
        mlflow.set_tag("parent_model_version", pytorch_model_version.version)
        mlflow.set_tag("parent_model_alias", args.alias)
        mlflow.set_tag("export_format", "onnx")
        mlflow.set_tag("export_stage", "between_gate1_and_gate2")
        
        print(f"✅ Export run started: {export_run.info.run_id[:8]}...")
        print(f"   Parent training run: {pytorch_model_version.run_id[:8]}...")
        
        print("-" * 70)
        
        # ===== STEP 3: PERFORM ONNX EXPORT =====
        # Temporary directory for ONNX files
        temp_dir = Path("/tmp/onnx_export")
        temp_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = temp_dir / "model.onnx"
        
        export_path, export_duration = export_to_onnx(
            pytorch_model,
            str(onnx_path),
            input_shape=input_shape
        )
        
        # Log export parameters
        mlflow.log_param("input_shape", str(input_shape))
        mlflow.log_param("opset_version", 14)
        mlflow.log_param("dynamic_axes", True)
        
        # Log export metrics
        mlflow.log_metric("export_duration_seconds", export_duration)
        
        file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        mlflow.log_metric("onnx_file_size_mb", file_size_mb)
        
        print("-" * 70)
        
        # ===== STEP 4: LOG ONNX MODEL TO MLFLOW =====
        # IMPORTANT: We use mlflow.onnx.log_model instead of log_artifact
        # This stores the model in the correct format for later loading
        
        print("Logging ONNX model to MLflow...")
        
        # Log the ONNX model file as artifact
        # artifact_path="model" means it ends up in the "model/" subdirectory
        mlflow.log_artifact(str(onnx_path), artifact_path="model")
        
        # Check if an external data file exists and log it too
        data_file = str(onnx_path) + ".data"
        if os.path.exists(data_file):
            mlflow.log_artifact(data_file, artifact_path="model")
            print(f"   Logged model.onnx and model.onnx.data")
        else:
            print(f"   Logged model.onnx")

        print(f"✅ ONNX model logged as artifact")
        print(f"   Artifact path: model/")
        print(f"   NOT registered yet (pending validation)")
        
        print("-" * 70)
        
        # ===== STEP 5: CREATE METADATA =====
        metadata = create_metadata(pytorch_model_version, export_run.info.run_id)
        
        # Add export-specific info
        metadata["onnx_file_size_mb"] = file_size_mb
        metadata["export_duration_seconds"] = export_duration
        
        # Log metadata as JSON artifact
        mlflow.log_dict(metadata, "metadata.json")
        
        print(f"✅ Metadata logged")
        
        print("-" * 70)
        
        # ===== STEP 6: OUTPUTS FOR ARGO =====
        export_run_id = export_run.info.run_id
        
        # Write Run ID to file for Argo
        # The next step (validate_onnx.py) needs this ID to load the
        # ONNX model from MLflow
        with open("/tmp/export_run_id", 'w') as f:
            f.write(export_run_id)
        
        print(f"Export run ID written to /tmp/export_run_id")
        
        # For Argo parsing
        print("")
        print(f"EXPORT_RUN_ID={export_run_id}")
        
    # Run is now completed (exited with block)
    
    print("=" * 70)
    print("✅ EXPORT COMPLETE")
    print("=" * 70)
    print(f"   ONNX Model: logged in MLflow run {export_run_id[:8]}...")
    print(f"   Status: Awaiting validation (Gate 2)")
    print(f"   Next: Run validate_onnx.py with --export-run-id {export_run_id[:8]}...")
    print("=" * 70)


if __name__ == "__main__":
    main()