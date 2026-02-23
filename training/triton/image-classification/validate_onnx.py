# scripts/validate_onnx.py
"""
ONNX Validation & Registration Script.

This script implements Gate 2 of the ML Pipeline: Technical Correctness Check.

It validates that the exported ONNX model is numerically equivalent to the
original PyTorch model. Only upon successful validation is the ONNX model
registered in the registry and given the 'deploy' alias.

This is a critical quality gate: It prevents broken ONNX models from entering
the deployment pipeline. If export or validation fails, the old, functioning
ONNX model with the 'deploy' alias remains intact.

The 'deploy' alias means: "This ONNX model is technically correct and ready
for production deployment." It does NOT mean that it is already deployed -
that comes in Gate 3 (deploy_to_triton.py).

Validation methodology:
- Generates 100 random input samples (no dataset download needed!)
- Runs inference with both models
- Compares outputs with strict tolerance (1e-5)
- Logs detailed statistics for debugging

Registration on success:
- Creates new version in the ONNX Registry
- Sets 'deploy' alias on this version
- Adds traceability tags (parent PyTorch model, validation results)
"""

import os
import sys
import argparse
import time

import numpy as np
import torch
import mlflow
import mlflow.onnx
import mlflow.pytorch
from mlflow.tracking import MlflowClient


def load_models_from_mlflow(export_run_id):
    """
    Loads ONNX and PyTorch models from MLflow for validation.

    This function fetches both models that need to be compared:

    1. ONNX Model: From the export run (the just-exported model)
    2. PyTorch Model: From the parent training run (the original)

    We use the parent-child relationship set in export_to_onnx.py
    to find the training run.

    Args:
        export_run_id: MLflow Run ID of the export run

    Returns:
        Tuple of (onnx_model_path, pytorch_model, parent_info)
        - onnx_model_path: Path to the ONNX model file (for onnxruntime)
        - pytorch_model: The PyTorch model (for comparison inference)
        - parent_info: Dict with parent model info for traceability

    Raises:
        SystemExit(1): If models cannot be loaded
    """
    client = MlflowClient()
    
    print(f"Loading models for validation")
    print(f"  Export Run ID: {export_run_id[:8]}...")
    
    try:
        # Get export run metadata
        export_run = client.get_run(export_run_id)
        
        # Extract parent run info from tags
        # These tags were set by export_to_onnx.py
        parent_run_id = export_run.data.tags.get('parent_run_id')
        parent_model_name = export_run.data.tags.get('parent_model_name')
        parent_model_version = export_run.data.tags.get('parent_model_version')
        
        if not parent_run_id:
            print(f"❌ ERROR: Export run has no 'parent_run_id' tag")
            print(f"   Make sure export_to_onnx.py ran correctly")
            sys.exit(1)
        
        print(f"  Parent Training Run: {parent_run_id[:8]}...")
        print(f"  Parent Model: {parent_model_name} v{parent_model_version}")
        
        print("-" * 70)
        
        # ===== LOAD ONNX MODEL =====
        print("Loading ONNX model from MLflow...")
        
        # The ONNX model was stored as an artifact in the export run
        # We use mlflow.artifacts.download_artifacts to fetch it
        onnx_model_uri = f"runs:/{export_run_id}/model"
        onnx_model_path = mlflow.artifacts.download_artifacts(
            artifact_uri=onnx_model_uri
        )
        
        # onnx_model_path now points to a local directory containing:
        # - model.onnx (the actual model)
        # - model.onnx.data (optional, for large models)
        # - MLmodel (MLflow metadata)
        
        onnx_file = f"{onnx_model_path}/model.onnx"
        print(f"✅ ONNX model downloaded to: {onnx_file}")
        
        # ===== LOAD PYTORCH MODEL =====
        print("Loading PyTorch model from MLflow...")
        
        # We load the PyTorch model from the parent training run
        pytorch_model_uri = f"runs:/{parent_run_id}/model"
        pytorch_model = mlflow.pytorch.load_model(
            pytorch_model_uri,
            map_location='cpu'  # Load on CPU for validation
        )
        
        # Important: Set model to eval() mode
        # This disables Dropout, BatchNorm uses running stats instead of batch stats
        pytorch_model.eval()
        
        print(f"✅ PyTorch model loaded and set to eval mode")
        
        # Parent info dictionary for later use
        parent_info = {
            'run_id': parent_run_id,
            'model_name': parent_model_name,
            'model_version': parent_model_version,
        }
        
        return onnx_file, pytorch_model, parent_info
        
    except Exception as e:
        print(f"❌ ERROR: Failed to load models")
        print(f"   Error: {e}")
        sys.exit(1)


def run_onnx_inference(onnx_model_path, inputs):
    """
    Runs inference with the ONNX model.

    ONNX models are not executed directly like PyTorch models.
    Instead, we need a "runtime" - a tool that interprets and executes
    ONNX models. We use ONNX Runtime (onnxruntime-gpu or onnxruntime,
    depending on availability).

    Args:
        onnx_model_path: Path to the model.onnx file
        inputs: NumPy array with input data [batch, channels, height, width]

    Returns:
        NumPy array with predictions [batch, num_classes]
    """
    import onnxruntime as ort
    
    # Create an inference session
    # This loads the ONNX model and compiles it for the current hardware
    # ort.InferenceSession is the equivalent of "model.eval()" in PyTorch
    session = ort.InferenceSession(onnx_model_path)
    
    # Get input names from the model
    # ONNX models have explicit input names (we used "input")
    input_name = session.get_inputs()[0].name
    
    # Run inference
    # session.run() returns a list of outputs (we only have one)
    outputs = session.run(
        None,  # None means "return all outputs"
        {input_name: inputs}  # Input as dictionary
    )
    
    # Extract the first (and only) output
    return outputs[0]


def validate_models(onnx_model_path, pytorch_model, num_samples=100):
    """
    Validates that ONNX and PyTorch models are numerically equivalent.

    This function is the core of Gate 2. It generates random inputs,
    runs inference with both models, and compares the outputs.

    Why random inputs instead of a real dataset?
    1. No dataset download needed (saves time and network)
    2. Random inputs test the model across the entire input space
    3. 100 samples are sufficient to detect numerical issues

    The validation is "strict": We only allow minimal differences (1e-5).
    These are typically just floating-point precision errors. Larger
    differences would indicate that the export has errors.

    Args:
        onnx_model_path: Path to the ONNX model file
        pytorch_model: The PyTorch model
        num_samples: Number of random samples to test (default: 100)

    Returns:
        Tuple of (passed, stats)
        - passed: Boolean - validation successful?
        - stats: Dict with detailed statistics
    """
    print(f"Running validation with {num_samples} random samples")
    print(f"  Input shape per sample: [1, 3, 224, 224]")
    print(f"  Total tensors to compare: {num_samples}")
    
    # Generate random inputs
    # Shape: [num_samples, channels, height, width]
    # We use the same normalization as in training
    # (mean=0, std=1 is close enough for validation)
    dummy_inputs = torch.randn(num_samples, 3, 224, 224)
    
    print("-" * 70)
    print("Running PyTorch inference...")
    
    # PyTorch inference
    start_time = time.time()
    
    with torch.no_grad():  # No gradients needed
        pytorch_outputs = pytorch_model(dummy_inputs).numpy()
    
    pytorch_time = time.time() - start_time
    print(f"✅ PyTorch inference completed in {pytorch_time:.3f}s")
    print(f"   Throughput: {num_samples / pytorch_time:.1f} samples/sec")
    
    print("-" * 70)
    print("Running ONNX inference...")
    
    # ONNX inference
    start_time = time.time()
    
    onnx_outputs = run_onnx_inference(
        onnx_model_path,
        dummy_inputs.numpy()  # ONNX Runtime requires NumPy arrays
    )
    
    onnx_time = time.time() - start_time
    print(f"✅ ONNX inference completed in {onnx_time:.3f}s")
    print(f"   Throughput: {num_samples / onnx_time:.1f} samples/sec")
    
    # Performance comparison
    speedup = pytorch_time / onnx_time
    print(f"   Speedup: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than PyTorch")
    
    print("-" * 70)
    print("Comparing outputs...")
    
    # Calculate differences
    diff = np.abs(pytorch_outputs - onnx_outputs)
    
    max_diff = diff.max()
    mean_diff = diff.mean()
    median_diff = np.median(diff)
    
    # Find the sample with the largest difference
    # This is useful for debugging if validation fails
    max_diff_idx = np.unravel_index(diff.argmax(), diff.shape)
    max_diff_sample = max_diff_idx[0]
    max_diff_class = max_diff_idx[1]
    
    print(f"Difference statistics:")
    print(f"  Max:    {max_diff:.2e}")
    print(f"  Mean:   {mean_diff:.2e}")
    print(f"  Median: {median_diff:.2e}")
    print(f"  Max diff location: Sample {max_diff_sample}, Class {max_diff_class}")
    
    # Tolerance Check
    tolerance = 1e-5
    passed = max_diff < tolerance
    
    print("-" * 70)
    
    if passed:
        print(f"✅ VALIDATION PASSED")
        print(f"   Max difference {max_diff:.2e} is below tolerance {tolerance:.2e}")
        print(f"   ONNX model is numerically equivalent to PyTorch model")
    else:
        print(f"❌ VALIDATION FAILED")
        print(f"   Max difference {max_diff:.2e} exceeds tolerance {tolerance:.2e}")
        print(f"   ONNX model is NOT equivalent to PyTorch model")
    
    # Collect statistics
    stats = {
        'num_samples': num_samples,
        'max_diff': float(max_diff),
        'mean_diff': float(mean_diff),
        'median_diff': float(median_diff),
        'tolerance': tolerance,
        'passed': passed,
        'pytorch_time_seconds': pytorch_time,
        'onnx_time_seconds': onnx_time,
        'onnx_speedup': speedup,
    }
    
    return passed, stats


def register_onnx_model(export_run_id, parent_info, validation_stats):
    """
    Registers the ONNX model in the registry and sets the 'deploy' alias.

    This function is ONLY called when validation was successful.
    It creates a new version in the ONNX Registry and marks it with
    the 'deploy' alias.

    This is the critical moment: From now on, the ONNX model is officially
    "ready for deployment". The old model with the 'deploy' alias (if present)
    loses this status.

    Args:
        export_run_id: MLflow Run ID of the export run
        parent_info: Dict with parent PyTorch model info
        validation_stats: Dict with validation statistics

    Returns:
        MLflow ModelVersion object of the newly registered version
    """
    client = MlflowClient()
    
    # Construct the ONNX model name
    # Convention: PyTorch model name + "-onnx" suffix
    pytorch_model_name = parent_info['model_name']
    onnx_model_name = f"{pytorch_model_name}-onnx"
    
    print("=" * 70)
    print("REGISTERING ONNX MODEL")
    print("=" * 70)
    print(f"Model Name: {onnx_model_name}")
    print(f"Source: Export run {export_run_id[:8]}...")
    print(f"Parent: {pytorch_model_name} v{parent_info['model_version']}")
    
    # Model URI points to the ONNX model in the export run
    onnx_model_uri = f"runs:/{export_run_id}/model"
    
    # Register the model
    # This creates a new version in the ONNX Registry
    # If the registry does not exist yet, it will be created
    model_version = mlflow.register_model(
        model_uri=onnx_model_uri,
        name=onnx_model_name
    )
    
    print(f"✅ Registered as version {model_version.version}")
    
    # Set 'deploy' alias on this version
    # If another version already had 'deploy', the alias is moved
    client.set_registered_model_alias(
        onnx_model_name,
        "deploy",
        model_version.version
    )
    
    print(f"✅ Alias 'deploy' set to version {model_version.version}")
    print(f"   This ONNX model is now ready for production deployment")
    
    # Set traceability tags in the registry
    # These tags connect the ONNX model back to the PyTorch model
    client.set_model_version_tag(
        onnx_model_name,
        model_version.version,
        "parent_pytorch_model",
        pytorch_model_name
    )
    
    client.set_model_version_tag(
        onnx_model_name,
        model_version.version,
        "parent_pytorch_version",
        parent_info['model_version']
    )
    
    client.set_model_version_tag(
        onnx_model_name,
        model_version.version,
        "parent_training_run_id",
        parent_info['run_id']
    )
    
    # Validation statistics as tags
    client.set_model_version_tag(
        onnx_model_name,
        model_version.version,
        "validation_max_diff",
        f"{validation_stats['max_diff']:.2e}"
    )
    
    client.set_model_version_tag(
        onnx_model_name,
        model_version.version,
        "validation_passed",
        "true"
    )
    
    print(f"✅ Traceability tags set")
    
    return model_version


def main():
    parser = argparse.ArgumentParser(
        description="Validate ONNX export and register if successful",
        epilog="This script implements Quality Gate 2: Technical Correctness"
    )
    parser.add_argument(
        "--export-run-id",
        required=True,
        help="MLflow Run ID of the export run"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of random samples for validation (default: 100)"
    )
    
    args = parser.parse_args()
    
    # MLflow Tracking URI Setup
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    
    print("=" * 70)
    print("ONNX VALIDATION & REGISTRATION")
    print("=" * 70)
    print(f"Export Run ID: {args.export_run_id[:8]}...")
    print(f"Validation Samples: {args.num_samples}")
    print(f"MLflow Tracking URI: {tracking_uri}")
    print("=" * 70)
    
    # ===== STEP 1: LOAD MODELS =====
    onnx_model_path, pytorch_model, parent_info = load_models_from_mlflow(
        args.export_run_id
    )
    
    # ===== STEP 2: PERFORM VALIDATION =====
    validation_passed, validation_stats = validate_models(
        onnx_model_path,
        pytorch_model,
        num_samples=args.num_samples
    )
    
    # ===== STEP 3: LOG VALIDATION RESULTS TO EXPORT RUN =====
    # We add the validation metrics to the export run
    # This makes the export run a complete "ONNX Model Card"
    print("-" * 70)
    print("Logging validation results to export run...")
    
    with mlflow.start_run(run_id=args.export_run_id):
        # Log all validation statistics
        for key, value in validation_stats.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"validation_{key}", value)
        
        # Set validation status tag
        mlflow.set_tag(
            "validation_status",
            "passed" if validation_passed else "failed"
        )
    
    print(f"✅ Validation results logged to export run")
    
    # ===== STEP 4: REGISTRATION ON SUCCESS =====
    if validation_passed:
        print("-" * 70)
        
        # Register the ONNX model and set the 'deploy' alias
        onnx_model_version = register_onnx_model(
            args.export_run_id,
            parent_info,
            validation_stats
        )
        
        # Outputs for Argo
        print("-" * 70)
        print("VALIDATION=passed")
        print(f"ONNX_MODEL_VERSION={onnx_model_version.version}")
        print(f"ONNX_MODEL_NAME={onnx_model_version.name}")
        
        print("=" * 70)
        print("✅ QUALITY GATE 2 PASSED")
        print("=" * 70)
        print(f"   ONNX Model: {onnx_model_version.name} v{onnx_model_version.version}")
        print(f"   Alias: 'deploy' (ready for production)")
        print(f"   Max Diff: {validation_stats['max_diff']:.2e}")
        print(f"   Next: Deploy to Triton with 'deploy' alias (Gate 3)")
        print("=" * 70)
        
        sys.exit(0)
    
    else:
        # Validation failed - NO registration
        print("-" * 70)
        print("❌ QUALITY GATE 2 FAILED")
        print("-" * 70)
        print(f"   ONNX model failed numerical validation")
        print(f"   Max difference {validation_stats['max_diff']:.2e} exceeded tolerance")
        print(f"   ONNX model will NOT be registered")
        
        # If an old model exists in the registry, it remains unchanged
        onnx_model_name = f"{parent_info['model_name']}-onnx"
        
        try:
            client = MlflowClient()
            current_deploy = client.get_model_version_by_alias(onnx_model_name, "deploy")
            print(f"   Current 'deploy' version unchanged: v{current_deploy.version}")
        except:
            print(f"   No 'deploy' version exists yet (first export failed)")
        
        # Optional: Create debug report
        debug_report = {
            'status': 'failed',
            'reason': 'numerical_validation_failed',
            'max_diff': validation_stats['max_diff'],
            'tolerance': validation_stats['tolerance'],
            'samples_tested': validation_stats['num_samples'],
        }
        
        with mlflow.start_run(run_id=args.export_run_id):
            mlflow.log_dict(debug_report, "validation_failure_report.json")
        
        print(f"   Debug report saved to export run artifacts")
        
        # Output for Argo
        print("")
        print("VALIDATION=failed")
        
        print("=" * 70)
        
        sys.exit(1)


if __name__ == "__main__":
    main()
    