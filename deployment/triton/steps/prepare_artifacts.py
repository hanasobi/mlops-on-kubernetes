#!/usr/bin/env python3
"""
Step 3: Prepare Deployment Artifacts

Creates the metadata.json file that is deployed with the model.
This metadata is our audit trail - it connects MLflow, Triton,
and S3 and gives us complete traceability over every deployment.

The metadata.json contains three main sections:
1. MLflow information (which model, which version, which run)
2. Triton information (version number, deployment timestamp)
3. Model information (format, input/output shapes, validation metrics)

Input Parameters:
    --mlflow-metadata: Path to mlflow_metadata.json from step 1
    --triton-version: Path to triton_version.txt from step 2
    --model-file: Path to model.onnx file from step 1
    --output-dir: Where the final artifacts should be saved

Output Artifacts:
    metadata.json: The complete deployment metadata
    model.onnx: Copy of the model (pass-through from step 1)

Environment Variables:
    ARGO_WORKFLOW_NAME: Optional - name of the workflow for audit trail
"""

import argparse
import json
import os
import sys
import shutil
from datetime import datetime, timezone


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Prepare deployment artifacts including metadata.json'
    )
    
    parser.add_argument(
        '--mlflow-metadata',
        required=True,
        help='Path to mlflow_metadata.json from step 1'
    )
    
    parser.add_argument(
        '--triton-version',
        required=True,
        help='Path to triton_version.txt from step 2'
    )
    
    parser.add_argument(
        '--model-file',
        required=True,
        help='Path to model.onnx from step 1'
    )
    
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Directory to save final artifacts'
    )
    
    return parser.parse_args()


def load_mlflow_metadata(metadata_path):
    """
    Loads the MLflow metadata that step 1 created.

    This file contains all information from the MLflow Registry -
    version number, Run ID, metrics, parameters, etc. We need
    this information to establish the connection between Triton
    deployment and MLflow run in the final metadata.json.

    Args:
        metadata_path: Path to mlflow_metadata.json

    Returns:
        Dictionary with MLflow metadata
    """
    print(f"Loading MLflow metadata from {metadata_path}")
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"MLflow metadata loaded successfully")
        print(f"  Model: {metadata.get('model_name')}")
        print(f"  Version: {metadata.get('version')}")
        print(f"  Run ID: {metadata.get('run_id')}")
        
        return metadata
        
    except FileNotFoundError:
        print(f"ERROR: MLflow metadata file not found at {metadata_path}")
        print("This file should have been created by step 1 (fetch_model)")
        raise
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in MLflow metadata file")
        print(f"Details: {e}")
        raise


def load_triton_version(version_path):
    """
    Loads the Triton version number that step 2 determined.

    This file contains a single line with the version number as a string.
    We convert it to an integer for calculations but also keep the string
    for JSON serialization.

    Args:
        version_path: Path to triton_version.txt

    Returns:
        String with the version number (e.g. "3")
    """
    print(f"Loading Triton version from {version_path}")
    
    try:
        with open(version_path, 'r') as f:
            version = f.read().strip()
        
        # Validate that it is a valid integer number
        try:
            version_int = int(version)
            if version_int <= 0:
                raise ValueError("Version must be positive")
        except ValueError as e:
            print(f"ERROR: Invalid version number '{version}'")
            print(f"Details: {e}")
            raise
        
        print(f"Triton version loaded: {version}")
        return version
        
    except FileNotFoundError:
        print(f"ERROR: Triton version file not found at {version_path}")
        print("This file should have been created by step 2 (determine_version)")
        raise


def extract_model_info(model_path):
    """
    Extracts technical information from the ONNX model.

    The ONNX format stores metadata about the model - input/output
    names, shapes, datatypes, etc. We extract this information
    programmatically instead of configuring it manually. This prevents
    errors where the config.pbtxt does not match the actual model.

    Args:
        model_path: Path to the model.onnx file

    Returns:
        Dictionary with model information
    """
    import onnx
    
    print(f"Extracting model information from {model_path}")
    
    try:
        # Load ONNX model
        model = onnx.load(model_path)
        
        # Extract input/output information
        # An ONNX model can have multiple inputs/outputs, but
        # our classification models typically have only one each
        inputs_info = []
        for input_tensor in model.graph.input:
            # Extract shape - can have symbolic dimensions (e.g. batch_size)
            shape = []
            for dim in input_tensor.type.tensor_type.shape.dim:
                if dim.dim_value:
                    shape.append(dim.dim_value)
                elif dim.dim_param:
                    # Symbolic dimension like "batch_size"
                    shape.append(dim.dim_param)
            
            inputs_info.append({
                'name': input_tensor.name,
                'shape': shape,
                'dtype': input_tensor.type.tensor_type.elem_type
            })
        
        outputs_info = []
        for output_tensor in model.graph.output:
            shape = []
            for dim in output_tensor.type.tensor_type.shape.dim:
                if dim.dim_value:
                    shape.append(dim.dim_value)
                elif dim.dim_param:
                    shape.append(dim.dim_param)
            
            outputs_info.append({
                'name': output_tensor.name,
                'shape': shape,
                'dtype': output_tensor.type.tensor_type.elem_type
            })
        
        # Calculate model size
        file_size_bytes = os.path.getsize(model_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        model_info = {
            'format': 'onnx',
            'file_size_mb': round(file_size_mb, 2),
            'inputs': inputs_info,
            'outputs': outputs_info,
            'opset_version': model.opset_import[0].version if model.opset_import else None
        }
        
        print(f"Model information extracted successfully")
        print(f"  Format: {model_info['format']}")
        print(f"  Size: {model_info['file_size_mb']} MB")
        print(f"  Inputs: {len(inputs_info)}")
        print(f"  Outputs: {len(outputs_info)}")
        
        return model_info
        
    except Exception as e:
        print(f"ERROR: Failed to extract model information")
        print(f"Details: {e}")
        # We return a minimal dictionary instead of failing
        # The deployment can still work, just without these details
        print("Warning: Continuing with minimal model info")
        return {
            'format': 'onnx',
            'file_size_mb': round(os.path.getsize(model_path) / (1024 * 1024), 2),
            'note': 'Could not extract detailed model information'
        }


def build_metadata(mlflow_metadata, triton_version, model_info):
    """
    Builds the final metadata.json dictionary.

    This is the core of the step - we take all the information we have
    collected and structure it into a clean, documented format. The structure
    is important because other scripts (like the rollback workflow) read
    this metadata later and depend on it having a consistent format.

    Args:
        mlflow_metadata: Dictionary from step 1
        triton_version: String from step 2
        model_info: Dictionary from extract_model_info()

    Returns:
        Dictionary with the complete deployment metadata
    """
    print("\nBuilding final metadata structure...")
    
    # Timestamp in ISO 8601 format with UTC timezone
    # This is important for audit trails - we want to know exactly WHEN deployment happened
    deployment_timestamp = datetime.now(timezone.utc).isoformat()
    
    # Get workflow name from environment variable if available
    # This gives us traceability of who/what triggered the deployment
    workflow_name = os.environ.get('ARGO_WORKFLOW_NAME', 'unknown')
    
    # The final structure - three main sections as in our design
    metadata = {
        'mlflow': {
            'model_name': mlflow_metadata.get('model_name'),
            'version': mlflow_metadata.get('version'),
            'run_id': mlflow_metadata.get('run_id'),
            'alias_at_deploy': mlflow_metadata.get('source'),
            'run_name': mlflow_metadata.get('run_name'),
        },
        'triton': {
            'version': triton_version,
            'deployed_at': deployment_timestamp,
            'deployed_by': workflow_name
        },
        'model': model_info,
        'validation': {}
    }
    
    # Optional: Add validation metrics from MLflow if available
    # These metrics come from training and show how well the model performs
    if mlflow_metadata.get('metrics'):
        metrics = mlflow_metadata['metrics']
        
        # Search for relevant validation metrics
        # The names can vary depending on how the training script logs them
        validation_metrics = {}
        for key, value in metrics.items():
            if 'val' in key.lower() or 'validation' in key.lower():
                validation_metrics[key] = value
        
        if validation_metrics:
            metadata['validation'] = validation_metrics
            print(f"  Added {len(validation_metrics)} validation metrics")
    
    print("Metadata structure built successfully")
    return metadata


def main():
    """Main logic of the prepare_artifacts step."""
    print("=" * 80)
    print("Step 3: Prepare Deployment Artifacts")
    print("=" * 80)
    
    # Step 1: Parse arguments
    args = parse_arguments()
    
    print(f"\nConfiguration:")
    print(f"  MLflow Metadata: {args.mlflow_metadata}")
    print(f"  Triton Version: {args.triton_version}")
    print(f"  Model File: {args.model_file}")
    print(f"  Output Directory: {args.output_dir}")
    
    # Step 2: Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 3: Load all input artifacts
    print("\nLoading input artifacts...")
    
    try:
        mlflow_metadata = load_mlflow_metadata(args.mlflow_metadata)
        triton_version = load_triton_version(args.triton_version)
        model_info = extract_model_info(args.model_file)
        
    except Exception as e:
        print(f"\nERROR: Failed to load input artifacts")
        print(f"Details: {e}")
        sys.exit(1)
    
    # Step 4: Build metadata.json
    try:
        metadata = build_metadata(mlflow_metadata, triton_version, model_info)
        
    except Exception as e:
        print(f"\nERROR: Failed to build metadata")
        print(f"Details: {e}")
        sys.exit(1)
    
    # Step 5: Write metadata.json to file
    metadata_output = os.path.join(args.output_dir, "metadata.json")
    
    try:
        print(f"\nWriting metadata to {metadata_output}")
        with open(metadata_output, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("Metadata written successfully")
        
    except Exception as e:
        print(f"\nERROR: Failed to write metadata file")
        print(f"Details: {e}")
        sys.exit(1)
    
    # Step 6: Copy model.onnx
    # We copy the model to output_dir so all artifacts
    # that should go to S3 are in one place
    model_output = os.path.join(args.output_dir, "model.onnx")
    
    try:
        print(f"\nCopying model to {model_output}")
        shutil.copy2(args.model_file, model_output)
        
        # Verify that the copy was successful
        if not os.path.exists(model_output):
            raise Exception("Model copy verification failed")
        
        # Verify that the size matches
        original_size = os.path.getsize(args.model_file)
        copied_size = os.path.getsize(model_output)
        
        if original_size != copied_size:
            raise Exception(
                f"Model copy size mismatch: original={original_size}, "
                f"copied={copied_size}"
            )
        
        print(f"Model copied successfully ({copied_size} bytes)")
        
    except Exception as e:
        print(f"\nERROR: Failed to copy model file")
        print(f"Details: {e}")
        sys.exit(1)
    
    # Step 7: Print summary
    print("\n" + "=" * 80)
    print("Step 3 completed successfully!")
    print("=" * 80)
    print(f"\nOutput Artifacts:")
    print(f"  metadata.json: {metadata_output}")
    print(f"  model.onnx: {model_output}")
    
    print(f"\nDeployment Metadata Summary:")
    print(f"  MLflow Model: {metadata['mlflow']['model_name']}")
    print(f"  MLflow Version: {metadata['mlflow']['version']}")
    print(f"  Triton Version: {metadata['triton']['version']}")
    print(f"  Deployed At: {metadata['triton']['deployed_at']}")
    print(f"  Deployed By: {metadata['triton']['deployed_by']}")
    
    # Success Exit Code
    sys.exit(0)


if __name__ == '__main__':
    main()