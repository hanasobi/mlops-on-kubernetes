#!/usr/bin/env python3
"""
Step 1: Fetch Model from MLflow

FIXED v2: Works with new mlflow_helpers v3 that returns file_paths separately
"""

import argparse
import json
import os
import sys
import shutil

# Add parent directory to path for utils import
sys.path.insert(0, '/scripts')

from utils.mlflow_helpers import MLflowHelper


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Fetch a model from MLflow Model Registry'
    )
    
    parser.add_argument('--model-name', required=True, help='Name of the registered model')
    parser.add_argument('--source', default='alias:deploy', help='Source to load from')
    parser.add_argument('--output-dir', required=True, help='Directory to save output artifacts')
    
    return parser.parse_args()


def validate_environment():
    """Validates that all required environment variables are set."""
    required_vars = ['MLFLOW_TRACKING_URI']
    missing_vars = [var for var in required_vars if var not in os.environ]
    
    if missing_vars:
        print(f"ERROR: Missing required environment variables: {missing_vars}")
        print("Please ensure the workflow sets MLFLOW_TRACKING_URI")
        sys.exit(1)
    
    print(f"MLflow Tracking URI: {os.environ['MLFLOW_TRACKING_URI']}")


def save_model_to_file(file_paths, output_path):
    """
    Saves the ONNX model to a file.

    Args:
        file_paths: Dictionary with 'onnx', 'onnx_data', 'temp_dir' keys
        output_path: Path where the .onnx file should be saved
    """
    print(f"Saving ONNX model to {output_path}")
    
    onnx_file = file_paths['onnx']
    onnx_data_file = file_paths.get('onnx_data')
    temp_dir = file_paths['temp_dir']
    
    try:
        # Copy ONNX file
        shutil.copy2(onnx_file, output_path)
        
        main_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        
        # Copy data file if present
        if onnx_data_file:
            data_output_path = output_path + ".data"
            shutil.copy2(onnx_data_file, data_output_path)
            
            data_size_mb = os.path.getsize(data_output_path) / (1024 * 1024)
            total_size_mb = main_size_mb + data_size_mb
            
            print(f"Model uses external data format")
            print(f"Model saved successfully:")
            print(f"  Main file: {main_size_mb:.2f} MB")
            print(f"  Data file: {data_size_mb:.2f} MB")
            print(f"  Total: {total_size_mb:.2f} MB")
        else:
            print(f"Model uses embedded data (single file)")
            print(f"Model saved successfully ({main_size_mb:.2f} MB)")
        
        # Verify
        if not os.path.exists(output_path):
            raise Exception(f"Failed to save model to {output_path}")
            
    finally:
        # Cleanup temp directory
        print(f"Cleaning up temporary files...")
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Main logic of the fetch_model step."""
    print("=" * 80)
    print("Step 1: Fetch Model from MLflow")
    print("=" * 80)
    
    args = parse_arguments()
    validate_environment()
    
    print(f"\nConfiguration:")
    print(f"  Model Name: {args.model_name}")
    print(f"  Source: {args.source}")
    print(f"  Output Directory: {args.output_dir}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\nInitializing MLflow Helper...")
    mlflow_helper = MLflowHelper()
    
    print(f"\nLoading model '{args.model_name}' from source '{args.source}'...")
    
    try:
        model, metadata, file_paths = mlflow_helper.load_model(args.model_name, args.source)
        print(f"Model loaded successfully!")
        print(f"  MLflow Version: {metadata['version']}")
        print(f"  Run ID: {metadata['run_id']}")
        print(f"  Status: {metadata['status']}")
        
    except Exception as e:
        print(f"\nERROR: Failed to load model from MLflow")
        print(f"Details: {e}")
        print("\nPossible causes:")
        print("  - Model name does not exist in registry")
        print("  - Alias/version does not exist")
        print("  - MLflow server is not reachable")
        print("  - Network connectivity issues")
        sys.exit(1)
    
    # Save model to file
    model_path = os.path.join(args.output_dir, "model.onnx")
    
    try:
        save_model_to_file(file_paths, model_path)
    except Exception as e:
        print(f"\nERROR: Failed to save model to file")
        print(f"Details: {e}")
        sys.exit(1)
    
    # Save MLflow metadata to JSON
    metadata_path = os.path.join(args.output_dir, "mlflow_metadata.json")
    
    try:
        print(f"\nSaving MLflow metadata to {metadata_path}")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("Metadata saved successfully")
        
    except Exception as e:
        print(f"\nERROR: Failed to save metadata")
        print(f"Details: {e}")
        sys.exit(1)
    
    # Summary
    print("\n" + "=" * 80)
    print("Step 1 completed successfully!")
    print("=" * 80)
    print(f"\nOutput Artifacts:")
    print(f"  model.onnx: {model_path}")
    
    data_path = model_path + ".data"
    if os.path.exists(data_path):
        print(f"  model.onnx.data: {data_path}")
    
    print(f"  mlflow_metadata.json: {metadata_path}")
    
    print(f"\nModel Information:")
    print(f"  Name: {metadata['model_name']}")
    print(f"  Version: {metadata['version']}")
    print(f"  Source: {metadata['source']}")
    
    if metadata.get('metrics'):
        print(f"\nModel Metrics:")
        for key, value in metadata['metrics'].items():
            print(f"  {key}: {value}")
    
    sys.exit(0)


if __name__ == '__main__':
    main()