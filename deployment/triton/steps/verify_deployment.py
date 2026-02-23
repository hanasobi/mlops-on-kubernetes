#!/usr/bin/env python3
"""
Step 7: Verify Deployment

Performs a functional test of the deployed model to ensure that it can
actually answer inference requests. This is the final safety check
before completing the deployment workflow.

This step does NOT test the content correctness of the predictions
(that would require a validation dataset with ground truth), but only
the structural correctness:
  - Can the model process requests at all?
  - Is the output shape correct?
  - Are the output values in the expected range?

We use a synthetic test image (random tensor) because we only want to test
the functionality, not the accuracy. This makes the test independent of
external data and fast to execute.

Input Parameters:
    --model-name: Name of the model in Triton
    --expected-input-shape: Expected input shape as JSON array
    --expected-output-shape: Expected output shape as JSON array
    --triton-url: Triton Server URL (default: http://triton-service:8000)
    --num-test-requests: Number of test requests (default: 3)
    --output-dir: Where output artifacts should be saved

Output Artifacts:
    verification_result.json: Detailed test results

Environment Variables:
    None - this step is self-contained
"""

import argparse
import json
import numpy as np
import os
import sys
import time

from utils.triton_client import TritonClient


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Verify deployed model functionality'
    )

    parser.add_argument(
        '--model-name',
        required=True,
        help='Name of the model in Triton'
    )

    parser.add_argument(
        '--expected-input-shape',
        required=True,
        help='Expected input shape as JSON array, e.g. "[1, 3, 224, 224]"'
    )

    parser.add_argument(
        '--expected-output-shape',
        required=True,
        help='Expected output shape as JSON array, e.g. "[1, 10]"'
    )

    parser.add_argument(
        '--triton-url',
        default='http://triton-service.ai-platform:8000',
        help='Triton Inference Server URL'
    )

    parser.add_argument(
        '--num-test-requests',
        type=int,
        default=3,
        help='Number of test inference requests to perform'
    )

    parser.add_argument(
        '--output-dir',
        required=True,
        help='Directory to save output artifacts'
    )

    return parser.parse_args()


def generate_synthetic_input(shape):
    """
    Generates a synthetic test image with the given shape.

    We use random noise because we only want to test the functionality,
    not the content correctness of the predictions. A real validation
    image would be overkill here and would introduce external data
    dependencies.

    The random values are drawn from a normal distribution with mean 0
    and standard deviation 1, which is typical for normalized image inputs
    in neural networks. This is realistic enough to ensure that the model
    has no edge cases with certain input values.

    Args:
        shape: Tuple or list with the desired tensor shape

    Returns:
        NumPy array with random float32 values
    """
    # Set NumPy random seed for reproducibility
    # This is useful for debugging - if a test fails
    # we can reproduce it with the same input
    np.random.seed(42)

    # Normal distribution (mean=0, std=1) is more realistic than uniform
    # because it is closer to normalized image data
    synthetic_input = np.random.randn(*shape).astype(np.float32)

    return synthetic_input


def validate_output_structure(output_data, expected_shape, model_name):
    """
    Validates that the output has the expected structure.

    We check several aspects here:
    1. Shape - is the dimensionality correct?
    2. Datatype - is it float32 as expected?
    3. Value range - are the values plausible?

    For classification models we typically expect:
    - Output shape [Batch, NumClasses]
    - Values between 0 and 1 (softmax output)
    - Sum over classes approx. 1.0

    Args:
        output_data: NumPy array with model output
        expected_shape: List/tuple with expected shape
        model_name: Name of the model (for error messages)

    Returns:
        Dictionary with validation results

    Raises:
        ValueError: If validation fails
    """
    validation_results = {}

    # Check 1: Shape
    actual_shape = output_data.shape
    validation_results['shape_matches'] = actual_shape == tuple(expected_shape)

    if not validation_results['shape_matches']:
        raise ValueError(
            f"Output shape mismatch for {model_name}: "
            f"expected {expected_shape}, got {actual_shape}"
        )

    print(f"  \u2713 Output shape is correct: {actual_shape}")

    # Check 2: Datatype
    validation_results['dtype'] = str(output_data.dtype)

    if output_data.dtype != np.float32:
        # This is only a warning, not a hard failure
        # Some models use float16 for efficiency
        print(f"  \u26a0 Output dtype is {output_data.dtype}, expected float32")
    else:
        print(f"  \u2713 Output dtype is correct: {output_data.dtype}")

    # Check 3: Value range
    # For classification models with softmax we expect values in [0, 1]
    min_val = float(np.min(output_data))
    max_val = float(np.max(output_data))

    validation_results['min_value'] = min_val
    validation_results['max_value'] = max_val

    # We allow a small tolerance due to floating-point precision
    if min_val < -0.01 or max_val > 1.01:
        print(f"  \u26a0 Output values outside expected range [0, 1]: "
              f"min={min_val:.6f}, max={max_val:.6f}")
        print(f"     This might indicate missing Softmax activation")
    else:
        print(f"  \u2713 Output values in expected range: "
              f"min={min_val:.6f}, max={max_val:.6f}")

    # Check 4: Summation (only meaningful for classification models)
    # With batch size 1 and output shape [1, NumClasses] the sum
    # over the class dimension should be approximately 1.0
    if len(expected_shape) == 2 and expected_shape[0] == 1:
        sum_over_classes = float(np.sum(output_data[0]))
        validation_results['sum_over_classes'] = sum_over_classes

        # Tolerance of 0.01 due to floating-point precision
        if abs(sum_over_classes - 1.0) > 0.01:
            print(f"  \u26a0 Sum over classes is {sum_over_classes:.6f}, expected ~1.0")
            print(f"     This might indicate missing Softmax activation")
        else:
            print(f"  \u2713 Sum over classes is correct: {sum_over_classes:.6f}")

    return validation_results


def run_single_inference(triton_client, model_name, input_shape, output_shape):
    """
    Performs a single inference request and validates the result.

    Args:
        triton_client: Initialized TritonClient
        model_name: Name of the model in Triton
        input_shape: Shape of the input tensor
        output_shape: Expected shape of the output tensor

    Returns:
        Dictionary with test results (success, latency_ms, validation)

    Raises:
        Exception: On inference errors or validation errors
    """
    # Step 1: Generate synthetic input
    test_input = generate_synthetic_input(input_shape)

    # Step 2: Prepare inference request
    # The format follows the KServe v2 protocol that Triton uses
    inputs = [{
        'name': 'input',  # Default input name for ONNX models
        'shape': list(input_shape),
        'datatype': 'FP32',
        'data': test_input.flatten().tolist()  # Flatten for wire format
    }]

    # Step 3: Perform inference and measure time
    start_time = time.time()

    try:
        response = triton_client.infer(model_name, inputs)
    except Exception as e:
        # Inference request failed - this is critical
        raise Exception(f"Inference request failed: {e}")

    latency_ms = (time.time() - start_time) * 1000

    # Step 4: Parse response
    # The output comes as a list, we take the first (and only) one
    if 'outputs' not in response or len(response['outputs']) == 0:
        raise Exception("Response has no outputs")

    output = response['outputs'][0]

    # The raw data comes as a flattened list, we need to reshape
    output_data = np.array(output['data'], dtype=np.float32)
    output_data = output_data.reshape(output_shape)

    # Step 5: Validate output
    validation_results = validate_output_structure(
        output_data,
        output_shape,
        model_name
    )

    return {
        'success': True,
        'latency_ms': round(latency_ms, 2),
        'validation': validation_results
    }


def main():
    """Main logic of the verify_deployment step."""
    print("=" * 80)
    print("Step 7: Verify Deployment")
    print("=" * 80)

    # Step 1: Parse arguments
    args = parse_arguments()

    # Convert shapes from JSON string to lists
    try:
        input_shape = json.loads(args.expected_input_shape)
        output_shape = json.loads(args.expected_output_shape)
    except json.JSONDecodeError as e:
        print(f"\nERROR: Invalid JSON in shape parameters")
        print(f"Details: {e}")
        sys.exit(1)

    print(f"\nConfiguration:")
    print(f"  Model Name: {args.model_name}")
    print(f"  Triton URL: {args.triton_url}")
    print(f"  Expected Input Shape: {input_shape}")
    print(f"  Expected Output Shape: {output_shape}")
    print(f"  Number of Test Requests: {args.num_test_requests}")
    print(f"  Output Directory: {args.output_dir}")

    # Step 2: Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Step 3: Initialize Triton client
    print("\n" + "-" * 80)
    print("Initializing Triton Client...")

    try:
        triton_client = TritonClient(base_url=args.triton_url)
    except Exception as e:
        print(f"\nERROR: Failed to initialize Triton client")
        print(f"Details: {e}")
        sys.exit(1)

    # Step 4: Server health check
    print("\n" + "-" * 80)
    print("Checking Triton Server health...")

    try:
        if not triton_client.is_server_ready():
            print(f"\nERROR: Triton server is not ready")
            print(f"Server URL: {args.triton_url}")
            sys.exit(1)

        print("  \u2713 Triton server is ready")

    except Exception as e:
        print(f"\nERROR: Failed to check server health")
        print(f"Details: {e}")
        sys.exit(1)

    # Step 5: Model ready check
    print("\n" + "-" * 80)
    print(f"Checking if model '{args.model_name}' is ready...")

    try:
        if not triton_client.is_model_ready(args.model_name):
            print(f"\nERROR: Model '{args.model_name}' is not ready")
            print(f"\nPossible causes:")
            print(f"  - Model failed to load (check Triton logs)")
            print(f"  - Wrong model name (check config.pbtxt)")
            print(f"  - Model is still initializing (unlikely at this stage)")
            sys.exit(1)

        print(f"  \u2713 Model '{args.model_name}' is ready")

    except Exception as e:
        print(f"\nERROR: Failed to check model readiness")
        print(f"Details: {e}")
        sys.exit(1)

    # Step 6: Run test inference requests
    print("\n" + "-" * 80)
    print(f"Running {args.num_test_requests} test inference requests...")

    test_results = []
    all_successful = True

    for i in range(args.num_test_requests):
        print(f"\nTest Request {i + 1}/{args.num_test_requests}:")

        try:
            result = run_single_inference(
                triton_client,
                args.model_name,
                input_shape,
                output_shape
            )

            test_results.append(result)
            print(f"  \u2713 Request successful (latency: {result['latency_ms']}ms)")

        except Exception as e:
            print(f"\n  \u2717 Request failed!")
            print(f"  Details: {e}")

            test_results.append({
                'success': False,
                'error': str(e)
            })

            all_successful = False

            # On error we abort immediately
            # If one fails, the model is not functional
            break

    # Step 7: Summarize results
    print("\n" + "-" * 80)
    print("Test Results Summary:")

    successful_count = sum(1 for r in test_results if r.get('success', False))

    print(f"  Successful: {successful_count}/{args.num_test_requests}")

    if successful_count > 0:
        latencies = [r['latency_ms'] for r in test_results if r.get('success', False)]
        avg_latency = sum(latencies) / len(latencies)
        print(f"  Average Latency: {avg_latency:.2f}ms")
        print(f"  Min Latency: {min(latencies):.2f}ms")
        print(f"  Max Latency: {max(latencies):.2f}ms")

    # Step 8: Write verification result artifact
    verification_result = {
        'model_name': args.model_name,
        'triton_url': args.triton_url,
        'input_shape': input_shape,
        'output_shape': output_shape,
        'num_test_requests': args.num_test_requests,
        'successful_requests': successful_count,
        'all_successful': all_successful,
        'test_results': test_results
    }

    output_path = os.path.join(args.output_dir, "verification_result.json")

    try:
        print(f"\nWriting verification results to {output_path}")
        with open(output_path, 'w') as f:
            json.dump(verification_result, f, indent=2)

        print("Verification results written successfully")

    except Exception as e:
        print(f"\nERROR: Failed to write verification results")
        print(f"Details: {e}")
        sys.exit(1)

    # Step 9: Final decision
    print("\n" + "=" * 80)

    if all_successful:
        print("Step 7 completed successfully!")
        print("=" * 80)
        print(f"\n\u2713 All {args.num_test_requests} inference requests succeeded")
        print(f"\u2713 Model '{args.model_name}' is fully functional")
        print(f"\u2713 Deployment verification passed")

        sys.exit(0)
    else:
        print("Step 7 FAILED!")
        print("=" * 80)
        print(f"\n\u2717 {args.num_test_requests - successful_count} out of {args.num_test_requests} requests failed")
        print(f"\u2717 Model '{args.model_name}' is NOT functional")
        print(f"\u2717 Deployment verification failed")
        print(f"\nThe model was loaded by Triton but cannot process inference requests")
        print(f"You need to investigate the failure before this deployment can be used")

        sys.exit(1)


if __name__ == '__main__':
    main()
