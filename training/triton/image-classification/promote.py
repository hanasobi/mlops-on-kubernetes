# scripts/promote.py
"""
Model Promotion Script with Champion-based Quality Gate.

This script implements Gate 1 of the ML pipeline: Training Performance Check.

It compares a candidate model (newly trained model) with the current
champion (best known model) and sets the 'champion' alias only when:
1. The candidate meets the absolute threshold (minimum quality)
2. The candidate is significantly better than the champion (minimum improvement)

The 'champion' alias marks the trained model with the best validation
performance. It is the "best model at this point in time" based on
metrics, regardless of whether it is deployed or not.

Conceptually important: 'champion' does NOT mean "in production". It only
means "best trained model". A model can be champion without ever being
deployed (e.g., if the ONNX export fails).

The clear separation between 'champion' (training performance) and 'deploy'
(technical readiness) makes the pipeline transparent and traceable.
"""

import mlflow
from mlflow.tracking import MlflowClient
import os
import sys
import yaml
import argparse


# Config Path relativ zu diesem Script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")


def load_config():
    """
    Loads the pipeline configuration from config.yaml.

    The config contains all important settings for the promotion logic:
    - Which metric is used for comparison (e.g. best_val_accuracy)
    - In which direction is better (greater_is_better: true/false)
    - Absolute threshold (minimum quality that must be reached)
    - Minimum improvement (how much better the candidate must be)

    This externalization of the config makes the pipeline adjustable without
    changing code. For different models you can simply adjust the config
    (e.g. different metrics or thresholds).
    """
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    return config


def register_model(run_id: str, model_name: str, alias: str = 'champion'):
    """
    Registers a model in the MLflow Model Registry and sets the champion alias.

    This function is the critical step of promotion: It takes a trained model
    (identified by its run_id) and "promotes" it to a registered model.
    This means:

    1. The model gets a version number in the registry
    2. The 'champion' alias is set to this version
    3. If another version was already 'champion', the alias is moved

    The 'champion' alias acts like a pointer: It always points to the version
    with the best performance. When a better model is trained, the pointer is
    simply moved to the new version. The old version remains in the registry
    (with its history), but loses the 'champion' status.

    Args:
        run_id: MLflow Run ID of the model to be registered
        model_name: Name of the registered model (e.g. "resnet18-imagenette")
        alias: Alias for this version (default: "champion")

    Returns:
        Model version number as string (e.g. "5")

    Side Effects:
        - Creates new model version in the registry (or uses existing one)
        - Sets/moves the 'champion' alias
        - Writes version number to /tmp/model_version for Argo
    """
    client = MlflowClient()
    
    # Model URI points to the trained model in the run
    # The format "runs:/<run_id>/model" is MLflow's standard convention
    # for referencing models from runs
    model_uri = f"runs:/{run_id}/model"
    
    print(f"Registering model from run {run_id[:8]}...")
    
    # mlflow.register_model does two things:
    # 1. If the model_name does not exist yet, it creates a new registry
    # 2. If it exists, it creates a new version
    # It returns a ModelVersion object with all metadata
    result = mlflow.register_model(model_uri=model_uri, name=model_name)
    
    print(f"✅ Created model version {result.version} in registry '{model_name}'")
    
    # Now we set the 'champion' alias to this new version
    # If another version was already 'champion', the alias is
    # automatically moved. MLflow only allows one alias per name.
    client.set_registered_model_alias(model_name, alias, result.version)
    
    print(f"✅ Alias '{alias}' now points to version {result.version}")
    print(f"   This is now the champion model (best training performance)")
    
    # Write version to file for Argo Workflows
    # Subsequent steps can read this version number when they need it
    with open("/tmp/model_version", 'w') as f:
        f.write(result.version)
    
    return result.version


def promote(run_id: str = None):
    """
    Main logic: Implements the champion-based quality gate.

    This function is the core of the promotion logic. It orchestrates
    the complete comparison and decision process:

    1. Identifies the candidate (the newly trained model)
    2. Finds the current champion (if one exists)
    3. Compares both based on the configured metric
    4. Decides whether promotion should occur
    5. Executes the promotion (or not)

    The function uses a "quality gate" approach: The candidate must
    pass TWO checks to be promoted:

    Check 1 (Absolute Threshold): Is the metric good enough?
    - Prevents bad models from becoming champion
    - Example: Accuracy must be at least 90%

    Check 2 (Relative Improvement): Is it SIGNIFICANTLY better than the champion?
    - Prevents minimal, insignificant improvements from being promoted
    - Example: Must be at least 0.5% better
    - Avoids "noise promotion" (when the difference is only random)

    Args:
        run_id: Optional - MLflow Run ID of the candidate
                If None: Automatically takes the last run in the experiment
                If set: Uses this specific run

    Returns:
        Nothing - exits with sys.exit(0) on success or sys.exit(1) on failure

    Side Effects:
        - Writes detailed logs to the terminal (for Argo logs)
        - Writes "RESULT=passed/failed" for Argo parsing
        - On success: Registers model and sets champion alias
        - On failure: Does not change anything in the registry
    """
    # MLflow Connection Setup
    # The tracking URI comes from the environment variable that is set in the
    # Argo Workflow (points to the in-cluster MLflow service)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    client = MlflowClient()
    
    # ===== STEP 1: LOAD CONFIG =====
    # All promotion criteria come from the config.yaml
    # This makes the logic configurable without code changes
    config = load_config()
    
    model_name = config["model"]["name"]   
    experiment_name = config["experiment"]["name"] 
    metric = config["metric"]["name"]
    greater_is_better = config["metric"]["greater_is_better"]
    threshold = config["promotion_criteria"]["threshold"]
    min_improvement = config["promotion_criteria"]["min_improvement"]

    # Header for the logs - makes Argo Workflow logs more readable
    print("=" * 70)
    print("MODEL PROMOTION - QUALITY GATE 1: TRAINING PERFORMANCE")
    print("=" * 70)
    print(f"Model Name: {model_name}")
    print(f"Experiment: {experiment_name}")
    print(f"Metric: {metric} (greater_is_better: {greater_is_better})")
    print(f"Absolute Threshold: {threshold}")
    print(f"Minimum Improvement: {min_improvement}")
    print("-" * 70)
    
    # ===== STEP 2: IDENTIFY CANDIDATE =====
    # The candidate is the model that was just trained and is now
    # to be promoted. There are two ways to identify it:

    # Option 1: Explicit Run ID was provided (from Argo train step)
    # This is the preferred way because it is explicit and unambiguous

    # Option 2: Automatic selection (fallback for local development)
    # Takes the last completed run in the experiment
    
    exp_id = client.get_experiment_by_name(experiment_name).experiment_id
    
    if run_id is not None:
        print(f"Using explicitly provided run_id: {run_id}")
        candidate_run_id = run_id
        candidate_run = client.get_run(run_id)
    else:
        print(f"No run_id provided - selecting last completed run")
        
        # Get all runs in the experiment
        runs = list(client.search_runs([exp_id]))
        
        if len(runs) == 0:
            print(f"❌ ERROR: No runs found in experiment '{experiment_name}'")
            print("   Make sure training completed successfully")
            print("RESULT=failed")
            sys.exit(1)
        
        # Sort by end_time (newest first) and take the first one
        # This is the run that was most recently completed
        candidate_run = sorted(
            runs,
            key=lambda r: r.info.end_time,
            reverse=True
        )[0]
        candidate_run_id = candidate_run.info.run_id
        print(f"Selected run: {candidate_run_id[:8]}...")
    
    # Get the metric from the candidate
    # If the metric does not exist, this is a critical error
    # (means that training did not log correctly)
    if metric not in candidate_run.data.metrics:
        print(f"❌ ERROR: Metric '{metric}' not found in run {candidate_run_id[:8]}")
        print(f"   Available metrics: {list(candidate_run.data.metrics.keys())}")
        print(f"   Make sure training script logs this metric")
        print("RESULT=failed")
        sys.exit(1)
    
    candidate_value = candidate_run.data.metrics[metric]
    print(f"Candidate performance: {metric} = {candidate_value:.4f}")
    
    # Initialize model_version file as empty
    # Only populated when promotion is successful
    with open("/tmp/model_version", 'w') as f:
        f.write("")
    
    # ===== STEP 3: FIND CHAMPION =====
    # Now we search for the current champion in the registry
    # There are three possible scenarios:

    # Scenario A: No champion exists (first promotion)
    # Scenario B: Champion exists and is a different run
    # Scenario C: Candidate IS already the champion (edge case)
    
    champion_value = None  # Default: No champion
    
    try:
        # Try to get the champion from the registry
        champion = client.get_model_version_by_alias(model_name, 'champion')
        champion_run_id = champion.run_id
        champion_run = client.get_run(champion_run_id)
        
        print(f"Current champion: run {champion_run_id[:8]}... (version {champion.version})")
        
        # Edge case check: Is the candidate already the champion?
        # This can happen when promote.py is called multiple times with the same run_id
        # (e.g. during a workflow retry)
        if candidate_run_id == champion_run_id:
            print("ℹ️  Candidate is already the champion")
            print("   No promotion needed (already promoted)")
            print("RESULT=failed")
            sys.exit(1)
        
        # Get champion metric
        # If the metric does not exist in the champion run (should not happen
        # but we handle it defensively), we treat it as "no champion"
        if metric not in champion_run.data.metrics:
            print(f"⚠️  WARNING: Metric '{metric}' not found in champion run")
            print(f"   Treating this as first-time promotion")
            champion_value = None
        else:
            champion_value = champion_run.data.metrics[metric]
            print(f"Champion performance: {metric} = {champion_value:.4f}")
        
    except Exception as e:
        # No champion exists in the registry
        # This is normal during the first promotion
        print(f"ℹ️  No registered model found with 'champion' alias")
        print(f"   This will be the first champion")
        champion_value = None
    
    # ===== STEP 4: PROMOTION DECISION LOGIC =====
    # Now we decide based on the metrics whether to promote
    # The logic distinguishes between "first promotion" and "champion challenge"
    
    should_promote = False
    
    print("-" * 70)
    print("EVALUATION")
    print("-" * 70)
    
    if champion_value is None:
        # CASE 1: No champion exists - First promotion
        # We only check against the absolute threshold
        # This prevents a bad model from becoming the first champion
        
        print("Scenario: First-time promotion (no champion exists)")
        print(f"Check: Does candidate meet absolute threshold?")
        
        if greater_is_better:
            # Metric should be high (e.g. accuracy)
            if candidate_value < threshold:
                print(f"❌ FAILED: Candidate below minimum threshold")
                print(f"   {candidate_value:.4f} < {threshold:.4f}")
                print(f"   Model must reach at least {threshold:.4f} to become champion")
            else:
                print(f"✅ PASSED: Candidate meets threshold")
                print(f"   {candidate_value:.4f} >= {threshold:.4f}")
                should_promote = True
        else:
            # Metric should be low (e.g. loss)
            if candidate_value > threshold:
                print(f"❌ FAILED: Candidate above maximum threshold")
                print(f"   {candidate_value:.4f} > {threshold:.4f}")
                print(f"   Model must be at most {threshold:.4f} to become champion")
            else:
                print(f"✅ PASSED: Candidate meets threshold")
                print(f"   {candidate_value:.4f} <= {threshold:.4f}")
                should_promote = True
    
    else:
        # CASE 2: Champion exists - Challenger scenario
        # Now we check against both the threshold and the champion
        # The candidate must be SIGNIFICANTLY better, not just minimally
        
        print("Scenario: Champion challenge (existing champion)")
        print(f"Check: Is candidate significantly better than champion?")
        
        # Calculate the actual improvement
        improvement = candidate_value - champion_value
        
        # Calculate percentage improvement for better readability
        # We use abs() because for "lower is better" metrics the improvement
        # is negative (e.g. loss from 0.5 to 0.4 is -0.1 improvement)
        improvement_pct = (abs(improvement) / abs(champion_value)) * 100
        
        if greater_is_better:
            # Metric should be higher (e.g. accuracy)
            # Required value = Champion + Minimum Improvement
            required_value = champion_value + min_improvement
            
            if candidate_value < required_value:
                print(f"❌ FAILED: Improvement not significant enough")
                print(f"   Candidate: {candidate_value:.4f}")
                print(f"   Champion:  {champion_value:.4f}")
                print(f"   Required:  {required_value:.4f}")
                print(f"   Actual improvement: {improvement:+.4f} ({improvement_pct:.2f}%)")
                print(f"   Minimum required:   +{min_improvement:.4f}")
                print("")
                print(f"   The candidate is {'better' if improvement > 0 else 'worse'} than champion,")
                print(f"   but the improvement is not significant enough to justify promotion.")
            else:
                print(f"✅ PASSED: Significant improvement over champion")
                print(f"   Candidate: {candidate_value:.4f}")
                print(f"   Champion:  {champion_value:.4f}")
                print(f"   Improvement: {improvement:+.4f} ({improvement_pct:.2f}%)")
                print(f"   Required:    +{min_improvement:.4f}")
                should_promote = True
        else:
            # Metric should be lower (e.g. loss)
            # Required value = Champion - Minimum Improvement
            required_value = champion_value - min_improvement
            
            if candidate_value > required_value:
                print(f"❌ FAILED: Improvement not significant enough")
                print(f"   Candidate: {candidate_value:.4f}")
                print(f"   Champion:  {champion_value:.4f}")
                print(f"   Required:  {required_value:.4f}")
                print(f"   Actual improvement: {improvement:+.4f} ({improvement_pct:.2f}%)")
                print(f"   Minimum required:   -{min_improvement:.4f}")
                print("")
                print(f"   The candidate is {'better' if improvement < 0 else 'worse'} than champion,")
                print(f"   but the improvement is not significant enough to justify promotion.")
            else:
                print(f"✅ PASSED: Significant improvement over champion")
                print(f"   Candidate: {candidate_value:.4f}")
                print(f"   Champion:  {champion_value:.4f}")
                print(f"   Improvement: {improvement:+.4f} ({improvement_pct:.2f}%)")
                print(f"   Required:    -{min_improvement:.4f}")
                should_promote = True
    
    # ===== STEP 5: EXECUTE PROMOTION =====
    # Based on the decision we now execute the promotion
    # or return a failure
    
    print("-" * 70)
    
    if should_promote:
        print("🚀 PROMOTING MODEL")
        print("-" * 70)
        
        # Register the model and set the champion alias
        model_version = register_model(candidate_run_id, model_name, alias='champion')
        
        # Outputs for Argo Workflow
        # These lines are parsed by Argo to determine the workflow status
        print("")
        print("RESULT=passed")
        print(f"MODEL_VERSION={model_version}")
        
        print("=" * 70)
        print("✅ PROMOTION SUCCESSFUL")
        print(f"   Model {model_name} version {model_version} is now the champion")
        print(f"   Ready for export to ONNX in next pipeline step")
        print("=" * 70)
        
        sys.exit(0)
    
    else:
        print("⛔ NOT PROMOTING")
        print("-" * 70)
        
        # On failure we do not change anything in the registry
        # The old champion remains champion
        # This is important: Failure does not mean "bad model",
        # but only "not good enough for promotion"
        
        if champion_value is not None:
            print(f"   Current champion remains: version with {metric} = {champion_value:.4f}")
        else:
            print(f"   No champion set yet (first model didn't meet threshold)")
        
        print("")
        print(f"   To become champion, train a model with:")
        if greater_is_better:
            if champion_value is not None:
                print(f"   - {metric} > {champion_value + min_improvement:.4f}")
            else:
                print(f"   - {metric} >= {threshold:.4f}")
        else:
            if champion_value is not None:
                print(f"   - {metric} < {champion_value - min_improvement:.4f}")
            else:
                print(f"   - {metric} <= {threshold:.4f}")
        
        # Output for Argo Workflow
        print("")
        print("RESULT=failed")
        
        print("=" * 70)
        
        # No technical error, just no promotion
        # Argo Workflow step checks for RESULT=failed to decide
        sys.exit(0) 


if __name__ == "__main__":
    # Command-line interface for explicit Run ID passing
    # This allows Argo Workflows to pass the run_id from the train step
    parser = argparse.ArgumentParser(
        description="Promote ML Model based on Champion comparison",
        epilog="This script implements Quality Gate 1: Training Performance"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="MLflow Run ID of the candidate model (optional - defaults to last run)"
    )
    
    args = parser.parse_args()
    
    # Call main function
    promote(run_id=args.run_id)