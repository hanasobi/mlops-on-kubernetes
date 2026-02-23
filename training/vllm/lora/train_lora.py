"""
LoRA Fine-tuning Training Script

This script fine-tunes Mistral 7B using LoRA (Low-Rank Adaptation) on AWS certification data.

The workflow:
1. Load base model (Mistral 7B) with 4-bit quantization (QLoRA)
2. Apply LoRA adapters to specified layers
3. Train only the LoRA parameters (< 1% of total parameters)
4. Track experiments in MLflow
5. Save LoRA adapters (much smaller than full model)

This is analogous to your CV transfer learning, but much more parameter-efficient.
"""

import os
import sys
import argparse
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import mlflow
import pandas as pd

# Add scripts directory to path for imports
sys.path.append(os.path.dirname(__file__))
from config import (
    DEFAULT_TRAINING_CONFIG,
    get_lora_config,
    estimate_trainable_parameters,
    LORA_CONFIGS
)
from utils import (
    create_dataset,
    load_jsonl,
    DataCollatorForInstructionTuning,
    count_parameters,
    print_model_parameters,
    setup_mlflow
)
from mlflow_callback import create_mlflow_callback


def load_model_and_tokenizer(config, lora_config_name):
    """
    Load base model with quantization and apply LoRA.
    
    This function demonstrates the key concepts:
    1. 4-bit quantization reduces memory from ~28GB to ~7GB
    2. LoRA adapters are added to frozen base model
    3. Only LoRA parameters are trainable
    
    Args:
        config: TrainingConfig instance
        lora_config_name: Name of LoRA configuration to use
    
    Returns:
        tuple: (model, tokenizer, lora_config)
    """
    print("\n" + "=" * 80)
    print(f"Loading model: {config.model_name}")
    print("=" * 80)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        use_fast=True  # Use fast tokenizer when available
    )
    
    # Set padding token (Mistral doesn't have one by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id
    
    # Print tokenizer info
    print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")
    print(f"Vocabulary size: {len(tokenizer)}")
    print(f"Special tokens: PAD={tokenizer.pad_token_id}, "
          f"EOS={tokenizer.eos_token_id}, "
          f"BOS={tokenizer.bos_token_id}")
    
    # Configure 4-bit quantization (QLoRA)
    # This is the key to fitting 7B model on 16GB GPU
    if config.use_4bit:
        print("\nConfiguring 4-bit quantization (QLoRA)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # Enable 4-bit loading
            bnb_4bit_quant_type="nf4",  # Normal Float 4-bit (better than standard int4)
            bnb_4bit_compute_dtype=torch.float16,  # Compute in fp16 for better precision
            bnb_4bit_use_double_quant=True,  # Nested quantization for additional memory savings
        )
    else:
        bnb_config = None
        print("\nWarning: Loading without quantization - may not fit on T4!")
    
    # Load base model
    print(f"\nLoading base model from HuggingFace Hub...")
    print("This may take a few minutes on first run (downloading ~14GB)...")
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",  # Automatically distribute model across available GPUs
        trust_remote_code=True,
        torch_dtype=torch.float16 if config.use_4bit else torch.float32,
    )
    
    print(f"Model loaded: {model.__class__.__name__}")
    print(f"Model dtype: {model.dtype}")
    print(f"Device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'Not available'}")
    
    # Prepare model for k-bit training
    # This sets up gradient checkpointing and other optimizations
    if config.use_4bit:
        print("\nPreparing model for k-bit training...")
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.use_gradient_checkpointing
        )
    
    # Print base model parameters before LoRA
    print("\nBase model (before LoRA):")
    stats = count_parameters(model)
    print(f"Total parameters: {stats['total']:,}")
    
    # Get LoRA configuration
    lora_cfg = get_lora_config(lora_config_name)
    print("\n" + "=" * 80)
    print(f"Applying LoRA Configuration: {lora_cfg.name}")
    print("=" * 80)
    print(lora_cfg)
    
    # Estimate parameters (this is theoretical, we'll verify after applying LoRA)
    param_estimate = estimate_trainable_parameters(lora_cfg, stats['total'])
    print(f"\nEstimated trainable parameters: {param_estimate['trainable_params']:,} "
          f"({param_estimate['percentage']:.4f}%)")
    
    # Create PEFT LoRA configuration
    # This tells PEFT library which layers to adapt and how
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # We're doing causal language modeling
        r=lora_cfg.rank,  # Rank of the low-rank decomposition
        lora_alpha=lora_cfg.alpha,  # Scaling factor
        lora_dropout=lora_cfg.dropout,  # Dropout for regularization
        target_modules=lora_cfg.target_modules,  # Which modules to adapt
        bias="none",  # Don't adapt bias terms
        inference_mode=False,  # We're training, not doing inference
    )
    
    # Apply LoRA to model
    print("\nApplying LoRA adapters...")
    model = get_peft_model(model, peft_config)
    
    # Verify actual trainable parameters
    print("\nModel after LoRA:")
    print_model_parameters(model, detailed=False)
    
    # Enable gradient checkpointing if configured
    # This trades compute for memory: recomputes activations during backward pass
    # instead of storing them, reducing memory usage significantly
    if config.use_gradient_checkpointing:
        print("\nEnabling gradient checkpointing...")
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled - trades compute for memory")
    
    return model, tokenizer, lora_cfg


def train(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    config,
    lora_cfg,
    experiment_id
):
    """
    Train the model using HuggingFace Trainer.
    
    The Trainer handles:
    - Training loop with gradient accumulation
    - Evaluation on validation set
    - Learning rate scheduling
    - Checkpointing
    - Logging
    
    Args:
        model: PEFT model with LoRA adapters
        tokenizer: HuggingFace tokenizer
        train_dataset: Training dataset
        eval_dataset: Validation dataset
        config: TrainingConfig instance
        lora_cfg: LoRA configuration
        experiment_id: MLflow experiment ID
    
    Returns:
        Trainer instance (can be used for additional evaluation)
    """
    print("\n" + "=" * 80)
    print("Setting up training")
    print("=" * 80)
    
    # Create output directory for this specific LoRA config
    output_dir = os.path.join(config.output_dir, lora_cfg.name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Training arguments
    # These control the training loop behavior
    training_args = TrainingArguments(
        # Output & logging
        output_dir=output_dir,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=config.logging_steps,
        logging_strategy="steps",
        
        # Evaluation
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        
        # Training
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        max_grad_norm=1.0,  # Gradient clipping for stability
        
        # Optimization
        optim=config.optim,  # paged_adamw_8bit for memory efficiency
        fp16=config.fp16,  # T4 supports fp16
        bf16=config.bf16,  # Set to False for T4
        
        # Checkpointing
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,  # Load best checkpoint at end
        metric_for_best_model="eval_loss",  # Use validation loss to determine best model
        greater_is_better=False,  # Lower loss is better
        
        # Misc
        report_to=[],
        run_name=lora_cfg.name,
        seed=42,
        dataloader_num_workers=4,  # Parallel data loading
        remove_unused_columns=False,  # Keep all columns
        ddp_find_unused_parameters=False,  # For distributed training (not used here)
    )
    
    print(f"\nTraining configuration:")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size per device: {config.per_device_train_batch_size}")
    print(f"  Gradient accumulation steps: {config.gradient_accumulation_steps}")
    print(f"  Effective batch size: {config.effective_batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  LR scheduler: {config.lr_scheduler_type}")
    print(f"  Warmup steps: {config.warmup_steps}")
    print(f"  Weight decay: {config.weight_decay}")
    print(f"  Optimizer: {config.optim}")
    print(f"  FP16: {config.fp16}")
    print(f"  Gradient checkpointing: {config.use_gradient_checkpointing}")
    
    # Data collator
    # This handles padding and instruction masking
    data_collator = DataCollatorForInstructionTuning(
        tokenizer=tokenizer,
        mask_instruction=True  # Only compute loss on response tokens
    )
    
    print(f"\nDataset sizes:")
    print(f"  Training: {len(train_dataset)} examples")
    print(f"  Validation: {len(eval_dataset)} examples")

    # Create custom MLflow callback
    # This gives us full control over what gets logged to MLflow and when
    mlflow_cb = create_mlflow_callback(log_model_checkpoints=False)
    
    # Calculate training steps
    steps_per_epoch = len(train_dataset) // (
        config.per_device_train_batch_size * config.gradient_accumulation_steps
    )
    total_steps = steps_per_epoch * config.num_epochs
    print(f"\nTraining steps:")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps}")
    print(f"  Evaluation every {config.eval_steps} steps")
    print(f"  Checkpoint every {config.save_steps} steps")
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[mlflow_cb],
    )
    
    # Log hyperparameters to MLflow
    if experiment_id:
        print("\nLogging hyperparameters to MLflow...")
        mlflow.log_params({
            "model_name": config.model_name,
            "lora_rank": lora_cfg.rank,
            "lora_alpha": lora_cfg.alpha,
            "lora_dropout": lora_cfg.dropout,
            "lora_target_modules": ",".join(lora_cfg.target_modules),
            "effective_scaling": lora_cfg.effective_scaling,
            "learning_rate": config.learning_rate,
            "batch_size": config.per_device_train_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "effective_batch_size": config.effective_batch_size,
            "num_epochs": config.num_epochs,
            "max_seq_length": config.max_seq_length,
            "use_4bit": config.use_4bit,
            "use_gradient_checkpointing": config.use_gradient_checkpointing,
            "warmup_steps": config.warmup_steps,
            "lr_scheduler": config.lr_scheduler_type,
        })
        
        # Log model architecture info as parameters (static, not time-series)
        param_stats = count_parameters(model)
        mlflow.log_params({
            "total_parameters": param_stats['total'],
            "trainable_parameters": param_stats['trainable'],
            "trainable_percentage": f"{param_stats['percentage']:.2f}",
        })
    
    # Start training
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")
    
    try:
        train_result = trainer.train()
        
        print("\n" + "=" * 80)
        print("Training completed successfully!")
        print("=" * 80)
        print(f"Final training loss: {train_result.training_loss:.4f}")
        print(f"Training time: {train_result.metrics['train_runtime']:.2f} seconds")
        print(f"Samples per second: {train_result.metrics['train_samples_per_second']:.2f}")
        
        # Final evaluation
        print("\nRunning final evaluation...")
        eval_results = trainer.evaluate()
        print(f"Final evaluation loss: {eval_results['eval_loss']:.4f}")
        print(f"Perplexity: {torch.exp(torch.tensor(eval_results['eval_loss'])):.2f}")
        
        # Log final metrics to MLflow
        if experiment_id:
            mlflow.log_metrics({
                "final_train_loss": train_result.training_loss,
                "final_eval_loss": eval_results['eval_loss'],
                "final_perplexity": torch.exp(torch.tensor(eval_results['eval_loss'])).item(),
                "train_runtime_seconds": train_result.metrics['train_runtime'],
                "samples_per_second": train_result.metrics['train_samples_per_second'],
            })
        
        # Save LoRA adapter (best checkpoint, loaded by load_best_model_at_end)
        print(f"\nSaving LoRA adapter to {output_dir}...")
        trainer.save_model(output_dir)

        # Files saved:
        # - adapter_model.safetensors (LoRA weights, ~30-100 MB)
        # - adapter_config.json (LoRA configuration)

        print("\n" + "=" * 80)
        print("Training pipeline completed successfully!")
        print("=" * 80 + "\n")

        return trainer
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"Training failed with error:")
        print(f"{'='*80}")
        print(f"{type(e).__name__}: {e}")
        print(f"{'='*80}\n")
        raise


def main():
    """
    Main training function.

    Usage:
        python train_lora.py --train-file data/train.jsonl --val-file data/val.jsonl \
            --data-version 1 --lora-config standard
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning for Mistral 7B")
    parser.add_argument(
        "--train-file",
        type=str,
        required=True,
        help="Path to training JSONL file"
    )
    parser.add_argument(
        "--val-file",
        type=str,
        required=True,
        help="Path to validation JSONL file"
    )
    parser.add_argument(
        "--data-version",
        type=str,
        required=True,
        help="Data version identifier (logged to MLflow for traceability)"
    )
    parser.add_argument(
        "--lora-config",
        type=str,
        default="standard",
        choices=list(LORA_CONFIGS.keys()),
        help="LoRA configuration to use"
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow logging"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Test mode: only train for a few steps"
    )
    parser.add_argument(
        "--log-adapter-to-mlflow",
        action="store_true",
        help="Log adapter artifacts to MLflow (use in workflows where local storage is ephemeral)"
    )
    # Optional hyperparameter overrides (default: use config.py values)
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate")
    parser.add_argument("--num-epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override per-device train batch size")
    parser.add_argument("--max-seq-length", type=int, default=None, help="Override max sequence length")

    args = parser.parse_args()

    # Load configuration and set data paths from CLI
    config = DEFAULT_TRAINING_CONFIG
    config.train_dataset_path = args.train_file
    config.val_dataset_path = args.val_file
    config.data_version = args.data_version

    # Apply hyperparameter overrides
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.num_epochs is not None:
        config.num_epochs = args.num_epochs
    if args.batch_size is not None:
        config.per_device_train_batch_size = args.batch_size
    if args.max_seq_length is not None:
        config.max_seq_length = args.max_seq_length

    # Test mode: reduce epochs and steps for quick testing
    if args.test_mode:
        print("\n" + "!" * 80)
        print("TEST MODE: Reduced training for quick verification")
        print("!" * 80 + "\n")
        config.num_epochs = 1
        config.eval_steps = 10
        config.save_steps = 10
        config.logging_steps = 1

    print("\n" + "=" * 80)
    print("LoRA Fine-tuning Pipeline")
    print("=" * 80)
    print(f"Data version: {args.data_version}")
    print(f"Train file:   {args.train_file}")
    print(f"Val file:     {args.val_file}")
    print(f"LoRA Config:  {args.lora_config}")
    print(f"MLflow:       {'Disabled' if args.no_mlflow else 'Enabled'}")
    print(f"Test Mode:    {args.test_mode}")
    print("=" * 80 + "\n")
    
    # Setup MLflow
    experiment_id = None
    if not args.no_mlflow:
        experiment_id = setup_mlflow(config)
        if experiment_id:
            # Enable automatic system metrics logging (CPU, GPU, Memory, Disk)
            # Requires: pip install psutil nvidia-ml-py
            os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
            # Start MLflow run
            mlflow.start_run(experiment_id=experiment_id, run_name=f"{config.mlflow_model_name}-{args.lora_config}")
            # Log data provenance
            mlflow.set_tag("data.version", args.data_version)
            mlflow.set_tag("data.source", os.environ.get("DATA_SOURCE", args.train_file))
    
    try:
        # Load model and tokenizer
        model, tokenizer, lora_cfg = load_model_and_tokenizer(config, args.lora_config)
        
        # Load datasets
        print("\n" + "=" * 80)
        print("Loading datasets")
        print("=" * 80)
        
        train_dataset = create_dataset(
            config.train_dataset_path,
            tokenizer,
            max_length=config.max_seq_length
        )
        
        eval_dataset = create_dataset(
            config.val_dataset_path,
            tokenizer,
            max_length=config.max_seq_length
        )

        # Log dataset lineage to MLflow
        if experiment_id:
            try:
                data_source = os.environ.get("DATA_SOURCE", "")

                raw_train = load_jsonl(config.train_dataset_path)
                train_mlflow_ds = mlflow.data.from_pandas(
                    pd.DataFrame(raw_train),
                    source=f"{data_source}/train.jsonl" if data_source else config.train_dataset_path,
                    name=f"train-v{args.data_version}"
                )
                mlflow.log_input(train_mlflow_ds, context="training")

                raw_val = load_jsonl(config.val_dataset_path)
                val_mlflow_ds = mlflow.data.from_pandas(
                    pd.DataFrame(raw_val),
                    source=f"{data_source}/val.jsonl" if data_source else config.val_dataset_path,
                    name=f"val-v{args.data_version}"
                )
                mlflow.log_input(val_mlflow_ds, context="validation")

                print(f"  Dataset lineage logged (train: {len(raw_train)} samples, val: {len(raw_val)} samples)")
            except Exception as e:
                print(f"  Warning: Dataset logging failed: {e}")

        # Train
        trainer = train(
            model,
            tokenizer,
            train_dataset,
            eval_dataset,
            config,
            lora_cfg,
            experiment_id
        )
        
        # Log adapter to MLflow if requested (workflow mode: local storage is ephemeral)
        if args.log_adapter_to_mlflow and experiment_id:
            adapter_dir = os.path.join(config.output_dir, lora_cfg.name)
            print(f"\nLogging adapter artifacts to MLflow...")
            for filename in ["adapter_model.safetensors", "adapter_config.json"]:
                filepath = os.path.join(adapter_dir, filename)
                if os.path.exists(filepath):
                    mlflow.log_artifact(filepath, "adapter")
                    print(f"  Logged: {filename}")

        # Write run-id for Argo Workflow output parameter
        if experiment_id:
            run_id = mlflow.active_run().info.run_id
            with open("/tmp/mlflow_run_id", "w") as f:
                f.write(run_id)
            print(f"\nMLflow run ID: {run_id}")
            print(f"View results: {config.mlflow_tracking_uri}")

        print("\nSuccess! Model trained and saved.")
        print(f"Adapter saved to: {config.output_dir}/{lora_cfg.name}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user (Ctrl+C)")
        print("Partial results may be saved in output directory")

    finally:
        # End MLflow run
        if experiment_id and not args.no_mlflow:
            mlflow.end_run()


if __name__ == "__main__":
    main()