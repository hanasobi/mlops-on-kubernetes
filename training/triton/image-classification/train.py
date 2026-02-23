#!/usr/bin/env python3
"""
ResNet18 Training on ImageNette with S3 Data and MLflow Tracking.

This script is designed to run in Argo Workflows and uses:
- S3 for dataset storage (via s3fs)
- MLflow for experiment tracking
- GPU if available, otherwise CPU
"""

import os
import time
import argparse
import subprocess
import yaml
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import mlflow
import mlflow.pytorch


def load_config(config_path='config.yaml'):
    """
    Loads the hyperparameter config from YAML.

    This function reads the config.yaml which contains default values for all
    hyperparameters. In the workflow we can override specific values
    via command-line args.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_mlflow(config):
    """
    Initializes MLflow Tracking with complete parameter logging.
    """
    tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow.ai-platform')
    mlflow.set_tracking_uri(tracking_uri)
    
    experiment_name = config['experiment']['name']
    mlflow.set_experiment(experiment_name)
    
    mlflow.start_run(run_name=config['model']['name'])
    
    # Log ALL config parameters for full reproducibility
    # We flatten the nested config into dot-notation
    mlflow.log_params({
        # Model Config
        'model.name': config['model']['name'],
        'model.architecture': config['model']['architecture'],
        
        # Experiment Config
        'experiment.name': config['experiment']['name'],
        
        # Training Config
        'training.batch_size': config['training']['batch_size'],
        'training.learning_rate': config['training']['learning_rate'],
        'training.epochs': config['training']['epochs'],
        'training.optimizer': config['training']['optimizer'],
        
        # Data Config
        'data.s3_path': config['data']['s3_path'],
        'data.num_workers': config['data']['num_workers'],

        # System Info
        'pytorch_version': torch.__version__,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'none',
    })

def download_dataset_from_s3(s3_path, local_path):
    """
    Downloads dataset from S3 to a local path if not already present.

    Uses aws s3 sync which is robust and only downloads missing files.
    """
    local_path = Path(local_path)
    
    # Check if dataset already exists locally
    train_path = local_path / "train"
    val_path = local_path / "val"
    
    if train_path.exists() and val_path.exists():
        # Quick sanity check - are there enough files?
        train_files = len(list(train_path.rglob("*.JPEG")))
        val_files = len(list(val_path.rglob("*.JPEG")))
        
        if train_files > 9000 and val_files > 3000:
            print(f"Dataset already exists locally at {local_path}")
            print(f"  Train: {train_files} images")
            print(f"  Val: {val_files} images")
            return str(local_path)
    
    # Dataset not present or incomplete - download from S3
    print(f"Downloading dataset from {s3_path} to {local_path}")
    
    # Create local directory
    local_path.mkdir(parents=True, exist_ok=True)
    
    # Use aws s3 sync - robust and efficient
    cmd = [
        "aws", "s3", "sync",
        s3_path,
        str(local_path),
        "--region", "eu-central-1",
        "--quiet"  # Less verbose output
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Download completed successfully")
        
        # Verify download
        train_files = len(list(train_path.rglob("*.JPEG")))
        val_files = len(list(val_path.rglob("*.JPEG")))
        print(f"  Train: {train_files} images")
        print(f"  Val: {val_files} images")
        
        return str(local_path)
        
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")
        print(f"Stderr: {e.stderr}")
        raise

def create_datasets(config):
    """
    Creates train and val datasets from local data.

    Downloads data from S3 if needed.
    """
    # S3 path from config
    s3_path = config['data']['s3_path']
    
    # Local path where we download the data to
    # /tmp on Kubernetes nodes is typically on the local disk
    local_data_path = "/tmp/imagenette-data"
    
    # Download from S3 (or skip if already present)
    dataset_root = download_dataset_from_s3(s3_path, local_data_path)
    
    print(f"Using dataset from local path: {dataset_root}")
    
    # Standard ImageNet Transforms
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Now we use local paths instead of S3
    train_dataset = datasets.ImageFolder(
        root=os.path.join(dataset_root, "train"),
        transform=train_transform
    )
    
    val_dataset = datasets.ImageFolder(
        root=os.path.join(dataset_root, "val"),
        transform=val_transform
    )
    
    print(f"Train dataset: {len(train_dataset)} images")
    print(f"Val dataset: {len(val_dataset)} images")
    
    return train_dataset, val_dataset

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    Trains the model for one epoch.

    This is almost identical to Phase 1, with additional performance
    tracking to see if S3 becomes a bottleneck.
    """
    model.train()
    running_loss = 0.0
    
    # Performance Tracking
    epoch_start = time.time()
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Progress logging every 50 batches
        if batch_idx % 50 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, "
                  f"Loss: {loss.item():.4f}")
    
    epoch_time = time.time() - epoch_start
    avg_loss = running_loss / len(train_loader)
    
    print(f"Epoch {epoch} completed in {epoch_time:.2f}s, Avg Loss: {avg_loss:.4f}")
    
    return avg_loss, epoch_time

def validate(model, val_loader, criterion, device):
    """
    Validates the model on the validation set.

    Standard validation loop - calculate accuracy and loss.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / len(val_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy

def main():
    """
    Main Training Function.

    Orchestrates the complete training process:
    1. Load config
    2. MLflow setup
    3. Device selection (GPU/CPU)
    4. Datasets & DataLoaders
    5. Model, loss, optimizer
    6. Training loop
    7. Best model tracking
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--epochs', type=int, help='Override epochs from config')
    parser.add_argument('--learning-rate', type=float, help='Override LR from config')
    parser.add_argument('--batch-size', type=int, help='Override batch size from config')
    parser.add_argument('--num-workers', type=int, help='Override num_workers from config')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Command-Line Overrides
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.num_workers:
        config['data']['num_workers'] = args.num_workers
    
    # MLflow Setup
    setup_mlflow(config)

    # These tags categorize the training approach and make it easy later
    # to filter and compare different approaches
    print("Setting experiment categorization tags...")
    mlflow.set_tag("approach", config['model']['approach'])
    mlflow.set_tag("pretrained", str(config['model']['pretrained']))
    
    if config['model']['frozen_layers'] is not None:
        mlflow.set_tag("frozen_layers", config['model']['frozen_layers'])    

    # Additional tag for the architecture (useful when we later
    # compare different architectures in the same experiment)
    mlflow.set_tag("architecture", config['model']['architecture'])
    
    print(f"  Approach: {config['model']['approach']}")
    print(f"  Pretrained: {config['model']['pretrained']}")
    print(f"  Architecture: {config['model']['architecture']}")        
    
    # Device Selection
    # CUDA is available on GPU nodes, falls back to CPU if not
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    mlflow.log_param('device', str(device))
    
    # Create datasets
    train_dataset, val_dataset = create_datasets(config)
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Create model
    # ResNet18 from scratch (no pretrained for this experiment)
    num_classes = len(train_dataset.classes)
    
    # Depending on the config we either load pretrained weights
    # or initialize randomly
    if config['model']['pretrained']:
        print(f"Loading {config['model']['architecture']} with pretrained ImageNet weights")
        weights = 'IMAGENET1K_V1'  # Standard pretrained weights for ResNet
    else:
        print(f"Initializing {config['model']['architecture']} from scratch")
        weights = None
    
    model = models.resnet18(weights=weights)

    # Adjust classifier for ImageNette (10 classes instead of 1000)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # If frozen_layers is set in the config, we freeze the
    # corresponding layers (for feature extraction)
    if config['model']['frozen_layers'] == 'backbone':
        print("Freezing backbone layers (feature extraction mode)")
        
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        
        # Make only the final FC layer trainable
        for param in model.fc.parameters():
            param.requires_grad = True
        
        # Log how many parameters are trainable
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    model = model.to(device)
    
    print(f"Model: ResNet18 with {num_classes} classes")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    
    if config['training']['optimizer'].lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), 
                              lr=config['training']['learning_rate'])
    else:
        optimizer = optim.SGD(model.parameters(), 
                             lr=config['training']['learning_rate'],
                             momentum=0.9)
    
    # Training Loop
    best_val_acc = 0.0
    best_model_state = None
    best_epoch = 0 
    
    for epoch in range(1, config['training']['epochs'] + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{config['training']['epochs']}")
        print(f"{'='*50}")
        
        # Train
        train_loss, epoch_time = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        
        # Best Model Tracking
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            print(f"New best validation accuracy: {val_acc:.2f}%")

            # Save the state of the best model
            # We make a true deep copy on CPU so that the weights
            # are not overwritten when training continues
            best_model_state = {
                key: value.cpu().clone() 
                for key, value in model.state_dict().items()
            }
            
            # Checkpoint with epoch in the name (for traceability)
            # These checkpoints are useful for debugging but are
            # not loaded by the Model Registry
            mlflow.pytorch.log_model(model, f"model_epoch_{epoch}")

        # Log metrics to MLflow
        mlflow.log_metrics({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'best_val_accuracy': best_val_acc,
            'epoch_time': epoch_time
        }, step=epoch)
        

    # Final Summary
    print(f"\n{'='*50}")
    print(f"Training completed!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"{'='*50}")

    if best_model_state is not None:
        print(f"Loaded best model (achieved {best_val_acc:.2f}%　accuracy)")

        # Move model to CPU if it was on GPU
        # This is important so that load_state_dict works consistently
        model = model.cpu()

        # Load the best weights back
        model.load_state_dict(best_model_state)

        print(f"✅ Best model loaded successfully")

    else:
        print("⚠️  Warning: No best model state found (this should not happen)")

    # Save the final best model with the standard name "model"
    # This is the artifact that is loaded by the Model Registry and export_to_onnx.py.
    # It is guaranteed to be the best model from training.
    print(f"Saving final best model as 'model' artifact for Model Registry")
    mlflow.pytorch.log_model(model, "model")
    
    # Best accuracy as final metric
    print(f"Saving final best model as 'model' artifact for Model Registry")
    mlflow.log_metric('best_val_accuracy', best_val_acc)
    mlflow.log_param('best_epoch', best_epoch)

    # Write Run ID for Argo output parameter
    run = mlflow.active_run()
    if run:
        run_id = run.info.run_id
        with open("/tmp/mlflow_run_id", 'w') as f:
            f.write(run_id)
        print(f"MLflow Run ID written for Argo: {run_id}")
    
    # End MLflow run
    mlflow.end_run()
    
    print(f"Results logged to MLflow: {mlflow.get_tracking_uri()}")

if __name__ == '__main__':
    main()