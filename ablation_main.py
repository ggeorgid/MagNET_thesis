import torch
import numpy as np
import argparse
from pathlib import Path
import random
import yaml
import matplotlib
import os
import wandb
import optuna
import json
matplotlib.use("Agg")  # Use Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
from torchinfo import summary
import datetime

# Import dataset-related functions from the `data/` folder
from data.download_dataset import download_dataset
from data.preprocess_dataset import check_dataset_exists, convert_to_npy, convert_to_tensors
from data.dataloaders_ablation import split_dataset, create_dataloaders, inspect_dataloader, check_dataloader_distribution, check_reproducibility
from data.voltage_coreloss_dataset import VoltageCoreLossDataset  # Updated import
# Import the model
from models.ablation_model import VoltageModel  # Updated import
from utils.train_utils import train_model

def parse_args():
    """Parse command-line arguments to override YAML hyperparameters or trigger Optuna optimization."""
    parser = argparse.ArgumentParser(description="Run Voltage Model Training or Optuna Optimization")
    parser.add_argument("--num_epoch", type=int, help="Number of epochs")
    parser.add_argument("--data_subset_size", type=int, help="Subset size of dataset")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--optimize", action="store_true", help="Run Optuna optimization")
    return parser.parse_args()

def get_current_date_str():
    """Return the current date as 'MonthDD', e.g., 'March17'."""
    now = datetime.datetime.now()
    month = now.strftime("%B")  # Full month name, e.g., "March"
    day = int(now.strftime("%d"))  # Day without leading zero, e.g., 17
    return f"{month}{day}"

def objective(trial, config, device, train_loader, valid_loader, test_loader):
    # Suggest hyperparameters
    config["NUM_EPOCH"] = trial.suggest_int("num_epoch", 10, 50)
    config["BATCH_SIZE"] = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    config["LEARNING_RATE"] = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)

    print(f"\n🔹 Running trial with: NUM_EPOCH={config['NUM_EPOCH']}, BATCH_SIZE={config['BATCH_SIZE']}, LEARNING_RATE={config['LEARNING_RATE']}")

    # Set project name with readable date
    date_str = get_current_date_str()
    project_name = f"Ablation_OptunaOptimization_{date_str}"

    # Define the project root as the directory where wavelet_main.py resides
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

    # Specify a unique save path for this model
    save_path = os.path.join(PROJECT_ROOT, 'ablation_best_model.pth')
    
    # Initialize wandb run
    wandb.init(
        project=project_name,
        name=f"trial_{trial.number}_epochs_{config['NUM_EPOCH']}_lr_{config['LEARNING_RATE']}_bs_{config['BATCH_SIZE']}",
        config=config
    )

    model = VoltageModel().to(device)  # Updated to VoltageModel
    trained_model, best_val_loss = train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        config=config,
        device=device,
        save_path=save_path,
        trial=trial
    )

    wandb.log({"Loss/Validation": best_val_loss})
    wandb.finish()

    return best_val_loss
    
def run_training(config, device, train_loader, valid_loader, test_loader):
    date_str = get_current_date_str()
    project_name = f"Ablation_ManualSweep_{date_str}"
    
    # Define the project root as the directory where wavelet_main.py resides
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

    # Specify a unique save path for this model
    save_path = os.path.join(PROJECT_ROOT, 'ablation_best_model.pth')

    wandb.init(
        project=project_name,
        name=f"epochs_{config['NUM_EPOCH']}_lr_{config['LEARNING_RATE']}_bs_{config['BATCH_SIZE']}",
        config=config
    )

    model = VoltageModel().to(device)  # Updated to VoltageModel
    trained_model, best_val_loss = train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        config=config,
        device=device,
        save_path=save_path
    )

    wandb.log({"Loss/Validation": best_val_loss})
    wandb.finish()  

def main():    
    # Setup
    print(f"\n🔹 Torch version: {torch.__version__}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔹 Selected device: {device}\n")

    # Load Hyperparameters
    with open("hyperparameters.yaml", "r") as file:
        config = yaml.safe_load(file)

    args = parse_args()
    
    config["NUM_EPOCH"] = args.num_epoch if args.num_epoch is not None else config["NUM_EPOCH"]
    config["DATA_SUBSET_SIZE"] = args.data_subset_size if args.data_subset_size is not None else config.get("DATA_SUBSET_SIZE", None)
    config["LEARNING_RATE"] = args.learning_rate if args.learning_rate is not None else config["LEARNING_RATE"]
    config["BATCH_SIZE"] = args.batch_size if args.batch_size is not None else config["BATCH_SIZE"]
    
    # Set random seeds
    torch.manual_seed(config["SEED"])
    torch.cuda.manual_seed_all(config["SEED"])
    np.random.seed(config["SEED"])
    random.seed(config["SEED"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🔹 [INFO] Random seed set to {config['SEED']}")
    
    print(f"\n🔹 Hyperparameters Loaded:")
    print(config)
    
    # Dataset Download & Processing
    raw_data_path = Path("data/raw")    
    drive_url = "https://drive.google.com/file/d/1syNCq6cr4P5rAEdkIZs5c9Ee39-NxKY9/view"    
    desired_filename = "dataset_raw"
    dataset_dir = raw_data_path / "dataset"  
    FIGURE_DIR = "figures"
    os.makedirs(FIGURE_DIR, exist_ok=True)

    if not config.get("DOWNLOAD_DATASET", True) and not check_dataset_exists(dataset_dir, check_for_npy=False):
        raise FileNotFoundError("[ERROR] Dataset not found. Set DOWNLOAD_DATASET to True or place the dataset manually.")

    if config.get("DOWNLOAD_DATASET", True):
        if not check_dataset_exists(dataset_dir, check_for_npy=False):
            print("[INFO] Raw dataset not found. Downloading...")
            download_dataset(drive_url, raw_data_path, desired_filename)
        else:
            print("[INFO] Raw dataset already exists. Skipping download.")

    preprocessed_data_path = Path("data/processed")
    preprocessed_data_path.mkdir(parents=True, exist_ok=True)

    if not check_dataset_exists(preprocessed_data_path, check_for_npy=True):
        print("[INFO] Processed dataset not found. Converting raw data to .npy format...")
        convert_to_npy(preprocessed_data_path, raw_data_path)
    else:
        print("[INFO] Processed dataset already exists. Skipping conversion.")

    dataset = np.load(preprocessed_data_path / "dataset.npy")
    print(f"\n🔹 Current dataset shape: {dataset.shape}") 
    
    # Dataset Subset Selection
    subset_indices_file = preprocessed_data_path / "subset_indices.npy"    

    if subset_indices_file.exists():
        indices = np.load(subset_indices_file, allow_pickle=True)
        if not isinstance(indices, np.ndarray) or indices.ndim == 0:
            print("[WARNING] Subset indices file is invalid. Regenerating subset.")
            indices = None
    else:
        indices = None

    if indices is None:
        print("[INFO] Generating new subset indices.")
        if config.get("DATA_SUBSET_SIZE", None) is None or config["DATA_SUBSET_SIZE"] > len(dataset):
            config["DATA_SUBSET_SIZE"] = len(dataset)
            print(f"[INFO] Using full dataset: {config['DATA_SUBSET_SIZE']} samples")
        seed = config.get("SEED", 42)
        print(f"[INFO] Using random seed: {seed}")
        np.random.seed(seed)
        indices = np.random.choice(len(dataset), config["DATA_SUBSET_SIZE"], replace=False)
        np.save(subset_indices_file, indices)
        print("[INFO] Saved new dataset subset indices for future runs.")

    dataset_subset = dataset[indices]         
    tensor_dataset = convert_to_tensors(dataset_subset)
    
    # Caching and Voltage-CoreLoss Processing
    core_loss_path = preprocessed_data_path / "core_loss.npy"

    if config.get("USE_CACHED_CORE_LOSS", False) and core_loss_path.exists():
        print("[INFO] Loading cached core loss.")
        voltage_dataset = VoltageCoreLossDataset(
            dataset_path=preprocessed_data_path / "dataset.npy",
            core_loss_path=core_loss_path
        )
    else:
        print("[INFO] Cached core loss not found or dataset changed. Computing core loss...")
        tensor_dataset = convert_to_tensors(dataset_subset)
        voltage_dataset = VoltageCoreLossDataset(V_I_dataset=tensor_dataset)
        if config.get("USE_CACHED_CORE_LOSS", False):
            np.save(core_loss_path, voltage_dataset.core_loss_values)
            print("[INFO] Cached new core loss for future use.")

    print(f"[INFO] Created VoltageCoreLossDataset with {len(voltage_dataset)} samples.")

    # Inspecting Voltage-CoreLoss Dataset
    # # Visualizing the first 5 samples
    # for i in range(5):
    #     voltage, core_loss = voltage_dataset[i]
    #     print(f"\n🔹 [DEBUG] Sample {i}:")
    #     print(f"   - Voltage Shape: {voltage.shape} (Expected: [1, 8000])")
    #     print(f"   - Core Loss Value: {core_loss.item()}")
    #     print(f"   - Voltage Sample Values (first 10 points):\n{voltage.squeeze().numpy()[:10]}")
    
    # # Inspecting Core Loss Statistics
    # core_loss_values = np.array([voltage_dataset[i][1].item() for i in range(len(voltage_dataset))])
    # print("\n🔹 [DEBUG] Core Loss Dataset Statistics:")
    # print(f"   - Mean: {np.mean(core_loss_values)}")
    # print(f"   - Min: {np.min(core_loss_values)}")
    # print(f"   - Max: {np.max(core_loss_values)}")
    # print(f"   - Std Dev: {np.std(core_loss_values)}")
    
    # # Visualizing Random Samples both in Voltage and Core Loss
    # random_indices = np.random.choice(len(voltage_dataset), 5, replace=False)
    # for idx in random_indices:
    #     voltage, core_loss = voltage_dataset[idx]
    #     print(f"\n🔹 [DEBUG] Random Sample {idx}:")
    #     print(f"   - Voltage Shape: {voltage.shape}")
    #     print(f"   - Core Loss Value: {core_loss.item()}")
    #     print(f"   - Voltage Mean: {voltage.mean().item()}, Variance: {voltage.var().item()}")

    # Visualize a Random Voltage Time Series
    random_idx = random.randint(0, len(voltage_dataset) - 1)
    voltage, _ = voltage_dataset[random_idx]
    print(f"🔹 Visualizing voltage time series from index {random_idx}...")

    plt.figure(figsize=(10, 4))
    plt.plot(voltage.squeeze().numpy())
    plt.title(f"Voltage Time Series Sample {random_idx}")
    plt.xlabel("Time Steps")
    plt.ylabel("Voltage")
    voltage_filename = os.path.join(FIGURE_DIR, f"voltage_sample_{random_idx}.png")
    plt.savefig(voltage_filename)
    plt.close()

    print(f"✅ Saved voltage time series visualization as {voltage_filename}")
    
    # Splitting the Dataset
    train_dataset, valid_dataset, test_dataset = split_dataset(
        voltage_dataset, 
        train_ratio=0.7, 
        valid_ratio=0.15, 
        test_ratio=0.15, 
        seed=config["SEED"], 
        save_dir=preprocessed_data_path
    )

    print(f"\n🔹 Train Set: {len(train_dataset)} samples")
    print(f"🔹 Validation Set: {len(valid_dataset)} samples")
    print(f"🔹 Test Set: {len(test_dataset)} samples")
    
    # Create DataLoaders
    train_loader, valid_loader, test_loader = create_dataloaders(
        train_dataset, 
        valid_dataset, 
        test_dataset, 
        batch_size=config["BATCH_SIZE"], 
        num_workers=4, 
        use_gpu=config["USE_GPU"]
    )
    
    print(f"🔹 [DEBUG] Train DataLoader Size: {len(train_loader.dataset)} (Expected: {len(train_dataset)})")
    print(f"🔹 [DEBUG] Valid DataLoader Size: {len(valid_loader.dataset)} (Expected: {len(valid_dataset)})")
    print(f"🔹 [DEBUG] Test DataLoader Size: {len(test_loader.dataset)} (Expected: {len(test_dataset)})")
    
    # Prints the first 5 samples inside the train, validation, and test dataloaders, limiting to 10 values for Voltage from 8000
    # inspect_dataloader(train_loader, "Train DataLoader")
    # inspect_dataloader(valid_loader, "Validation DataLoader")
    # inspect_dataloader(test_loader, "Test DataLoader")
    
    # Creates a histogram plot of the core loss distribution for each DataLoader in figures directory
    check_dataloader_distribution(train_loader, voltage_dataset, "Train DataLoader")
    check_dataloader_distribution(valid_loader, voltage_dataset, "Validation DataLoader")
    check_dataloader_distribution(test_loader, voltage_dataset, "Test DataLoader")
    
    # Decide to train or optimize
    if args.optimize:
        study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
        study.optimize(lambda trial: objective(trial, config, device, train_loader, valid_loader, test_loader), n_trials=100)
        print("\n✅ Best Hyperparameters:", study.best_params)
        with open("best_hyperparameters.json", "w") as f:
            json.dump(study.best_params, f)
        print("✅ Best hyperparameters saved to best_hyperparameters.json")
    else:
        run_training(config, device, train_loader, valid_loader, test_loader)
    
    # Clean Up
    train_indices_file = preprocessed_data_path / "train_indices.npy"
    valid_indices_file = preprocessed_data_path / "valid_indices.npy"
    test_indices_file = preprocessed_data_path / "test_indices.npy"

    for file in [subset_indices_file, train_indices_file, valid_indices_file, test_indices_file]:
        if file.exists():
            print(f"\n[INFO] Deleting {file.name} to enforce seed-based reproducibility in next runs.")
            file.unlink()

if __name__ == "__main__":
    main()