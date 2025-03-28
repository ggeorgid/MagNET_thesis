import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

# Ensure figures directory exists
FIGURE_DIR = "figures"
os.makedirs(FIGURE_DIR, exist_ok=True)

# Split Dataset Utility Functions
def split_dataset(dataset, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15, seed=42, save_dir=Path("data/processed")):
    """
    Splits dataset into train, validation, and test sets while ensuring consistency across runs.
    
    Args:
        dataset (Dataset): The dataset to be split.
        train_ratio (float): Proportion for training set.
        valid_ratio (float): Proportion for validation set.
        test_ratio (float): Proportion for test set.
        seed (int): Random seed for reproducibility.
        save_dir (Path): Directory to save/load dataset splits.
    
    Returns:
        tuple: Train, validation, and test dataset subsets.
    """
    total_size = len(dataset)

    split_files = {
        "train": save_dir / "train_indices.npy",
        "valid": save_dir / "valid_indices.npy",
        "test": save_dir / "test_indices.npy",
    }

    if all(f.exists() for f in split_files.values()):
        train_indices = np.load(split_files["train"])
        valid_indices = np.load(split_files["valid"])
        test_indices = np.load(split_files["test"])

        if total_size == (len(train_indices) + len(valid_indices) + len(test_indices)):
            print("[INFO] Loaded dataset split indices from previous runs.")
        else:
            print("[WARNING] Dataset size has changed. Regenerating split indices.")
    else:
        np.random.seed(seed)
        indices = np.random.permutation(total_size)
        train_end = int(total_size * train_ratio)
        valid_end = train_end + int(total_size * valid_ratio)
        train_indices, valid_indices, test_indices = np.split(indices, [train_end, valid_end])

        np.save(split_files["train"], train_indices)
        np.save(split_files["valid"], valid_indices)
        np.save(split_files["test"], test_indices)
        print("[INFO] Saved dataset split indices for future runs.")

    train_subset = torch.utils.data.Subset(dataset, train_indices)
    valid_subset = torch.utils.data.Subset(dataset, valid_indices)
    test_subset = torch.utils.data.Subset(dataset, test_indices)

    check_core_loss_distribution(train_subset, valid_subset, test_subset, dataset)

    return train_subset, valid_subset, test_subset

def check_core_loss_distribution(train_subset, valid_subset, test_subset, dataset):
    """
    Prints core loss statistics for train, validation, and test sets.
    """
    train_core_loss = np.array([dataset.core_loss_values[i] for i in train_subset.indices])
    valid_core_loss = np.array([dataset.core_loss_values[i] for i in valid_subset.indices])
    test_core_loss = np.array([dataset.core_loss_values[i] for i in test_subset.indices])

    print("\n🔹 [DEBUG] Core Loss Distribution Across Splits:")
    print(f"   - Train: Mean = {np.mean(train_core_loss):.5e}, Variance = {np.var(train_core_loss):.5e}")
    print(f"   - Validation: Mean = {np.mean(valid_core_loss):.5e}, Variance = {np.var(valid_core_loss):.5e}")
    print(f"   - Test: Mean = {np.mean(test_core_loss):.5e}, Variance = {np.var(test_core_loss):.5e}")
    
    print(f"   - Train: Min = {np.min(train_core_loss):.5e}, Max = {np.max(train_core_loss):.5e}")
    print(f"   - Validation: Min = {np.min(valid_core_loss):.5e}, Max = {np.max(valid_core_loss):.5e}")
    print(f"   - Test: Min = {np.min(test_core_loss):.5e}, Max = {np.max(test_core_loss):.5e}")

    plot_core_loss_distribution(train_core_loss, valid_core_loss, test_core_loss)

def plot_core_loss_distribution(train_core_loss, valid_core_loss, test_core_loss):
    """
    Plots and saves core loss distribution for train, validation, and test sets using Seaborn.
    """
    plt.figure(figsize=(12, 5))

    sns.histplot(train_core_loss, bins=50, kde=True, label="Train", color="blue", alpha=0.6)
    sns.histplot(valid_core_loss, bins=50, kde=True, label="Validation", color="orange", alpha=0.6)
    sns.histplot(test_core_loss, bins=50, kde=True, label="Test", color="green", alpha=0.6)

    plt.xlabel("Core Loss Value")
    plt.ylabel("Density")
    plt.title("Core Loss Distribution Across Train/Valid/Test Splits")
    plt.legend()

    plot_filename = os.path.join(FIGURE_DIR, "core_loss_distribution.png")
    plt.savefig(plot_filename)
    plt.close()

    print(f"✅ Saved core loss distribution plot as {plot_filename}") 

# DataLoader Utility Functions
def create_dataloaders(train_dataset, valid_dataset, test_dataset, batch_size, num_workers=min(4, os.cpu_count()), use_gpu=True):
    """
    Creates DataLoader objects for training, validation, and testing datasets.

    Args:
        train_dataset (Dataset): Dataset for training.
        valid_dataset (Dataset): Dataset for validation.
        test_dataset (Dataset): Dataset for testing.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses to use for data loading.
        use_gpu (bool): If True, enables pinned memory for faster GPU transfers.

    Returns:
        tuple: DataLoaders for training, validation, and testing.
    """
    if len(train_dataset) == 0 or len(valid_dataset) == 0 or len(test_dataset) == 0:
        raise ValueError("[ERROR] One or more datasets are empty! Check data preprocessing.")

    kwargs = {'num_workers': num_workers}
    if torch.cuda.is_available() and use_gpu:
        kwargs['pin_memory'] = True

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, valid_loader, test_loader

def inspect_dataloader(dataloader, name="DataLoader", num_samples=5):
    """
    Inspects a DataLoader by printing a few sample values.

    Args:
        dataloader: The DataLoader object to inspect.
        name (str): Name of the DataLoader (train, validation, test).
        num_samples (int): Number of samples to print.

    Returns:
        None (Prints sample data).
    """
    print(f"\n🔹 [DEBUG] Inspecting {name}...")
    
    batch = next(iter(dataloader))
    voltages, core_loss_values = batch

    print(f"   - Voltage Batch Shape: {voltages.shape} (Expected: [batch_size, 1, 8000])")
    print(f"   - Core Loss Batch Shape: {core_loss_values.shape} (Expected: [batch_size, 1])")

    for i in range(min(num_samples, len(voltages))):
        print(f"\n🔹 [DEBUG] Sample {i} in {name}:")
        print(f"   - Core Loss Value: {core_loss_values[i].item()}")
        print(f"   - Voltage Sample Values (first 10 points):\n{voltages[i, 0, :10].cpu().numpy()}")

def check_dataloader_distribution(dataloader, dataset, name="DataLoader"):
    """
    Checks the core loss distribution inside a DataLoader and saves the plot.

    Args:
        dataloader: The DataLoader object to inspect.
        dataset: The original dataset (VoltageCoreLossDataset).
        name (str): Name of the DataLoader (train, validation, test).

    Returns:
        None (Prints core loss distribution and saves a histogram plot).
    """
    core_loss_values = []
    
    for batch in dataloader:
        _, core_loss_batch = batch
        core_loss_values.extend(core_loss_batch.numpy().flatten())

    core_loss_values = np.array(core_loss_values)

    print(f"\n🔹 [DEBUG] {name} Core Loss Distribution:")
    print(f"   - Mean: {core_loss_values.mean():.5e}")
    print(f"   - Min: {core_loss_values.min():.5e}")
    print(f"   - Max: {core_loss_values.max():.5e}")
    print(f"   - Variance: {core_loss_values.var():.5e}")

    dataset_core_loss = np.array(dataset.core_loss_values)
    print(f"   - Full Dataset Mean: {dataset_core_loss.mean():.5e}, Variance: {dataset_core_loss.var():.5e}")

    plt.figure(figsize=(8, 5))
    plt.hist(core_loss_values, bins=50, alpha=0.6, label=name, color="blue", edgecolor="black")
    plt.xlabel("Core Loss Value")
    plt.ylabel("Frequency")
    plt.title(f"Core Loss Distribution - {name}")
    plt.legend()

    plot_filename = os.path.join(FIGURE_DIR, f"{name.replace(' ', '_').lower()}_distribution.png")
    plt.savefig(plot_filename)
    plt.close()

    print(f"✅ Saved {name} core loss distribution plot as {plot_filename}")
    
def check_reproducibility(dataloader, name="DataLoader", seed=42):
    """
    Checks reproducibility by retrieving the first sample multiple times.

    Args:
        dataloader: The DataLoader object to check.
        name (str): Name of the DataLoader (train, validation, test).
        seed (int): Random seed for reproducibility.

    Returns:
        None (Prints sample values multiple times).
    """
    torch.manual_seed(seed)
    batch = next(iter(dataloader))
    voltages, core_loss_values = batch

    print(f"\n🔹 [DEBUG] Checking Reproducibility in {name} (Seed={seed}):")
    for _ in range(3):
        print(f"   - First Sample Core Loss: {core_loss_values[0].item()}")
        print(f"   - First Sample Voltage (first 10 points):\n{voltages[0, 0, :10].cpu().numpy()}")