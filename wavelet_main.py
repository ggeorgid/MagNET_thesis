import torch
import numpy as np
import argparse
from pathlib import Path
import random
import yaml
import matplotlib
import os
matplotlib.use("Agg")  # Use Qt6 because TkAgg had conflicts with num_of_workers variable
import matplotlib.pyplot as plt
from torchinfo import summary

# Import dataset-related functions from the `data/` folder
from data.download_dataset import download_dataset
from data.preprocess_dataset import check_dataset_exists, convert_to_npy, convert_to_tensors
from data.dataloaders_split import split_dataset, create_dataloaders, inspect_dataloader, check_dataloader_distribution, check_reproducibility
from data.wavelet_coreloss_dataset import WaveletCoreLossDataset
# Import the model
from wavelet_model import WaveletModel  
from utils.train_utils import train_model

def parse_args():
    """Parse command-line arguments to override YAML hyperparameters."""
    parser = argparse.ArgumentParser(description="Run Wavelet Model Training")
    parser.add_argument("--num_epoch", type=int, help="Number of epochs")
    parser.add_argument("--data_subset_size", type=int, help="Subset size of dataset")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    return parser.parse_args()
    
def main():    
    
    # --------------------------------- Setup ---------------------------------
    print(f"\n🔹 Torch version: {torch.__version__}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔹 Selected device: {device}\n")

    # ---------------------- Load Hyperparameters ----------------------
    with open("hyperparameters.yaml", "r") as file:
        config = yaml.safe_load(file)

    args = parse_args()
    
    # Override YAML hyperparameters with command-line arguments
    config["NUM_EPOCH"] = args.num_epoch if args.num_epoch is not None else config["NUM_EPOCH"]
    config["DATA_SUBSET_SIZE"] = args.data_subset_size if args.data_subset_size is not None else config.get("DATA_SUBSET_SIZE", None)
    config["LEARNING_RATE"] = args.learning_rate if args.learning_rate is not None else config["LEARNING_RATE"]
    config["BATCH_SIZE"] = args.batch_size if args.batch_size is not None else config["BATCH_SIZE"]
    
    # Set all necessary random seeds
    torch.manual_seed(config["SEED"])              # Ensure PyTorch uses the same seed
    torch.cuda.manual_seed_all(config["SEED"])     # Ensure CUDA operations are deterministic
    np.random.seed(config["SEED"])                 # Set NumPy seed
    random.seed(config["SEED"])                    # Set Python's built-in random module seed
    torch.backends.cudnn.deterministic = True      # Ensure CUDA backend is deterministic
    torch.backends.cudnn.benchmark = False         # Ensure reproducibility
    print(f"🔹 [INFO] Random seed set to {config['SEED']}")
    
    print(f"\n🔹 Hyperparameters Loaded:")
    print(config)
    
    # ---------------------- Dataset Download & Processing ----------------------
    raw_data_path = Path("data/raw")    
    drive_url = "https://drive.google.com/file/d/1syNCq6cr4P5rAEdkIZs5c9Ee39-NxKY9/view"    
    desired_filename = "dataset_raw"
    dataset_dir = raw_data_path / "dataset"  
    # Ensure figures directory exists
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
    
    # ---------------------- Dataset Subset Selection ----------------------
    subset_indices_file = preprocessed_data_path / "subset_indices.npy"    

    # If the subset indices file exists, attempt to load it
    if subset_indices_file.exists():
        indices = np.load(subset_indices_file, allow_pickle=True)

        # ✅ Ensure `indices` is always a NumPy array
        if not isinstance(indices, np.ndarray) or indices.ndim == 0:
            print("[WARNING] Subset indices file is invalid. Regenerating subset.")
            indices = None  # Force regeneration

    # If indices are invalid or file doesn't exist, generate a new subset
    if not subset_indices_file.exists() or indices is None:
        print("[INFO] Generating new subset indices.")

        # Make sure DATA_SUBSET_SIZE is within valid range
        if config.get("DATA_SUBSET_SIZE", None) is None or config["DATA_SUBSET_SIZE"] > len(dataset):
            config["DATA_SUBSET_SIZE"] = len(dataset)
            print(f"[INFO] Using full dataset: {config['DATA_SUBSET_SIZE']} samples")

        # ✅ Use SEED from config.yaml for reproducibility
        seed = config.get("SEED", 42)  # Default to 42 if missing
        print(f"[INFO] Using random seed: {seed}")
        np.random.seed(seed)  # Apply the correct seed

        indices = np.random.choice(len(dataset), config["DATA_SUBSET_SIZE"], replace=False)

        # ✅ Save indices correctly to prevent corruption
        np.save(subset_indices_file, indices)
        print("[INFO] Saved new dataset subset indices for future runs.")

    # Debug prints to verify
    #print(f"[DEBUG] Final Indices Shape: {indices.shape}")
    #print(f"[DEBUG] Dataset Shape: {dataset.shape}")

    # Select the subset
    dataset_subset = dataset[indices]         
    tensor_dataset = convert_to_tensors(dataset_subset)
    
        
    # ---------------------- Caching and Scalogram Processing ----------------------
    scalograms_path = preprocessed_data_path / "scalograms.npy"
    core_loss_path = preprocessed_data_path / "core_loss.npy"

    if config.get("USE_CACHED_SCALOGRAMS", False) and scalograms_path.exists() and core_loss_path.exists():
        print("[INFO] Loading cached scalograms and core loss.")
        wavelet_dataset = WaveletCoreLossDataset(
            scalograms_path=scalograms_path,
            core_loss_path=core_loss_path
        )
    else:
        print("[INFO] Cached scalograms not found or dataset changed. Computing scalograms...")
        tensor_dataset = convert_to_tensors(dataset)
        wavelet_dataset = WaveletCoreLossDataset(V_I_dataset=tensor_dataset)

        # Save new scalograms if caching is enabled
        if config.get("USE_CACHED_SCALOGRAMS", False):
            np.save(scalograms_path, wavelet_dataset.scalograms)  # ✅ No .numpy() needed
            np.save(core_loss_path, wavelet_dataset.core_loss_values)  # ✅ No .numpy() needed
            print("[INFO] Cached new scalograms for future use.")

    print(f"[INFO] Created WaveletCoreLossDataset with {len(wavelet_dataset)} samples.")

    #----------------Inspecting Scalograms-CoreLoss Dataset----------------------------------------
    # Print first 5 sample scalograms and core loss values
    # for i in range(5):
    #     scalogram, core_loss = wavelet_dataset[i]  # Get dataset sample

    #     print(f"\n🔹 [DEBUG] Sample {i}:")
    #     print(f"   - Scalogram Shape: {scalogram.shape} (Expected: [1, 24, 24])")
    #     print(f"   - Core Loss Value: {core_loss.item()}")

    #     # Print a small section of the scalogram (first 3×3 block)
    #     print(f"   - Scalogram Sample Values (first 3×3 block):\n{scalogram.squeeze().numpy()[:3, :3]}")
        
    # Compute statistics
    # core_loss_values = np.array([wavelet_dataset[i][1].item() for i in range(len(wavelet_dataset))])

    # print("\n🔹 [DEBUG] Core Loss Dataset Statistics:")
    # print(f"   - Mean: {np.mean(core_loss_values)}")
    # print(f"   - Min: {np.min(core_loss_values)}")
    # print(f"   - Max: {np.max(core_loss_values)}")
    # print(f"   - Std Dev: {np.std(core_loss_values)}")
    
    # # Extract 5 random samples and check their scalogram shapes
    # random_indices = np.random.choice(len(wavelet_dataset), 5, replace=False)

    # for idx in random_indices:
    #     scalogram, core_loss = wavelet_dataset[idx]
        
    #     print(f"\n🔹 [DEBUG] Random Sample {idx}:")
    #     print(f"   - Scalogram Shape: {scalogram.shape}")
    #     print(f"   - Core Loss Value: {core_loss.item()}")

    #     # Check mean and variance of the scalogram
    #     print(f"   - Scalogram Mean: {scalogram.mean().item()}, Variance: {scalogram.var().item()}")
        
            
    # ---------------------- Visualize a Random Scalogram ----------------------
    random_idx = random.randint(0, len(wavelet_dataset) - 1)
    raw_scalogram = wavelet_dataset.get_raw_scalogram(random_idx)
    print(f"🔹 Visualizing scalogram from index {random_idx}...")

    plt.figure(figsize=(6, 4))
    plt.imshow(raw_scalogram, cmap='jet', aspect='auto')
    plt.colorbar()
    plt.title(f"Scalogram Sample {random_idx}")

    # Save the figure
    scalogram_filename = os.path.join(FIGURE_DIR, f"scalogram_sample_{random_idx}.png")
    plt.savefig(scalogram_filename)
    plt.close()

    print(f"✅ Saved scalogram visualization as {scalogram_filename}")
    
    # -------------------------- Splitting the Dataset --------------------------
    # Split dataset
    train_dataset, valid_dataset, test_dataset = split_dataset(
        wavelet_dataset, 
        train_ratio=0.7, 
        valid_ratio=0.15, 
        test_ratio=0.15, 
        seed=config["SEED"],  # ✅ Ensure seed comes from YAML 
        save_dir=preprocessed_data_path
    )

    print(f"\n🔹 Train Set: {len(train_dataset)} samples")
    print(f"🔹 Validation Set: {len(valid_dataset)} samples")
    print(f"🔹 Test Set: {len(test_dataset)} samples")
    
    # ---------------------- Create DataLoaders ---------------------------------
    
    train_loader, valid_loader, test_loader = create_dataloaders(
        train_dataset, 
        valid_dataset, 
        test_dataset, 
        batch_size=config["BATCH_SIZE"], 
        num_workers=4, 
        use_gpu=config["USE_GPU"]
    )
    
    # ✅ Check dataloader sizes to see if they are the same as the dataset split sizes
    # print(f"🔹 [DEBUG] Train DataLoader Size: {len(train_loader.dataset)} (Expected: {len(train_dataset)})")
    # print(f"🔹 [DEBUG] Valid DataLoader Size: {len(valid_loader.dataset)} (Expected: {len(valid_dataset)})")
    # print(f"🔹 [DEBUG] Test DataLoader Size: {len(test_loader.dataset)} (Expected: {len(test_dataset)})")
    
    # # Inspect each DataLoader
    # inspect_dataloader(train_loader, "Train DataLoader")
    # inspect_dataloader(valid_loader, "Validation DataLoader")
    # inspect_dataloader(test_loader, "Test DataLoader")
    
    # Check each DataLoader's core loss distribution
    check_dataloader_distribution(train_loader, wavelet_dataset, "Train DataLoader")
    check_dataloader_distribution(valid_loader, wavelet_dataset, "Validation DataLoader")
    check_dataloader_distribution(test_loader, wavelet_dataset, "Test DataLoader")
    
    # Check reproducibility to ensure the seed works correctly
    # check_reproducibility(train_loader, "Train DataLoader", seed=config["SEED"])
    # check_reproducibility(valid_loader, "Validation DataLoader", seed=config["SEED"])
    # check_reproducibility(test_loader, "Test DataLoader", seed=config["SEED"])
    
   

    
    # ---------------------- Model Training ----------------------
    model = WaveletModel().to(device)
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        config=config,
        device=device
    )
    
    
    
    
    # ---------------------- Clean Up ----------------------
    train_indices_file = preprocessed_data_path / "train_indices.npy"
    valid_indices_file = preprocessed_data_path / "valid_indices.npy"
    test_indices_file = preprocessed_data_path / "test_indices.npy"

    for file in [subset_indices_file, train_indices_file, valid_indices_file, test_indices_file]:
        if file.exists():
            print(f"\n[INFO] Deleting {file.name} to enforce seed-based reproducibility in next runs.")
            file.unlink()
            
            
            
if __name__ == "__main__":
    main()

    

