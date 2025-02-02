import torch
import numpy as np
from pathlib import Path
import random
import yaml
import matplotlib
matplotlib.use("QtAgg") # Use Qt6 because TkAgg had conflicts with num_of_workers variable <-weird
import matplotlib.pyplot as plt

# Import dataset-related functions from the `data/` folder
from data.download_dataset import download_dataset
from data.preprocess_dataset import check_dataset_exists, check_cached_scalograms, convert_to_npy, convert_to_tensors, calculate_core_loss, split_dataset, create_dataloaders
from data.wavelet_coreloss_dataset import WaveletCoreLossDataset

def main():
    # -------------------------------- Printing Essentials --------------------------------

    # Check PyTorch version
    print(f"\n🔹 Torch version: {torch.__version__}")

    # Setup device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔹 Selected device: {device}\n")

    # -------------------------------- Importing Hyperparameters --------------------------------

    with open("hyperparameters.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Load values from config
    seed = config["SEED"]
    use_gpu = config["USE_GPU"]
    num_epoch = config["NUM_EPOCH"]
    batch_size = config["BATCH_SIZE"]    
    lr = config["LEARNING_RATE"]
    download_dataset_flag = config.get("DOWNLOAD_DATASET", True)
    use_cached_scalograms = config.get("USE_CACHED_SCALOGRAMS", True)
    data_subset_size = config.get("DATA_SUBSET_SIZE", None)
        
    print(f"\n🔹 Hyperparameters Loaded:")
    print(f"   - SEED: {seed}")
    print(f"   - USE_GPU: {use_gpu}")
    print(f"   - NUM_EPOCH: {num_epoch}")
    print(f"   - BATCH_SIZE: {batch_size}")
    print(f"   - LEARNING_RATE: {lr}")
    print(f"   - DOWNLOAD_DATASET: {download_dataset_flag}\n")    
    print(f"   - USE_CACHED_SCALOGRAMS: {use_cached_scalograms}\n")
        
    # -------------------------------- Storing the Dataset --------------------------------

    raw_data_path = Path("data/raw")    
    drive_url = "https://drive.google.com/file/d/1syNCq6cr4P5rAEdkIZs5c9Ee39-NxKY9/view"    
    desired_filename = "dataset_raw"
    dataset_dir = raw_data_path / "dataset"  

    if download_dataset_flag:
        if check_dataset_exists(dataset_dir):
            print(f"[INFO] Dataset already exists at {dataset_dir}. Skipping download.\n")
        else:
            download_dataset(drive_url, raw_data_path, desired_filename)
    else:
        if check_dataset_exists(dataset_dir):
            print(f"[INFO] Dataset found at {dataset_dir}. Skipping download.\n")
        else:
            raise FileNotFoundError(
                f"[ERROR] Dataset not found in {dataset_dir}. "
                "Either set DOWNLOAD_DATASET to True or place the dataset manually in this directory."
            )

    # -------------------------------- Preprocessing the Dataset --------------------------------
    # Specify preprocessed_data directory and convert dataset to numpy array
    preprocessed_data_path = Path("data/processed")
    preprocessed_data_path.mkdir(parents=True, exist_ok=True)  # Ensure directory exists       
    
    convert_to_npy(preprocessed_data_path, raw_data_path)
    
    # Load the .npy dataset
    dataset = np.load(f"{preprocessed_data_path}/dataset.npy")
    print(f"\n🔹 Current dataset shape: {dataset.shape}")  

    # Transform data to PyTorch tensors
    tensor_dataset = convert_to_tensors(dataset)
    
    # -------------------------------- Inspecting the Dataset --------------------------------
    
    print(f"\n🔹 Length of tensor_dataset: {len(tensor_dataset)}")
    print(f"🔹 Type of tensor_dataset: {type(tensor_dataset)}\n")    
    
    # Inspect first sample in the dataset (voltage and current)
    sample = tensor_dataset[0]
    print(f"🔹 First sample in the dataset:\n{sample}\n")

    # Initialize min and max tracking variables
    min_value, max_value = float('inf'), float('-inf')
    min_tensor, max_tensor = None, None

    # Iterate over TensorDataset to find min and max values
    for voltage, current in tensor_dataset:
        voltage_min, current_min = voltage.min().item(), current.min().item()
        voltage_max, current_max = voltage.max().item(), current.max().item()

        # Update min and max
        if voltage_min < min_value:
            min_value, min_tensor = voltage_min, voltage
        if current_min < min_value:
            min_value, min_tensor = current_min, current
        if voltage_max > max_value:
            max_value, max_tensor = voltage_max, voltage
        if current_max > max_value:
            max_value, max_tensor = current_max, current

    print(f"\n🔹 Min value in tensor_dataset: {min_value}")
    print(f"🔹 Max value in tensor_dataset: {max_value}\n")

    print("🔹 Tensor with the minimum value:\n", min_tensor, "\n")
    print("🔹 Tensor with the maximum value:\n", max_tensor, "\n")
    
    #-------------------------------------Calculating the Magnetic Core Loss----------------------------
    #Important parameters(hyperparameters maybe? <-ask this)
    DATA_LENGTH = 8192 #400 se aytous gia kapoio logo ?????????    
    SAMPLE_RATE = 2e-6
    
    core_loss_dataset = calculate_core_loss(DATA_LENGTH, SAMPLE_RATE, tensor_dataset)
    voltage_tensor, core_loss_tensor = core_loss_dataset.tensors  # Unpack dataset tensors

    print(f"\n🔹 Length of core_loss_dataset: {len(core_loss_dataset)}")
    print(f"🔹 Voltage tensor shape: {voltage_tensor.shape}")
    print(f"🔹 Core loss tensor shape: {core_loss_tensor.shape}\n")

    # Inspect first sample in core loss dataset
    sample = core_loss_dataset[0]
    print(f"🔹 First sample in the core loss dataset:\n{sample}\n")
    
    # Apply dataset subsetting if specified using random permutation ordering e.g. tensor([523, 198, 754, 12, 899, ...])
    if data_subset_size:
        print(f"[INFO] Subsetting enabled. Preparing to select {data_subset_size} samples.")
        indices = torch.randperm(len(tensor_dataset))[:data_subset_size]
        tensor_dataset = torch.utils.data.Subset(tensor_dataset, indices)
        print(f"[INFO] Using a subset of {data_subset_size} samples for training/debugging.\n")
            
    # -------------------------------- Handling Scalograms & Core Loss --------------------------------
    SAMPLE_RATE = 2e-6
    scalograms_path = preprocessed_data_path / "scalograms.npy"
    core_loss_path = preprocessed_data_path / "core_loss.npy"

    if use_cached_scalograms and check_cached_scalograms(preprocessed_data_path):
        print("[INFO] Loading cached scalograms and core loss from disk.\n")
        wavelet_dataset = WaveletCoreLossDataset(
            scalograms_path=scalograms_path,
            core_loss_path=core_loss_path
        )
    else:
        print("[INFO] Creating WaveletCoreLossDataset and calculating scalograms and core loss.\n")
        wavelet_dataset = WaveletCoreLossDataset(
            V_I_dataset=tensor_dataset,
            sample_rate=SAMPLE_RATE
        )

        try:
            np.save(scalograms_path, wavelet_dataset.scalograms.numpy())
            np.save(core_loss_path, wavelet_dataset.core_loss_values.numpy())
            print("[INFO] Scalograms and core loss saved successfully.\n")
        except Exception as e:
            print(f"[ERROR] Failed to save scalograms or core loss: {e}")
        
            
    # -------------------------------- Visualize a Random Scalogram --------------------------------
    random_idx = random.randint(0, len(wavelet_dataset) - 1)  # Pick a random index

    raw_scalogram = wavelet_dataset.get_raw_scalogram(random_idx)

    print(f"🔹 Visualizing scalogram from index {random_idx}...\n")

    # Plot a raw scalogram
    plt.ion()  # Turn on interactive mode
    plt.imshow(raw_scalogram, cmap='jet', aspect='auto')
    plt.colorbar()
    plt.title(f"Scalogram Sample {random_idx}")
    plt.show(block=False)  # Show plot without blocking
    plt.pause(10)  # Keep the plot open for 10 seconds
    plt.close()  # Automatically close the plot
      
    # -------------------------------- Train/Validation/Test Split & Loading Dataloaders --------------------------------
    train_dataset, valid_dataset, test_dataset = split_dataset(wavelet_dataset)
    train_loader, valid_loader, test_loader = create_dataloaders(
        train_dataset, valid_dataset, test_dataset, batch_size=batch_size, use_gpu=use_gpu
    )
    
    # Move datasets to device
    train_dataset = [(scalogram.to(device), core_loss.to(device)) for scalogram, core_loss in train_dataset]
    valid_dataset = [(scalogram.to(device), core_loss.to(device)) for scalogram, core_loss in valid_dataset]
    test_dataset = [(scalogram.to(device), core_loss.to(device)) for scalogram, core_loss in test_dataset]
    
    # Example DataLoader Validation
    for scalogram_batch, core_loss_batch in train_loader:
        scalogram_batch = scalogram_batch.to(device)
        core_loss_batch = core_loss_batch.to(device)
        print(f"Batch Loaded - Scalogram Shape: {scalogram_batch.shape}, Core Loss Shape: {core_loss_batch.shape}")
        break

    print(f"Train Loader: {len(train_loader)}, Validation Loader: {len(valid_loader)}, Test Loader: {len(test_loader)}")
    
    
    
if __name__ == "__main__":
    main()
