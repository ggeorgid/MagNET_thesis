import torch
import os
import numpy as np
from pathlib import Path
import random
import yaml
import pywt
import matplotlib
matplotlib.use("TkAgg")  # Use an interactive backend
import matplotlib.pyplot as plt

# Import dataset-related functions from the `data/` folder
from data.download_dataset import download_dataset
from data.preprocess_dataset import check_dataset_exists, check_cached_scalograms, convert_to_npy, convert_to_tensors, calculate_core_loss, calculate_scalograms 
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
        
           
    # # -------------------------------- Calculating Wavelets & Scalograms --------------------------------
    
    # SAMPLE_RATE = 2e-6
    # wave_name = 'cgau8'  # Complex Gaussian wavelet

    # # Initialize dataset with scalograms
    # wavelet_dataset = WaveletCoreLossDataset(V_I_dataset=tensor_dataset, raw_dataset=dataset, sample_rate=SAMPLE_RATE, wave_name=wave_name)
    # print(f"[INFO] WaveletCoreLossDataset length: {len(wavelet_dataset)}")
    
    # # -------------------------------- Handling and Storing Scalograms --------------------------------

    # if use_cached_scalograms and check_cached_scalograms(preprocessed_data_path):
    #     print("[INFO] Loading cached scalograms and core loss from disk.\n")
    #     scalograms = torch.tensor(np.load(preprocessed_data_path / "scalograms.npy"))
    #     core_loss = torch.tensor(np.load(preprocessed_data_path / "core_loss.npy"))
    # else:
    #     print("[INFO] Calculating scalograms and core loss. This may take some time...\n")
    #     core_loss_dataset = calculate_core_loss(8192, 2e-6, tensor_dataset)
    #     scalograms = calculate_scalograms(dataset)

    #     # Save scalograms and core loss for future runs
    #     np.save(preprocessed_data_path / "scalograms.npy", scalograms.numpy())
    #     np.save(preprocessed_data_path / "core_loss.npy", core_loss_dataset.tensors[1].numpy())
    #     core_loss = core_loss_dataset.tensors[1]
        
    # print(f"I'm currenty here ")   
    # print(f"Dataset min: {dataset.min()}, max: {dataset.max()}")
    # print(f"Sample data:\n{dataset[0]}")
    
    # -------------------------------- Handling Scalograms & Core Loss --------------------------------

    SAMPLE_RATE = 2e-6
    wave_name = 'cgau8'  # Complex Gaussian wavelet

    if use_cached_scalograms and check_cached_scalograms(preprocessed_data_path):
        print("[INFO] Loading cached scalograms and core loss from disk.\n")
        scalograms = torch.tensor(np.load(preprocessed_data_path / "scalograms.npy"))
        core_loss = torch.tensor(np.load(preprocessed_data_path / "core_loss.npy"))
        print(f"[INFO] Loaded scalograms shape: {scalograms.shape}")
        print(f"[INFO] Loaded core loss shape: {core_loss.shape}")
    else:
        print("[INFO] Creating WaveletCoreLossDataset and calculating scalograms and core loss. This may take some time...\n")
        wavelet_dataset = WaveletCoreLossDataset(V_I_dataset=tensor_dataset, raw_dataset=dataset, sample_rate=SAMPLE_RATE, wave_name=wave_name)

        # Extract data from wavelet_dataset
        scalograms = torch.stack([wavelet_dataset[i][0] for i in range(len(wavelet_dataset))])
        core_loss = torch.stack([wavelet_dataset[i][1] for i in range(len(wavelet_dataset))])

        # Save scalograms and core loss for future runs with error handling
        try:
            np.save(preprocessed_data_path / "scalograms.npy", scalograms.numpy())
            np.save(preprocessed_data_path / "core_loss.npy", core_loss.numpy())
            print("[INFO] Scalograms and core loss saved successfully.\n")
        except Exception as e:
            print(f"[ERROR] Failed to save scalograms or core loss: {e}")

    # -------------------------------- Visualize a Random Scalogram --------------------------------

    random_idx = random.randint(0, len(scalograms) - 1)  # Pick a random index

    if 'wavelet_dataset' in locals():
        raw_scalogram = wavelet_dataset.get_raw_scalogram(random_idx)
    else:
        raw_scalogram = scalograms[random_idx].squeeze().numpy()

    print(f"🔹 Visualizing scalogram from index {random_idx}...\n")

    # Plot a raw scalogram
    plt.ion()  # Turn on interactive mode
    plt.imshow(raw_scalogram, cmap='jet', aspect='auto')
    plt.colorbar()
    plt.title(f"Scalogram Sample {random_idx}")
    plt.show(block=False)  # Show plot without blocking
    plt.pause(10)  # Keep the plot open for 10 seconds
    plt.close()  # Automatically close the plot

    # -------------------------------- DataLoader Setup ------------------------------------------

    dataloader = torch.utils.data.DataLoader(list(zip(scalograms, core_loss)), batch_size=batch_size, shuffle=True)
    print(f"🔹 DataLoader initialized with batch size {batch_size}\n")

    # Check a batch
    for scalogram_batch, core_loss_batch in dataloader:
        print(f"🔹 Batch Loaded - Scalogram Shape: {scalogram_batch.shape}, Core Loss Shape: {core_loss_batch.shape}\n")

        # ------------------------- Dataset Validation -------------------------
        assert scalogram_batch.shape[0] == core_loss_batch.shape[0], "[ERROR] Batch size mismatch!"
        assert scalogram_batch.shape[1:] == (1, 24, 24), f"[ERROR] Unexpected scalogram shape: {scalogram_batch.shape[1:]}"
        assert core_loss_batch.ndim == 2 and core_loss_batch.shape[1] == 1, f"[ERROR] Unexpected core loss shape: {core_loss_batch.shape}"

        print("[INFO] Dataset validation passed successfully!\n")
        break



   
   
   
   
if __name__ == "__main__":
    main()
