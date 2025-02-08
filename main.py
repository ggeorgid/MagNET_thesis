import torch
import numpy as np
from pathlib import Path
import random
import yaml
import matplotlib
matplotlib.use("QtAgg") # Use Qt6 because TkAgg had conflicts with num_of_workers variable <-weird
import matplotlib.pyplot as plt
from torchinfo import summary  # Added for model summary


# Import dataset-related functions from the `data/` folder
from data.download_dataset import download_dataset
from data.preprocess_dataset import check_dataset_exists, convert_to_npy, convert_to_tensors, split_dataset, create_dataloaders
from data.wavelet_coreloss_dataset import WaveletCoreLossDataset
# Import the model
from wavelet_model import WaveletModel  
from utils.train_utils import train_model


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
    
    
    # Check if dataset.npy exists before processing
    dataset_file = preprocessed_data_path / "dataset.npy"
    if not dataset_file.exists():
        convert_to_npy(preprocessed_data_path, raw_data_path)
    
    
    # Load the .npy dataset
    dataset = np.load(f"{preprocessed_data_path}/dataset.npy")
    print(f"\n🔹 Current dataset shape: {dataset.shape}")  

    # Transform data to PyTorch tensors
    tensor_dataset = convert_to_tensors(dataset)

    # -------------------------------- Prevent Data Leakage: Random Subset Selection --------------------------------
    if data_subset_size is not None:
        indices = torch.randperm(len(tensor_dataset))[:data_subset_size]  # Random selection to avoid leakage
        tensor_dataset = torch.utils.data.Subset(tensor_dataset, indices)

    # -------------------------------- Handling Scalograms & Core Loss --------------------------------
    SAMPLE_RATE = 2e-6 # 2 µs period
    scalograms_path = preprocessed_data_path / "scalograms.npy"
    core_loss_path = preprocessed_data_path / "core_loss.npy"

    if use_cached_scalograms and scalograms_path.exists() and core_loss_path.exists():
        cached_scalograms = np.load(scalograms_path)
        cached_core_loss = np.load(core_loss_path)
        
        if len(cached_scalograms) >= data_subset_size:
            print(f"[INFO] Using a subset of cached scalograms for subset size {data_subset_size}.")
            indices = np.random.choice(len(cached_scalograms), data_subset_size, replace=False)
            selected_scalograms = cached_scalograms[indices]
            selected_core_loss = cached_core_loss[indices]
            
            np.save(scalograms_path, selected_scalograms)
            np.save(core_loss_path, selected_core_loss)
            
            wavelet_dataset = WaveletCoreLossDataset(
                scalograms_path=scalograms_path,
                core_loss_path=core_loss_path
            )
        else:
            print(f"[WARNING] Cached scalograms size {len(cached_scalograms)} is smaller than requested {data_subset_size}. Recomputing...")
            use_cached_scalograms = False  # Force recalculation
    
    if not use_cached_scalograms:
        print(f"[INFO] Calculating new scalograms for subset size {data_subset_size}.")
        wavelet_dataset = WaveletCoreLossDataset(
            V_I_dataset=tensor_dataset,
            sample_rate=SAMPLE_RATE
        )
        np.save(scalograms_path, wavelet_dataset.scalograms.numpy())
        np.save(core_loss_path, wavelet_dataset.core_loss_values.numpy())
        print(f"[INFO] Saved scalograms for subset size {data_subset_size}.")

                
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
    
    print(f"🔹 Dataset Size Used for Training: {len(wavelet_dataset)}")
      
    # -------------------------------- Train/Validation/Test Split & Loading Dataloaders --------------------------------
    train_dataset, valid_dataset, test_dataset = split_dataset(wavelet_dataset)
    train_loader, valid_loader, test_loader = create_dataloaders(
        train_dataset, valid_dataset, test_dataset, batch_size=batch_size, use_gpu=use_gpu
    )
    
    # Example DataLoader Validation
    for scalogram_batch, core_loss_batch in train_loader:
        scalogram_batch = scalogram_batch.to(device)
        core_loss_batch = core_loss_batch.to(device)
        print(f"Batch Loaded - Scalogram Shape: {scalogram_batch.shape}, Core Loss Shape: {core_loss_batch.shape}")
        break

    print(f"Train Loader: {len(train_loader)}, Validation Loader: {len(valid_loader)}, Test Loader: {len(test_loader)}")
    
    # -------------------------------- Model Initialization --------------------------------
    model = WaveletModel().to(device)
    
    # -------------------------------- Step 3: Forward Pass of a Single Batch --------------------------------
    for scalogram_batch, core_loss_batch in train_loader:
        scalogram_batch = scalogram_batch.to(device)
        core_loss_batch = core_loss_batch.to(device)

        with torch.no_grad():  # No need to track gradients for this inspection
            output = model(scalogram_batch)

        print(f"\n🔹 Forward Pass Output Shape: {output.shape}")
        print(f"🔹 Forward Pass Output Sample: {output[:5]}\n")  # Show first 5 predictions
        break  # Only inspect the first batch

    # -------------------------------- Step 4: Model Summary Using torchinfo --------------------------------
    print("🔹 Model Summary:\n")
    summary(model, input_size=(batch_size, 1, 24, 24))
    
    
    # -------------------------------- Training the Model --------------------------------
    
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        config=config,  # Now always uses correct values
        device=device
    )

    
if __name__ == "__main__":
    main()
