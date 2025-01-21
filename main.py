import torch
import os
import numpy as np
from pathlib import Path
from data.download_dataset import download_dataset
from data.preprocess_dataset import convert_to_npy,convert_to_tensors,calculate_core_loss

def main():
    #--------------------------------Printing Essentials--------------------------------------------------
    
    # Check for the PyTorch version
    print(f"torch version: {torch.__version__}")
    
    # Setup device agnostic code and check the device parameter
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Selected device is {device}")
    
    #---------------------------------Storing the dataset--------------------------------------------------
    
    # Specify raw_data directory and untar it using download_dataset.py    
    raw_data_path = Path("data/raw")    
    drive_url = "https://drive.google.com/file/d/1syNCq6cr4P5rAEdkIZs5c9Ee39-NxKY9/view"    
    desired_filename = "dataset_raw"    
    download_dataset(drive_url, raw_data_path, desired_filename)
    
    # Specify preprocessed_data directory and convert dataset to numpy array
    preprocessed_data_path = Path("data/processed")
    convert_to_npy(preprocessed_data_path,raw_data_path)
    
    # Check the size of the .npy array
    dataset = np.load(f"{preprocessed_data_path}/dataset.npy")
    print(f"Current shape of the dataset is {dataset.shape}")  
    
    #Transform data to Pytorch tensors
    tensor_dataset = convert_to_tensors(preprocessed_data_path, dataset)
    
    #----------------------------------Inspecting the dataset--------------------------------------------
        
    # Check the length of the dataset (number of samples)
    print(f"Length of tensor_dataset: {len(tensor_dataset)}\n")
    print(type(tensor_dataset))    
    
    # Inspect the first sample in the dataset (voltage and current)
    sample = tensor_dataset[0]
    print(f"First sample in the dataset: {sample}")
    
    # Initialize variables to track the global min and max values and the corresponding tensors
    min_value = float('inf')  # Set initial to a very high value
    max_value = float('-inf')  # Set initial to a very low value
    min_tensor = None  # To store the tensor corresponding to the min value
    max_tensor = None  # To store the tensor corresponding to the max value

    # Iterate over the TensorDataset and find the min and max
    for voltage, current in tensor_dataset:
        # Check for min value in voltage and current tensors
        voltage_min, current_min = voltage.min().item(), current.min().item()
        voltage_max, current_max = voltage.max().item(), current.max().item()
        
        # Update the global min and max values and store the corresponding tensors
        if voltage_min < min_value:
            min_value = voltage_min
            min_tensor = voltage
        if current_min < min_value:
            min_value = current_min
            min_tensor = current
        
        if voltage_max > max_value:
            max_value = voltage_max
            max_tensor = voltage
        if current_max > max_value:
            max_value = current_max
            max_tensor = current

    # Print the results
    print(f"Min value in tensor_dataset: {min_value}")
    print(f"Max value in tensor_dataset: {max_value}")

    # Print the full tensors that have the min and max values
    print("\nTensor with the minimum value:")
    print(min_tensor)
    print("\nTensor with the maximum value:")
    print(max_tensor)
    
    #-------------------------------------Calculating the Magnetic Core Loss----------------------------
    #Important parameters(hyperparameters maybe? <-ask this)
    DATA_LENGTH = 8192 #400 se aytous gia kapoio logo ?????????
    SAMPLE_RATE = 2e-6
    
    core_loss_dataset = calculate_core_loss(DATA_LENGTH, SAMPLE_RATE, tensor_dataset)
    
    #----------------------------------Inspecting the dataset--------------------------------------------
        
    # Check the length of the dataset (number of samples)
    print(f"Length of core_loss_dataset: {len(core_loss_dataset)}\n")    
    
    # Inspect the first sample in the dataset (voltage and core_loss)
    sample = core_loss_dataset[0]
    print(f"First sample in the dataset: {sample}")
    
    # Initialize variables to track the global min and max values and the corresponding tensors
    min_value = float('inf')  # Set initial to a very high value
    max_value = float('-inf')  # Set initial to a very low value
    min_tensor = None  # To store the tensor corresponding to the min value
    max_tensor = None  # To store the tensor corresponding to the max value

    # Iterate over the TensorDataset and find the min and max
    for voltage, core_loss in core_loss_dataset:
        # Check for min value in voltage and current tensors
        voltage_min, core_loss_min = voltage.min().item(), core_loss.min().item()
        voltage_max, core_loss_max = voltage.max().item(), core_loss.max().item()
        
        # Update the global min and max values and store the corresponding tensors
        if voltage_min < min_value:
            min_value = voltage_min
            min_tensor = voltage
        if core_loss_min < min_value:
            min_value = core_loss_min
            min_tensor = core_loss
        
        if voltage_max > max_value:
            max_value = voltage_max
            max_tensor = voltage
        if core_loss_max > max_value:
            max_value = core_loss_max
            max_tensor = core_loss

    # Print the results
    print(f"Min value in tensor_dataset: {min_value}")
    print(f"Max value in tensor_dataset: {max_value}")

    # Print the full tensors that have the min and max values
    print("\nTensor with the minimum value:")
    print(min_tensor)
    print("\nTensor with the maximum value:")
    print(max_tensor)
   
if __name__ == "__main__":
    main()
