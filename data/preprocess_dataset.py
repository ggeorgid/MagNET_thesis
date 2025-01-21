import numpy as np
import torch
import logging
import glob
import pandas as pd
import os

def convert_to_npy(preprocessed_data_path: str, raw_data_path: str):
    """
    Load and process CSV files from a given directory, concatenate them into a single dataset,
    and save the result as a numpy array.

    Parameters
    ----------
    preprocessed_data_path : str
        Path to the directory where the final numpy array will be saved.
    raw_data_path : str
        Path to the directory containing the raw CSV files for processing.    
    """
    csv_paths = glob.glob(os.path.join(raw_data_path, "dataset/*.csv"))
    all_data = []

    # Store warnings and only print once
    warnings = []

    for csv_path in csv_paths:
        # Skip the meta-info CSV
        if "info.csv" in csv_path:
            continue

        # Load CSV data using pandas, specifying dtype to avoid mixed type warnings
        df = pd.read_csv(csv_path, header=None, dtype={0: str, 1: np.float64})
        
        try:
            sample_period, sample_length = df.iloc[0]
        except ValueError as e:
            warnings.append(f"Error in reading first row from {csv_path}: {e}")
            continue

        # Ensure sample_length is an integer
        sample_length = int(sample_length)

        # Validate sample length
        if sample_length != 8192:
            warnings.append(f"[MagNet] WARNING: {csv_path} has sample length {sample_length}, which is not the expected 8192.")
            continue

        # Process the data
        data = df.iloc[1:].values.astype(np.float64)
        num_samples = data.shape[0] // sample_length
        num_samples = int(num_samples)  # Ensure num_samples is an integer

        try:
            data = np.reshape(data, (num_samples, sample_length, -1))  # Reshape the data to the correct shape
        except ValueError as e:
            warnings.append(f"Error reshaping data from {csv_path}: {e}")
            continue

        # Append processed data to the list
        all_data.append(data)

    # Concatenate all processed data into one array
    if all_data:
        final_data = np.concatenate(all_data, axis=0)
        # Save the final dataset as a numpy array
        np.save(os.path.join(preprocessed_data_path, "dataset.npy"), final_data)

    # Print all warnings at the end
    if warnings:
        for warning in warnings:
            logging.warning(warning)


def convert_to_tensors(dataset_dir: str, dataset: np.ndarray) -> torch.utils.data.TensorDataset:
    """
    Convert a NumPy array to PyTorch tensors and create a TensorDataset.

    Args:    
        dataset_dir : str
            Directory where the dataset is stored or processed.
        dataset : np.ndarray
            The dataset to be converted, expected as a 3D NumPy array
            with shape (samples, time_steps, 2). The third dimension
            represents voltage and current.
        
    Returns:    
        torch.utils.data.TensorDataset
            A PyTorch TensorDataset containing two tensors: voltage and current.
    """       
    if torch is None:
        raise ImportError("PyTorch is not installed. Please install it to use this function.")

    # No need to load dataset from a file, as it is already provided as a NumPy array
    try:
        voltage = torch.FloatTensor(dataset[:, :, 0])  # Convert voltage to a tensor
        current = torch.FloatTensor(dataset[:, :, 1])  # Convert current to a tensor
    except Exception as e:
        raise RuntimeError(f"Failed to process the dataset: {e}")

    # Create TensorDataset
    tensor_dataset = torch.utils.data.TensorDataset(voltage, current)

    logging.info("PyTorch dataset created successfully.")
    return tensor_dataset

def calculate_core_loss(datalength: int, 
                        sample_rate: float,
                        V_I_dataset: torch.utils.data.TensorDataset) -> torch.utils.data.TensorDataset:
    """
    Calculate core loss for a given dataset by integrating the instantaneous power.

    Parameters:
    ----------
    datalength : int
        The length of each data sample (e.g., number of time steps).
    sample_rate : float
        The sample rate (time between each data point).
    V_I_dataset : torch.utils.data.TensorDataset
        The dataset containing voltage and current tensors.

    Returns:
    -------
    torch.utils.data.TensorDataset
        A dataset containing the voltage and the computed core loss.
    """
    
    # Create lists to store the processed voltage and core_loss
    voltage_list = []
    core_loss_list = []

    # Iterate through the TensorDataset
    for voltage, current in V_I_dataset:
        # Ensure the data is sliced correctly based on `datalength`
        voltage = voltage[:datalength]
        current = current[:datalength]

        # Calculate instantaneous power
        power = voltage * current
        
        # Time array for integration
        t = np.arange(0, (datalength - 0.5) * sample_rate, sample_rate)
        
        # Numerical integration using the trapezoidal rule
        core_loss = np.trapz(power.numpy(), t, axis=0) / (sample_rate * datalength)
        
        # Append the results to the lists
        voltage_list.append(voltage)
        core_loss_list.append(torch.FloatTensor([core_loss]))  # Make core_loss a tensor
    
    # Convert lists of tensors back into a single TensorDataset
    voltage_tensor = torch.stack(voltage_list)
    core_loss_tensor = torch.stack(core_loss_list)

    # Create a TensorDataset with the voltage and the computed core loss
    dataset = torch.utils.data.TensorDataset(voltage_tensor, core_loss_tensor)

    return dataset