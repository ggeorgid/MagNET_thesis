import os
import pywt
import torch
import logging
import glob
import pandas as pd
import numpy as np
from skimage.transform import resize
from tqdm import tqdm  # For progress visualization

def check_dataset_exists(dataset_path):
    """
    Check if the dataset directory exists and is not empty.
    Handles both directory and file checks.
    """
    return dataset_path.is_dir() and any(dataset_path.iterdir())


def check_cached_scalograms(preprocessed_data_path):
    """
    Check if cached scalograms and core loss files exist.
    """
    scalogram_file = preprocessed_data_path / "scalograms.npy"
    core_loss_file = preprocessed_data_path / "core_loss.npy"
    return scalogram_file.exists() and core_loss_file.exists()

def convert_to_npy(preprocessed_data_path: str, raw_data_path: str) -> None:
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
    
    def process_csv(csv_path: str) -> np.ndarray | None:
        """Helper function to process a single CSV file."""
        if "info.csv" in csv_path:
            return None
        
        try:
            df = pd.read_csv(csv_path, header=None, dtype={0: str, 1: np.float64})
            sample_period, sample_length = df.iloc[0]
            sample_length = int(sample_length)  # Ensure sample_length is an integer
            
            if sample_length != 8192:
                logging.warning(f"[MagNet] WARNING: {csv_path} has sample length {sample_length}, expected 8192.")
                return None

            data = df.iloc[1:].values.astype(np.float64)
            num_samples = data.shape[0] // sample_length
            
            return data.reshape(num_samples, sample_length, -1)

        except (ValueError, IndexError) as e:
            logging.warning(f"Error processing {csv_path}: {e}")
            return None

    # Process all CSVs and filter out None values
    # Be careful: uses the := walrus operator which is available in Python 3.8 and later
    all_data = [data for csv_path in csv_paths if (data := process_csv(csv_path)) is not None]

    if all_data:
        final_data = np.concatenate(all_data, axis=0)
        np.save(os.path.join(preprocessed_data_path, "dataset.npy"), final_data)


def convert_to_tensors(dataset: np.ndarray) -> torch.utils.data.TensorDataset:
    """
    Convert a NumPy array to PyTorch tensors and create a TensorDataset.

    Args:    
        dataset : np.ndarray
            The dataset to be converted, expected as a 3D NumPy array
            with shape (samples, time_steps, 2). The third dimension
            represents voltage and current.
        
    Returns:    
        torch.utils.data.TensorDataset
            A PyTorch TensorDataset containing two tensors: voltage and current.
    """    
    if not isinstance(dataset, np.ndarray):
        raise TypeError(f"Expected dataset to be a NumPy array, got {type(dataset)} instead.")

    if dataset.ndim != 3 or dataset.shape[-1] != 2:
        raise ValueError(f"Expected dataset shape (samples, time_steps, 2), got {dataset.shape} instead.")

    # Convert voltage and current to PyTorch tensors
    voltage = torch.tensor(dataset[:, :, 0], dtype=torch.float32)
    current = torch.tensor(dataset[:, :, 1], dtype=torch.float32)

    logging.info(f"PyTorch dataset created successfully with shape: {dataset.shape}")

    return torch.utils.data.TensorDataset(voltage, current)

def calculate_core_loss(datalength: int, 
                        sample_rate: float,
                        V_I_dataset: torch.utils.data.TensorDataset) -> torch.utils.data.TensorDataset:
    """
    Optimized calculation of core loss using vectorized PyTorch operations.

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
        core_loss_dataset.tensors returns a tuple of PyTorch tensors.
    """
    
    # Unpack dataset: (num_samples, datalength)
    voltage_tensor, current_tensor = V_I_dataset.tensors  # Extract tensors
    voltage_tensor = voltage_tensor[:, :datalength]  # Ensure correct slicing
    current_tensor = current_tensor[:, :datalength]

    # Compute instantaneous power (element-wise multiplication)
    power = voltage_tensor * current_tensor  # Shape: (num_samples, datalength)

    # Generate time array
    t = torch.linspace(0, (datalength - 0.5) * sample_rate, datalength, device=voltage_tensor.device)

    # Use PyTorch integration with trapezoidal rule (vectorized)
    core_loss = torch.trapz(power, t, dim=1) / (sample_rate * datalength)  # Shape: (num_samples,)

    # Expand dimensions for consistency with PyTorch dataset
    core_loss = core_loss.unsqueeze(1)  # Shape: (num_samples, 1)

    # Create and return optimized TensorDataset
    dataset = torch.utils.data.TensorDataset(voltage_tensor, core_loss)

    return dataset


def calculate_scalograms(dataset: np.ndarray, 
                         wave_name: str = 'cgau8', 
                         sample_length: int = None,  # Default is None
                         sample_rate: float = 200e3):
    """
    Computes scalograms from a dataset using Continuous Wavelet Transform (CWT).
    
    Parameters:
        dataset (np.ndarray): Input dataset of shape (num_samples, time_steps, 2).
        wave_name (str): The wavelet type (default 'cgau8').
        sample_length (int, optional): Number of time steps to use. Defaults to full length.
        sample_rate (float): Sampling frequency in Hz (default 200e3).
    
    Returns:
        torch.Tensor: Scalogram tensor of shape (num_samples, 1, 24, 24).
    """

    # Validate dataset shape
    if dataset.ndim != 3 or dataset.shape[2] != 2:
        raise ValueError("Dataset should have shape (num_samples, time_steps, 2).")

    # Use full dataset length if `sample_length` is not specified
    if sample_length is None:
        sample_length = dataset.shape[1]  # Take the full available time length

    # Trim dataset to `sample_length`
    dataset = dataset[:, :sample_length, :]

    # Compute wavelet scales
    total_scale = 30
    fc = pywt.central_frequency(wave_name)
    fmax = 10e3
    cparam = (1 / sample_rate) / fmax * fc * total_scale
    scales = cparam / np.arange(total_scale, 1, -1)

    # Prepare output array
    num_samples = dataset.shape[0]
    image_size = 24  # Target scalogram size
    scalograms = np.empty((num_samples, image_size, image_size), dtype=np.float32)  # Use `np.empty` for efficiency

    # Compute scalograms efficiently
    for index in tqdm(range(num_samples), desc="Generating Scalograms"):
        row = dataset[index, :, 0]  # Extract voltage signal
        cwtmatr, _ = pywt.cwt(row, scales, wave_name, sample_rate)  # Wavelet transform
        scalograms[index] = resize(abs(cwtmatr), (image_size, image_size), anti_aliasing=True)

    # Convert to PyTorch tensor and reshape for CNN input
    scalogram_tensor = torch.tensor(scalograms).unsqueeze(1)  # Shape: (num_samples, 1, 24, 24)

    return scalogram_tensor
