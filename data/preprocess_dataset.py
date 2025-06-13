import os
import pywt
import torch
import pandas as pd
import numpy as np
from skimage.transform import resize
from tqdm import tqdm
from pathlib import Path

def check_dataset_exists(dataset_path, check_for_npy=True):
    """
    Checks if the dataset directory exists and contains valid data.

    Args:
        dataset_path (Path): Path to the dataset directory.
        check_for_npy (bool): If True, checks for dataset.npy (processed data).
                              If False, checks for raw dataset files.

    Returns:
        bool: True if dataset exists, otherwise False.
    """
    if check_for_npy:
        dataset_file = dataset_path / "dataset.npy"
        if dataset_file.exists():
            return True
        print("[INFO] Processed dataset not found.")
        return False
    else:
        # Check if raw dataset directory contains any files
        if dataset_path.exists() and any(dataset_path.iterdir()):
            return True
        print("[INFO] Raw dataset directory is empty.")
        return False


def process_csv(csv_path: str) -> np.ndarray | None:
    """
    Reads a CSV file and converts it into a NumPy array while ensuring consistent length based on the shortest sample.
    """
    if "info.csv" in str(csv_path):
        return None
    
    print(f"[DEBUG] Processing file: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path, header=None, dtype={0: str, 1: np.float64})
        sample_period, sample_length = df.iloc[0]
        sample_length = int(sample_length)
        
        print(f"[DEBUG] {csv_path} -> Sample Period: {sample_period}, Sample Length: {sample_length}")
        
        data = df.iloc[1:].values.astype(np.float64)
        num_samples = data.shape[0] // sample_length
        
        print(f"[DEBUG] {csv_path} -> Data Shape Before Reshaping: {data.shape}")
        
        data = data.reshape(num_samples, sample_length, -1)
        
        print(f"[DEBUG] {csv_path} -> Data Shape After Reshaping: {data.shape}")
        
        return data, sample_length
    
    except (ValueError, IndexError) as e:
        print(f"[ERROR] Failed to process {csv_path}: {e}")
        return None, None

def convert_to_npy(preprocessed_data_path: Path, raw_data_path: Path) -> None:
    """
    Converts all CSV files in raw_data_path to a single NumPy dataset, adjusting all time series to the shortest length.
    """
    csv_paths = list(raw_data_path.glob("dataset/*.csv"))
    print(f"[DEBUG] Found CSV files: {csv_paths}")
    processed_data = [process_csv(csv) for csv in csv_paths]
    
    # ✅ Filter out None values **before** unpacking
    processed_data = [entry for entry in processed_data if entry is not None]

    if not processed_data:
        raise RuntimeError("[ERROR] No valid CSV files found for dataset conversion.")

    # ✅ Unpack only valid entries
    all_data, all_lengths = zip(*processed_data)

    min_length = min(all_lengths)
    print(f"[INFO] Shortest time series length found: {min_length}. Truncating all to this length.")
    
    all_data = [data[:, :min_length, :] for data in all_data]
    
    final_data = np.concatenate(all_data, axis=0)
    np.save(preprocessed_data_path / "dataset.npy", final_data)
    print(f"[INFO] Successfully saved dataset.npy with shape {final_data.shape}")


def convert_to_tensors(dataset: np.ndarray, device="cpu") -> torch.utils.data.TensorDataset:
    """
    Converts a NumPy dataset to PyTorch tensors and creates a TensorDataset.

    Args:
        dataset (np.ndarray): Input dataset of shape (samples, time_steps, 2).
        device (str): The device to store the tensors (default: "cpu").

    Returns:
        torch.utils.data.TensorDataset: Tensor dataset with voltage and current tensors.
    """
    try:
        dataset = np.asarray(dataset)  # Ensures input is a NumPy array

        # Ensure dataset is in (samples, time_steps, 2) format
        if dataset.ndim != 3 or dataset.shape[-1] != 2:
            raise ValueError(f"[ERROR] Expected dataset shape (samples, time_steps, 2), got {dataset.shape}")

        # Convert voltage and current to PyTorch tensors
        voltage = torch.tensor(dataset[:, :, 0], dtype=torch.float32, device=device)
        current = torch.tensor(dataset[:, :, 1], dtype=torch.float32, device=device)

        print(f"[INFO] PyTorch dataset created successfully with shape: {dataset.shape}")

        return torch.utils.data.TensorDataset(voltage, current)

    except Exception as e:
        print(f"[ERROR] Failed to convert dataset to tensors: {e}")
        raise


def calculate_core_loss(
    V_I_dataset: torch.utils.data.Dataset,
    sample_period: float = 2e-6,  # ✅ Default to 2 µs (sampling period from dataset)
    track_gradients: bool = False
) -> torch.utils.data.TensorDataset:
    """
    Computes core loss using trapezoidal numerical integration on voltage-current time-series data.

    Args:
        V_I_dataset (torch.utils.data.Dataset): Dataset containing voltage and current tensors.
        sample_period (float): The sample period (in seconds), defaults to 2 µs.
        track_gradients (bool): Whether to keep computation graphs for core loss.

    Returns:
        torch.utils.data.TensorDataset: A dataset containing voltage and the computed core loss.
    """

    # Extract tensors
    if isinstance(V_I_dataset, torch.utils.data.Subset):
        base_dataset = V_I_dataset.dataset
        indices = V_I_dataset.indices
        voltage_tensor, current_tensor = base_dataset.tensors[0][indices], base_dataset.tensors[1][indices]
    else:
        voltage_tensor, current_tensor = V_I_dataset.tensors

    # Determine the actual length of each sample
    datalength = voltage_tensor.shape[1]

    # Ensure correct slicing
    voltage_tensor = voltage_tensor[:, :datalength]
    current_tensor = current_tensor[:, :datalength]

    # Compute instantaneous power
    power = voltage_tensor * current_tensor  # Shape: (num_samples, datalength)

    # DEBUG: Print some power values before integration
    #print(f"[DEBUG] Sample Power Values (first 10x10 block): \n{power[:10, :10].cpu().numpy()}")

    # Use trapezoidal integration which in our case is the same as taking the mean of power
    t = torch.linspace(0, (datalength - 1) * sample_period, datalength, device=voltage_tensor.device)
    energy = torch.trapz(power, t, dim=1)  # Total energy in joules
    total_time = (datalength - 1) * sample_period  # Total time in seconds
    core_loss = energy / total_time  # Average power in watts

    # Ensure correct output shape
    core_loss = core_loss.unsqueeze(1)  # Shape: (num_samples, 1)

    # DEBUG: Print first 10 core loss values
    #print(f"[DEBUG] Sample Core Loss Values: {core_loss[:10].cpu().numpy().flatten()}")

    # DEBUG: Print min, max, and mean core loss values
    #print(f"[INFO] Core Loss Stats - Mean: {core_loss.mean().item()} | Min: {core_loss.min().item()} | Max: {core_loss.max().item()}")

    #print(f"[INFO] Computed core loss with shape: {core_loss.shape}, Sample Length: {datalength}")

    # Return dataset with voltage and computed core loss
    return torch.utils.data.TensorDataset(voltage_tensor, core_loss)



# Core Loss Calculation Alternative without the trapezoidal integration
# def calculate_core_loss(
#     V_I_dataset: torch.utils.data.Dataset,
#     sample_period: float = 2e-6,  # Sample period in seconds
#     track_gradients: bool = False
# ) -> torch.utils.data.TensorDataset:
#     """
#     Computes core loss as the average power from voltage and current time-series data.

#     Args:
#         V_I_dataset (torch.utils.data.Dataset): Dataset with voltage and current tensors.
#         sample_period (float): Time between samples (seconds), defaults to 2 µs.
#         track_gradients (bool): Whether to retain computation graphs for gradients.

#     Returns:
#         torch.utils.data.TensorDataset: Dataset with voltage and computed core loss.
#     """
#     # Extract voltage and current tensors
#     if isinstance(V_I_dataset, torch.utils.data.Subset):
#         base_dataset = V_I_dataset.dataset
#         indices = V_I_dataset.indices
#         voltage_tensor = base_dataset.tensors[0][indices]
#         current_tensor = base_dataset.tensors[1][indices]
#     else:
#         voltage_tensor, current_tensor = V_I_dataset.tensors

#     # Dynamically determine number of time steps per sample
#     datalength = voltage_tensor.shape[1]

#     # Compute instantaneous power
#     power = voltage_tensor * current_tensor  # Shape: (num_samples, datalength)

#     # Calculate core loss as average power
#     with torch.no_grad() if not track_gradients else torch.enable_grad():
#         core_loss = power.mean(dim=1, keepdim=True)  # Shape: (num_samples, 1)

#     # Debugging outputs
#     print(f"[DEBUG] Sample Core Loss Values: {core_loss[:10].cpu().numpy().flatten()}")
#     print(f"[INFO] Core Loss Stats - Mean: {core_loss.mean().item()} | Min: {core_loss.min().item()} | Max: {core_loss.max().item()}")
#     print(f"[INFO] Computed core loss with shape: {core_loss.shape}, Sample Length: {datalength}")

#     return torch.utils.data.TensorDataset(voltage_tensor, core_loss)


#-----------------Scalogram Calculation-----------------#
def calculate_scalograms(
    dataset: np.ndarray,
    sampling_period: float,
    wave_name: str = 'morl',
    sample_length: int = None,
    total_scale: int = 40,
    fmax: float = 10e3,
    image_size: int = 24,
    save_path: str = "data/processed/scalograms_memmap.dat"
) -> np.memmap:
    """
    Computes scalograms from a dataset using Continuous Wavelet Transform (CWT) and stores them in a memory-mapped array.

    Args:
        dataset (np.ndarray): Input dataset of shape (num_samples, time_steps, 2).
        sampling_period (float): Time between samples in seconds.
        wave_name (str): Wavelet type (default 'morl').
        sample_length (int, optional): Number of time steps to use. Defaults to full length.
        total_scale (int): Number of scales for CWT (default 40).
        fmax (float): Maximum frequency of interest in Hz (default 10e3).
        image_size (int): Size to resize scalograms to (default 24 for 24x24).
        save_path (str): Path to store the memory-mapped scalogram array (default 'data/processed/scalograms_memmap.dat').

    Returns:
        np.memmap: Memory-mapped array of scalograms with shape (num_samples, 1, image_size, image_size).
    """
    # Validate dataset shape
    if dataset.ndim != 3 or dataset.shape[2] != 2:
        raise ValueError("Dataset must have shape (num_samples, time_steps, 2).")

    # Determine sample length
    if sample_length is None:
        sample_length = dataset.shape[1]
    dataset = dataset[:, :sample_length, :]

    print(f"[DEBUG] Total_Scales: {total_scale} and Wavelet: {wave_name} and Image_size: {image_size} that reached here.")

    # Compute wavelet scales
    fc = pywt.central_frequency(wave_name)
    cparam = (1 / sampling_period) / fmax * fc * total_scale  # 1/sampling_period = sampling frequency
    scales = cparam / np.arange(total_scale, 1, -1)

    print(f"[DEBUG] Computed Scales: {scales}")
    print(f"[DEBUG] Sampling Period: {sampling_period} s | Scale Range: {scales.min()} - {scales.max()}")

    # Initialize output array
    num_samples = dataset.shape[0]
    scalograms = np.memmap(save_path, dtype='float32', mode='w+', shape=(num_samples, 1, image_size, image_size))

    # Compute scalograms
    for index in tqdm(range(num_samples), desc="Generating Scalograms"):
        voltage_signal = dataset[index, :, 0]
        cwtmatr, _ = pywt.cwt(voltage_signal, scales, wave_name, sampling_period)
        scalograms[index] = resize(abs(cwtmatr), (image_size, image_size), anti_aliasing=True) #υπαρχει ενα [index,0]

        if index == 0:
            print(f"[DEBUG] First Scalogram Shape: {cwtmatr.shape} | Resized to: {scalograms[index].shape}")

    # Ensure data is written to disk
    scalograms.flush()
    return scalograms


# Newer Version with logarithmic scales (11/6/2025)

# def calculate_scalograms(
#     dataset: np.ndarray,
#     sampling_period: float,
#     wave_name: str = 'morl',
#     sample_length: int = None,
#     total_scale: int = 40,
#     fmax: float = 10e3,
#     fmin: float = 1e3,  # I tried 1 here and the time to compute the scalograms exploded to 40 hours
#     image_size: int = 24
# ) -> torch.Tensor:
#     """
#     Computes scalograms from a dataset using Continuous Wavelet Transform (CWT) with real Morlet wavelet.

#     Args:
#         dataset (np.ndarray): Input dataset of shape (num_samples, time_steps, 2).
#         sampling_period (float): Time between samples in seconds (e.g., 2e-6 s).
#         wave_name (str): Wavelet type (default 'morl' for real Morlet).
#         sample_length (int, optional): Number of time steps to use. Defaults to full length.
#         total_scale (int): Number of scales for CWT (default 40).
#         fmax (float): Maximum frequency of interest in Hz (default 10e3).
#         fmin (float): Minimum frequency of interest in Hz (default 1.0, adjustable).
#         image_size (int): Size to resize scalograms to (default 24 for 24x24).

#     Returns:
#         torch.Tensor: Scalogram tensor of shape (num_samples, 1, image_size, image_size).
#     """
#     # Validate dataset shape
#     if dataset.ndim != 3 or dataset.shape[2] != 2:
#         raise ValueError("Dataset must have shape (num_samples, time_steps, 2).")

#     # Determine sample length
#     if sample_length is None:
#         sample_length = dataset.shape[1]
#     dataset = dataset[:, :sample_length, :]

#     # Compute logarithmic frequency range
#     freqs = np.geomspace(fmax, fmin, num=total_scale)
#     print(f"[DEBUG] Frequency Range: {freqs.min()} Hz to {freqs.max()} Hz")

#     # Convert frequencies to scales using Morlet central frequency (0.8125 Hz)
#     scales = pywt.frequency2scale(wave_name, freqs * sampling_period)
#     print(f"[DEBUG] Computed Scales: {scales.min()} to {scales.max()}")

#     # Initialize output array
#     num_samples = dataset.shape[0]
#     scalograms = np.empty((num_samples, image_size, image_size), dtype=np.float32)

#     # Compute scalograms
#     for index in tqdm(range(num_samples), desc="Generating Scalograms"):
#         voltage_signal = dataset[index, :, 0]
#         cwtmatr, _ = pywt.cwt(voltage_signal, scales, wave_name, sampling_period)
#         scalogram = np.abs(cwtmatr)  # Absolute value for scalogram
#         scalograms[index] = resize(scalogram, (image_size, image_size), anti_aliasing=True)

#         if index == 0:
#             print(f"[DEBUG] First Scalogram Shape: {cwtmatr.shape} | Resized to: {scalograms[index].shape}")

#     # Convert to tensor
#     scalogram_tensor = torch.tensor(scalograms, dtype=torch.float32).unsqueeze(1)
#     return scalogram_tensor


# def calculate_scalograms(
#     dataset: np.ndarray,
#     sampling_period: float,
#     wave_name: str = 'morl',
#     sample_length: int = None,
#     total_scale: int = 40,
#     fmax: float = 10e3,
#     fmin: float = 1e3,
#     image_size: int = 24,
#     save_path: str = "data/processed/scalograms_memmap.dat"
# ) -> np.memmap:
#     """
#     Computes scalograms using CWT and stores them in a memory-mapped array.

#     Args:
#         dataset (np.ndarray): Shape (num_samples, time_steps, 2).
#         sampling_period (float): Time between samples in seconds.
#         wave_name (str): Wavelet type (default 'morl').
#         sample_length (int, optional): Number of time steps.
#         total_scale (int): Number of scales (default 40).
#         fmax (float): Max frequency (default 10e3 Hz).
#         fmin (float): Min frequency (default 1e3 Hz).
#         image_size (int): Output size (default 24).
#         save_path (str): Path for memory-mapped file.

#     Returns:
#         np.memmap: Memory-mapped array of shape (num_samples, image_size, image_size).
#     """
#     if dataset.ndim != 3 or dataset.shape[2] != 2:
#         raise ValueError("Dataset must have shape (num_samples, time_steps, 2).")

#     print(f"[DEBUG] Total_Scales: {total_scale} and Image_size: {image_size} that reached here.")

#     sample_length = sample_length or dataset.shape[1]
#     dataset = dataset[:, :sample_length, :]
#     num_samples = dataset.shape[0]

#     freqs = np.geomspace(fmax, fmin, num=total_scale)
#     scales = pywt.frequency2scale(wave_name, freqs * sampling_period)

#     # Create memory-mapped array
#     scalograms = np.memmap(save_path, dtype='float32', mode='w+', shape=(num_samples, image_size, image_size))

#     for index in tqdm(range(num_samples), desc="Generating Scalograms"):
#         voltage_signal = dataset[index, :, 0]
#         cwtmatr, _ = pywt.cwt(voltage_signal, scales, wave_name, sampling_period)
#         scalogram = np.abs(cwtmatr)
#         scalograms[index] = resize(scalogram, (image_size, image_size), anti_aliasing=True)

#         if index == 0:
#             print(f"[DEBUG] First Scalogram Shape: {cwtmatr.shape} | Resized to: {scalograms[index].shape}")

#     scalograms.flush()  # Write to disk
#     return scalograms



# def calculate_scalograms(
#     dataset: np.ndarray,
#     sampling_period: float,
#     wave_name: str = 'morl',  # or 'cgau8' if preferred
#     sample_length: int = None,
#     total_scale: int = 40,
#     fmax: float = 10e3,
#     image_size: int = 24,
#     save_path: str = "data/processed/scalograms_memmap.dat"
# ) -> np.memmap:
#     if dataset.ndim != 3 or dataset.shape[2] != 2:
#         raise ValueError("Dataset must have shape (num_samples, time_steps, 2).")

#     sample_length = sample_length or dataset.shape[1]
#     dataset = dataset[:, :sample_length, :]
#     num_samples = dataset.shape[0]

#     # Check if the wavelet is continuous or discrete
#     if wave_name in pywt.wavelist(kind='continuous'):
#         # Use ContinuousWavelet for continuous wavelets
#         wavelet = pywt.ContinuousWavelet(wave_name)
#         # Define central frequency manually for continuous wavelets
#         if wave_name == 'morl':
#             fc = 0.8125  # Default central frequency for Morlet wavelet in PyWavelets
#         else:
#             raise NotImplementedError(f"Central frequency for {wave_name} not implemented yet.")
#     else:
#         # Use Wavelet for discrete wavelets
#         wavelet = pywt.Wavelet(wave_name)
#         fc = wavelet.center_frequency

#     # Compute scales
#     cparam = (1 / sampling_period) / fmax * fc * total_scale
#     scales = cparam / np.arange(total_scale, 1, -1)  # Small to large scales

#     scalograms = np.memmap(save_path, dtype='float32', mode='w+', shape=(num_samples, image_size, image_size))

#     for index in tqdm(range(num_samples), desc="Generating Scalograms"):
#         voltage_signal = dataset[index, :, 0]
#         cwtmatr, _ = pywt.cwt(voltage_signal, scales, wavelet, sampling_period)
#         scalograms[index] = resize(np.abs(cwtmatr), (image_size, image_size), anti_aliasing=True)

#     scalograms.flush()
#     return scalograms

# def calculate_scalograms(
#     dataset: np.ndarray,
#     sampling_period: float,
#     wave_name: str = 'morl',
#     sample_length: int = None,
#     total_scale: int = 40,
#     fmax: float = 10e3,
#     image_size: int = 24,
#     save_path: str = "data/processed/scalograms_memmap.dat"
# ) -> np.memmap:
#     """
#     Computes scalograms from a dataset using Continuous Wavelet Transform (CWT) and stores them in a memory-mapped array.

#     Args:
#         dataset (np.ndarray): Input dataset of shape (num_samples, time_steps, 2).
#         sampling_period (float): Time between samples in seconds.
#         wave_name (str): Wavelet type (default 'morl').
#         sample_length (int, optional): Number of time steps to use. Defaults to full length.
#         total_scale (int): Number of scales for CWT (default 40).
#         fmax (float): Maximum frequency of interest in Hz (default 10e3).
#         image_size (int): Size to resize scalograms to (default 24 for 24x24).
#         save_path (str): Path to store the memory-mapped scalogram array (default 'data/processed/scalograms_memmap.dat').

#     Returns:
#         np.memmap: Memory-mapped array of scalograms with shape (num_samples, 1, image_size, image_size).
#     """
#     # Validate dataset shape
#     if dataset.ndim != 3 or dataset.shape[2] != 2:
#         raise ValueError("Dataset must have shape (num_samples, time_steps, 2).")

#     # Determine sample length
#     if sample_length is None:
#         sample_length = dataset.shape[1]
#     dataset = dataset[:, :sample_length, :]

#     # Compute wavelet scales
#     fc = pywt.central_frequency(wave_name)
#     cparam = (1 / sampling_period) / fmax * fc * total_scale  # 1/sampling_period = sampling frequency
#     scales = cparam / np.arange(total_scale, 1, -1)

#     print(f"[DEBUG] Computed Scales: {scales}")
#     print(f"[DEBUG] Sampling Period: {sampling_period} s | Scale Range: {scales.min()} - {scales.max()}")

#     # Initialize memory-mapped array with channel dimension
#     num_samples = dataset.shape[0]
#     scalograms = np.memmap(save_path, dtype='float32', mode='w+', shape=(num_samples, 1, image_size, image_size))

#     # Compute scalograms
#     for index in tqdm(range(num_samples), desc="Generating Scalograms"):
#         voltage_signal = dataset[index, :, 0]
#         cwtmatr, _ = pywt.cwt(voltage_signal, scales, wave_name, sampling_period)
#         resized_scalogram = resize(abs(cwtmatr), (image_size, image_size), anti_aliasing=True)
#         scalograms[index, 0] = resized_scalogram

#         if index == 0:
#             print(f"[DEBUG] First Scalogram Shape: {cwtmatr.shape} | Resized to: {resized_scalogram.shape}")

#     # Ensure data is written to disk
#     scalograms.flush()
#     return scalograms
