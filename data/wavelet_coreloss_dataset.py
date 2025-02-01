import torch
import numpy as np
from torch.utils.data import Dataset
from data.preprocess_dataset import calculate_core_loss, calculate_scalograms

class WaveletCoreLossDataset(Dataset):
    def __init__(self, V_I_dataset, raw_dataset, sample_rate, sample_length=None, transform=None, wave_name='cgau8'):
        """
        Args:
            V_I_dataset (torch.utils.data.TensorDataset): Dataset containing voltage & current tensors.
            raw_dataset (np.ndarray): Original dataset of shape (num_samples, time_steps, 2).
            sample_rate (float): Sampling frequency in Hz.
            sample_length (int, optional): Number of time steps to use. Defaults to full length.
            transform (callable, optional): Transformations to apply to scalograms.
        """
        # Compute core loss using the provided function
        self.core_loss_dataset = calculate_core_loss(
            datalength=sample_length or raw_dataset.shape[1],  # Defaults to full time length
            sample_rate=sample_rate,
            V_I_dataset=V_I_dataset
        )
        
        # Compute scalograms using the provided function
        self.scalograms = calculate_scalograms(
            dataset=raw_dataset,
            wave_name=wave_name,
            sample_length=sample_length,
            sample_rate=sample_rate
        )

        # Extract tensors
        self.voltage_data, self.core_loss_values = self.core_loss_dataset.tensors

        self.transform = transform

    def __len__(self):
        return len(self.scalograms)

    def __getitem__(self, idx):
        """
        Returns:
            Processed scalogram (with transforms if applied) and core loss value.
        """
        scalogram = self.scalograms[idx]  # Shape: (1, 24, 24)
        core_loss = self.core_loss_values[idx]  # Shape: (1,)

        if self.transform:
            scalogram = self.transform(scalogram)

        return scalogram, core_loss

    def get_raw_scalogram(self, idx):
        """Returns raw scalogram data as a NumPy array for plotting."""
        return self.scalograms[idx].squeeze(0).numpy()  # Safe NumPy conversion
    

