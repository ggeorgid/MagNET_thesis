import torch
import numpy as np
from torch.utils.data import Dataset
from data.preprocess_dataset import calculate_core_loss, calculate_scalograms

class WaveletCoreLossDataset(Dataset):
    def __init__(self, V_I_dataset=None, sample_rate=None, sample_length=None, transform=None, wave_name='cgau8', 
                 scalograms_path=None, core_loss_path=None):
        """
        Args:
            V_I_dataset (torch.utils.data.TensorDataset, optional): Dataset containing voltage & current tensors.
            sample_rate (float, optional): Sampling frequency in Hz.
            sample_length (int, optional): Number of time steps to use. Defaults to full length.
            transform (callable, optional): Transformations to apply to scalograms.
            wave_name (str): Wavelet name for CWT.
            scalograms_path (str, optional): Path to precomputed scalograms (.npy file).
            core_loss_path (str, optional): Path to precomputed core loss (.npy file).
        """
        self.transform = transform

        if scalograms_path and core_loss_path:
            # Load precomputed scalograms and core loss
            print("[INFO] Loading precomputed scalograms and core loss.")
            self.scalograms = torch.tensor(np.load(scalograms_path), dtype=torch.float32)
            self.core_loss_values = torch.tensor(np.load(core_loss_path), dtype=torch.float32)
        elif V_I_dataset and sample_rate:
            # Compute core loss using the provided function
            print("[INFO] Calculating core loss and scalograms from raw data.")
            self.core_loss_dataset = calculate_core_loss(
                datalength=sample_length or V_I_dataset.tensors[0].shape[1],
                sample_rate=sample_rate,
                V_I_dataset=V_I_dataset
            )

            # Compute scalograms using the voltage data from V_I_dataset
            voltage_data = V_I_dataset.tensors[0].numpy()
            self.scalograms = calculate_scalograms(
                dataset=np.stack((voltage_data, np.zeros_like(voltage_data)), axis=2),
                wave_name=wave_name,
                sample_length=sample_length,
                sample_rate=sample_rate
            )

            # Extract tensors
            self.voltage_data, self.core_loss_values = self.core_loss_dataset.tensors
        else:
            raise ValueError("Provide either (V_I_dataset and sample_rate) or (scalograms_path and core_loss_path).")

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
        return self.scalograms[idx].squeeze(0).numpy()


