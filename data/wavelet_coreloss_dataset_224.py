import torch
import numpy as np
from torch.utils.data import Dataset
from data.preprocess_dataset import calculate_core_loss, calculate_scalograms

class WaveletCoreLossDataset224(Dataset):
    def __init__(self, V_I_dataset=None, sample_rate=2e-6, sample_length=None, transform=None, 
                 wave_name='cgau8', total_scale=224, fmax=10e3, image_size=224,
                 scalograms_path=None, core_loss_path=None, verbose=True):
        """
        Dataset class for computing or loading 224x224 scalograms and corresponding core loss values.

        Args:
            V_I_dataset (TensorDataset, optional): Voltage-current dataset to compute scalograms and core loss.
            sample_rate (float): Sampling period in seconds (default: 2e-6).
            sample_length (int, optional): Length of each sample; inferred if None.
            transform (callable, optional): Transformations to apply to scalograms.
            wave_name (str): Wavelet type (default: 'cgau8').
            total_scale (int): Number of scales for wavelet transform (default: 224).
            fmax (float): Maximum frequency for scalogram (default: 10e3).
            image_size (int): Output scalogram size (default: 224 for 224x224).
            scalograms_path (Path, optional): Path to precomputed scalograms (.npy file).
            core_loss_path (Path, optional): Path to precomputed core loss values (.npy file).
            verbose (bool): Print debug/info messages (default: True).
        """
        self.transform = transform
        self.verbose = verbose
        self.sample_rate = sample_rate
        self.sample_length = sample_length
        self.total_scale = total_scale
        self.fmax = fmax
        self.image_size = image_size

        if scalograms_path and core_loss_path:
            if self.verbose:
                print("[INFO] Loading precomputed scalograms and core loss.")
            self.scalograms = np.load(scalograms_path, mmap_mode='r')
            self.core_loss_values = np.load(core_loss_path, mmap_mode='r')
            # Ensure core loss values are non-negative
            self.core_loss_values = np.maximum(self.core_loss_values, 0)
        elif V_I_dataset is not None:
            voltage_data = V_I_dataset.tensors[0]  # Voltage tensor
            
            if self.verbose:
                sample_len = sample_length or voltage_data.shape[1]
                print(f"[INFO] Computing core loss & scalograms. Sample Length: {sample_len}, Sample Rate: {sample_rate}")
            
            # Compute core loss
            core_loss_tensor = calculate_core_loss(V_I_dataset=V_I_dataset).tensors[1]
            self.core_loss_values = core_loss_tensor.numpy()
            self.core_loss_values = np.maximum(self.core_loss_values, 0)
            
            # Compute 224x224 scalograms directly
            self.scalograms = calculate_scalograms(
                dataset=np.stack((voltage_data.numpy(), np.zeros_like(voltage_data.numpy())), axis=2),
                sampling_period=sample_rate,
                wave_name=wave_name,
                sample_length=sample_length,
                total_scale=total_scale,
                fmax=fmax,
                image_size=image_size
            )
        else:
            raise ValueError("Must provide either V_I_dataset or both scalograms_path and core_loss_path.")

        if self.verbose:
            print(f"[DEBUG] Core Loss Stats - Mean: {np.mean(self.core_loss_values):.4f}, "
                  f"Min: {np.min(self.core_loss_values):.4f}, Max: {np.max(self.core_loss_values):.4f}")
    
    def __len__(self):
        return len(self.scalograms)

    def __getitem__(self, idx):
        scalogram = self.scalograms[idx]  # Shape: (1, 224, 224) or (224, 224)
        core_loss = self.core_loss_values[idx]
        
        # Convert scalogram to tensor and ensure it’s writable
        if isinstance(scalogram, np.ndarray):
            if not scalogram.flags.writeable:
                scalogram = scalogram.copy()
            scalogram = torch.from_numpy(scalogram).float()
        elif isinstance(scalogram, torch.Tensor):
            scalogram = scalogram.clone().detach().float()
        
        # Adjust dimensions: ensure at least 3D with a channel
        if scalogram.ndim == 2:  # [224, 224]
            scalogram = scalogram.unsqueeze(0)  # [1, 224, 224]
        elif scalogram.ndim == 3 and scalogram.shape[0] == 1:  # [1, 224, 224]
            pass  # Already correct
        else:
            raise ValueError(f"Unexpected scalogram shape: {scalogram.shape}")
        
        # Apply transforms if provided
        if self.transform:
            scalogram = self.transform(scalogram)  # Now [3, 224, 224] after Lambda
        
        # Ensure core_loss is a tensor
        core_loss = torch.tensor(core_loss, dtype=torch.float32)
        return scalogram, core_loss

    def get_raw_scalogram(self, idx):
        """Return the raw scalogram reshaped to (image_size, image_size)."""
        scalogram = self.scalograms[idx]
        if scalogram.ndim == 3 and scalogram.shape[0] == 1:
            scalogram = scalogram.squeeze(0)
        return scalogram.reshape(self.image_size, self.image_size)