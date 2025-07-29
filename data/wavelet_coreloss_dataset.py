import torch
import numpy as np
from torch.utils.data import Dataset
from data.preprocess_dataset import calculate_core_loss, calculate_scalograms

class WaveletCoreLossDataset(Dataset):
    def __init__(self, V_I_dataset=None, sample_rate=2e-6, sample_length=None, transform=None,
                 wave_name='morl', total_scale=41, fmax=10e3, image_size=24, #wave_name='cgau8',
                 scalograms_path=None, core_loss_path=None, verbose=True):
        """
        Initializes the dataset class.
        - Loads precomputed scalograms and core loss if paths are provided.
        - Otherwise, computes core loss and scalograms from `V_I_dataset`.
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
            # Clamp loaded core loss values to non-negative
            self.core_loss_values = np.maximum(self.core_loss_values, 0)
        elif V_I_dataset is not None:
            voltage_data = V_I_dataset.tensors[0]  # Keep as tensor
            
            if self.verbose:
                print(f"[INFO] Computing core loss & scalograms from raw dataset. Sample Length: {sample_length or voltage_data.shape[1]}, Sample Rate: {sample_rate}")
            
            # Compute core loss
            core_loss_tensor = calculate_core_loss(V_I_dataset=V_I_dataset).tensors[1]
            self.core_loss_values = core_loss_tensor.numpy()
            # Clamp computed core loss values to non-negative
            self.core_loss_values = np.maximum(self.core_loss_values, 0)
            
            # Compute scalograms and convert tensor to NumPy array
            scalograms_memmap_path = "data/processed/scalograms_memmap.dat"
            self.scalograms = calculate_scalograms(
                dataset=np.stack((voltage_data.numpy(), np.zeros_like(voltage_data.numpy())), axis=2),
                sampling_period=sample_rate,
                wave_name=wave_name,
                sample_length=sample_length,
                total_scale=total_scale,
                fmax=fmax,
                image_size=image_size,
                save_path=scalograms_memmap_path
            )
            #self.scalograms = scalograms_tensor.numpy()  # Convert tensor to NumPy array here
        else:
            raise ValueError("Provide either (V_I_dataset) or (scalograms_path and core_loss_path).")
        
        if self.verbose:
            print(f"[DEBUG] Core Loss Stats - Mean: {np.mean(self.core_loss_values):.4f}, "
                  f"Min: {np.min(self.core_loss_values):.4f}, Max: {np.max(self.core_loss_values):.4f}")
    
    def __len__(self):
        return len(self.scalograms)

    def __getitem__(self, idx):
        scalogram = self.scalograms[idx]  # Shape: (1, 24, 24), now always a NumPy array
        core_loss = self.core_loss_values[idx]
        
        # Convert to tensor, handling np.memmap writability
        if isinstance(scalogram, np.ndarray):
            if not scalogram.flags.writeable:
                scalogram = scalogram.copy()
            scalogram = torch.from_numpy(scalogram).float()
        elif isinstance(scalogram, torch.Tensor):
            scalogram = scalogram.clone().detach().float()
                        
        # Apply transform if provided, or convert to tensor
        if self.transform:
            scalogram = self.transform(scalogram)  # e.g., ToPILImage(), Resize(224, 224), ToTensor()        
        
        core_loss = torch.tensor(core_loss, dtype=torch.float32)
        return scalogram, core_loss

    def get_raw_scalogram(self, idx):
        return self.scalograms[idx].squeeze().reshape(24, 24)







