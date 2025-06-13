import torch
import numpy as np
from torch.utils.data import Dataset
from data.preprocess_dataset import calculate_core_loss

class VoltageCoreLossDataset(Dataset):
    def __init__(self, dataset_path=None, V_I_dataset=None, core_loss_path=None, verbose=True):
        """
        Dataset class for predicting core loss directly from raw voltage time series.
        - Loads precomputed core loss if a path is provided.
        - Otherwise, computes core loss from `V_I_dataset`.
        """
        self.verbose = verbose

        # Load or set voltage data
        if dataset_path:
            if self.verbose:
                print("[INFO] Loading preprocessed dataset from file.")
            full_data = np.load(dataset_path, mmap_mode='r')  # [9936, 8000, 2]
            self.voltage_data = full_data[:, :, 0]  # Extract voltage: [9936, 8000]
        elif V_I_dataset is not None:
            if self.verbose:
                print("[INFO] Using voltage data from V_I_dataset.")
            self.voltage_data = V_I_dataset.tensors[0].numpy()  # [9936, 8000]
        else:
            raise ValueError("Provide either `dataset_path` or `V_I_dataset`.")

        # Load or compute core loss
        if core_loss_path:
            if self.verbose:
                print("[INFO] Loading precomputed core loss.")
            self.core_loss_values = np.load(core_loss_path, mmap_mode='r')  # [9936]
        elif V_I_dataset is not None:
            if self.verbose:
                print("[INFO] Computing core loss from V_I_dataset.")
            core_loss_tensor = calculate_core_loss(V_I_dataset=V_I_dataset).tensors[1]  # Use full V_I_dataset
            self.core_loss_values = core_loss_tensor.numpy()  # [9936]
        else:
            raise ValueError("Core loss values must be provided or computed from `V_I_dataset`.")

        # Clamp core loss values to non-negative
        self.core_loss_values = np.maximum(self.core_loss_values, 0)

        if self.verbose:
            print(f"[INFO] Dataset Loaded - Samples: {len(self.voltage_data)}")
            print(f"[DEBUG] Core Loss Stats - Mean: {np.mean(self.core_loss_values)}, Min: {np.min(self.core_loss_values)}, Max: {np.max(self.core_loss_values)}")

    def __len__(self):
        return len(self.voltage_data)

    def __getitem__(self, idx):
        """
        Returns voltage time series and core loss value for a given index.
        - Voltage shape: [1, 8000] (single-channel time series for 1D convolution).
        - Core loss shape: scalar (batched as [batch_size] by dataloader).
        """        
        voltage = torch.from_numpy(self.voltage_data[idx].copy()).float().unsqueeze(0)  # [1, 8000] added copy() to avoid warning
        core_loss = torch.tensor(self.core_loss_values[idx], dtype=torch.float32)  # scalar
        return voltage, core_loss


