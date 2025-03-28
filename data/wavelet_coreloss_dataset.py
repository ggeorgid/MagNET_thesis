import torch
import numpy as np
from torch.utils.data import Dataset
from data.preprocess_dataset import calculate_core_loss, calculate_scalograms

class WaveletCoreLossDataset(Dataset):
    def __init__(self, V_I_dataset=None, sample_rate=2e-6, sample_length=None, transform=None, wave_name='cgau8', 
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
            
            # Compute scalograms
            self.scalograms = calculate_scalograms(
                dataset=np.stack((voltage_data.numpy(), np.zeros_like(voltage_data.numpy())), axis=2),
                sampling_period=sample_rate,
                wave_name=wave_name,
                sample_length=sample_length
            )
        else:
            raise ValueError("Provide either (V_I_dataset) or (scalograms_path and core_loss_path).")
        
        if self.verbose:
            print(f"[DEBUG] Core Loss Stats - Mean: {np.mean(self.core_loss_values)}, Min: {np.min(self.core_loss_values)}, Max: {np.max(self.core_loss_values)}")
    
    def __len__(self):
        return len(self.scalograms)

    def __getitem__(self, idx):
        scalogram = self.scalograms[idx]
        core_loss = self.core_loss_values[idx]
        if self.transform:
            scalogram = self.transform(scalogram)
        if isinstance(scalogram, torch.Tensor):
            scalogram = scalogram.clone().detach().to(dtype=torch.float32)
        else:
            scalogram = torch.tensor(scalogram, dtype=torch.float32)
        core_loss = torch.tensor(core_loss, dtype=torch.float32)
        return scalogram, core_loss

    def get_raw_scalogram(self, idx):
        return self.scalograms[idx].squeeze().reshape(24, 24)



# import torch
# import numpy as np
# from torch.utils.data import Dataset
# from data.preprocess_dataset import calculate_core_loss, calculate_scalograms

# class WaveletCoreLossDataset(Dataset):
#     def __init__(self, V_I_dataset=None, sample_rate=2e-6, sample_length=None, transform=None, wave_name='cgau8', 
#                  scalograms_path=None, core_loss_path=None, verbose=True):
#         """
#         Initializes the dataset class.
#         - Loads precomputed scalograms and core loss if paths are provided.
#         - Otherwise, computes core loss and scalograms from `V_I_dataset`.
#         """
#         self.transform = transform
#         self.verbose = verbose
#         self.sample_rate = sample_rate
#         self.sample_length = sample_length

#         if scalograms_path and core_loss_path:
#             if self.verbose:
#                 print("[INFO] Loading precomputed scalograms and core loss.")
#             self.scalograms = np.load(scalograms_path, mmap_mode='r')
#             self.core_loss_values = np.load(core_loss_path, mmap_mode='r')
#         elif V_I_dataset is not None:
#             voltage_data = V_I_dataset.tensors[0]  # Keep as a tensor to avoid extra numpy conversion
            
#             if self.verbose:
#                 print(f"[INFO] Computing core loss & scalograms from raw dataset. Sample Length: {sample_length or voltage_data.shape[1]}, Sample Rate: {sample_rate}")
            
#             # ✅ Compute Core Loss using the correct sample rate (500 kHz, unchanged)
#             core_loss_tensor = calculate_core_loss(V_I_dataset=V_I_dataset).tensors[1]
#             self.core_loss_values = core_loss_tensor.numpy()  # Convert only when necessary
            
#             # ✅ Compute Scalograms using the correct sample period (2e-6s)
#             self.scalograms = calculate_scalograms(
#                 dataset=np.stack((voltage_data.numpy(), np.zeros_like(voltage_data.numpy())), axis=2),
#                 sampling_period=sample_rate,
#                 wave_name=wave_name,
#                 sample_length=sample_length
#             )
#         else:
#             raise ValueError("Provide either (V_I_dataset) or (scalograms_path and core_loss_path).")
        
#         # ✅ **DEBUG: Ensure Core Loss Values Have Variance After Loading**
#         #print(f"[DEBUG] Dataset Core Loss - Mean: {np.mean(self.core_loss_values)}, Variance: {np.var(self.core_loss_values)}")
    
#     def __len__(self):
#         return len(self.scalograms)

#     def __getitem__(self, idx):
#         """
#         Returns the scalogram and core loss value for a given index.
#         """
#         scalogram = self.scalograms[idx]
#         core_loss = self.core_loss_values[idx]

#         # Apply transform if provided
#         if self.transform:
#             scalogram = self.transform(scalogram)

#         # Convert scalogram to tensor if necessary
#         if isinstance(scalogram, torch.Tensor):
#             scalogram = scalogram.clone().detach().to(dtype=torch.float32)
#         else:
#             scalogram = torch.tensor(scalogram, dtype=torch.float32)

#         # Convert core_loss to tensor
#         core_loss = torch.tensor(core_loss, dtype=torch.float32)

#         # Warning: If you uncomment the following debug statements, make sure you're not training the model
#         # as it will print a lot of information to the console.
#         # ✅ **DEBUG PRINT STATEMENT: Check the First 10 Samples** 
#         # if idx < 10:
#         #     print(f"[DEBUG] Sample {idx} - Scalogram Shape: {scalogram.shape}, Core Loss: {core_loss}")
        
#         # ✅ **DEBUG PRINT: Check if Core Loss is Changing for Different Indices**
#         # if idx % 500 == 0:  # Print every 500 samples to check consistency
#         #     print(f"[DEBUG] Sample {idx} - Core Loss Value: {core_loss}")
        
#         return scalogram, core_loss

#     def get_raw_scalogram(self, idx):
#         """
#         Returns the raw scalogram reshaped to 24x24.
#         """
#         return self.scalograms[idx].squeeze().reshape(24, 24)




