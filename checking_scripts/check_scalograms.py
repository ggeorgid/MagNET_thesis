import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from pathlib import Path
import argparse
import random
import os

# Set up argument parser for configurable parameters
parser = argparse.ArgumentParser(description='Scalogram Verification Script')
parser.add_argument('--scalograms_path', type=str, default='/home/ggeorgid/my_projects/MagNET_thesis/data/processed/scalograms.npy', help='Path to scalograms.npy')
parser.add_argument('--resize_size', type=int, default=224, help='Target size for resized scalograms (e.g., 224 for ResNet)')
parser.add_argument('--num_samples', type=int, default=5, help='Number of random samples to visualize')
args = parser.parse_args()

# Ensure figures directory exists
os.makedirs('figures', exist_ok=True)

# Load scalograms.npy
scalograms = np.load(args.scalograms_path)
print(f"🔹 Loaded scalograms with shape: {scalograms.shape}")

# Check expected shape (N, 1, 24, 24)
expected_shape = (slice(None), 1, 24, 24)  # N can vary
if scalograms.ndim != 4 or scalograms.shape[1:] != expected_shape[1:]:
    raise ValueError(f"Unexpected shape for scalograms: {scalograms.shape}. Expected (N, 1, 24, 24).")

# Convert to tensor for consistency
scalograms_tensor = torch.tensor(scalograms, dtype=torch.float32)

# Calculate statistics for original scalograms
min_val = torch.min(scalograms_tensor).item()
max_val = torch.max(scalograms_tensor).item()
mean_val = torch.mean(scalograms_tensor).item()
num_neg = (scalograms_tensor < 0).sum().item()
perc_neg = (num_neg / scalograms_tensor.numel()) * 100

# Check for NaNs and Infs
nans = torch.isnan(scalograms_tensor).sum().item()
infs = torch.isinf(scalograms_tensor).sum().item()
print(f"🔹 NaNs in scalograms: {nans}, Infs: {infs}")

print(f"🔹 Original Scalogram Statistics - Min: {min_val:.6f}, Max: {max_val:.6f}, Mean: {mean_val:.6f}")
print(f"🔹 Negative Values: {num_neg} ({perc_neg:.2f}%)")

# Define the transform for resizing (mimics dataset preprocessing)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.resize_size, args.resize_size)),
    transforms.ToTensor()
])

# Select random samples to visualize
N = scalograms.shape[0]
random_indices = random.sample(range(N), min(args.num_samples, N))

# Visualize and compare original and resized scalograms
for idx in random_indices:
    # Original scalogram (24x24)
    original_scalogram = scalograms_tensor[idx].squeeze()  # Shape: (24, 24)
    
    # Resized scalogram (224x224)
    resized_scalogram = transform(original_scalogram.numpy()).squeeze()  # Shape: (224, 224)
    
    # Calculate statistics for resized scalogram
    resized_min = resized_scalogram.min().item()
    resized_max = resized_scalogram.max().item()
    
    # Plot original scalogram
    plt.figure(figsize=(6, 4))
    plt.imshow(original_scalogram.numpy(), cmap='jet', aspect='auto')
    plt.colorbar()
    plt.title(f"Original Scalogram Sample {idx} (24x24)\nMin: {min_val:.6f}, Max: {max_val:.6f}")
    plt.savefig(f"figures/original_scalogram_sample_{idx}.png")
    plt.close()
    
    # Plot resized scalogram
    plt.figure(figsize=(6, 4))
    plt.imshow(resized_scalogram.numpy(), cmap='jet', aspect='auto')
    plt.colorbar()
    plt.title(f"Resized Scalogram Sample {idx} (224x224)\nMin: {resized_min:.6f}, Max: {resized_max:.6f}")
    plt.savefig(f"figures/resized_scalogram_sample_{idx}.png")
    plt.close()
    
    print(f"✅ Visualized sample {idx}: Original Max: {max_val:.6f}, Resized Max: {resized_max:.6f}")

# Save statistics to a file
stats_path = Path('scalogram_stats.txt')
with open(stats_path, 'w') as f:
    f.write(f"Original Min: {min_val:.6f}\n")
    f.write(f"Original Max: {max_val:.6f}\n")
    f.write(f"Mean: {mean_val:.6f}\n")
    f.write(f"Negative Values: {num_neg} ({perc_neg:.2f}%)\n")
    f.write(f"NaNs: {nans}, Infs: {infs}\n")
print(f"✅ Saved statistics to '{stats_path}'")

# Evaluation and Analysis
print("\n=== Evaluation & Analysis ===")
print(f"1. **Shape Check:** Scalograms shape {scalograms.shape} matches expected (N, 1, 24, 24).")
print(f"2. **Range Check:** Original scalograms range from {min_val:.6f} to {max_val:.6f} (mean {mean_val:.6f}).")
print(f"   - After resizing, expect [0, 1] due to ToTensor(). Verify visually.")
print(f"3. **Negatives:** {num_neg} values ({perc_neg:.2f}%) are negative.")
print(f"   - Suggestion: If problematic, clamp negatives with `torch.clamp(scalograms, min=0)`.") if num_neg > 0 else None
print(f"4. **Scaling Factor:** Original max is {max_val:.6f}. Post-ToTensor() max should be 1.0.")
print(f"   - To denormalize resized scalograms, multiply by {max_val:.6f}.")
print("5. **Verdict:** Check figures for feature preservation in resized scalograms.")


# Run the Script:

#     Execute it from the terminal:
#     bash

# python3 checking_scripts/check_scalograms.py --scalograms_path /home/ggeorgid/my_projects/MagNET_thesis/data/processed/scalograms.npy --resize_size 224 --num_samples 5
# Adjust the arguments:

#     --scalograms_path: Path to your scalograms.npy file.
#     --resize_size: Size for resized scalograms (default: 224).
#     --num_samples: Number of random samples to visualize (default: 5).


# What the Script Does
# 1. Loads and Verifies Scalograms

#     Loads scalograms.npy from data/processed/ and checks if its shape is (N, 1, 24, 24).

# 2. Calculates Statistics

#     Computes the minimum, maximum, and mean values of the original scalograms.
#     Checks for negative values, NaNs, and Infs, reporting their counts and percentages.

# 3. Visualizes Scalograms

#     Selects random samples (default: 5) and plots:
#         Original (24x24): Displays the raw scalogram.
#         Resized (224x224): Applies the same resizing transform as in your dataset (e.g., for ResNet) and shows the result.
#     Saves plots to figures/ for inspection.

# 4. Verifies the Original Max

#     Reports the original maximum value to determine the scaling factor.
#     After ToTensor() (which normalizes to [0, 1]), the resized max should be 1.0. To denormalize back to the original range, multiply by the original max.

# 5. Provides Analysis

#     Summarizes findings, including shape correctness, range, negatives, and scaling factor guidance.
#     Suggests clamping negatives if present and advises checking feature preservation in resized scalograms.
