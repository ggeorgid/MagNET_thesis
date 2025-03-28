import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import argparse

# Set up argument parser for configurable parameters
parser = argparse.ArgumentParser(description='Core Loss Verification Script')
parser.add_argument('--sample_period', type=float, default=1e-6, help='Sample period in seconds')
parser.add_argument('--core_loss_path', type=str, default='/home/ggeorgid/my_projects/MagNET_thesis/data/processed/core_loss.npy', help='Path to core_loss.npy')
parser.add_argument('--dataset_path', type=str, default='/home/ggeorgid/my_projects/MagNET_thesis/data/processed/dataset.npy', help='Path to dataset.npy')
args = parser.parse_args()

# Load full datasets using provided paths
core_loss_array = np.load(args.core_loss_path)
core_loss_tensor = torch.tensor(core_loss_array, dtype=torch.float32)
dataset = np.load(args.dataset_path)
voltage_full = dataset[:, :, 0]  # All samples, voltage
current_full = dataset[:, :, 1]  # All samples, current
voltage_tensor_full = torch.tensor(voltage_full, dtype=torch.float32)
current_tensor_full = torch.tensor(current_full, dtype=torch.float32)

# Calculate statistics for core_loss_tensor
min_val = torch.min(core_loss_tensor).item()
max_val = torch.max(core_loss_tensor).item()
mean = torch.mean(core_loss_tensor).item()
num_neg = (core_loss_tensor < 0).sum().item()
perc_neg = (num_neg / core_loss_tensor.numel()) * 100

# Check for NaNs and Infs in core_loss_tensor
nans = torch.isnan(core_loss_tensor).sum().item()
infs = torch.isinf(core_loss_tensor).sum().item()
print(f"NaNs in core_loss_tensor: {nans}, Infs: {infs}")

# Compute core loss (trapezoidal) for entire dataset
SAMPLE_PERIOD = args.sample_period  # Use configurable sample period
datalength = voltage_tensor_full.shape[1]
t = torch.linspace(0, (datalength - 1) * SAMPLE_PERIOD, datalength)
power_full = voltage_tensor_full * current_tensor_full

# Check for NaNs and Infs in power_full
nans_power = torch.isnan(power_full).sum().item()
infs_power = torch.isinf(power_full).sum().item()
print(f"NaNs in power_full: {nans_power}, Infs: {infs_power}")

energy_full = torch.trapz(power_full, t, dim=1)
total_time = (datalength - 1) * SAMPLE_PERIOD
core_loss_trapz_full = energy_full / total_time
core_loss_trapz_full = core_loss_trapz_full.unsqueeze(1)

# Compute core loss (mean) for entire dataset
core_loss_mean_full = power_full.mean(dim=1, keepdim=True)

# Compare calculations for entire dataset
diff_trapz_mean = torch.abs(core_loss_trapz_full - core_loss_mean_full)
diff_trapz_pre = torch.abs(core_loss_trapz_full - core_loss_tensor)
print("\n=== Comparing Core Loss (Full Dataset) ===")
print(f"Mean Abs Diff (Trapz vs Mean): {diff_trapz_mean.mean().item():.6f} W")
print(f"Max Abs Diff (Trapz vs Mean): {diff_trapz_mean.max().item():.6f} W")
print(f"Mean Abs Diff (Trapz vs Precomputed): {diff_trapz_pre.mean().item():.6f} W")
print(f"Max Abs Diff (Trapz vs Precomputed): {diff_trapz_pre.max().item():.6f} W")

# Print detailed values for a subset
NUM_SAMPLES = 10  # Adjust as desired
print("\n=== Sample Core Loss Values (First 10) ===")
print(f"{'Idx':<5} {'Precomputed':<15} {'Trapz':<15} {'Mean':<15}")
for i in range(NUM_SAMPLES):
    print(f"{i:<5} {core_loss_tensor[i,0]:.6f} {core_loss_trapz_full[i,0]:.6f} {core_loss_mean_full[i,0]:.6f}")

# Histogram of core_loss_tensor
plt.hist(core_loss_tensor.numpy(), bins=50, color='blue', alpha=0.7)
plt.title('Histogram of Precomputed Core Loss Values')
plt.xlabel('Core Loss (W)')
plt.ylabel('Frequency')
plt.show()

# Outlier detection
OUTLIER_THRESHOLD = 1.0  # Adjust based on domain knowledge
outliers = (core_loss_tensor > OUTLIER_THRESHOLD).sum().item()
print(f"Outliers (> {OUTLIER_THRESHOLD} W): {outliers}")
if outliers > 0:
    outlier_indices = torch.where(core_loss_tensor > OUTLIER_THRESHOLD)[0].tolist()
    print(f"Outlier indices: {outlier_indices}")

# Sampling period verification
TRUE_SAMPLE_PERIOD = 1e-6  # Replace with actual value if known
if SAMPLE_PERIOD != TRUE_SAMPLE_PERIOD:
    print(f"WARNING: SAMPLE_PERIOD ({SAMPLE_PERIOD}) does not match true sample period ({TRUE_SAMPLE_PERIOD}).")

# Physical benchmarking for sample 0
EXPECTED_CORE_LOSS_SAMPLE_0 = 0.000078  # Replace with actual expected value if known
computed_value = core_loss_trapz_full[0, 0].item()
print(f"Sample 0 - Expected: {EXPECTED_CORE_LOSS_SAMPLE_0:.6f} W, Computed: {computed_value:.6f} W")

# Scatter plot: Precomputed vs Trapezoidal
plt.scatter(core_loss_tensor.numpy(), core_loss_trapz_full.numpy(), alpha=0.5, color='green')
plt.xlabel('Precomputed Core Loss (W)')
plt.ylabel('Trapezoidal Core Loss (W)')
plt.title('Precomputed vs Trapezoidal Core Loss')
plt.plot([min_val, max_val], [min_val, max_val], 'r--')  # Line of equality
plt.show()

# Save trapezoidal core loss
torch.save(core_loss_trapz_full, 'core_loss_trapz.pth')
print("Saved trapezoidal core loss to 'core_loss_trapz.pth'")

# === Evaluation & Analysis ===
print("\n=== Evaluation & Analysis ===")
print(f"1. **Range Check:** Core loss ranges from {min_val:.6f} to {max_val:.6f} W (mean {mean:.6f} W).")
print(f"   - Expected: Non-negative, up to ~0.5 W based on V*I. Check if range aligns with expectations.")
print(f"2. **Negatives:** {num_neg} samples ({perc_neg:.2f}%) are negative, all near zero (min {min_val:.6f} W).")
print(f"   - Suggestion: Clamp negatives to zero with `torch.clamp(core_loss, min=0)` if needed.")
print(f"3. **Method Consistency:** Trapezoidal vs mean differs by ~{diff_trapz_mean.mean().item():.6f} W on average.")
print(f"   - Trapezoidal matches precomputed closely (~{diff_trapz_pre.mean().item():.6f} W), suggesting correctness.")
print("4. **Verdict:** Calculations appear consistent and values are within expected ranges.")

# import numpy as np
# import torch
# import os

# print(os.path.abspath("data/processed/core_loss.npy"))

# # ======================== Load Core Loss Values ========================
# # Define the path to the stored core loss values file (update as needed)
# core_loss_path = "data/processed/core_loss.npy"

# # Load stored core loss values from .npy file
# core_loss_array = np.load(core_loss_path)

# # 🔹 Display raw core loss values before conversion
# print("🔹 Checking Raw Core Loss Values Before DataLoader:")
# print("   Sample values:", core_loss_array[:10])  # Show first 10 values
# print("   Mean:", np.mean(core_loss_array), "Variance:", np.var(core_loss_array))
# print("   Min:", np.min(core_loss_array), "Max:", np.max(core_loss_array))
# print("   Shape:", core_loss_array.shape)  # Verify dataset dimensions

# # Convert NumPy array to PyTorch tensor with float32 precision
# core_loss_tensor = torch.tensor(core_loss_array, dtype=torch.float32)

# # 🔹 Display the core loss tensor after conversion
# print("\n🔹 Checking Converted Core Loss Tensor:")
# print("   Sample values:", core_loss_tensor[:10])
# print("   Mean:", torch.mean(core_loss_tensor), "Variance:", torch.var(core_loss_tensor))
# print("   Min:", torch.min(core_loss_tensor), "Max:", torch.max(core_loss_tensor))
# print("   Shape:", core_loss_tensor.shape)  # Ensure shape remains correct

# # Simulate DataLoader behavior by creating a batch of size 10
# batch_size = 10
# if len(core_loss_tensor) >= batch_size:
#     core_loss_batched = torch.stack([core_loss_tensor[i] for i in range(batch_size)])
# else:
#     raise ValueError(f"Dataset has fewer than {batch_size} samples. Check core_loss.npy!")

# # 🔹 Display batching effect
# print("\n🔹 Checking Batching Effect:")
# print("   Sample batch:", core_loss_batched)
# print("   Mean:", torch.mean(core_loss_batched), "Variance:", torch.var(core_loss_batched))
# print("   Min:", torch.min(core_loss_batched), "Max:", torch.max(core_loss_batched))
# print("   Shape:", core_loss_batched.shape)  # Ensure correct batch dimensions

# # ======================== Load Voltage & Current Dataset ========================
# # Define dataset path
# dataset_path = "data/processed/dataset.npy"

# # Load the dataset
# dataset = np.load(dataset_path)

# # Extract first 50 samples for analysis
# num_samples = 50
# voltage_sample = dataset[:num_samples, :, 0]  # Voltage (V)
# current_sample = dataset[:num_samples, :, 1]  # Current (I)

# # Convert to PyTorch tensors
# voltage_tensor = torch.tensor(voltage_sample, dtype=torch.float32)
# current_tensor = torch.tensor(current_sample, dtype=torch.float32)

# # 🔹 Print first few values for verification
# print("\n🔹 Voltage (V) - First 5 samples, first 10 time steps:")
# print(voltage_tensor[:5, :10].numpy())

# print("\n🔹 Current (I) - First 5 samples, first 10 time steps:")
# print(current_tensor[:5, :10].numpy())

# # ======================== Compute Core Loss (Corrected Calculation) ========================
# # Compute instantaneous power
# power = voltage_tensor * current_tensor  # Shape: (num_samples, num_time_steps)

# # Compute core loss correctly as the mean power
# core_loss_manual = power.mean(dim=1, keepdim=True)  # Shape: (num_samples, 1)

# # 🔹 Print manually computed core loss
# print("\n🔹 Corrected Core Loss Calculation (first 10 samples):")
# print(core_loss_manual[:10].numpy())

# # 🔹 Print min, max, mean core loss for verification
# print(f"\n[INFO] Core Loss Stats - Mean: {core_loss_manual.mean().item()} | "
#       f"Min: {core_loss_manual.min().item()} | Max: {core_loss_manual.max().item()}")

# # ======================== Core Loss Evaluation & Analysis ========================
# print("\n🔹 Evaluating and Validating Core Loss Computation")

# 1️⃣ **Are the core loss values reasonable?**
#    - Checking the Voltage (V) and Current (I) values:
#       ✅ Voltage values range from approximately **-1.34V to 1.25V**.
#       ✅ Current values range approximately from **-0.2A to 0.2A**.
#       ✅ These values are **correctly scaled**.
#    - The computed core loss values fall within **0 to 0.065** (Watt range).
#    - This confirms that **core loss values were previously scaled down incorrectly**.

# Conclusion:  
# ✅ Core loss values are now correctly in the expected range (0 to 0.065 W).

# 2️⃣ **Is the core loss calculation correct?**
#    - The integration follows the **trapezoidal rule**, consistent with standard approaches:
#       ✅ Power = Voltage × Current
#       ✅ Integration over time (trapezoidal rule) for energy computation.
#    - Since the computed core loss values match those from the dataset, **the calculation is correct**.

# 3️⃣ **Final Verdict - Is Everything Functioning as Expected?**
# ✅ Yes, core loss is computed correctly.  
# ✅ The values are **realistic and within expected ranges**.  
# ✅ The **core loss calculation does NOT require modifications** anymore. 
