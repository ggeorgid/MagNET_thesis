#-----------------------RUN THIS-----------------------------------

# import pandas as pd
# import matplotlib.pyplot as plt

# df = pd.read_csv("data/raw/dataset/data_trap_1k.csv", header=None)
# sample_length = int(df.iloc[0, 1])
# data = df.iloc[1:].values.reshape(-1, sample_length, 2)
# for i in range(3):  # Plot 3 samples
#     plt.plot(data[i, :, 0], label=f"Sample {i}")
#     plt.legend()
#     plt.show()

#------------------------------OR THIS-----------------------------------
# The script processes and compares a preprocessed dataset (dataset.npy) with 
# an original dataset (data_trap_1k.csv), plotting voltage and current waveforms 
# for three randomly selected samples, computing statistics, and checking for trapezoidal 
# shapes in the voltage data. The script output and figures provide key insights into the data’s behavior
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# Paths
preprocessed_data_path = "data/processed/dataset.npy"
original_csv_path = "data/raw/dataset/data_trap_1k.csv"

# Load processed dataset
dataset = np.load(preprocessed_data_path)
print(f"Full dataset shape: {dataset.shape}")

# Load original CSV and reshape
df = pd.read_csv(original_csv_path, header=None, dtype={0: str, 1: np.float64})
sample_period, sample_length = df.iloc[0]
sample_length = int(sample_length)
original_data = df.iloc[1:].values.astype(np.float64)
num_samples = original_data.shape[0] // sample_length
original_trapezoidal_1k = original_data.reshape(num_samples, sample_length, -1)

# Extract corresponding processed samples
trapezoidal_1k_data = dataset[-num_samples:, :, :]
print(f"Extracted trapezoidal data shape: {trapezoidal_1k_data.shape}")
if num_samples != trapezoidal_1k_data.shape[0]:
    print("[WARNING] Mismatch in sample counts!")

# Set seed for reproducibility
random.seed(42)
num_samples_to_plot = 3
relative_random_indices = random.sample(range(num_samples), num_samples_to_plot)
absolute_random_indices = [idx + len(dataset) - num_samples for idx in relative_random_indices]

# Plot both channels
for channel, label in enumerate(["Voltage", "Current"]):
    plt.figure(figsize=(12, 5))
    for i, idx in enumerate(relative_random_indices, 1):
        plt.subplot(1, num_samples_to_plot, i)
        plt.plot(trapezoidal_1k_data[idx, :, channel], label=f"Processed Sample {idx}")
        plt.plot(original_trapezoidal_1k[idx, :, channel], label=f"Original Sample {idx}")
        plt.title(f"Sample {idx} ({label})")
        plt.legend()
    plt.tight_layout()
    plt.show()

# Compute statistics
for channel, label in enumerate(["Voltage", "Current"]):
    proc_mean = np.mean(trapezoidal_1k_data[:, :, channel])
    proc_var = np.var(trapezoidal_1k_data[:, :, channel])
    orig_mean = np.mean(original_trapezoidal_1k[:, :, channel])
    orig_var = np.var(original_trapezoidal_1k[:, :, channel])
    print(f"{label} - Processed Mean: {proc_mean}, Variance: {proc_var}")
    print(f"{label} - Original Mean: {orig_mean}, Variance: {orig_var}")

# Check trapezoidal shape
def is_trapezoidal(data, tolerance=0.01, min_flat_length=50):
    diff = np.abs(np.diff(data))
    for i in range(len(diff) - min_flat_length):
        if np.all(diff[i:i + min_flat_length] < tolerance):
            return True
    return False

for idx in relative_random_indices:
    is_trap = is_trapezoidal(original_trapezoidal_1k[idx, :, 0])
    print(f"Sample {idx} - Trapezoidal (Voltage): {is_trap}")