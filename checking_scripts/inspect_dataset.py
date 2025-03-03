import pandas as pd
import os
import numpy as np

# ======================== Dataset Folder Path ========================
# Define the path to the raw dataset folder containing CSV files.
dataset_folder = "/home/ggeorgid/my_projects/MagNET_thesis/data/raw/dataset"

# ======================== List All CSV Files ========================
# Retrieve all CSV files in the dataset folder (excluding subdirectories).
csv_files = [f for f in os.listdir(dataset_folder) if f.endswith(".csv")]

# Print the found CSV files for verification.
print("\n🔹 Found CSV files in dataset folder:")
for file in csv_files:
    print(f"  - {file}")

# ======================== Load Metadata from info.csv ========================
# The 'info.csv' file contains metadata about the dataset.
info_path = os.path.join(dataset_folder, "info.csv")

if "info.csv" in csv_files:
    print("\n🔹 Loading metadata from info.csv...")
    
    # Read the metadata file into a DataFrame.
    info_df = pd.read_csv(info_path)

    # Print full contents of the metadata file for reference.
    print(info_df)
else:
    print("\n⚠️  Warning: info.csv not found! Skipping metadata load.")

# ======================== Process Each CSV File ========================
# Loop through each dataset file (excluding 'info.csv').
for csv_file in csv_files:
    if csv_file == "info.csv":
        continue  # Skip metadata file

    # Construct the full file path for reading.
    file_path = os.path.join(dataset_folder, csv_file)
    print(f"\n🔹 Inspecting {csv_file}...")

    try:
        # Read CSV file while skipping the first row (assumed to contain column names).
        df = pd.read_csv(file_path, skiprows=1, header=None)

        # Determine the expected number of time steps per sample.
        # Some dataset files (e.g., containing 'sin' or 'tri') have 8192 time steps.
        # Otherwise, assume 8000 time steps per sample.
        num_time_steps = 8192 if "sin" in csv_file or "tri" in csv_file else 8000

        # Compute the number of samples in the dataset.
        num_samples = df.shape[0] // num_time_steps  # Integer division
        if df.shape[0] % num_time_steps != 0:
            print(f"⚠️ Warning: {csv_file} has extra rows that don't fit the expected shape!")

        # Reshape the dataset into the required format: (num_samples, num_time_steps, 2)
        # - num_samples: Number of full waveform samples.
        # - num_time_steps: Number of time steps per sample.
        # - 2: Two data channels (Voltage, Current).
        dataset_array = df.to_numpy().reshape(num_samples, num_time_steps, 2)

        print(f"📌 Reshaped to: {dataset_array.shape} (Samples={num_samples}, Time Steps={num_time_steps}, Channels=2)")

        # ======================== Extract & Analyze Data ========================
        # Extract Voltage (V) and Current (I) as separate NumPy arrays.
        V = dataset_array[:, :, 0]  # Voltage channel
        I = dataset_array[:, :, 1]  # Current channel

        # Compute and print statistical properties for both Voltage and Current.
        print("\n📊 Value Statistics:")
        print(f"  - Voltage (V): Min={V.min():.6f}, Max={V.max():.6f}, Mean={V.mean():.6f}, Std={V.std():.6f}")
        print(f"  - Current (I): Min={I.min():.6f}, Max={I.max():.6f}, Mean={I.mean():.6f}, Std={I.std():.6f}")

    except Exception as e:
        print(f"⚠️ Error loading {csv_file}: {e}")

# ======================== Final Message ========================
print("\n✅ Dataset inspection complete!")


# Key Conclusions from the Dataset Analysis

# After inspecting the Min, Max, Mean, and Standard Deviation tables, we can make the following observations:
# 1️⃣ Voltage and Current Ranges Differ Across Waveform Types

#     Sinusoidal (sin) waveforms have a relatively small range:
#         Voltage (V): Typically between -3 and 3 volts.
#         Current (I): Ranges from about -0.83 to 0.83 A.
#         This suggests a smooth variation with limited fluctuations.
#     Triangular (tri) waveforms show more variation:
#         Voltage (V): Some datasets go up to ±10V (tri_5k).
#         Current (I): Slightly more variability, but still within ±1A.
#         Indicates a sharper, more varying signal compared to sinusoids.
#     Trapezoidal (trap) waveforms have the largest range:
#         Voltage (V): Reaches ±10V, especially in trap_5k.
#         Current (I): Some values exceed ±1.5A, meaning stronger variations.
#         Suggests a waveform with sharp transitions and potential high peaks.

# 2️⃣ Standard Deviation Analysis Shows Variability

#     High standard deviation (Voltage Std > 3V) in:
#         tri_5k.csv, trap_5k.csv, and sin_5k.csv → means high fluctuations.
#     Lower standard deviation (Voltage Std < 1V) in:
#         tri_1k.csv and sin_1k.csv → means more stable waveforms.

# 🔹 Takeaway: We might need to normalize the dataset before feeding it into the model. Otherwise, waveforms with larger ranges (e.g., trap_5k.csv) might dominate the training process.
# 3️⃣ Mean Values Indicate No Major Offset Issues

#     The mean voltage (V) and current (I) values are close to zero in most cases.
#     Some slight offsets exist in trapezoidal datasets, where mean ≈ 0.06V.
#     Why is this important?
#         A strong nonzero mean could indicate a DC offset, which might need to be corrected.
#         However, since most values are close to zero, no major DC bias correction seems necessary.

# 4️⃣ Dataset Size Distribution
# Waveform Type	Total Samples
# Sinusoidal (sin)	1050
# Triangular (tri)	4712
# Trapezoidal (trap)	4174

#     Triangular waveforms dominate the dataset (~47% of total time series).
#     Trapezoidal waveforms are also significant, though we are currently skipping them.
#     Sinusoidal waveforms are underrepresented (~10% of total data).

# 🔹 Takeaway: If we train a model on sin + tri waveforms only (as we're doing now), it may learn better on tri waveforms since they're more frequent. If necessary, we could balance the dataset later.
# 5️⃣ Potential Next Steps

# Now that we understand the dataset, we can make some informed decisions:

#     Normalization:
#         Since voltage varies between ±10V and current is around ±1A, we might consider scaling or normalizing the dataset.
#         Options: Min-Max scaling ([0,1] or [-1,1]) or Z-score standardization.

#     Ensuring Dataset Correctness Before Training
#         Before uncommenting the training loop in wavelet_main.py, we should confirm that scalograms and core loss calculations are working correctly.

#     Dataset Balancing (If Needed Later)
#         If we include the trapezoidal series later, we should check whether we need to balance the dataset to avoid overfitting on one waveform type.

# 📌 Summary of Findings

# ✔ Voltage ranges from ~-10V to 10V, and current from ~-1A to 1A in most cases.
# ✔ Triangular waveforms dominate (~47%), followed by trapezoidal (~42%) and sinusoidal (~10%).
# ✔ Standard deviations vary, meaning some datasets fluctuate more than others.
# ✔ Mean values are close to zero, indicating no major DC bias issues.
# ✔ Next steps: Check dataset transformations (scalograms, core loss), then move toward training.
# 🚀 What Do You Want to Do Next?

#     Do you want to inspect the scalogram and core loss calculations in wavelet_main.py?
#     Or do you want to move forward with debugging the dataset pipeline in wavelet_main.py?

# Let me know how you’d like to proceed! 🚀
