import sys
import itertools
import subprocess

# Check OS type for correct Python command
python_cmd = "python3" if sys.platform != "win32" else "python"

# Define hyperparameter values for the sweep
num_epochs = [100]  # Single fixed value
learning_rates = [0.001, 0.0005]
batch_sizes = [32, 64, 128, 256]

# Iterate over all combinations
for lr, batch_size in itertools.product(learning_rates, batch_sizes):
    print(f"🔹 Running with: NUM_EPOCH={num_epochs[0]}, LEARNING_RATE={lr}, BATCH_SIZE={batch_size}")

    # Call wavelet_main.py and stream output in real-time
    process = subprocess.Popen(
        [
            python_cmd, "wavelet_main.py",
            "--num_epoch", str(num_epochs[0]),
            "--learning_rate", str(lr),
            "--batch_size", str(batch_size)
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Print output line by line as it's generated
    for line in process.stdout:
        print(line, end='')  # end='' prevents double newlines

    # Print errors if any
    for line in process.stderr:
        print(f"❌ ERROR: {line}", end='')

    # Wait for process to complete
    process.wait()

    if process.returncode == 0:
        print(f"✅ Completed run: NUM_EPOCH={num_epochs[0]}, LEARNING_RATE={lr}, BATCH_SIZE={batch_size}\n")
    else:
        print(f"❌ Run failed with error code {process.returncode}\n")
        
        
# The automated wandb sweep setting was not working because agent was not initializing properly.
# Here is the code for consistency. 
# program: main.py  # Ensure the correct entry point for the sweep
# method: grid

# project: sweep_experiment  # <-- Add this line to explicitly define project name

# metric:
#   name: validation_loss  # The metric we optimize
#   goal: minimize

# parameters:
#   NUM_EPOCH:
#     values: [50]
#   DATA_SUBSET_SIZE:
#     values: [128, 256, 1024, 4096]
#   LEARNING_RATE:
#     values: [0.001, 0.0005]
#   BATCH_SIZE:
#     values: [32, 64]
