import sys
import itertools
import subprocess
import datetime
import time

# Check OS type for correct Python command
python_cmd = "python3" if sys.platform != "win32" else "python"

# Define hyperparameter values for the sweep
num_epochs = [50]
learning_rates = [0.001, 0.0005]
batch_sizes = [32, 64, 128, 256]

# Generate a timestamp for the log file
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"transfer_sweep_{timestamp}.txt"

# Open the log file in append mode
with open(log_file, "a") as f:
    for lr, batch_size in itertools.product(learning_rates, batch_sizes):
        f.write(f"🔹 Starting run: NUM_EPOCH={num_epochs[0]}, LEARNING_RATE={lr}, BATCH_SIZE={batch_size} at {datetime.datetime.now()}\n")

        # Start the subprocess with unbuffered output
        process = subprocess.Popen(
            [
                python_cmd, "-u", "transfer_main.py",  # -u for unbuffered output
                "--num_epoch", str(num_epochs[0]),
                "--learning_rate", str(lr),
                "--batch_size", str(batch_size)
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Capture stdout in real-time
        for line in iter(process.stdout.readline, ''):
            f.write(f"[STDOUT] {line}")

        # Capture stderr in real-time
        for line in iter(process.stderr.readline, ''):
            f.write(f"[STDERR] {line}")

        # Wait for the process to complete with a timeout
        try:
            process.wait(timeout=3600)  # 1-hour timeout
            if process.returncode == 0:
                f.write(f"✅ Completed run: NUM_EPOCH={num_epochs[0]}, LEARNING_RATE={lr}, BATCH_SIZE={batch_size} at {datetime.datetime.now()}\n\n")
            else:
                f.write(f"❌ Run failed with error code {process.returncode} at {datetime.datetime.now()}\n\n")
        except subprocess.TimeoutExpired:
            process.kill()
            f.write(f"[ERROR] Run timed out and was killed at {datetime.datetime.now()}\n\n")

        time.sleep(5)  # Delay to release resources

# Final log entry
with open(log_file, "a") as f:
    f.write(f"[DEBUG] Script finished at {datetime.datetime.now()}\n")