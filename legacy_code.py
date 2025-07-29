#This script is not meant to be run. It is a collection of code snippets that were used in the development of the project.
#The code snippets are just here in case we need them again in the future or as a guide-reference.
#This script is not meant to be in the final version of the project.
#First code cleanup: 2025-02-08 

#--------------------------------------------Unused functions---------------------------------
def check_cached_scalograms(preprocessed_data_path):
    """
    Check if the precomputed scalograms and core loss files exist in the given directory.

    Args:
        preprocessed_data_path (Path): The path to the directory containing preprocessed data.

    Returns:
        bool: True if both 'scalograms.npy' and 'core_loss.npy' exist, False otherwise.
    """
    scalogram_file = preprocessed_data_path / "scalograms.npy"
    core_loss_file = preprocessed_data_path / "core_loss.npy"
    return scalogram_file.exists() and core_loss_file.exists()



#--------------------------------- Inspecting the Dataset --------------------------------
    
print(f"\n🔹 Length of tensor_dataset: {len(tensor_dataset)}")
print(f"🔹 Type of tensor_dataset: {type(tensor_dataset)}\n")    
    
# Inspect first sample in the dataset (voltage and current)
sample = tensor_dataset[0]
print(f"🔹 First sample in the dataset:\n{sample}\n")

# Initialize min and max tracking variables
min_value, max_value = float('inf'), float('-inf')
min_tensor, max_tensor = None, None

# Iterate over TensorDataset to find min and max values
for voltage, current in tensor_dataset:
    voltage_min, current_min = voltage.min().item(), current.min().item()
    voltage_max, current_max = voltage.max().item(), current.max().item()

    # Update min and max
    if voltage_min < min_value:
        min_value, min_tensor = voltage_min, voltage
    if current_min < min_value:
        min_value, min_tensor = current_min, current
    if voltage_max > max_value:
        max_value, max_tensor = voltage_max, voltage
    if current_max > max_value:
        max_value, max_tensor = current_max, current

print(f"\n🔹 Min value in tensor_dataset: {min_value}")
print(f"🔹 Max value in tensor_dataset: {max_value}\n")

print("🔹 Tensor with the minimum value:\n", min_tensor, "\n")
print("🔹 Tensor with the maximum value:\n", max_tensor, "\n")

#-------------------------------------Calculating the Magnetic Core Loss----------------------------
#Important parameters(hyperparameters maybe? <-ask this)
DATA_LENGTH = 8192 #400 se aytous gia kapoio logo ?????????    
SAMPLE_RATE = 2e-6

core_loss_dataset = calculate_core_loss(DATA_LENGTH, SAMPLE_RATE, tensor_dataset)
voltage_tensor, core_loss_tensor = core_loss_dataset.tensors  # Unpack dataset tensors

print(f"\n🔹 Length of core_loss_dataset: {len(core_loss_dataset)}")
print(f"🔹 Voltage tensor shape: {voltage_tensor.shape}")
print(f"🔹 Core loss tensor shape: {core_loss_tensor.shape}\n")

# Inspect first sample in core loss dataset
sample = core_loss_dataset[0]
print(f"🔹 First sample in the core loss dataset:\n{sample}\n")

# Apply dataset subsetting if specified using random permutation ordering e.g. tensor([523, 198, 754, 12, 899, ...])
if data_subset_size:
    print(f"[INFO] Subsetting enabled. Preparing to select {data_subset_size} samples.")
    indices = torch.randperm(len(tensor_dataset))[:data_subset_size]
    tensor_dataset = torch.utils.data.Subset(tensor_dataset, indices)
    print(f"[INFO] Using a subset of {data_subset_size} samples for training/debugging.\n")
    
    
    
    
#---------------------Old Creating Scalograms and Coreloss Dataset Version----------------------------
if use_cached_scalograms and check_cached_scalograms(preprocessed_data_path):
    print("[INFO] Loading cached scalograms and core loss from disk.\n")
    wavelet_dataset = WaveletCoreLossDataset(
        scalograms_path=scalograms_path,
        core_loss_path=core_loss_path
    )
else:
    print("[INFO] Creating WaveletCoreLossDataset and calculating scalograms and core loss.\n")
    wavelet_dataset = WaveletCoreLossDataset(
        V_I_dataset=tensor_dataset,
        sample_rate=SAMPLE_RATE
    )
    try:
        np.save(scalograms_path, wavelet_dataset.scalograms.numpy())
        np.save(core_loss_path, wavelet_dataset.core_loss_values.numpy())
        print("[INFO] Scalograms and core loss saved successfully.\n")
    except Exception as e:
        print(f"[ERROR] Failed to save scalograms or core loss: {e}")
        
           
#-----------------------------------Older way of training the model--------------------------------
#-------------------------------- Training/Testing the Model --------------------------------
trained_model = train_model(
    model=model,
    train_loader=train_loader,
    valid_loader=valid_loader,
    test_loader=test_loader,
    config=config,
    device=device
)
 
#---------------------------Explicitly handling of a wandb sweep configuration--------------------- 
# ✅ Explicitly handle whether a sweep is running or not
if wandb.run:
    config["NUM_EPOCH"] = wandb.config.NUM_EPOCH
    config["LEARNING_RATE"] = wandb.config.LEARNING_RATE
    config["BATCH_SIZE"] = wandb.config.BATCH_SIZE
    config["DATA_SUBSET_SIZE"] = wandb.config.DATA_SUBSET_SIZE

# ✅ Debugging: Print config before training
print(f"🔹 Final Config Used: {config}")

#-----------------------Loading .npy whole dataset-----------------------------------
convert_to_npy(preprocessed_data_path, raw_data_path)


#-----------------------Older way of training the model WITH NO WANDB -------------------
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt  # Added for visualization

def plot_results(y_true, y_pred):
    # 1️⃣ Scatter Plot: Predicted vs Actual
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='b')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')  # Diagonal line
    plt.xlabel('Actual Core Loss')
    plt.ylabel('Predicted Core Loss')
    plt.title('Predicted vs Actual Core Loss')

    # 2️⃣ Error Distribution Plot
    plt.subplot(1, 2, 2)
    errors = y_pred - y_true
    plt.hist(errors, bins=20, edgecolor='black')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')

    plt.tight_layout()
    plt.show(block=True)  # Keep the plot window open until manually closed

def train_model(model, train_loader, valid_loader, test_loader, config, device):
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])
    num_epochs = config["NUM_EPOCH"]

    # Training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_train_loss = 0
        y_train_true, y_train_pred = [], []

        for scalograms, core_loss in train_loader:
            scalograms, core_loss = scalograms.to(device), core_loss.to(device)

            optimizer.zero_grad()           # Zero the gradients
            outputs = model(scalograms)     # Forward pass
            loss = criterion(outputs, core_loss)  # Compute loss
            loss.backward()                 # Backpropagation
            optimizer.step()                # Update weights

            epoch_train_loss += loss.item()
            y_train_true.append(core_loss.detach().cpu())
            y_train_pred.append(outputs.detach().cpu())

        # Validation
        model.eval()
        epoch_valid_loss = 0
        y_valid_true, y_valid_pred = [], []

        with torch.inference_mode():
            for scalograms, core_loss in valid_loader:
                scalograms, core_loss = scalograms.to(device), core_loss.to(device)
                outputs = model(scalograms)
                loss = criterion(outputs, core_loss)
                epoch_valid_loss += loss.item()

                y_valid_true.append(core_loss.cpu())
                y_valid_pred.append(outputs.cpu())

        # Metrics Calculation
        y_train_true = torch.cat(y_train_true).numpy()
        y_train_pred = torch.cat(y_train_pred).numpy()
        y_valid_true = torch.cat(y_valid_true).numpy()
        y_valid_pred = torch.cat(y_valid_pred).numpy()

        train_r2 = r2_score(y_train_true, y_train_pred)
        valid_r2 = r2_score(y_valid_true, y_valid_pred)
        train_mae = mean_absolute_error(y_train_true, y_train_pred)
        valid_mae = mean_absolute_error(y_valid_true, y_valid_pred)

        # Logging
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_valid_loss = epoch_valid_loss / len(valid_loader)

        print(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.6f} | Validation Loss: {avg_valid_loss:.6f} | "
              f"Train R²: {train_r2:.4f} | Validation R²: {valid_r2:.4f} | "
              f"Train MAE: {train_mae:.6f} | Validation MAE: {valid_mae:.6f}")

    # Testing
    model.eval()
    y_true, y_pred = [], []
    with torch.inference_mode():
        for scalograms, core_loss in test_loader:
            scalograms, core_loss = scalograms.to(device), core_loss.to(device)
            predictions = model(scalograms)

            y_true.append(core_loss)
            y_pred.append(predictions)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_pred = torch.cat(y_pred, dim=0).cpu().numpy()

    # Test Loss and Metrics
    test_loss = criterion(torch.tensor(y_pred), torch.tensor(y_true)).item()
    test_r2 = r2_score(y_true, y_pred)
    test_mae = mean_absolute_error(y_true, y_pred)

    print(f"\nFinal Test Loss: {test_loss:.6f}")
    print(f"Final Test R² Score: {test_r2:.4f}")
    print(f"Final Test MAE: {test_mae:.6f}")

    # Visualization of Predictions
    plot_results(y_true, y_pred)

    return model


#-------------------------------Older way of training the model WITH WANDB(BETTER LOGGING) -------------------
""" import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt  # Added for visualization
import wandb  # Added for experiment tracking

def plot_results(y_true, y_pred):
    # 1️⃣ Scatter Plot: Predicted vs Actual
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='b')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')  # Diagonal line
    plt.xlabel('Actual Core Loss')
    plt.ylabel('Predicted Core Loss')
    plt.title('Predicted vs Actual Core Loss')

    # 2️⃣ Error Distribution Plot
    plt.subplot(1, 2, 2)
    errors = y_pred - y_true
    plt.hist(errors, bins=20, edgecolor='black')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')

    plt.tight_layout()
    plt.show(block=True)  # Keep the plot window open until manually closed

def train_model(model, train_loader, valid_loader, test_loader, config, device):
    # Initialize wandb
    wandb.init(
        project="wavelet-coreloss",  # Change this to your project name
        config=config  # Logs all hyperparameters
    )

    print(f"wandB run active: {wandb.run is not None}")

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])
    num_epochs = config["NUM_EPOCH"]

    # Training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_train_loss = 0
        y_train_true, y_train_pred = [], []

        for scalograms, core_loss in train_loader:
            scalograms, core_loss = scalograms.to(device), core_loss.to(device)

            optimizer.zero_grad()           # Zero the gradients
            outputs = model(scalograms)     # Forward pass
            loss = criterion(outputs, core_loss)  # Compute loss
            loss.backward()                 # Backpropagation
            optimizer.step()                # Update weights

            epoch_train_loss += loss.item()
            y_train_true.append(core_loss.detach().cpu())
            y_train_pred.append(outputs.detach().cpu())

        # Validation
        model.eval()
        epoch_valid_loss = 0
        y_valid_true, y_valid_pred = [], []

        with torch.inference_mode():
            for scalograms, core_loss in valid_loader:
                scalograms, core_loss = scalograms.to(device), core_loss.to(device)
                outputs = model(scalograms)
                loss = criterion(outputs, core_loss)
                epoch_valid_loss += loss.item()

                y_valid_true.append(core_loss.cpu())
                y_valid_pred.append(outputs.cpu())

        # Metrics Calculation
        y_train_true = torch.cat(y_train_true).numpy()
        y_train_pred = torch.cat(y_train_pred).numpy()
        y_valid_true = torch.cat(y_valid_true).numpy()
        y_valid_pred = torch.cat(y_valid_pred).numpy()

        train_r2 = r2_score(y_train_true, y_train_pred)
        valid_r2 = r2_score(y_valid_true, y_valid_pred)
        train_mae = mean_absolute_error(y_train_true, y_train_pred)
        valid_mae = mean_absolute_error(y_valid_true, y_valid_pred)

        # Test Evaluation After Each Epoch
        model.eval()
        y_true, y_pred = [], []
        with torch.inference_mode():
            for scalograms, core_loss in test_loader:
                scalograms, core_loss = scalograms.to(device), core_loss.to(device)
                predictions = model(scalograms)
                y_true.append(core_loss)
                y_pred.append(predictions)

        y_true = torch.cat(y_true, dim=0).cpu().numpy()
        y_pred = torch.cat(y_pred, dim=0).cpu().numpy()

        test_loss = criterion(torch.tensor(y_pred), torch.tensor(y_true)).item()
        test_r2 = r2_score(y_true, y_pred)
        test_mae = mean_absolute_error(y_true, y_pred)

        # Combined Logging
        wandb.log({
            "epoch": epoch,
            "train_loss": epoch_train_loss / len(train_loader),
            "valid_loss": epoch_valid_loss / len(valid_loader),
            "train_r2": train_r2,
            "valid_r2": valid_r2,
            "train_mae": train_mae,
            "valid_mae": valid_mae,
            "test_loss": test_loss,
            "test_r2": test_r2,
            "test_mae": test_mae
        }, step=epoch)

        print(f"Epoch {epoch:02d} | Train Loss: {epoch_train_loss:.6f} | Validation Loss: {epoch_valid_loss:.6f} | "
              f"Train R²: {train_r2:.4f} | Validation R²: {valid_r2:.4f} | "
              f"Train MAE: {train_mae:.6f} | Validation MAE: {valid_mae:.6f}")

    # Visualization of Predictions
    plot_results(y_true, y_pred)

    # Save model to wandb
    wandb.watch(model, log="all")
    torch.save(model.state_dict(), "trained_model.pth")
    wandb.save("trained_model.pth")

    return model """
    
    
    
#--------------------Older way of training the model WITH WANDB (EVEN NEWER VERSION BEST LOGGING I THINK) ---------------    

""" # import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.metrics import r2_score, mean_absolute_error
# import matplotlib.pyplot as plt  # Added for visualization
# import wandb  # Added for experiment tracking

# def plot_results(y_true, y_pred):
#     # 1️⃣ Scatter Plot: Predicted vs Actual
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='b')
#     plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')  # Diagonal line
#     plt.xlabel('Actual Core Loss')
#     plt.ylabel('Predicted Core Loss')
#     plt.title('Predicted vs Actual Core Loss')

#     # 2️⃣ Error Distribution Plot
#     plt.subplot(1, 2, 2)
#     errors = y_pred - y_true
#     plt.hist(errors, bins=20, edgecolor='black')
#     plt.xlabel('Prediction Error')
#     plt.ylabel('Frequency')
#     plt.title('Error Distribution')

#     plt.tight_layout()
#     plt.show(block=False)
#     plt.pause(10)
#     plt.close()  # Keep the plot window open until manually closed

# def train_model(model, train_loader, valid_loader, test_loader, config, device):
#     # Initialize wandb
#     wandb.init(
#         project="wavelet-coreloss",
#         name=f"Subset_size{wandb.config.DATA_SUBSET_SIZE}_Epoch{wandb.config.NUM_EPOCH}_LR{wandb.config.LEARNING_RATE}_Batch_size{wandb.config.BATCH_SIZE}",
#         config=config
#     )

#     # Overwrite config with wandb sweep values
#     config["NUM_EPOCH"] = wandb.config.NUM_EPOCH
#     config["LEARNING_RATE"] = wandb.config.LEARNING_RATE
#     config["BATCH_SIZE"] = wandb.config.BATCH_SIZE
#     config["DATA_SUBSET_SIZE"] = wandb.config.DATA_SUBSET_SIZE

#     print(f"wandB run active: {wandb.run is not None}")

#     # Loss function and optimizer
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])
#     num_epochs = config["NUM_EPOCH"]

#     # Training loop
#     for epoch in range(1, num_epochs + 1):
#         model.train()
#         epoch_train_loss = 0
#         y_train_true, y_train_pred = [], []

#         for scalograms, core_loss in train_loader:
#             scalograms, core_loss = scalograms.to(device), core_loss.to(device)

#             optimizer.zero_grad()           # Zero the gradients
#             outputs = model(scalograms)     # Forward pass
#             loss = criterion(outputs, core_loss)  # Compute loss
#             loss.backward()                 # Backpropagation
#             optimizer.step()                # Update weights

#             epoch_train_loss += loss.item()
#             y_train_true.append(core_loss.detach().cpu())
#             y_train_pred.append(outputs.detach().cpu())

#         # Validation
#         model.eval()
#         epoch_valid_loss = 0
#         y_valid_true, y_valid_pred = [], []

#         with torch.inference_mode():
#             for scalograms, core_loss in valid_loader:
#                 scalograms, core_loss = scalograms.to(device), core_loss.to(device)
#                 outputs = model(scalograms)
#                 loss = criterion(outputs, core_loss)
#                 epoch_valid_loss += loss.item()

#                 y_valid_true.append(core_loss.cpu())
#                 y_valid_pred.append(outputs.cpu())

#         # Metrics Calculation
#         y_train_true = torch.cat(y_train_true).numpy()
#         y_train_pred = torch.cat(y_train_pred).numpy()
#         y_valid_true = torch.cat(y_valid_true).numpy()
#         y_valid_pred = torch.cat(y_valid_pred).numpy()

#         train_r2 = r2_score(y_train_true, y_train_pred)
#         valid_r2 = r2_score(y_valid_true, y_valid_pred)
#         train_mae = mean_absolute_error(y_train_true, y_train_pred)
#         valid_mae = mean_absolute_error(y_valid_true, y_valid_pred)

#         # Test Evaluation After Each Epoch
#         model.eval()
#         y_true, y_pred = [], []
#         with torch.inference_mode():
#             for scalograms, core_loss in test_loader:
#                 scalograms, core_loss = scalograms.to(device), core_loss.to(device)
#                 predictions = model(scalograms)
#                 y_true.append(core_loss)
#                 y_pred.append(predictions)

#         y_true = torch.cat(y_true, dim=0).cpu().numpy()
#         y_pred = torch.cat(y_pred, dim=0).cpu().numpy()

#         test_loss = criterion(torch.tensor(y_pred), torch.tensor(y_true)).item()
#         test_r2 = r2_score(y_true, y_pred)
#         test_mae = mean_absolute_error(y_true, y_pred)

#         # Combined Logging
#         wandb.log({
#             "epoch": epoch,
#             "train_loss": epoch_train_loss / len(train_loader),
#             "valid_loss": epoch_valid_loss / len(valid_loader),
#             "train_r2": train_r2,
#             "valid_r2": valid_r2,
#             "train_mae": train_mae,
#             "valid_mae": valid_mae,
#             "test_loss": test_loss,
#             "test_r2": test_r2,
#             "test_mae": test_mae
#         }, step=epoch)

#         print(f"Epoch {epoch:02d} | Train Loss: {epoch_train_loss:.6f} | Validation Loss: {epoch_valid_loss:.6f} | "
#               f"Train R²: {train_r2:.4f} | Validation R²: {valid_r2:.4f} | "
#               f"Train MAE: {train_mae:.6f} | Validation MAE: {valid_mae:.6f}")

#     # Visualization of Predictions
#     plot_results(y_true, y_pred)

#     # Save model to wandb
#     wandb.watch(model, log="all")
#     torch.save(model.state_dict(), "trained_model.pth")
#     wandb.save("trained_model.pth")

#     return model
 """
    
#----------------------Final way of training with WANDB USING A SWEEP(WAS NOT WORKING)---------------------
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt  # Added for visualization
import wandb  # Added for experiment tracking

def plot_results(y_true, y_pred):
    # 1️⃣ Scatter Plot: Predicted vs Actual
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='b')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')  # Diagonal line
    plt.xlabel('Actual Core Loss')
    plt.ylabel('Predicted Core Loss')
    plt.title('Predicted vs Actual Core Loss')

    # 2️⃣ Error Distribution Plot
    plt.subplot(1, 2, 2)
    errors = y_pred - y_true
    plt.hist(errors, bins=20, edgecolor='black')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(10)
    plt.close()  # Keep the plot window open until manually closed

def train_model(model, train_loader, valid_loader, test_loader, config, device):
    # Ensure wandb is properly initialized
    if wandb.run is not None:
        wandb.finish()

    wandb.init(
        entity="ggeorgid",
        project="wavelet-coreloss",
        name=f"Subset_size{config.get('DATA_SUBSET_SIZE', 'default')}_Epoch{config.get('NUM_EPOCH', 'default')}_LR{config.get('LEARNING_RATE', 'default')}_Batch_size{config.get('BATCH_SIZE', 'default')}",
        config=config
    )

    print(f"wandB run active: {wandb.run is not None}")

    # ✅ Update config values explicitly (Handles both normal run & sweeps)
    config["NUM_EPOCH"] = wandb.config.NUM_EPOCH if wandb.run else config["NUM_EPOCH"]
    config["LEARNING_RATE"] = wandb.config.LEARNING_RATE if wandb.run else config["LEARNING_RATE"]
    config["BATCH_SIZE"] = wandb.config.BATCH_SIZE if wandb.run else config["BATCH_SIZE"]
    config["DATA_SUBSET_SIZE"] = wandb.config.DATA_SUBSET_SIZE if wandb.run else config["DATA_SUBSET_SIZE"]

    print(f"🔹 Updated Config: {config}")  # Debugging: Print to verify correct values

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])
    num_epochs = config["NUM_EPOCH"]

    # Training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_train_loss = 0
        y_train_true, y_train_pred = [], []

        for scalograms, core_loss in train_loader:
            scalograms, core_loss = scalograms.to(device), core_loss.to(device)

            optimizer.zero_grad()           # Zero the gradients
            outputs = model(scalograms)     # Forward pass
            loss = criterion(outputs, core_loss)  # Compute loss
            loss.backward()                 # Backpropagation
            optimizer.step()                # Update weights

            epoch_train_loss += loss.item()
            y_train_true.append(core_loss.detach().cpu())
            y_train_pred.append(outputs.detach().cpu())

        # Validation
        model.eval()
        epoch_valid_loss = 0
        y_valid_true, y_valid_pred = [], []

        with torch.inference_mode():
            for scalograms, core_loss in valid_loader:
                scalograms, core_loss = scalograms.to(device), core_loss.to(device)
                outputs = model(scalograms)
                loss = criterion(outputs, core_loss)
                epoch_valid_loss += loss.item()

                y_valid_true.append(core_loss.cpu())
                y_valid_pred.append(outputs.cpu())

        # Metrics Calculation
        y_train_true = torch.cat(y_train_true).numpy()
        y_train_pred = torch.cat(y_train_pred).numpy()
        y_valid_true = torch.cat(y_valid_true).numpy()
        y_valid_pred = torch.cat(y_valid_pred).numpy()

        train_r2 = r2_score(y_train_true, y_train_pred)
        valid_r2 = r2_score(y_valid_true, y_valid_pred)
        train_mae = mean_absolute_error(y_train_true, y_train_pred)
        valid_mae = mean_absolute_error(y_valid_true, y_valid_pred)

        # Test Evaluation After Each Epoch
        model.eval()
        y_true, y_pred = [], []
        with torch.inference_mode():
            for scalograms, core_loss in test_loader:
                scalograms, core_loss = scalograms.to(device), core_loss.to(device)
                predictions = model(scalograms)
                y_true.append(core_loss)
                y_pred.append(predictions)

        y_true = torch.cat(y_true, dim=0).cpu().numpy()
        y_pred = torch.cat(y_pred, dim=0).cpu().numpy()

        test_loss = criterion(torch.tensor(y_pred), torch.tensor(y_true)).item()
        test_r2 = r2_score(y_true, y_pred)
        test_mae = mean_absolute_error(y_true, y_pred)

        # Combined Logging
        wandb.log({
            "epoch": epoch,
            "train_loss": epoch_train_loss / len(train_loader),
            "valid_loss": epoch_valid_loss / len(valid_loader),
            "train_r2": train_r2,
            "valid_r2": valid_r2,
            "train_mae": train_mae,
            "valid_mae": valid_mae,
            "test_loss": test_loss,
            "test_r2": test_r2,
            "test_mae": test_mae
        }, step=epoch)

        print(f"Epoch {epoch:02d} | Train Loss: {epoch_train_loss:.6f} | Validation Loss: {epoch_valid_loss:.6f} | "
              f"Train R²: {train_r2:.4f} | Validation R²: {valid_r2:.4f} | "
              f"Train MAE: {train_mae:.6f} | Validation MAE: {valid_mae:.6f}")

    # Visualization of Predictions
    plot_results(y_true, y_pred)

    # Save model to wandb
    wandb.watch(model, log="all")
    torch.save(model.state_dict(), "trained_model.pth")
    wandb.save("trained_model.pth")

    return model


#Older version of calculating scalograms and checking sizes that doesn't take into account 
#that the DATA_SUBSET_SIZE might be increased in value compared to what's stored as chached(e.g. 256 -> 512)

# -------------------------------- Handling Scalograms & Core Loss --------------------------------
SAMPLE_RATE = 2e-6 # 2 µs period
scalograms_path = preprocessed_data_path / "scalograms.npy"
core_loss_path = preprocessed_data_path / "core_loss.npy"

# Apply dataset subset BEFORE computing scalograms
if data_subset_size is not None:
    tensor_dataset = torch.utils.data.Subset(tensor_dataset, range(data_subset_size))

# Check if cached scalograms exist for the current subset size
if use_cached_scalograms and scalograms_path.exists() and core_loss_path.exists():
    print(f"[INFO] Loading cached scalograms for subset size {data_subset_size}.")
    wavelet_dataset = WaveletCoreLossDataset(
        scalograms_path=scalograms_path,
        core_loss_path=core_loss_path
    )
else:
    print(f"[INFO] Calculating new scalograms for subset size {data_subset_size}.")
    wavelet_dataset = WaveletCoreLossDataset(
        V_I_dataset=tensor_dataset,
        sample_rate=SAMPLE_RATE
    )
    # Save new scalograms and core loss for this subset size
    np.save(scalograms_path, wavelet_dataset.scalograms.numpy())
    np.save(core_loss_path, wavelet_dataset.core_loss_values.numpy())
    print(f"[INFO] Saved scalograms for subset size {data_subset_size}.")
    
    
#--------------------------------Simplified version of main.py------------------------------------------
  
# -------------------------------- Handling Scalograms & Core Loss --------------------------------
    SAMPLE_RATE = 2e-6
    wavelet_dataset = WaveletCoreLossDataset(
        V_I_dataset=tensor_dataset,
        sample_rate=SAMPLE_RATE
    )
    
    # -------------------------------- Train/Validation/Test Split & Loading Dataloaders --------------------------------
    train_dataset, valid_dataset, test_dataset = split_dataset(wavelet_dataset)
    train_loader, valid_loader, test_loader = create_dataloaders(
        train_dataset, valid_dataset, test_dataset, batch_size=batch_size, use_gpu=use_gpu
    )
    
    model = WaveletModel().to(device)
    
    print("🔹 Model Summary:\n")
    summary(model, input_size=(batch_size, 1, 24, 24))
    
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        config={
            "NUM_EPOCH": num_epoch,
            "DATA_SUBSET_SIZE": data_subset_size,
            "LEARNING_RATE": lr,
            "BATCH_SIZE": batch_size
        },
        device=device
    )
    
#------------------------------Handling Trapezoid Time Series with 8000 samples instead of 8192--------------------------
#The following two functions will need to be changed
def process_csv(csv_path: str, target_length=8192) -> np.ndarray | None:
    """
    Reads a CSV file and converts it into a NumPy array, either trimming or padding it to `target_length`.

    Args:
        csv_path (str): Path to the CSV file.
        target_length (int): The number of time steps to standardize all samples (default: 8192).
    
    Returns:
        np.ndarray or None: Processed data if valid, otherwise None.
    """
    if "info.csv" in csv_path:
        return None  # Skip metadata files

    try:
        df = pd.read_csv(csv_path, header=None, dtype=np.float64, skiprows=1).values
        sample_length = df.shape[0]

        if sample_length not in (8000, 8192):
            print(f"[WARNING] {csv_path} has an unexpected sample length {sample_length}, skipping.")
            return None

        # Handle trimming (if too long) or padding (if too short)
        if sample_length > target_length:
            trimmed_data = df[:target_length]
        elif sample_length < target_length:
            padding = np.zeros((target_length - sample_length, df.shape[1]))
            trimmed_data = np.vstack((df, padding))
        else:
            trimmed_data = df  # No modification needed

        return trimmed_data.reshape(-1, target_length, trimmed_data.shape[1])

    except (ValueError, IndexError, pd.errors.ParserError) as e:
        print(f"[ERROR] Failed to process {csv_path}: {e}")
        return None

def convert_to_npy(preprocessed_data_path: Path, raw_data_path: Path, use_short=False) -> None:
    """
    Converts all CSV files in raw_data_path to a single NumPy dataset.

    Args:
        preprocessed_data_path (Path): Directory to save processed dataset.
        raw_data_path (Path): Directory containing raw CSV files.
        use_short (bool): If True, trims all time series to 8000. If False, keeps original sizes (8192).
    """
    csv_paths = list(raw_data_path.glob("dataset/*.csv"))

    target_length = 8000 if use_short else 8192  # Determine target length dynamically
    all_data = [process_csv(csv, target_length) for csv in csv_paths if process_csv(csv, target_length) is not None]

    if not all_data:
        raise RuntimeError("[ERROR] No valid CSV files found for dataset conversion.")

    final_data = np.concatenate(all_data, axis=0)
    np.save(preprocessed_data_path / f"dataset_{target_length}.npy", final_data)
    print(f"[INFO] Successfully saved dataset_{target_length}.npy with shape {final_data.shape}")

    
#--------------------Older way of creating the .npy dataset before adding the trapezoid time series-------------------
# The following two functions have been updated to handle the new dataset structure
# and to ensure that the sample length is consistent across all samples. 
# Total dataset size went from [5762,8192,2] to [9936,8000,2].
def process_csv(csv_path: str) -> np.ndarray | None:
    """
    Reads a CSV file and converts it into a NumPy array.
    
    Args:
        csv_path (str): Path to the CSV file.
    
    Returns:
        np.ndarray or None: Processed data if valid, otherwise None.
    """
    if "info.csv" in str(csv_path):  # ✅ Convert Path to string
        return None
    
    try:
        df = pd.read_csv(csv_path, header=None, dtype={0: str, 1: np.float64})
        sample_period, sample_length = df.iloc[0]
        sample_length = int(sample_length)
        
        if sample_length != 8192:
            print(f"[WARNING] {csv_path} has unexpected sample length {sample_length}, skipping.")
            return None

        data = df.iloc[1:].values.astype(np.float64)
        num_samples = data.shape[0] // sample_length

        return data.reshape(num_samples, sample_length, -1)
    
    except (ValueError, IndexError) as e:
        print(f"[ERROR] Failed to process {csv_path}: {e}")
        return None

def convert_to_npy(preprocessed_data_path: Path, raw_data_path: Path) -> None:
    """
    Converts all CSV files in raw_data_path to a single NumPy dataset.
    
    Args:
        preprocessed_data_path (Path): Directory to save processed dataset.
        raw_data_path (Path): Directory containing raw CSV files.
    """
    csv_paths = list(raw_data_path.glob("dataset/*.csv"))
    all_data = [process_csv(csv) for csv in csv_paths if process_csv(csv) is not None]

    if not all_data:
        raise RuntimeError("[ERROR] No valid CSV files found for dataset conversion.")

    final_data = np.concatenate(all_data, axis=0)
    np.save(preprocessed_data_path / "dataset.npy", final_data)
    print("[INFO] Successfully saved dataset.npy")


#-----------------------------------Older way of calculating scalograms-----------------------------------
# Newer Version with logarithmic scales (11/6/2025)

# def calculate_scalograms(
#     dataset: np.ndarray,
#     sampling_period: float,
#     wave_name: str = 'morl',
#     sample_length: int = None,
#     total_scale: int = 40,
#     fmax: float = 10e3,
#     fmin: float = 1e3,  # I tried 1 here and the time to compute the scalograms exploded to 40 hours
#     image_size: int = 24
# ) -> torch.Tensor:
#     """
#     Computes scalograms from a dataset using Continuous Wavelet Transform (CWT) with real Morlet wavelet.

#     Args:
#         dataset (np.ndarray): Input dataset of shape (num_samples, time_steps, 2).
#         sampling_period (float): Time between samples in seconds (e.g., 2e-6 s).
#         wave_name (str): Wavelet type (default 'morl' for real Morlet).
#         sample_length (int, optional): Number of time steps to use. Defaults to full length.
#         total_scale (int): Number of scales for CWT (default 40).
#         fmax (float): Maximum frequency of interest in Hz (default 10e3).
#         fmin (float): Minimum frequency of interest in Hz (default 1.0, adjustable).
#         image_size (int): Size to resize scalograms to (default 24 for 24x24).

#     Returns:
#         torch.Tensor: Scalogram tensor of shape (num_samples, 1, image_size, image_size).
#     """
#     # Validate dataset shape
#     if dataset.ndim != 3 or dataset.shape[2] != 2:
#         raise ValueError("Dataset must have shape (num_samples, time_steps, 2).")

#     # Determine sample length
#     if sample_length is None:
#         sample_length = dataset.shape[1]
#     dataset = dataset[:, :sample_length, :]

#     # Compute logarithmic frequency range
#     freqs = np.geomspace(fmax, fmin, num=total_scale)
#     print(f"[DEBUG] Frequency Range: {freqs.min()} Hz to {freqs.max()} Hz")

#     # Convert frequencies to scales using Morlet central frequency (0.8125 Hz)
#     scales = pywt.frequency2scale(wave_name, freqs * sampling_period)
#     print(f"[DEBUG] Computed Scales: {scales.min()} to {scales.max()}")

#     # Initialize output array
#     num_samples = dataset.shape[0]
#     scalograms = np.empty((num_samples, image_size, image_size), dtype=np.float32)

#     # Compute scalograms
#     for index in tqdm(range(num_samples), desc="Generating Scalograms"):
#         voltage_signal = dataset[index, :, 0]
#         cwtmatr, _ = pywt.cwt(voltage_signal, scales, wave_name, sampling_period)
#         scalogram = np.abs(cwtmatr)  # Absolute value for scalogram
#         scalograms[index] = resize(scalogram, (image_size, image_size), anti_aliasing=True)

#         if index == 0:
#             print(f"[DEBUG] First Scalogram Shape: {cwtmatr.shape} | Resized to: {scalograms[index].shape}")

#     # Convert to tensor
#     scalogram_tensor = torch.tensor(scalograms, dtype=torch.float32).unsqueeze(1)
#     return scalogram_tensor


# def calculate_scalograms(
#     dataset: np.ndarray,
#     sampling_period: float,
#     wave_name: str = 'morl',
#     sample_length: int = None,
#     total_scale: int = 40,
#     fmax: float = 10e3,
#     fmin: float = 1e3,
#     image_size: int = 24,
#     save_path: str = "data/processed/scalograms_memmap.dat"
# ) -> np.memmap:
#     """
#     Computes scalograms using CWT and stores them in a memory-mapped array.

#     Args:
#         dataset (np.ndarray): Shape (num_samples, time_steps, 2).
#         sampling_period (float): Time between samples in seconds.
#         wave_name (str): Wavelet type (default 'morl').
#         sample_length (int, optional): Number of time steps.
#         total_scale (int): Number of scales (default 41).
#         fmax (float): Max frequency (default 10e3 Hz).
#         fmin (float): Min frequency (default 1e3 Hz).
#         image_size (int): Output size (default 24).
#         save_path (str): Path for memory-mapped file.

#     Returns:
#         np.memmap: Memory-mapped array of shape (num_samples, image_size, image_size).
#     """
#     if dataset.ndim != 3 or dataset.shape[2] != 2:
#         raise ValueError("Dataset must have shape (num_samples, time_steps, 2).")

#     print(f"[DEBUG] Total_Scales: {total_scale} and Image_size: {image_size} that reached here.")

#     sample_length = sample_length or dataset.shape[1]
#     dataset = dataset[:, :sample_length, :]
#     num_samples = dataset.shape[0]

#     freqs = np.geomspace(fmax, fmin, num=total_scale)
#     scales = pywt.frequency2scale(wave_name, freqs * sampling_period)

#     # Create memory-mapped array
#     scalograms = np.memmap(save_path, dtype='float32', mode='w+', shape=(num_samples, image_size, image_size))

#     for index in tqdm(range(num_samples), desc="Generating Scalograms"):
#         voltage_signal = dataset[index, :, 0]
#         cwtmatr, _ = pywt.cwt(voltage_signal, scales, wave_name, sampling_period)
#         scalogram = np.abs(cwtmatr)
#         scalograms[index] = resize(scalogram, (image_size, image_size), anti_aliasing=True)

#         if index == 0:
#             print(f"[DEBUG] First Scalogram Shape: {cwtmatr.shape} | Resized to: {scalograms[index].shape}")

#     scalograms.flush()  # Write to disk
#     return scalograms



# def calculate_scalograms(
#     dataset: np.ndarray,
#     sampling_period: float,
#     wave_name: str = 'morl',  # or 'cgau8' if preferred
#     sample_length: int = None,
#     total_scale: int = 41,
#     fmax: float = 10e3,
#     image_size: int = 24,
#     save_path: str = "data/processed/scalograms_memmap.dat"
# ) -> np.memmap:
#     if dataset.ndim != 3 or dataset.shape[2] != 2:
#         raise ValueError("Dataset must have shape (num_samples, time_steps, 2).")

#     sample_length = sample_length or dataset.shape[1]
#     dataset = dataset[:, :sample_length, :]
#     num_samples = dataset.shape[0]

#     # Check if the wavelet is continuous or discrete
#     if wave_name in pywt.wavelist(kind='continuous'):
#         # Use ContinuousWavelet for continuous wavelets
#         wavelet = pywt.ContinuousWavelet(wave_name)
#         # Define central frequency manually for continuous wavelets
#         if wave_name == 'morl':
#             fc = 0.8125  # Default central frequency for Morlet wavelet in PyWavelets
#         else:
#             raise NotImplementedError(f"Central frequency for {wave_name} not implemented yet.")
#     else:
#         # Use Wavelet for discrete wavelets
#         wavelet = pywt.Wavelet(wave_name)
#         fc = wavelet.center_frequency

#     # Compute scales
#     cparam = (1 / sampling_period) / fmax * fc * total_scale
#     scales = cparam / np.arange(total_scale, 1, -1)  # Small to large scales

#     scalograms = np.memmap(save_path, dtype='float32', mode='w+', shape=(num_samples, image_size, image_size))

#     for index in tqdm(range(num_samples), desc="Generating Scalograms"):
#         voltage_signal = dataset[index, :, 0]
#         cwtmatr, _ = pywt.cwt(voltage_signal, scales, wavelet, sampling_period)
#         scalograms[index] = resize(np.abs(cwtmatr), (image_size, image_size), anti_aliasing=True)

#     scalograms.flush()
#     return scalograms

