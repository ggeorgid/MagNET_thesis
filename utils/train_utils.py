import torch
import os
import torch.nn as nn
import torch.optim as optim
import wandb
import numpy as np  # Added for variance check
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt  # Added for visualization
import optuna

# Ensure figures directory exists
FIGURE_DIR = "figures"
os.makedirs(FIGURE_DIR, exist_ok=True)

def plot_results(y_true, y_pred, relative_errors, are, absolute_errors, mae, title="Core Loss Prediction Results"):
    """Plots and saves a figure with four subplots: Measured vs Predicted, Relative Error vs Measured, Absolute Error vs Measured, and Error Distribution."""
    plt.figure(figsize=(20, 5))  # Increased width for four subplots
    
    # Subplot 1: Measured vs Predicted Core Loss
    plt.subplot(1, 4, 1)  # Changed from (1, 3, 1) to (1, 4, 1)
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='b', label='Predictions')
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
    plt.xlabel('Measured Core Loss [W]')
    plt.ylabel('Predicted Core Loss [W]')
    plt.title('Measured vs Predicted')
    plt.legend()
    plt.grid(True)
    
    # Subplot 2: Relative Error vs Measured Core Loss
    plt.subplot(1, 4, 2)  # Changed from (1, 3, 2) to (1, 4, 2)
    plt.scatter(y_true, relative_errors, alpha=0.6, edgecolors='b', label='Relative Errors')
    plt.axhline(y=np.mean(relative_errors), color='r', linestyle='--', label=f"Avg = {np.mean(relative_errors):.2f}%")
    plt.xlabel('Measured Core Loss [W]')
    plt.ylabel('Relative Error [%]')
    plt.title('Relative Error Distribution')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 50)
    
    # Subplot 3: Absolute Error vs Measured Core Loss
    plt.subplot(1, 4, 3)  # Remains (1, 4, 3)
    plt.scatter(y_true, absolute_errors, alpha=0.6, edgecolors='b', label='Absolute Errors')
    plt.axhline(y=mae, color='r', linestyle='--', label=f"MAE = {mae:.6f} W")
    plt.xlabel('Measured Core Loss [W]')
    plt.ylabel('Absolute Error [W]')
    plt.title('Absolute Error Distribution')
    plt.legend()
    plt.grid(True)
    
    # Subplot 4: Error Distribution (Histogram)
    plt.subplot(1, 4, 4)  # Remains (1, 4, 4)
    errors = y_pred - y_true
    plt.hist(errors, bins=20, edgecolor='black')
    plt.xlabel('Prediction Error [W]')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(FIGURE_DIR, "core_loss_predictions.png"))
    plt.close()


def train_model(model, train_loader, valid_loader, test_loader, config, device, trial=None):
    """
    Trains the model and logs metrics to wandb.
    """
    use_wandb_sweep = wandb.run is not None  # Detect if inside a wandb sweep
    
    #--------Optuna Pruning Step--------
    #wandb.watch(model, log="all", log_freq=10)  # Log model gradients and parameters
    best_val_loss = float("inf")
    early_stopping_patience = 5
    patience_counter = 0
    #-------------------------------------
           
    print("\n✅ Training with the following hyperparameters:")
    print(config)
    
    # 🔹 Step 1: Check if core_loss has variation BEFORE training
    core_loss_values = []
    for _, core_loss in train_loader:
        core_loss_values.append(core_loss.numpy())  # Convert tensor to NumPy array
    
    core_loss_values = np.concatenate(core_loss_values, axis=0)  # Flatten
    print(f"🔹 Before training - Core loss mean: {core_loss_values.mean():.6f}, variance: {core_loss_values.var():.6f}")
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=config["LEARNING_RATE"])#only pass the trainable parameters
    num_epochs = config["NUM_EPOCH"]
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        # Training Phase
        model.train()
        epoch_train_loss = 0
        y_train_true, y_train_pred = [], []
        
        for inputs, core_loss in train_loader:
            inputs, core_loss = inputs.to(device), core_loss.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, core_loss)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item() * inputs.size(0) / len(train_loader.dataset)
            y_train_true.append(core_loss.detach().cpu())
            y_train_pred.append(outputs.detach().cpu())
        
        # Validation Phase
        model.eval()
        epoch_valid_loss = 0
        y_valid_true, y_valid_pred = [], []
        
        with torch.inference_mode():
            for inputs, core_loss in valid_loader:
                inputs, core_loss = inputs.to(device), core_loss.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, core_loss)
                epoch_valid_loss += loss.item() * inputs.size(0) / len(valid_loader.dataset)
                
                y_valid_true.append(core_loss.cpu())
                y_valid_pred.append(outputs.cpu())
        
        # Test Phase
        epoch_test_loss = 0
        y_test_true, y_test_pred = [], []
        
        with torch.inference_mode():
            for inputs, core_loss in test_loader:
                inputs, core_loss = inputs.to(device), core_loss.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, core_loss)
                epoch_test_loss += loss.item() * inputs.size(0) / len(test_loader.dataset)
                
                y_test_true.append(core_loss.cpu())
                y_test_pred.append(outputs.cpu())
        
        # Compute Metrics for Train, Validation, and Test Sets
        y_train_true = torch.cat(y_train_true).numpy().flatten()
        y_train_pred = torch.cat(y_train_pred).numpy().flatten()
        y_valid_true = torch.cat(y_valid_true).numpy().flatten()
        y_valid_pred = torch.cat(y_valid_pred).numpy().flatten()
        y_test_true = torch.cat(y_test_true).numpy().flatten()
        y_test_pred = torch.cat(y_test_pred).numpy().flatten()

        # Compute R² using sklearn
        train_r2 = r2_score(y_train_true, y_train_pred)
        valid_r2 = r2_score(y_valid_true, y_valid_pred)
        test_r2 = r2_score(y_test_true, y_test_pred)

        train_mae = mean_absolute_error(y_train_true, y_train_pred)
        valid_mae = mean_absolute_error(y_valid_true, y_valid_pred)
        test_mae = mean_absolute_error(y_test_true, y_test_pred)

        train_mse = mean_squared_error(y_train_true, y_train_pred)
        valid_mse = mean_squared_error(y_valid_true, y_valid_pred)
        test_mse = mean_squared_error(y_test_true, y_test_pred)

        # Compute Relative Errors and ARE
        epsilon = 0.001  # Avoid division by zero        
        # Train set ARE
        relative_errors_train = np.abs(y_train_pred - y_train_true) / (y_train_true + epsilon) * 100
        train_are = np.mean(relative_errors_train)
        # Validation set ARE
        relative_errors_valid = np.abs(y_valid_pred - y_valid_true) / (y_valid_true + epsilon) * 100
        valid_are = np.mean(relative_errors_valid)
        # Test set ARE
        relative_errors = np.abs(y_test_pred - y_test_true) / (y_test_true + epsilon) * 100
        test_are = np.mean(relative_errors)
        
        # Compute Absolute Errors (NEW)
        absolute_errors_train = np.abs(y_train_pred - y_train_true)
        absolute_errors_valid = np.abs(y_valid_pred - y_valid_true)
        absolute_errors_test = np.abs(y_test_pred - y_test_true)
        
        # Log metrics to wandb
        wandb.log({
            "_step": epoch,
            "epoch": epoch,
            "Loss/Train": epoch_train_loss,
            "Loss/Validation": epoch_valid_loss,
            "Loss/Test": epoch_test_loss,
            "MSE/Train": train_mse,
            "MSE/Validation": valid_mse,
            "MSE/Test": test_mse,
            "MAE/Train": train_mae,
            "MAE/Validation": valid_mae,
            "MAE/Test": test_mae,
            "R2/Train": train_r2,
            "R2/Validation": valid_r2,
            "R2/Test": test_r2,
            "ARE/Train": train_are,
            "ARE/Validation": valid_are,
            "ARE/Test": test_are,
        })
        
        print(f"Epoch {epoch:02d} | Train Loss: {epoch_train_loss:.6f} | Validation Loss: {epoch_valid_loss:.6f} | "
               f"Test Loss: {epoch_test_loss:.6f} | Train R²: {train_r2:.4f} | Validation R²: {valid_r2:.4f} | "
               f"Test R²: {test_r2:.4f} | Train MAE: {train_mae:.6f} | Validation MAE: {valid_mae:.6f} | "
               f"Test MAE: {test_mae:.6f} | Train MSE: {train_mse:.6f} | Validation MSE: {valid_mse:.6f} | "
               f"Test MSE: {test_mse:.6f} | Train ARE: {train_are:.2f}% | Validation ARE: {valid_are:.2f}% | "
               f"Test ARE: {test_are:.2f}%")                

        # Report to Optuna for pruning (only if trial is provided)
        if trial:
            trial.report(epoch_valid_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Early stopping logic (only if trial is provided)
        if trial:
            if epoch_valid_loss < best_val_loss:
                best_val_loss = epoch_valid_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"[INFO] Early stopping triggered at epoch {epoch}.")
                break
        
    torch.save(model.state_dict(), "trained_model.pth")
    
    # Log final validation loss correctly when NOT using Optuna
    print(f"Final Validation Loss: {epoch_valid_loss}")
    if trial is None:
        final_epoch_loss = epoch_valid_loss
        
    # Save test set true and predicted values for sanity check
    np.save("y_test_true.npy", y_test_true)
    np.save("y_test_pred.npy", y_test_pred)
    print("\n✅ Saved y_test_true.npy and y_test_pred.npy for sanity check.")
    
    # Plot results (updated call)
    plot_results(
        y_test_true,
        y_test_pred,
        relative_errors,
        test_are,
        absolute_errors_test,  # New argument
        test_mae,              # New argument
        title="Core Loss Prediction Results on Test Set"
    )
    
    # Return model and best validation loss    
    if trial:
        return model, best_val_loss
    else:
        return model, final_epoch_loss







# inputs.size(0) is the first dimension of the inputs tensor, which is the batch size for each batch.

#epoch_train_loss += loss.item() * inputs.size(0) / len(train_loader.dataset)
#epoch_valid_loss += loss.item() * inputs.size(0) / len(valid_loader.dataset)

# Why is this change necessary? (from ChatGTP) <- Xrhsto help explain this please 

#     Previous issue: Loss values were being summed over all batches. Since batch sizes can vary 
#           (especially in the last batch), this caused inconsistency in reported loss values.
#     Fix: Dividing by the dataset size ensures that the loss reflects an average per sample 
#           rather than an accumulation across batches.

# How does it help?

#     Loss values are now comparable between training, validation, and test phases.
#     Ensures correct tracking when batch sizes vary.

#-------------------------------------------------------------------------------------------------------------
#test_loss = criterion(torch.tensor(y_test_pred), torch.tensor(y_test_true)).item()
# Why was this needed? (from ChatGTP) <- Xrhsto help explain this please

#     Previous issue: The test loss was incorrectly scaled by the dataset size, inflating the loss value.
#     Fix: Compute the average test loss directly from the model’s predictions without multiplying by the dataset size.

# How does it help?

#     Ensures test loss is correctly calculated as an average loss per sample.
#     Maintains consistency across training, validation, and test metrics.

#----------------------------More Context---------------------------------------------------------------------
# Issue: <- Xrhsto help explain this please

#     The training and validation loss values are being summed instead of averaged, which makes them dependent on batch size.
#     The test loss is being multiplied by len(y_test_true), which inflates the loss artificially.

#-------------------Considering the whole test block, if it functions correctly-------------------------------
# Why this approach is correct:

#     Maintains consistency with train/validation loss calculation:
#         In both training and validation phases, we accumulate the loss weighted by the batch size and then normalize by the total dataset size.
#         Applying the same logic to the test phase ensures consistent loss scaling across all three sets.

#     Fixes the issue with directly computing loss on tensors:
#         criterion(torch.tensor(y_test_pred), torch.tensor(y_test_true)).item() computes the loss in one step but does not handle batched computation properly.
#         If the test set is large, computing loss on the entire dataset at once may cause memory overflow issues.

#     Prevents over/under-estimation of loss values:
#         If we don't normalize by len(test_loader.dataset), the test loss can increase linearly with dataset size, making it not directly comparable to train/valid losses.

# Why the other approaches are incorrect:

#     test_loss = criterion(torch.tensor(y_test_pred), torch.tensor(y_test_true)).item()
#         Issues:
#             This approach directly computes the loss after collecting all predictions.
#             It does not account for batch-wise loss accumulation.
#             It could introduce issues if y_test_pred and y_test_true are very large (out-of-memory risk).
#             Does not follow the same normalization process as train/valid loss.

#     Keeping test_loss = criterion(predictions, core_loss).item() inside the loop without accumulation
#         Issues:
#             Each batch loss is not accumulated over the entire test dataset.
#             This means we would only retain the last batch's loss, which is not representative of the whole dataset.

# Point 2: Training Loss Calculation

# This concerns the formula inside train_utils.py during the training loop:

# epoch_train_loss += loss.item() * inputs.size(0) / len(train_loader.dataset)

# Let’s break it down:
# How Loss is Computed in PyTorch

#     Each batch computes its loss:
#     loss = criterion(outputs, core_loss)
#         This gives an average loss per batch.
#     You accumulate the total epoch loss:
#         You multiply the loss by inputs.size(0), which is the batch size.
#         Then divide by the total dataset size len(train_loader.dataset).
#         This ensures that the final epoch_train_loss is an average over all batches.

# Potential Issue (Minor)

#     If the last batch is smaller (e.g., dataset size is not perfectly divisible by batch size), this formula may slightly underestimate the epoch loss.
#     But this is generally fine, and PyTorch’s approach already averages correctly per batch.

# ✅ Conclusion:
# Your loss calculation is valid.
# However, if we want to be precise, we could calculate a weighted batch contribution instead of averaging over all batches directly.





