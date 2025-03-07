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

def plot_results(y_true, y_pred, title="Predicted vs Actual Core Loss"):
    """Plots and saves predicted vs actual core loss and error distribution."""
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='b')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')  # Diagonal line
    plt.xlabel('Actual Core Loss')
    plt.ylabel('Predicted Core Loss')
    plt.title(title)

    # Error Distribution Plot
    plt.subplot(1, 2, 2)
    errors = y_pred - y_true
    plt.hist(errors, bins=20, edgecolor='black')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "model_predictions.png"))
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
    
    # # Initialize wandb run if not already initialized
    # if wandb.run is None:
    #     wandb.init(
    #         project="No_trapezoids_sweep_March3rd",
    #         name=f"epochs_{config['NUM_EPOCH']}_lr_{config['LEARNING_RATE']}_bs_{config['BATCH_SIZE']}",
    #         config=config
    #     )
    
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
    optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])
    num_epochs = config["NUM_EPOCH"]
    
    for epoch in range(1, num_epochs + 1):
        # Training Phase
        model.train()
        epoch_train_loss = 0
        y_train_true, y_train_pred = [], []
        
        for scalograms, core_loss in train_loader:
            scalograms, core_loss = scalograms.to(device), core_loss.to(device)
            
            optimizer.zero_grad()
            outputs = model(scalograms)
            loss = criterion(outputs, core_loss)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item() * scalograms.size(0) / len(train_loader.dataset)
            y_train_true.append(core_loss.detach().cpu())
            y_train_pred.append(outputs.detach().cpu())
        
        # Validation Phase
        model.eval()
        epoch_valid_loss = 0
        y_valid_true, y_valid_pred = [], []
        
        with torch.inference_mode():
            for scalograms, core_loss in valid_loader:
                scalograms, core_loss = scalograms.to(device), core_loss.to(device)
                outputs = model(scalograms)
                loss = criterion(outputs, core_loss)
                epoch_valid_loss += loss.item() * scalograms.size(0) / len(valid_loader.dataset)
                
                y_valid_true.append(core_loss.cpu())
                y_valid_pred.append(outputs.cpu())
        
        # Test Phase
        epoch_test_loss = 0
        y_test_true, y_test_pred = [], []
        
        with torch.inference_mode():
            for scalograms, core_loss in test_loader:
                scalograms, core_loss = scalograms.to(device), core_loss.to(device)
                outputs = model(scalograms)
                loss = criterion(outputs, core_loss)
                epoch_test_loss += loss.item() * scalograms.size(0) / len(test_loader.dataset)
                
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
        })
        
        print(f"Epoch {epoch:02d} | Train Loss: {epoch_train_loss:.6f} | Validation Loss: {epoch_valid_loss:.6f} | "
               f"Test Loss: {epoch_test_loss:.6f} | Train R²: {train_r2:.4f} | Validation R²: {valid_r2:.4f} | "
               f"Test R²: {test_r2:.4f} | Train MAE: {train_mae:.6f} | Validation MAE: {valid_mae:.6f} | "
               f"Test MAE: {test_mae:.6f} | Train MSE: {train_mse:.6f} | Validation MSE: {valid_mse:.6f} | "
               f"Test MSE: {test_mse:.6f}")

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
    
    # Plot results
    plot_results(y_test_true, y_test_pred, title="Test Set Predictions")
    
    # Return model and best validation loss    
    if trial:
        return model, best_val_loss
    else:
        return model, final_epoch_loss







# scalograms.size(0) is the first dimension of the scalograms tensor, which is the batch size for each batch.

#epoch_train_loss += loss.item() * scalograms.size(0) / len(train_loader.dataset)
#epoch_valid_loss += loss.item() * scalograms.size(0) / len(valid_loader.dataset)

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

# epoch_train_loss += loss.item() * scalograms.size(0) / len(train_loader.dataset)

# Let’s break it down:
# How Loss is Computed in PyTorch

#     Each batch computes its loss:
#     loss = criterion(outputs, core_loss)
#         This gives an average loss per batch.
#     You accumulate the total epoch loss:
#         You multiply the loss by scalograms.size(0), which is the batch size.
#         Then divide by the total dataset size len(train_loader.dataset).
#         This ensures that the final epoch_train_loss is an average over all batches.

# Potential Issue (Minor)

#     If the last batch is smaller (e.g., dataset size is not perfectly divisible by batch size), this formula may slightly underestimate the epoch loss.
#     But this is generally fine, and PyTorch’s approach already averages correctly per batch.

# ✅ Conclusion:
# Your loss calculation is valid.
# However, if we want to be precise, we could calculate a weighted batch contribution instead of averaging over all batches directly.





