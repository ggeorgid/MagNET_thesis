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
    plt.figure(figsize=(20, 5))  
    
    # Subplot 1: Measured vs Predicted Core Loss
    plt.subplot(1, 4, 1)  
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
    plt.subplot(1, 4, 2)  
    plt.scatter(y_true, relative_errors, alpha=0.6, edgecolors='b', label='Relative Errors')
    plt.axhline(y=np.mean(relative_errors), color='r', linestyle='--', label=f"Avg = {np.mean(relative_errors):.2f}%")
    plt.xlabel('Measured Core Loss [W]')
    plt.ylabel('Relative Error [%]')
    plt.title('Relative Error Distribution')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 50)
    
    # Subplot 3: Absolute Error vs Measured Core Loss
    plt.subplot(1, 4, 3)  
    plt.scatter(y_true, absolute_errors, alpha=0.6, edgecolors='b', label='Absolute Errors')
    plt.axhline(y=mae, color='r', linestyle='--', label=f"MAE = {mae:.6f} W")
    plt.xlabel('Measured Core Loss [W]')
    plt.ylabel('Absolute Error [W]')
    plt.title('Absolute Error Distribution')
    plt.legend()
    plt.grid(True)
    
    # Subplot 4: Error Distribution (Histogram)
    plt.subplot(1, 4, 4)  
    errors = y_pred - y_true
    plt.hist(errors, bins=20, edgecolor='black')
    plt.xlabel('Prediction Error [W]')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(FIGURE_DIR, "core_loss_predictions.png"))
    plt.close()


def train_model(model, train_loader, valid_loader, test_loader, config, device, save_path='best_model.pth', trial=None):
    """
    Trains the model and logs metrics to wandb.
    """
    use_wandb_sweep = wandb.run is not None  # Detect if inside a wandb sweep
    
    # Added print statement to track USE_PRETRAINED at the start
    print(f"🔹 [DEBUG] USE_PRETRAINED at start of train_model: {config.get('USE_PRETRAINED', 'Not found')}")

    # 🔹 Step 0: Initialize variables to track and graph the model with the best validation loss
    # Initialize best validation loss 
    best_val_loss = float('inf')    

    #--------Optuna Pruning Step--------
    #wandb.watch(model, log="all", log_freq=10)  # Log model gradients and parameters    
    early_stopping_patience = 10
    patience_counter = 0
    #-------------------------------------
           
    print("\n✅ Training with the following hyperparameters:")
    print(config)
    
    # 🔹 Step 1: Check if core_loss has variation BEFORE training <- Optional Step. Can be commented out 
    core_loss_values = []
    for _, core_loss in train_loader:
        core_loss_values.append(core_loss.numpy())  # Convert tensor to NumPy array
    
    core_loss_values = np.concatenate(core_loss_values, axis=0)  # Flatten
    print(f"🔹 Before training - Core loss mean: {core_loss_values.mean():.6f}, variance: {core_loss_values.var():.6f}")
    
    # 🔹 Step 2: Define loss function and optimizer
    # Define loss function
    criterion = nn.MSELoss()

    # Added print statement to track USE_PRETRAINED before optimizer setup
    print(f"🔹 [DEBUG] USE_PRETRAINED before optimizer setup: {config.get('USE_PRETRAINED', 'Not found')}")
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["LEARNING_RATE"]
    )
    # # Set up optimizer with conditional logic
    # if config.get("USE_PRETRAINED", False):  # Check if USE_PRETRAINED is True
    #     # For transfer learning model, use differential learning rates
    #     layer4_params = [param for name, param in model.named_parameters() if name.startswith("layer4.") and param.requires_grad]
    #     fc_params = [param for name, param in model.named_parameters() if name.startswith("fc.") and param.requires_grad]
    #     optimizer = optim.Adam([
    #         {'params': layer4_params, 'lr': config["LEARNING_RATE"] /0.5},  # Smaller LR for layer4
    #         {'params': fc_params, 'lr': config["LEARNING_RATE"] *0.5 }            # Full LR for fc
    #     ])
    #     # Added print statement to confirm differential learning rates
    #     print("🔹 [DEBUG] Using differential learning rates for transfer learning")
    # else:
    #     # For other models (wavelet, ablation), use standard optimizer
    #     optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=config["LEARNING_RATE"])
    #     # Added print statement to confirm standard optimizer
    #     print("🔹 [DEBUG] Using standard optimizer for other models")
    num_epochs = config["NUM_EPOCH"]
    
    # 🔹 Step 3: Training loop
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
        
        # Check for best model
        if epoch_valid_loss < best_val_loss:
            best_val_loss = epoch_valid_loss
            try:
                torch.save(model.state_dict(), save_path)
                print(f"✅ Epoch {epoch}: New best model saved with val_loss: {best_val_loss:.6f}")
            except IOError as e:
                print(f"❌ Error saving model: {e}")            
            patience_counter = 0
        else:
            patience_counter += 1

        
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
        
        # Compute Metrics for Train, Validation, and Test Sets for all epochs(models)
        y_train_true = torch.cat(y_train_true).numpy().flatten()
        y_train_pred = torch.cat(y_train_pred).numpy().flatten()
        y_valid_true = torch.cat(y_valid_true).numpy().flatten()
        y_valid_pred = torch.cat(y_valid_pred).numpy().flatten()
        y_test_true = torch.cat(y_test_true).numpy().flatten()
        y_test_pred = torch.cat(y_test_pred).numpy().flatten()

        # Compute R² using sklearn for all epochs(models)
        train_r2 = r2_score(y_train_true, y_train_pred)
        valid_r2 = r2_score(y_valid_true, y_valid_pred)
        test_r2 = r2_score(y_test_true, y_test_pred)

        # Compute MAE for all epochs(models)
        train_mae = mean_absolute_error(y_train_true, y_train_pred)
        valid_mae = mean_absolute_error(y_valid_true, y_valid_pred)
        test_mae = mean_absolute_error(y_test_true, y_test_pred)

        # Compute MSE for all epochs(models)
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
        if wandb.run is not None:
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
                "ARE/Test": test_are                
            })
        
        print(f"Epoch {epoch:02d} | Train Loss: {epoch_train_loss:.6f} | Validation Loss: {epoch_valid_loss:.6f} | "
               f"Test Loss: {epoch_test_loss:.6f} | Train R²: {train_r2:.4f} | Validation R²: {valid_r2:.4f} | "
               f"Test R²: {test_r2:.4f} | Train MAE: {train_mae:.6f} | Validation MAE: {valid_mae:.6f} | "
               f"Test MAE: {test_mae:.6f} | Train MSE: {train_mse:.6f} | Validation MSE: {valid_mse:.6f} | "
               f"Test MSE: {test_mse:.6f} | Train ARE: {train_are:.2f}% | Validation ARE: {valid_are:.2f}% | "
               f"Test ARE: {test_are:.2f}% ")                


        # Report to Optuna for pruning (only if trial is provided)
        if trial:
            trial.report(epoch_valid_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Early stopping logic 
        if patience_counter >= early_stopping_patience:
            print(f"[INFO] Early stopping triggered at epoch {epoch}.")
            break

    print(f"Model saved to: {os.path.abspath(save_path)}")
    # 🔹 Step 4: Load the best model to prepare for graphs  
    # Load the best model state
    model.load_state_dict(torch.load(save_path, weights_only=True)) # weights_only=True parameter to avoid warning about malicious intent
    print("✅ Best model loaded for evaluation and plotting.")
    
    # Evaluate on test set with the best model
    model.eval()
    y_test_true, y_test_pred_best = [], []
    with torch.inference_mode():
        for inputs, core_loss in test_loader:
            inputs, core_loss = inputs.to(device), core_loss.to(device)
            outputs = model(inputs)
            y_test_true.append(core_loss.cpu())
            y_test_pred_best.append(outputs.cpu())
    y_test_true = torch.cat(y_test_true).numpy().flatten()
    y_test_pred_best = torch.cat(y_test_pred_best).numpy().flatten()

    # Compute metrics for plotting
    epsilon = 0.001
    relative_errors_best = np.abs(y_test_pred_best - y_test_true) / (y_test_true + epsilon) * 100
    are_best = np.mean(relative_errors_best)
    absolute_errors_best = np.abs(y_test_pred_best - y_test_true) #an array of absolute errors used for the graph
    mae_best = mean_absolute_error(y_test_true, y_test_pred_best) #the mean absolute error metric value
        
    # Save test set true and predicted values for sanity check
    np.save("y_test_true.npy", y_test_true)
    np.save("y_test_pred.npy", y_test_pred)
    print("\n✅ Saved y_test_true.npy and y_test_pred.npy for sanity check.")
    
    # Plot results for the best performing epoch-model <-Updated needs checking
    plot_results(
        y_test_true,
        y_test_pred_best,
        relative_errors_best,
        are_best,
        absolute_errors_best,  
        mae_best,              
        title="Core Loss Prediction Results on Test Set"
    )
    
    # Return the best model and best validation loss
    return model, best_val_loss







