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
