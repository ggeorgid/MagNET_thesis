# Script to be used only after training the model and predicting the test set
# This script is used to check the predictions of the model on the test set
# It compares the actual values of the test set to the predicted values
# It also provides some basic statistics about the test set
# It also plots the actual values vs the predicted values

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os

# Ensure figures directory exists
FIGURE_DIR = "figures"
os.makedirs(FIGURE_DIR, exist_ok=True)

# Load test set predictions
y_test_true = np.load("y_test_true.npy")
y_test_pred = np.load("y_test_pred.npy")

print(f"y_test_true shape: {y_test_true.shape}")
print(f"y_test_pred shape: {y_test_pred.shape}")

# Basic Statistics
print("\n✅ Sanity Check on Test Set:")
print(f"   - y_test_true Mean: {y_test_true.mean():.6f}, Std: {y_test_true.std():.6f}")
print(f"   - y_test_pred Mean: {y_test_pred.mean():.6f}, Std: {y_test_pred.std():.6f}")

# Check if any values are all zeros (which could indicate an issue)
if np.all(y_test_true == 0) or np.all(y_test_pred == 0):
    print("⚠️ Warning: Some predictions or labels are all zeros!")

# Compute R², MAE, and MSE for additional validation
test_r2 = r2_score(y_test_true, y_test_pred)
test_mae = mean_absolute_error(y_test_true, y_test_pred)
test_mse = mean_squared_error(y_test_true, y_test_pred)

print("\n📊 Test Set Metrics:")
print(f"   - R² Score: {test_r2:.6f}")
print(f"   - Mean Absolute Error (MAE): {test_mae:.6f}")
print(f"   - Mean Squared Error (MSE): {test_mse:.6f}")

# Show first 10 predictions for a manual check
print("\n🔹 First 10 test set predictions vs actual values:")
for i in range(min(10, len(y_test_true))):
    print(f"   - Actual: {y_test_true[i]:.6e}, Predicted: {y_test_pred[i]:.6e}")

# Scatter Plot: Actual vs Predicted
plt.figure(figsize=(6, 6))
plt.scatter(y_test_true, y_test_pred, alpha=0.6, edgecolors='b')
plt.plot([min(y_test_true), max(y_test_true)], [min(y_test_true), max(y_test_true)], 'r--')  # Diagonal line
plt.xlabel('Actual Core Loss')
plt.ylabel('Predicted Core Loss')
plt.title('Sanity Check: Predicted vs Actual Core Loss')

# Save the plot
scatter_plot_path = os.path.join(FIGURE_DIR, "test_predictions_sanity_check.png")
plt.savefig(scatter_plot_path)
print(f"\n✅ Saved scatter plot: {scatter_plot_path}")

plt.show()

