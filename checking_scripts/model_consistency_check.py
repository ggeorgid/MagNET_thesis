#Script that helps compare two models and their datasets for consistency.
#The script provides two functions:
#1. check_model_weights_consistency(model_path1, model_path2): Checks if two saved PyTorch model state dictionaries have identical weights.
#2. check_dataset_consistency(core_loss_path1, core_loss_path2, scalogram_path1, scalogram_path2): Checks if two dataset runs have identical core loss values and scalograms.

import torch
import numpy as np

def check_model_weights_consistency(model_path1, model_path2):
    """
    Checks if two saved PyTorch model state dictionaries have identical weights.
    
    Args:
        model_path1 (str): Path to the first saved model state dictionary (.pth file).
        model_path2 (str): Path to the second saved model state dictionary (.pth file).
    
    Returns:
        bool: True if all corresponding model weights in both state dictionaries are identical, False otherwise.
    
    Prints:
        "Model Weights Consistent: True/False" indicating if the models are identical.
    """
    # Load saved model state dictionaries
    model1 = torch.load(model_path1)
    model2 = torch.load(model_path2)
    
    # Check if all keys (layer names) exist in both models and their corresponding weights are identical
    consistent = all(torch.equal(model1[k], model2[k]) for k in model1.keys())
    
    print("Model Weights Consistent:", consistent)
    return consistent

def check_dataset_consistency(core_loss_path1, core_loss_path2, scalogram_path1, scalogram_path2):
    """
    Checks if two dataset runs have identical core loss values and scalograms.
    
    Args:
        core_loss_path1 (str): Path to the first NumPy file containing core loss values.
        core_loss_path2 (str): Path to the second NumPy file containing core loss values.
        scalogram_path1 (str): Path to the first NumPy file containing scalograms.
        scalogram_path2 (str): Path to the second NumPy file containing scalograms.
    
    Returns:
        tuple: (bool, bool) indicating whether core loss values and scalograms are consistent, respectively.
    
    Prints:
        - "Core Loss Consistency: True/False" indicating if the core loss values are identical.
        - "Scalogram Consistency: True/False" indicating if the scalograms are identical.
    """
    # Load datasets from .npy files
    core_loss_1 = np.load(core_loss_path1)
    core_loss_2 = np.load(core_loss_path2)
    scalograms_1 = np.load(scalogram_path1)
    scalograms_2 = np.load(scalogram_path2)
    
    # Check if core loss values are numerically close (with floating-point tolerance)
    core_loss_consistent = np.allclose(core_loss_1, core_loss_2)
    
    # Check if scalograms are numerically close (with floating-point tolerance)
    scalogram_consistent = np.allclose(scalograms_1, scalograms_2)
    
    print("Core Loss Consistency:", core_loss_consistent)
    print("Scalogram Consistency:", scalogram_consistent)
    
    return core_loss_consistent, scalogram_consistent

# Example usage when running the script directly
if __name__ == "__main__":
    # Example file paths (modify as needed)
    model_file_1 = "trained_model_run1.pth" # Path to the first saved model state dictionary
    model_file_2 = "trained_model_run2.pth" # Path to the second saved model state dictionary
    core_loss_file_1 = "run1_core_loss.npy" # Path to the first NumPy file containing core loss values
    core_loss_file_2 = "run2_core_loss.npy" # Path to the second NumPy file containing core loss values
    scalogram_file_1 = "run1_scalograms.npy" # Path to the first NumPy file containing scalograms
    scalogram_file_2 = "run2_scalograms.npy" # Path to the second NumPy file containing scalograms
    
    # Check if the models have identical weights
    check_model_weights_consistency(model_file_1, model_file_2)
    
    # Check if the datasets (core loss and scalograms) are identical
    check_dataset_consistency(core_loss_file_1, core_loss_file_2, scalogram_file_1, scalogram_file_2)
