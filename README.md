# **MagNET Thesis**

📡 **A Thesis Project Based on the MagNET Database**

This repository contains my work on applying machine learning techniques to analyze and model data from the **MagNET** database. The project focuses on **predicting magnetic core loss using wavelet-based transformations and deep learning models**, with an additional **ablation study** to evaluate the impact of scalograms.

🚧 **This repository is still a work in progress.** Stay tuned for updates!

---

## **🔹 How to Run the Project**

### **1️⃣ Single Run (Scalogram Model)**
To train and evaluate the **scalogram-based model** with specific hyperparameters, run:

```bash
python3 wavelet_main.py --num_epoch 50 --learning_rate 0.001 --batch_size 128
```

### **2️⃣ Single Run (Ablation Model)**
To train and evaluate the **ablation model** (using raw voltage data, no scalograms) with specific hyperparameters, run:

```bash
python3 ablation_main.py --num_epoch 50 --learning_rate 0.001 --batch_size 128
```
### **3️⃣ Single Run (Transfer Learning Model)**

To train and evaluate the transfer learning model (based on ResNet18 with 224x224 scalograms) with specific hyperparameters, run:
```bash
python3 transfer_main.py --num_epoch 50 --learning_rate 0.001 --batch_size 128
```
    Use --use_pretrained to load pre-trained ResNet18 weights (default: True).
    Use --freeze_layers to freeze all layers except the final fully connected layer (default: False).

### **4️⃣ Multiple Runs (Hyperparameter Sweep)**

To perform multiple runs with different hyperparameter values:

• Scalogram Model:
```bash
python3 manual_sweep.py
```
• Ablation Model:
```bash
python3 ablation_manual_sweep.py
```
• Transfer Learning Model:
```bash
python3 transfer_manual_sweep.py
```
These scripts execute experiments with preset hyperparameter combinations and log the results for evaluation.

### **5️⃣ Optimization Run**

To run hyperparameter optimization using Optuna:

• Scalogram Model:
```bash
python3 wavelet_main.py --optimize
```
• Ablation Model:
```bash
python3 ablation_main.py --optimize
```
• Transfer Learning Model:
```bash
python3 transfer_main.py --optimize
```

---

## **🔹 Repository Structure**

📁 **/data/** → Processed and raw datasets (not included in the repo) and dataset-related scripts  
📁 **/models/** → Contains model architecture definitions  
📁 **/checking_scripts/** → Directory containing helper scripts for dataset preprocessing, visualization, and consistency checks  
📁 **/utils/** → Model training scripts  

### **Main Files**
- `wavelet_main.py` → **Main script for training the scalogram-based model**
- `ablation_main.py` → **Main script for training the ablation model using raw voltage data (no scalograms)**
- `transfer_main.py` → **Main script for training the transfer learning model (ResNet18-based)**
- `manual_sweep.py` → **Performs multiple runs for hyperparameter tuning of the scalogram model**
- `ablation_manual_sweep.py` → **Performs multiple runs for hyperparameter tuning of the ablation model**
- `transfer_manual_sweep.py` → **Performs multiple runs for hyperparameter tuning for the transfer learning model**
### **Data Processing Files**
- `data/wavelet_coreloss_dataset.py` → **Processes and transforms data into scalograms for the scalogram model**
- `data/voltage_coreloss_dataset.py` → **Processes raw voltage data for the ablation model**
- `data/wavelet_coreloss_dataset_224.py` → **Processes and transforms data into 224x224 scalograms for the transfer learning model**
- `data/data_processing.py` → **Splits dataset into train/validation/test, checks core loss distribution, and creates DataLoaders (used by scalogram model)**
- `data/dataloaders_ablation.py` → **Splits dataset and creates DataLoaders specifically for the ablation model**
### **Model Architecture Files**
- `models/wavelet_model.py` → **Neural network architecture for the scalogram-based model**
- `models/ablation_model.py` → **Neural network architecture for the ablation model (no scalograms)**
- `models/transfer_model.py` → **Neural network architecture for the transfer learning model (ResNet18-based)**
### **Training Scripts**
- `utils/train_utils.py` → **Training and evaluation utilities shared across models**
### **Checking Scripts**
- `checking_scripts/coreloss_verification.py` → **Validates core loss calculations**
- `checking_scripts/model_consistency_check.py` → **Ensures model weight consistency across runs**
- `checking_scripts/inspect_dataset.py` → **Explores and analyzes the dataset**
- `checking_scripts/sanity_check.py` → **Checks model predictions on the test set, compares actual vs. predicted values, and provides statistics (e.g., R², MAE, MSE)**
- `checking_scripts/trapcheck.py` → **Compares preprocessed and original datasets, checks waveform consistency**
- `checking_scripts/check_scalograms.py` → **Visualizes and verifies original and resized scalograms for the transfer learning model**
---

## **🔹 Requirements**

Ensure you have all dependencies installed before running the scripts:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, install the key dependencies manually:

```bash
pip install numpy torch torchvision matplotlib pandas tqdm optuna
```

---

## **🔹 Current Progress**

✔️ Implemented wavelet transformations for input data  
✔️ Created dataset classes for efficient loading (scalogram and raw voltage)  
✔️ Built deep learning models for core loss prediction (scalogram and ablation)  
✔️ Added Weights & Biases (wandb) logging for experiment tracking  
✔️ Implemented ablation study to evaluate the impact of scalograms  
✔️ Integrated transfer learning with a pre-trained ResNet18 model and 224x224 scalograms
🚧 **Next Steps:** Optimize all models and finalize hyperparameter tuning  

---

## **🔹 Contributing**

👨‍💻 Contributions, suggestions, and feedback are welcome! Feel free to open an issue or submit a pull request.

---

## **🔹 License**

📜 This project is for academic and research purposes. Please cite appropriately if you use any part of it in your work.

---

## **🔹 Acknowledgments**

Special thanks to the **MagNET database** for providing the data used in this thesis.

