# **MagNET Thesis**
📡 **A Thesis Project Based on the MagNET Database**

This repository contains my work on applying machine learning techniques to analyze and model data from the **MagNET** database. The project focuses on **predicting magnetic core loss using wavelet-based transformations and deep learning models**.

🚧 **This repository is still a work in progress.** Stay tuned for updates!

---

## **🔹 How to Run the Project**
### **1️⃣ Single Run**
To train and evaluate the model with a single set of hyperparameters, run:

```bash
python3 wavelet_main.py
```

### **2️⃣ Multiple Runs (Hyperparameter Sweep)**
To perform multiple runs with different hyperparameter values, execute:

```bash
python3 manual_sweep.py
```

This script runs experiments with different hyperparameters and logs results for evaluation.

### **3️⃣ Optimization Run**
To run the program with hyperparameter optimization, execute:

```bash
python3 wavelet_main.py --optimize
```

---

## **🔹 Repository Structure**

📁 **/data/** → Processed and raw datasets (not included in the repo) and dataset-related scripts  
📁 **/checking_scripts/** → Helper scripts for dataset preprocessing, visualization, and consistency checks  
📁 **/utils/** → Model training scripts  

### **Main Files**
- `wavelet_main.py` → **Main script for training the model**
- `manual_sweep.py` → **Performs multiple runs for hyperparameter tuning**
- `wavelet_coreloss_dataset.py` → **Dataset processing & transformations**
- `wavelet_model.py` → **Neural network architecture**
- `train_utils.py` → **Training and evaluation utilities**
- `coreloss_verification.py` → **Script for validating core loss calculations**
- `model_consistency_check.py` → **Ensures model weight consistency across runs**
- `inspect_dataset.py` → **Explores and analyzes the dataset**
- `data_processing.py` → **Splits dataset into train/validation/test, checks core loss distribution, and creates DataLoaders**
- `checking_scripts.py` → **Contains scripts for checking various results**

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
✔️ Created dataset class for efficient loading  
✔️ Built initial deep learning model for core loss prediction  
✔️ Added Weights & Biases (wandb) logging for experiment tracking  
🚧 Next Steps: Optimize the model and finalize hyperparameter tuning  

---

## **🔹 Contributing**
👨‍💻 Contributions, suggestions, and feedback are welcome! Feel free to open an issue or submit a pull request.

---

## **🔹 License**
📜 This project is for academic and research purposes. Please cite appropriately if you use any part of it in your work.

---

## **🔹 Acknowledgments**
Special thanks to the MagNET database for providing the data used in this thesis.

