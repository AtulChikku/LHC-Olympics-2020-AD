# Unsupervised Anomaly Detection on the LHC Olympics 2020 Dataset

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project explores and compares two deep learning methods for unsupervised anomaly detection in high energy physics, using the LHC Olympics 2020 R&D Dataset. The goal is to identify anomalous "signal" events (indicative of new physics) from a large "background" of known Standard Model processes.

## 🎯 Motivation & Goal

At the Large Hadron Collider (LHC), searches for new physics often involve looking for rare, exotic events (anomalies) hidden within an overwhelming amount of background data. Since the signature of new physics is unknown, we cannot use traditional supervised classification.

This project implements **unsupervised anomaly detection** by training models *only* on known background data. Anomalies are then identified as data points that the models find difficult to reconstruct.

We compare two distinct approaches:
1.  **A Feature-Based Approach:** Using Autoencoders (AEs) and Variational Autoencoders (VAEs) on high-level kinematic features.
2.  **An Image-Based Approach:** Constructing "jet images" from low-level particle data and using a U-Net-based Convolutional Autoencoder for reconstruction.

## 💾 Dataset

The project uses the **LHC Olympics 2020 R&D Dataset**. This is a simulated dataset containing a large "background" set (Standard Model di-jets) and several hidden "signal" sets for evaluation. The data is provided in two forms:

* **High-Level Features (HLF):** A simple vector of pre-calculated features for each event (e.g., jet mass, $p_T$, $\tau_{21}$).
* **Low-Level Features (LLF):** The raw constituent particles ($p_T, \eta, \phi$) for each jet, allowing for more complex feature engineering.

## 🛠️ Methodology

The core principle for both methods is to train an autoencoder on the background data to learn a compressed representation. The **reconstruction loss** (e.g., MSE) is then used as an anomaly score.

### Approach 1: Autoencoders on High-Level Features

This approach uses a simple, fully-connected neural network architecture to reconstruct the provided high-level features.

* **Models:** Standard Autoencoder (AE) and Variational Autoencoder (VAE).
* **Input:** A 1D vector of the high-level features for each event.
* **Tuning:** Hyperparameters (e.g., latent dimension, layer size, learning rate) were optimized using **Optuna** to maximize the final anomaly detection performance.

### Approach 2: Convolutional Autoencoders on Jet Images

This approach uses the richer, low-level data to perform a more complex analysis.

1.  **Feature Engineering:** The low-level particle constituents are binned into 2D histograms based on their ($\eta, \phi$) coordinates, with pixel intensity weighted by particle $p_T$. This creates a "jet image" for each event, representing the energy deposition in the calorimeter.
2.  **Model:** A **Convolutional Autoencoder (CAE) with a U-Net architecture** was used. The U-Net's skip connections are highly effective for image reconstruction tasks, preserving fine-grained spatial details, which is ideal for identifying subtle anomalies in the jet's energy distribution.
3.  **Tuning:** The model's complex hyperparameters (e.g., filter count, kernel size, dropout) were also optimized using **Optuna**.

## 📊 Results & Analysis

The models were evaluated on their ability to distinguish the hidden signal events from the background, quantified by the **Area Under the ROC Curve (AUC)**.

| Method                                | Anomaly Score         | Max AUC (Tuned) |
| :------------------------------------ | :-------------------- | :-------------- |
| **AE / VAE (High-Level Features)** | Vector MSE Loss       | **0.80** |
| **U-Net CAE (Jet Images)** | Pixel-wise MSE Loss   | **0.92** |

The results show that while the simple AE/VAE on high-level features provides a strong baseline, the convolutional model operating on jet images achieves significantly better performance. This indicates that the low-level particle data contains granular information crucial for anomaly detection, which is successfully exploited by the convolutional architecture.

![Example ROC Curves](results/figures/roc_curves.png)
*(Note: Replace with the actual path to your saved plot in the repository.)*

## 🚀 How to Use

### Prerequisites
- Python 3.8+
- PyTorch
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn
- Optuna
- (Optional: `awkward-array`, `vector` for HEP data handling)

### Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/](https://github.com/)<your-username>/<your-repo-name>.git
    cd <your-repo-name>
    ```

2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Code

1.  **Data Analysis & Exploration:**
    See the notebooks in the `/notebooks` directory for data loading, preprocessing, and results analysis.
    ```bash
    jupyter notebook notebooks/01_data_exploration.ipynb
    ```

2.  **Train Models:**
    Use the main training scripts in the `/src` directory.
    ```bash
    # Train the VAE on High-Level Features
    python src/train.py --model vae --data hlf

    # Train the U-Net on Jet Images
    python src/train.py --model unet --data llf
    ```

3.  **Hyperparameter Tuning:**
    To run an Optuna optimization study:
    ```bash
    python src/tune.py --model unet
    ```
