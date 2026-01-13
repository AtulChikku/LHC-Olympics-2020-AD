# Enhanced Resonant Anomaly Detection on the LHC Olympics 2020 Dataset

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

In this project, I present an **optimized implementation of the ANODE (Anomaly Detection with Normalizing Flows) framework**, introducing specific refinements in numerical stability and regularization to detect new physics as localized over-densities in the di-jet invariant mass spectrum ($m_{jj}$).

## 🚀 The Challenge & My Approach

The standard ANODE method identifies "bumps" by comparing density estimates $p(x|m)$ in signal and control regions. However, I found that high-dimensional feature spaces ($D=134$) often lead to **density collapse** and **numerical instability** in vanilla implementations.

### ✨ Key Innovations in this Implementation

While utilizing the ANODE core architecture, I introduced several critical improvements to enhance performance and stability:

* **Robust Density Ensembling**: I implemented a **Log-Sum-Exp averaged ensemble** over the final 10 training epochs. This effectively blurs transient density spikes caused by overfitting while reinforcing the persistent physical resonance.
* **Averaged Log-Likelihood Ratio (ALR)**: I use the log-ratio $R(\mathbf{x}|m) = \log \text{avg}[p_{data}] - \log \text{avg}[p_{bg}]$ for anomaly scoring. This ensures numerical stability and prevents the "SIC explosions" common in raw probability ratios.
* **Optimized Regularization Balance**: I identified that Spectral Normalization was too restrictive for anomaly detection. I replaced it with a tuned **Weight Decay ($10^{-6}$)** strategy to maintain the "sharpness" required to detect narrow resonances without triggering model collapse.
* **Constituent-Based Global Representations**: I transitioned from sparse 2D "jet images" to **kT-ordered, centered sequences** to preserve the hard-parton substructure of the jet.



## 💡 Architectural Evolution: My Journey to Density Manifolds

This project was an iterative journey through four distinct phases of representation learning as I worked to overcome the unique challenges of particle physics data.

#### Phase 1: The Sparsity Crisis (Image-Based VAEs/ViTs)
* **My Strategy**: Initially, I treated jets as 2D energy depositions ($\eta, \phi$ grids) using **U-Nets** or **Vision Transformers** for reconstruction.
* **The Failure**: I discovered that jet images were over 95% sparse. My models suffered from "posterior collapse," where they learned to "cheat" by simply predicting blank black images to achieve a low MSE loss without learning any actual substructure.

#### Phase 2: Solving Sparsity with Submanifold Sparse Convolutions (SSCN)
* **My Strategy**: To solve the sparsity issue, I implemented **Submanifold Sparse Convolutions** to focus computation only on active pixels.
* **The Breakthrough**: This worked significantly better; I was able to reconstruct detailed jet images with high fidelity.
* **The Limitation**: However, I found that the reconstruction error alone was not a strong discriminator. My background-trained model became "too good" at reconstruction, meaning the distinction between background and signal was not evident from the error alone.

#### Phase 3: Hybrid Sequence Modeling (SSCN Features)
* **My Strategy**: Instead of raw features, I moved to sequence modeling using the latent features learned from my sparse convolutions.
* **The Result**: This provided slightly better results than raw features alone, as the features were more physically representative.
* **The Dead End**: I realized I was still limited by the **Reconstruction Paradigm**. Reconstruction loss is designed to find outliers, but resonant signals are often **over-densities** rather than structural outliers.

#### Phase 4: Pivot to Density Estimation (Enhanced ANODE)
* **My Strategy**: Finally, I abandoned reconstruction in favor of **Explicit Density Estimation** using Masked Autoregressive Flows (MAF).
* **The Success**: By estimating $p(x)$ directly, I was able to identify the 3.5 TeV resonance as an over-density. I stabilized this by introducing **Log-Sum-Exp averaging**, which brought my SIC scores into a scientifically valid range.



## 📈 Results & Analysis

| Method | Metric | Result |
| :--- | :--- | :--- |
| **Baseline (VAE)** | Max AUC | 0.80 |
| **My Enhanced ANODE** | **Max AUC** | **0.83** |
| **My Enhanced ANODE** | **Max SIC** | **5.20** |

## 💻 Installation & Usage

1. **Clone**: `git clone https://github.com/AtulChikku/LHC-Olympics-2020-AD.git`
2. **Install**: `pip install -r requirements.txt`
3. **Run**: Detailed training and evaluation scripts are located in the `/scripts` directory.

## 📜 References
* ANODE Paper: [arXiv:2001.04990](https://arxiv.org/abs/2001.04990)
* LHC Olympics 2020: [arXiv:2101.08320](https://arxiv.org/abs/2101.08320)
