# 🔬 Resonant Anomaly Detection on the LHC Olympics 2020 Dataset

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

In this project, I present **two complementary approaches for anomaly detection** to discover new physics as localized over-densities in the di-jet invariant mass spectrum ($m_{jj}$): an enhanced **ANODE framework using Masked Autoregressive Flows (MAF)** and a **semi-supervised representation learning approach using Sparse Autoencoders with XGBoost**.

---

## 🚀 The Challenge & My Approach

The standard ANODE method identifies "bumps" by comparing density estimates $p(x|m)$ in signal and control regions. However, I found that high-dimensional feature spaces ($D=134$ to $D=2010$) often lead to **density collapse** and **numerical instability** in vanilla implementations.

### ✨ Approach 1: Enhanced ANODE with Masked Autoregressive Flows

While utilizing the ANODE core architecture, I introduced several critical improvements to enhance performance and stability:

* **Robust Density Ensembling**: Implemented a **Log-Sum-Exp averaged ensemble** over the final 10 training epochs. This effectively blurs transient density spikes caused by overfitting while reinforcing the persistent physical resonance.
* **Averaged Log-Likelihood Ratio (ALR)**: Using the log-ratio $R(\mathbf{x}|m) = \log \text{avg}[p_{data}] - \log \text{avg}[p_{bg}]$ for anomaly scoring ensures numerical stability and prevents instabilities common in raw probability ratios.
* **Optimized Regularization Balance**: Identified that Spectral Normalization was too restrictive for anomaly detection. Replaced it with a tuned **Weight Decay ($10^{-6}$)** strategy to maintain the "sharpness" required to detect narrow resonances without triggering model collapse.
* **Constituent-Based Global Representations**: Transitioned from sparse 2D "jet images" to **kT-ordered, centered sequences** to preserve the hard-parton substructure of the jet.

### ✨ Approach 2: Semi-Supervised Sparse Feature Learning with XGBoost

After extensive experimentation with pure density estimation, I recognized that **learned latent representations** could capture discriminative features that density models might miss. This led to a hybrid deep learning + gradient boosting pipeline:

* **3-Channel Sparse Autoencoder**: Built a sparse convolutional autoencoder operating on jet images with channels for $p_T$, Energy, and $\log(p_T)$. The model learns compressed bottleneck representations while handling the extreme sparsity (~95%) of jet images.
* **Bottleneck Feature Extraction**: Rather than using reconstruction error as the anomaly score, I extract the **128-dimensional bottleneck features** as rich, physics-informed representations of each jet.
* **XGBoost Classification**: These learned features feed into an XGBoost classifier, combining the representation power of deep learning with the robustness and interpretability of gradient boosting.
* **Semi-Supervised Training**: The autoencoder is trained only on background data, meaning signal-like patterns remain "out-of-distribution" in the latent space—making them easier for the downstream classifier to identify.

---

## 💡 Architectural Evolution: My Journey Through Representation Learning

This project was an iterative journey through five distinct phases as I worked to overcome the unique challenges of particle physics data.

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
* **My Strategy**: I abandoned reconstruction in favor of **Explicit Density Estimation** using Masked Autoregressive Flows (MAF).
* **The Success**: By estimating $p(x)$ directly, I was able to identify the 3.5 TeV resonance as an over-density. I stabilized this by introducing **Log-Sum-Exp averaging**, which brought my results into a scientifically valid range.

#### Phase 5: Leveraging Learned Representations (Sparse AE + XGBoost)
* **My Reasoning**: While density estimation worked well, I hypothesized that the **bottleneck features** from my sparse autoencoder might contain discriminative information that pure density estimation could miss. The autoencoder learns to compress background jets into a latent manifold—signal jets, being structurally different (e.g., 2-prong vs 1-prong substructure), should occupy different regions of this space.
* **My Strategy**: I trained a 3-channel sparse autoencoder on background data, extracted bottleneck features, and used them as input to an XGBoost classifier.
* **The Success**: This approach achieved the highest AUC, validating that learned representations combined with gradient boosting can be highly effective for anomaly detection.

---

## 📈 Results & Analysis

| Method | Max AUC | Resonance Detection |
| :--- | :---: | :---: |
| **Baseline (High-Level Feature based)** | **0.79** |  |
| **Enhanced ANODE (MAF)** | **0.83** | ✅ 3.5 TeV |
| **Sparse AE + XGBoost** | **0.85** | ✅ 3.5 TeV |

---

## 💻 Installation & Usage

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- ~16 GB RAM for full sequence training

### Installation

```bash
# Clone the repository
git clone https://github.com/AtulChikku/LHC-Olympics-2020-AD.git
cd LHC-Olympics-2020-AD

# Install dependencies
pip install -r requirements.txt

# For sparse convolutions
pip install spconv-cu118  # Adjust for your CUDA version
```

### Usage

Training and evaluation scripts are available in the repository. Run notebooks for interactive exploration or use the Python scripts for batch training.

---

## 📜 References

* **ANODE Paper**: [arXiv:2001.04990](https://arxiv.org/abs/2001.04990) — *Simulation-Based Anomaly Detection*
* **LHC Olympics 2020**: [arXiv:2101.08320](https://arxiv.org/abs/2101.08320) — *The LHC Olympics Black Box Challenge*
* **Masked Autoregressive Flows**: [arXiv:1705.07057](https://arxiv.org/abs/1705.07057)
* **Submanifold Sparse Convolutions**: [arXiv:1706.01307](https://arxiv.org/abs/1706.01307)

---

<p align="center">
  <i>Developed as part of research into machine learning methods for new physics discovery at the LHC.</i>
</p>
