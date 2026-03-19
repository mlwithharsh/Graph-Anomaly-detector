# ML4SCI GSoC 2026 — Jet Classification using Deep Learning & Graph Neural Networks

## Overview

This repository presents the implementation of all tasks for the **ML4SCI GSoC 2026 selection process**, focusing on jet classification (quark vs gluon) using deep learning and graph-based methods.

The project progresses through three stages:

1. **Autoencoder-based representation learning**
2. **Graph Neural Network (GNN) classification**
3. **Contrastive Learning for improved graph representations**

The goal is to build robust, high-quality representations of jet data and evaluate their effectiveness in classification tasks.

---
<img width="1184" height="389" alt="image" src="https://github.com/user-attachments/assets/c2b05639-7e28-4cb7-b08b-bde74c36bf8a" />

## Key Contributions

* Implemented an **end-to-end ML pipeline** from raw jet images to graph-based learning
* Designed a **Graph Neural Network architecture** for jet classification
* Applied **contrastive learning** to improve representation quality
* Successfully avoided **representation collapse** (critical challenge in self-supervised learning)
* Achieved **strong classification performance** with stable training

---

## Repository Structure

```
ML4SCI-GSoC/
│
├── Task1_Autoencoder.ipynb
├── Task2_GNN.ipynb
├── Task3_Contrastive_Learning.ipynb
│
├── data/
│   └── graph_dataset.pt
│
├── results/
│   ├── roc_curve.png
│   └── loss_plots.png
│
├── requirements.txt
└── README.md
```

---

# Task 1 — Autoencoder for Jet Representation

## Objective

Learn compressed representations of jet images using an autoencoder.

## Methodology

* Input: Jet images (energy distributions)
* Model: Convolutional Autoencoder
* Loss: Reconstruction loss (MSE)
<img width="950" height="320" alt="image" src="https://github.com/user-attachments/assets/994ee744-8d72-48ba-b9b6-5bfceb7d425d" />
<img width="1153" height="593" alt="image" src="https://github.com/user-attachments/assets/59fcde97-11ea-47c8-b78c-8ff38a4ad0a6" />

## Results

* Final Loss: ~0.15
* Stable training with no divergence
* Reconstruction quality preserved key jet structures
<img width="790" height="390" alt="image" src="https://github.com/user-attachments/assets/615ec970-a7dc-4e26-9faf-ea274e15c2be" />

## Insight

The autoencoder successfully captures meaningful latent features, forming a strong foundation for downstream tasks.

---

# Task 2 — Graph Neural Network (GNN)

## Objective

Convert jet images into graphs and perform classification using GNNs.

## Pipeline

1. Convert images → point clouds
2. Construct graphs using **KNN connectivity**
3. Node features: `(x, y, energy)`
4. Train a GNN classifier

## Architecture

* GCN layers
* Non-linear activations (ReLU)
* Graph pooling
* Fully connected classifier
<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/f3991a6c-8892-4191-a5e5-2a189124e124" />


## Results

* Classification Loss: ~0.58
* ROC-AUC: ~0.74
<img width="660" height="499" alt="image" src="https://github.com/user-attachments/assets/74b1c0db-dcd3-4c12-ab1c-210b9f4444ff" />

## Insight

Graph representation captures spatial relationships better than raw pixel-based methods, improving classification performance.

---

# Task 3 — Contrastive Learning on Graphs

## Objective

Learn robust graph representations using self-supervised contrastive learning.

---

## Key Challenge

A major issue encountered:

```text
Representation Collapse (embedding std ≈ 0)
```

This was resolved through:

* Batch normalization
* Proper projection head design
* Improved pooling (Mean + Max pooling)
* Feature normalization

---

## Architecture

### 🔹 Graph Encoder

* Multiple GCN layers
* BatchNorm + ReLU
* Dropout regularization
* Mean + Max pooling

### 🔹 Projection Head

* MLP with BatchNorm
* Maps embeddings to contrastive space

---

## Augmentation Strategy

* Feature noise injection
* Random feature masking

---

## Contrastive Training Results

* Initial Loss: ~2.5
* Final Loss: ~0.02
* Embedding Std: **~1.7 (healthy, non-collapsed)**

---

## Classification Performance

### Frozen Encoder

* Loss: ~0.59

### Fine-tuned Model

* Loss: ~0.56
* ROC-AUC: **~0.7456**

---

##  ROC Curve

The ROC curve demonstrates strong discriminative performance across thresholds.

* AUC ≈ 0.75 → Good classification capability
* Significant improvement over random baseline (0.5)
<img width="536" height="547" alt="image" src="https://github.com/user-attachments/assets/2b4b8318-739a-4fb5-827b-01a93ad15e23" />

---

## Key Insights

* Contrastive learning significantly improves representation quality
* Preventing collapse is critical for meaningful embeddings
* Pooling strategy (Mean + Max) enhances feature richness
* Fine-tuning improves downstream performance

---

#  Final Summary

| Task   | Method            | Result          |
| ------ | ----------------- | --------------- |
| Task 1 | Autoencoder       | Loss ~0.15      |
| Task 2 | GNN               | ROC-AUC ~0.74   |
| Task 3 | Contrastive + GNN | ROC-AUC ~0.7456 |

---

# Technologies Used

* **PyTorch**
* **PyTorch Geometric**
* NumPy
* h5py
* Matplotlib
* scikit-learn

---

# Installation

```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install numpy matplotlib scikit-learn h5py
```

---

# How to Run

1. Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

2. Open notebooks:

```bash
jupyter notebook
```

3. Run tasks in order:

* Task 1 → Task 2 → Task 3

---

#  Learning Outcomes

This project demonstrates:

* Strong understanding of **deep learning fundamentals**
* Practical experience with **Graph Neural Networks**
* Ability to implement and debug **contrastive learning**
* Handling of **real-world ML challenges** such as:

  * CUDA issues
  * Shape mismatches
  * Representation collapse

---

# Conclusion

This work successfully builds a complete pipeline for jet classification using both supervised and self-supervised techniques.

Contrastive learning enhances representation quality and improves classification performance, demonstrating its effectiveness for graph-based physics problems.

---

# Acknowledgements

* ML4SCI GSoC 2026 Program
* PyTorch & PyTorch Geometric communities

---

# Contact

For any queries or discussions, feel free to connect.

---

⭐ If you found this project useful, consider giving it a star!
