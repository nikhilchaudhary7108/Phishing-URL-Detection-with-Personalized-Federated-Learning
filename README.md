
## 👥 Team Members
- **Ashutosh Ranjan**  
- **Nikhil Chaudhary**  
- **Pankaj Kumar**  
- **Nikhil Kumar**  
- **Sapavath Hanumanth**
# Phishing & Malicious URL Detection using Deep Learning

## 📌 Project Overview

This project presents an **end-to-end centralized and federated learning framework for phishing and malicious URL detection**, designed to operate across **heterogeneous, decentralized datasets** while minimizing per-device computational cost.

The system is built around **three tightly integrated pipelines**:

* **Unified Data & Feature Pipeline**: Multiple phishing and malicious URL datasets are preprocessed independently at different clients using a common character-level (ASCII/byte) encoding. This ensures schema consistency while preserving non-IID data distributions through local train/validation/test splits.

* **Hybrid Deep URL Representation Pipeline**: URLs are modeled using a lightweight hybrid architecture combining character embeddings, residual multi-kernel CNNs, BiLSTM layers, and dual-attention pooling to capture both local patterns and long-range dependencies with high efficiency.

* **Federated & Personalized Training Pipeline**: A federated meta-learning strategy is employed where clients train locally on private data, only shared parameters are aggregated on the server, and personalization layers remain client-specific. Soft updates and quick fine-tuning enable stable coordination across non-IID clients without raw data sharing.

---

## ✨ Research Novelty & Key Contributions

1. **Federated Multi-Dataset URL Learning Framework**

   * Proposes a unified framework that integrates centralized and federated learning for phishing URL detection across multiple heterogeneous public datasets.

2. **Personalized Federated Training for Non-IID Data**

   * Introduces a separation of shared (global) and client-specific (personalized) parameters, enabling effective learning under non-IID data distributions while reducing per-device computational load.

3. **Hybrid and Computation-Efficient URL Encoder**

   * Designs a lightweight yet expressive architecture combining residual multi-kernel CNNs, BiLSTM, and attention mechanisms to capture both structural and sequential URL patterns.

4. **Privacy-Preserving Cross-Dataset Coordination**

   * Enables collaboration across datasets and clients without raw data sharing, improving robustness and scalability in decentralized environments.

---

## 📊 Datasets Used

| Dataset ID | Source                  | Description                      |
| ---------- | ----------------------- | -------------------------------- |
| Dataset 1  | Kaggle (Malicious URLs) | Mixed malicious and benign URLs  |
| Dataset 2  | Kaggle (PhiUSIIL)       | Phishing-focused URLs            |
| Dataset 3  | HuggingFace (kmack)     | Large-scale phishing URL corpus  |
| Dataset 4  | Kaggle (Tarun Tiwari)   | Real-world phishing website URLs |

All datasets are standardized to:

```text
url   → string
label → 1 (phishing/malicious), 0 (benign)
```

---

## 🧪 Training Pipeline (Stage-wise)

### **Stage 1: Data Preprocessing** (`01_dataset_preprocessing_pipeline.py`)

* Dataset download (Kaggle + HuggingFace)
* Column normalization
* Label filtering & encoding
* Duplicate URL removal
* Stratified splitting (Train / Validation / Test)

📊 **Result Visualization**:

**Figure:** Class distribution and train/validation/test split consistency after preprocessing.

---

### **Stage 2: Exploratory Data Analysis** (`02_exploratory_dataset_analysis.ipynb`)

* Class imbalance analysis
* URL length distribution
* Phishing vs benign structural patterns

📊 **Result Visualization**:

**Figure:** Exploratory analysis highlighting structural differences between phishing and benign URLs.

---

### **Stage 3: Centralized & Client-Specific Model Training** (`03_multi_dataset_training_experiments.ipynb`)

* Each dataset is trained independently to analyze dataset-specific learning behavior and bias.
* A **hybrid URL encoder** (Embedding → Residual Multi-Kernel CNN → BiLSTM → Dual-Attention Pooling) is optimized end-to-end.
* Model parameters are explicitly divided into shared (global) and personalized (local) layers.
* Separate optimizers and schedulers are applied to improve stability and convergence.
* Threshold tuning is performed on the validation set to optimize F1-score.

<h3 align="center">📊 Result Visualization</h3>




**Figure:** Training and validation accuracy/loss curves demonstrating stable convergence across datasets.

---

### **Stage 4: Cross-Dataset Evaluation & Comparison** (`04_model_performance_comparison.ipynb`)

* Compare models trained on different datasets
* Evaluate generalization performance under dataset shift

📊 **Result Visualization**:

**Figure:** Cross-dataset evaluation highlighting generalization performance and dataset bias.

---

## 📈 Experimental Results

### **TABLE III. Cross-Dataset Validation Accuracy**

*Each cell reports ****accuracy**** when a model trained on the row dataset is evaluated on the column dataset.*

| **Train \ Test**                | **Dataset 1****(Malicious URLs)** | **Dataset 2****(PhiUSIIL)** | **Dataset 3****(KMack)** | **Dataset 4****(Kaggle Phishing)** |
| ------------------------------- | --------------------------------- | --------------------------- | ------------------------ | ---------------------------------- |
| **Dataset 1 (Malicious URLs)**  | 0.9846                            | 0.5736                      | 0.5195                   | 0.6960                             |
| **Dataset 2 (PhiUSIIL)**        | 0.8518                            | 0.9980                      | 0.4221                   | 0.7761                             |
| **Dataset 3 (KMack)**           | 0.8638                            | 0.1333                      | 0.9128                   | 0.8857                             |
| **Dataset 4 (Kaggle Phishing)** | 0.7849                            | 0.5918                      | 0.7427                   | 0.9816                             |

---

### **TABLE I. Performance Comparison of the Proposed Model Across Four Phishing URL Datasets**

| Dataset                              | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| ------------------------------------ | -------- | --------- | ------ | -------- | ------- |
| Dataset 1: Malicious URLs            | 0.9848   | 0.9824    | 0.9322 | 0.9566   | 0.9969  |
| Dataset 2: PhiUSIIL Phishing         | 0.9978   | 0.9967    | 0.9994 | 0.9981   | 0.9986  |
| Dataset 3: KMack Phishing URLs       | 0.9170   | 0.9094    | 0.9262 | 0.9177   | 0.9739  |
| Dataset 4: Kaggle Phishing Site URLs | 0.9817   | 0.9768    | 0.9410 | 0.9586   | 0.9966  |

---

<h3 align="center">📊 Validation Accuracy Comparison (Per Dataset)</h3>

<table align="center">
  <tr>
    <td align="center">
      <img src="comparison_graphs/val_acc_d1.png" width="320"><br>
      <i>(a) Dataset-1: Malicious URLs</i>
    </td>
    <td align="center">
      <img src="comparison_graphs/val_acc_d2.png" width="320"><br>
      <i>(b) Dataset-2: PhishTank</i>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="comparison_graphs/val_acc_d3.png" width="320"><br>
      <i>(c) Dataset-3: OpenPhish</i>
    </td>
    <td align="center">
      <img src="comparison_graphs/val_acc_d4.png" width="320"><br>
      <i>(d) Dataset-4: Combined Dataset</i>
    </td>
  </tr>
</table>

<p align="center">
  <b>Fig.</b> Validation accuracy versus epochs for different ablation models evaluated across four phishing URL datasets.
</p>

<h3 align="center">📉 Validation Loss Comparison (Per Dataset)</h3>

<table align="center">
  <tr>
    <td align="center">
      <img src="comparison_graphs/val_loss_d1.png" width="320"><br>
      <i>(a) Dataset-1: Malicious URLs</i>
    </td>
    <td align="center">
      <img src="comparison_graphs/val_loss_d2.png" width="320"><br>
      <i>(b) Dataset-2: PhishTank</i>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="comparison_graphs/val_loss_d3.png" width="320"><br>
      <i>(c) Dataset-3: OpenPhish</i>
    </td>
    <td align="center">
      <img src="comparison_graphs/val_loss_d4.png" width="320"><br>
      <i>(d) Dataset-4: Combined Dataset</i>
    </td>
  </tr>
</table>

<p align="center">
  <b>Fig.</b> Validation loss versus epochs for different ablation models evaluated across four phishing URL datasets.
</p>





### **TABLE II. Performance Comparison of Ablation Models on Phishing URL Detection**

| Model                          | Accuracy   | Precision  | Recall     | F1-score   | ROC-AUC    | R²     |
| ------------------------------ | ---------- | ---------- | ---------- | ---------- | ---------- | ------ |
| MLP on URL Embeddings          | 0.9518     | 0.8974     | 0.8878     | 0.8926     | 0.9781     | 0.7491 |
| Residual Multi-Kernel CNN      | 0.9778     | 0.9742     | 0.9262     | 0.9496     | 0.9945     | 0.8947 |
| Temporal BiLSTM                | 0.9796     | 0.9750     | 0.9332     | 0.9536     | 0.9956     | 0.9068 |
| CNN + BiLSTM Hybrid            | 0.9797     | 0.9786     | 0.9300     | 0.9537     | 0.9959     | 0.9083 |
| SE + Attention Pooling         | 0.9795     | 0.9671     | 0.9412     | 0.9540     | 0.9957     | 0.9074 |
| DAP + CNN + BiLSTM             | 0.9821     | 0.9743     | 0.9456     | 0.9597     | 0.9966     | 0.9208 |
| transformer Based              | 0.9793     | 0.9660     | 0.9413     | 0.9535     | 0.9957     | 0.9072 |
| **Proposed Best Hybrid Model** | **0.9829** | **0.9717** | **0.9518** | **0.9616** | **0.9970** | —      |
---

<p align="center">
  <img src="comparison_graphs/val_loss_vs_epochs.png" width="32%">
  <img src="comparison_graphs/val_loss_vs_time.png" width="32%">
  <img src="comparison_graphs/val_accuracy_vs_epochs.png" width="32%">
</p>

<p align="center">
  <b>Fig.</b> Validation loss versus epochs (left), validation loss versus training time (middle), and validation accuracy versus epochs (right) for different ablation models on the phishing URL detection task.
</p>


## 🧠 Key Observations

* Validation accuracy plateaus earlier for smaller datasets
* Dataset bias significantly impacts cross-dataset testing
* CNN-based URL models benefit from diverse training data

---

## 🔮 Future Work

* Transformer-based URL encoders
* Federated learning across datasets
* Adversarial URL robustness testing
* Real-time deployment pipeline

---

## 📜 Citation (Suggested)

If you use this work in your research, please cite:

> *Multi-Dataset Deep Learning Framework for Phishing URL Detection*, 2026

---

## 👤 Author

**Ashutosh Ranjan**
Faculty of Technology, University of Delhi

---

✅ *This repository is designed to be reproducible, extensible, and research-review ready.*
