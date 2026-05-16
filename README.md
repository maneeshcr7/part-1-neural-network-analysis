# Part 1: Neural Network Fundamentals and Training Behavior Analysis

## Overview
This repository contains the implementation for Part 1 of the Applied Neural Networks, CNN, NLP, and AI Solution Design project. The goal is to build and analyse a feed-forward neural network to predict customer churn, while clearly demonstrating how neural networks learn through forward pass, loss calculation, backpropagation, and parameter updates.

## Dataset
**Source:** [Synthetic AI Project Datasets – Google Drive](https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs?usp=sharing)  
**File used:** `part_1_neural_network_analysis/customer_churn_nn.csv`  
**Rows:** 2,000 | **Features:** 15 | **Target:** `churn` (binary: 0 = retained, 1 = churned)

> ⚠️ Dataset files are **not uploaded** to this repository per submission guidelines. Download from the link above and place `customer_churn_nn.csv` in the project root before running.

## Repository Structure
```
part-1-neural-network-analysis/
├── README.md
├── notebook.ipynb
├── requirements.txt
└── results/
    ├── model_comparison_table.csv
    ├── model_comparison_table.png
    ├── part1_training_curves.png
    ├── part1_confusion_matrix.png
    └── part1_eda.png
```

## Approach

### Task 1 – Dataset Understanding
- 2,000 rows, 16 columns (1 identifier, 4 categorical, 11 numerical, 1 target)
- Target variable: `churn` — highly imbalanced (~1.55% churn rate)
- No missing values detected
- Key features: `tenure_months`, `monthly_charges_inr`, `satisfaction_score`, `support_tickets_last_90_days`

### Task 2 – Data Preprocessing
- Dropped `customer_id` (non-predictive identifier)
- Label-encoded 4 categorical columns: `region`, `plan_type`, `contract_type`, `payment_method`
- Scaled all numerical features using `StandardScaler`
- Stratified 80/20 train-test split to preserve class ratio
- Applied class weighting (`class_weight`) to handle imbalance

### Task 3 – Neural Network Model
Built using **TensorFlow / Keras**:
- **Input layer:** 15 features
- **Hidden layers:** Dense(64) → Dropout(0.3) → Dense(32) → Dropout(0.3)
- **Activation:** ReLU (hidden), Sigmoid (output)
- **Loss function:** Binary Cross-Entropy
- **Optimizer:** Adam (lr=0.001)

### Task 4 – Training and Evaluation
- Trained for 50 epochs, batch size 32
- Evaluated using Accuracy, F1-Score, and Confusion Matrix
- F1-Score used as primary metric due to class imbalance

### Task 5 – Hyperparameter Experiments

| Experiment | Architecture | LR | Activation | Test Accuracy | F1 Score |
|---|---|---|---|---|---|
| Baseline | [64, 32] | 0.001 | ReLU | ~0.9425 | ~0.26 |
| Deeper Network | [128, 64, 32] | 0.001 | ReLU | ~0.9425 | ~0.26 |
| High Learning Rate | [64, 32] | 0.01 | ReLU | ~0.9425 | ~0.09 |
| tanh Activation | [64, 32] | 0.001 | tanh | ~0.9425 | ~0.25 |

### Task 6 – Key Reflections
- **Weights & Biases:** Weights scale the influence of each input; biases shift activations allowing non-zero-centred fits. Both are updated via backpropagation.
- **Activation functions:** Without non-linearity, stacked layers collapse into a single linear transformation — the network cannot learn complex boundaries.
- **Learning rate:** Too high → overshooting minima, unstable loss. Too low → extremely slow convergence, risk of local minima.
- **Overfitting/Underfitting:** High test accuracy (~94%) is misleading; low F1 (~0.26) reveals that the model largely predicts the majority class. The dataset's severe imbalance is the primary bottleneck, not model capacity.

## Results
| Metric | Value |
|---|---|
| Test Accuracy | ~94.25% |
| Best F1 Score | ~0.26 (Baseline) |
| Training epochs | 50 |
| Class imbalance ratio | ~64:1 (retained:churned) |

## Setup and Installation

```bash
git clone <repo-url>
cd part-1-neural-network-analysis
pip install -r requirements.txt
# Place customer_churn_nn.csv in the project root
jupyter notebook notebook.ipynb
```

## Requirements
```
tensorflow>=2.10
scikit-learn>=1.2
pandas>=1.5
numpy>=1.23
matplotlib>=3.6
seaborn>=0.12
```

## Observations
The severe class imbalance (98.45% retained, 1.55% churned) dominates model behaviour. Accuracy is not a reliable metric here — F1 Score is the honest evaluator. Future improvements would include SMOTE oversampling, threshold tuning, or a specialised imbalanced learning library.
