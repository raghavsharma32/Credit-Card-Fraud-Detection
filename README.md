# 💳 Credit Card Fraud Detection using XGBoost

This project implements a machine learning model to detect fraudulent credit card transactions using the [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). It uses the low-level XGBoost API with DMatrix objects for high-performance training and evaluation.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Modeling Approach](#modeling-approach)
- [Results](#results)
- [Installation & Usage](#installation--usage)
- [Project Structure](#project-structure)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## 📖 Overview

Credit card fraud is a growing issue worldwide. This project builds a binary classification model to predict fraudulent transactions using real, anonymized transaction data. Key challenges include extreme class imbalance and the need for high recall to minimize false negatives.

---

## 📊 Dataset

- 📁 Source: [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 🧾 Records: 284,807 transactions
- 🎯 Positive class (fraud): 492 (~0.172%)
- 🔒 Features are anonymized due to confidentiality (`V1`, `V2`, ..., `V28`, plus `Time` and `Amount`)

---

## 🧠 Modeling Approach

- ✅ Data split into `train`, `validation`, and `test` sets (70/15/15)
- ✅ Feature scaling (if applicable) on `Amount` and `Time`
- ✅ Model: XGBoost 
- ✅ Objective: `binary:logistic`
- ✅ Evaluation Metric: `AUC` (Area Under ROC Curve)
- ✅ Early stopping using validation AUC

### Hyperparameters:
```python
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'eta': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}
# Credit-Card-Fraud-Detection
# Credit-Card-Fraud-Detection
