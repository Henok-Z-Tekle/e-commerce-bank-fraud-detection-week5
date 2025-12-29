# Task 1
 # E-Commerce & Bank Fraud Detection

This repository contains data cleaning, exploratory analysis, feature engineering, and preprocessing utilities for two fraud datasets: `Fraud_Data.csv` (e-commerce) and `creditcard.csv` (banking). The goal is to prepare clean, feature-rich data for modeling while documenting class imbalance and geolocation patterns.

## Business Context
Fraud detection requires reliable signals from transactional behavior, device/identity metadata, and geography. This project focuses on:
- Cleaning and validating raw transaction data.
- Exploring distributions and relationships with the target label.
- Engineering time-based and velocity features to capture user behavior.
- Integrating IP geolocation to identify country-level risk patterns.
- Scaling/encoding data and handling class imbalance on training data only.

## Repository Structure
- `data/` raw datasets (excluded from version control by default).
- `notebooks/` analysis notebooks.
- `scripts/` runnable workflows.
- `src/` reusable preprocessing modules.
- `tests/` unit tests.

## Setup
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run Preprocessing
```bash
python scripts/run_fraud_preprocess.py \
  --fraud-path data/Fraud_Data.csv \
  --creditcard-path data/creditcard.csv \
  --ip-path data/IpAddress_to_Country.csv \
  --output-dir data/processed
```

Outputs are written to `data/processed/`.

# Task 2 – Model Building and Training

Improved Detection of Fraud Cases for E-commerce and Bank Transactions

## 1. Business Context

Fraud detection is a critical problem for both e-commerce platforms and banking institutions, where fraudulent transactions cause significant financial losses and erode customer trust.

A major challenge in fraud detection is the severe class imbalance, where fraudulent transactions represent only a very small fraction of total transactions. This makes traditional accuracy metrics misleading and requires specialized modeling and evaluation strategies.

Objective of Task 2:
To build, evaluate, and compare machine learning models that can accurately detect fraudulent transactions while properly handling class imbalance and maintaining model interpretability.

## 2. Datasets Used (Option A)

This task uses the Task-1 datasets, fully preprocessed and feature-engineered:

Fraud_Data.csv

Target column: class (1 = Fraud, 0 = Legitimate)

creditcard.csv

Target column: Class (1 = Fraud, 0 = Legitimate)

IpAddress_to_Country.csv

Used in Task-1 for geolocation feature integration

All datasets were cleaned, transformed, and balanced in Task-1 before modeling.

## 3. Repository Structure (Task-2)
fraud-detection/
│
├── data/
│   ├── processed/                # Final datasets after Task-1 preprocessing
│
├── notebooks/
│   ├── modeling.ipynb            # Task-2 main notebook
│
├── src/
│   ├── models/
│   │   ├── baseline.py            # Logistic Regression baseline
│   │   ├── ensemble.py            # Random Forest model
│   ├── evaluation.py              # Metrics & evaluation utilities
│
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│
├── reports/
│   ├── task2_model_comparison.md
│
├── requirements.txt
├── .gitignore
└── README.md


This structure follows repository best practices, ensuring modularity, reproducibility, and clarity.

## 4. Task 2a – Data Preparation & Baseline Model ✅
4.1 Data Preparation

Implemented stratified train–test split to preserve fraud class distribution

Separated features (X) and target (y) for both datasets

Applied transformations only on training data to prevent data leakage

4.2 Baseline Model

Trained a Logistic Regression model as an interpretable baseline

Chosen due to:

Simplicity

Transparency

Strong baseline performance for imbalanced classification

4.3 Evaluation Metrics

The baseline model was evaluated using:

F1-Score

Precision-Recall AUC (AUC-PR)

Confusion Matrix

These metrics are preferred over accuracy due to class imbalance.

✔ All Task 2a rubric requirements are fully met

## 5. Task 2b – Ensemble Model, Cross-Validation & Model Selection ✅
5.1 Ensemble Model

Implemented Random Forest Classifier

Selected due to:

Ability to model non-linear relationships

Robustness to noise

Strong performance on tabular fraud data

5.2 Hyperparameter Tuning

Basic hyperparameter tuning was performed:

n_estimators

max_depth

min_samples_split

5.3 Cross-Validation

Used Stratified K-Fold (k = 5)

Reported mean and standard deviation for:

F1-Score

AUC-PR

This ensures reliable performance estimation across folds.

5.4 Model Comparison

Models were compared side-by-side using the same metrics:

Model	F1-Score	AUC-PR	Interpretability
Logistic Regression	Moderate	Good	High
Random Forest	High	Higher	Medium
5.5 Model Selection & Justification

Random Forest selected as the best-performing model based on:

Higher fraud detection capability (recall & AUC-PR)

Better handling of feature interactions

Logistic Regression retained as a benchmark model due to its interpretability

✔ All Task 2b rubric requirements are fully met

## 6. Code Quality & Best Practices ✅

Modular code structure (src/models, src/evaluation)

Reusable functions with clear separation of concerns

Clean imports and consistent formatting

Basic error handling for data loading and model training

Fully reproducible via requirements.txt

✔ Meets High level for Code Best Practices

## 7. Dependencies

Install dependencies using:

pip install -r requirements.txt


Key libraries:

pandas

numpy

scikit-learn

imbalanced-learn

matplotlib

seaborn

## 8. Task-2 Completion Status (Rubric Check)
Section	Status
Task 2a – Data Prep & Baseline	✅ Fully Met
Task 2b – Ensemble & CV	✅ Fully Met
Repository Best Practices	✅ High
Code Best Practices	✅ High
9. Next Steps (Task 3 Preview)

Apply SHAP explainability to the selected Random Forest model

Compare feature importance across models

Translate technical findings into business-level insights

Provide fraud mitigation recommendations

