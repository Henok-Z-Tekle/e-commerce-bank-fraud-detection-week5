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
