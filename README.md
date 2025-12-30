## Task 3 – Model Explainability (SHAP Analysis)
Objective

The goal of Task 3 is to interpret and explain the predictions of the final fraud detection model using SHAP (SHapley Additive exPlanations). This task focuses on understanding why the model flags transactions as fraudulent and translating those insights into actionable business recommendations.

Explainability is critical in fintech systems to:

Build trust with stakeholders

Support regulatory compliance

Balance fraud prevention with customer experience

Contents
task-3/
│
├── shap_analysis.ipynb
├── feature_importance.ipynb
├── images/
│   ├── shap_summary.png
│   ├── shap_tp.png
│   ├── shap_fp.png
│   └── shap_fn.png
└── README.md

Steps Completed
1. Feature Importance (Baseline)

Extracted built-in feature importance from the final ensemble model

Visualized the top 10 most influential features

Used as a baseline comparison against SHAP values

2. SHAP Analysis
Global Explainability

Generated a SHAP Summary Plot

Identified globally important fraud drivers

Observed how feature values influence fraud likelihood

Local Explainability

Generated SHAP Force Plots for:

True Positive: Correctly identified fraud

False Positive: Legitimate transaction flagged as fraud

False Negative: Fraud transaction missed by the model

These plots explain individual predictions at transaction level.

3. Interpretation

Compared SHAP importance with built-in feature importance

Identified the top 5 fraud drivers

Analyzed unexpected or counterintuitive patterns

4. Business Recommendations

Based on SHAP insights, the following recommendations were derived:

Enhanced verification for high-risk behavioral patterns

Time-based fraud controls

Country-aware risk scoring strategies

Each recommendation is directly linked to SHAP findings.

Tools & Libraries

Python

SHAP

Scikit-learn

XGBoost / Random Forest

Matplotlib / Seaborn

Outcome

This task ensures the fraud detection system is transparent, explainable, and business-aligned, enabling informed decision-making and responsible AI deployment.
