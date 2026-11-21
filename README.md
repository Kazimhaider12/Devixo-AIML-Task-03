# Customer Churn Prediction — End-to-End Machine Learning Pipeline

**Devixo Solutions Summer Internship — AI/ML Track (Week 3: Task 03)**  
**Author:** Kazim Haider Syed  

## Project Overview
An industry-oriented, end-to-end Machine Learning solution predicting telecom customer churn using a reusable Scikit-learn `Pipeline`. Covers raw data cleaning, domain feature engineering, automated feature selection, scaling/encoding, 5-model benchmarking with 5-fold Stratified Cross-Validation, `GridSearchCV` hyperparameter optimization, feature importance analysis, dual serialization (`Joblib` + `Pickle`), and an interactive **Streamlit** web application.

## Repository Structure
```text
devixo-week3-customer-churn-ml/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # 7,043+ rows (IBM/Kaggle Telco Churn schema)
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py                  # Part 1: Cleaning, Feature Engineering & ColumnTransformer
│   └── train.py                               # Parts 2-4: Training, 5-Fold CV, Tuning, Plots & Serialization
├── models/
│   ├── best_model.joblib                      # Serialized best pipeline (Joblib)
│   └── best_model.pkl                         # Serialized best pipeline (Pickle)
├── outputs/
│   ├── metrics_summary.json                   # Complete experiment metrics
│   └── plots/
│       ├── model_comparison.png
│       ├── roc_curves.png
│       ├── confusion_matrix.png
│       └── feature_importance.png
├── notebooks/
│   └── week3_end_to_end_ml.ipynb              # Interactive end-to-end walkthrough
├── app.py                                     # Bonus: Streamlit real-time prediction app
├── requirements.txt
└── README.md
```

## Key Results (5-Fold Stratified CV & 20% Holdout Test Set)

| Model | CV ROC-AUC | Test ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting (Tuned)** | **0.8562** | **0.8514** | **0.8197** | **0.6864** | 0.4737 | 0.5606 |
| Logistic Regression | 0.8634 | 0.8573 | 0.7615 | 0.5057 | 0.7719 | 0.6111 |
| Random Forest | 0.8477 | 0.8380 | 0.7729 | 0.5247 | 0.6842 | 0.5939 |
| Gradient Boosting (Base) | 0.8461 | 0.8425 | 0.8126 | 0.6403 | 0.5205 | 0.5742 |
| Support Vector Machine (RBF) | 0.8438 | 0.8336 | 0.7431 | 0.4824 | 0.8012 | 0.6022 |
| Decision Tree | 0.8303 | 0.8280 | 0.7395 | 0.4772 | 0.7661 | 0.5881 |

> **Dataset Note:** The included CSV mirrors the exact 7,043-row Kaggle IBM Telco Customer Churn schema (with realistic missing values and duplicate rows for pipeline verification). To run on the official Kaggle download, drop `WA_Fn-UseC_-Telco-Customer-Churn.csv` into `data/` and run `python src/train.py`.

## Quickstart
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the complete end-to-end training, tuning, evaluation & model export pipeline
python src/train.py

# 3. Launch the Bonus Streamlit Prediction Interface
streamlit run app.py
```
