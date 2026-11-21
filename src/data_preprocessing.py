"""
Part 1: Data Preparation, Feature Engineering, Feature Selection & Scaling.
Reusable preprocessing module for both training (src/train.py) and inference (app.py).
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
]


def load_and_clean_data(csv_path: str) -> tuple[pd.DataFrame, dict]:
    """Load raw CSV, fix data types, handle missing values, and remove duplicates."""
    df = pd.read_csv(csv_path)
    stats = {"raw_rows": int(len(df)), "raw_cols": int(df.shape[1])}

    # Remove exact duplicate rows
    dup_count = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    stats["duplicates_removed"] = dup_count

    # Convert TotalCharges from string/whitespace to numeric
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    missing_before = {k: int(v) for k, v in df.isnull().sum().items() if v > 0}
    stats["missing_before_imputation"] = missing_before

    # Impute missing numeric values using median
    for col in ["TotalCharges", "MonthlyCharges"]:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    stats["clean_rows"] = int(len(df))
    stats["missing_after_imputation"] = int(df.isnull().sum().sum())
    return df, stats


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain-specific customer churn features."""
    df = df.copy()

    # 1. Average monthly spend over customer lifetime
    df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1.0)

    # 2. Current monthly charge relative to historical average
    df["ChargeToAvgRatio"] = df["MonthlyCharges"] / (df["AvgMonthlySpend"] + 1e-5)

    # 3. Total number of subscribed add-on services
    df["TotalServices"] = sum((df[col] == "Yes").astype(int) for col in SERVICE_COLS if col in df.columns)

    # 4. Security/Support protection flag
    df["HasProtection"] = (
        (df["OnlineSecurity"] == "Yes") | (df["TechSupport"] == "Yes")
    ).astype(int)

    # 5. Automatic payment flag
    df["IsAutoPay"] = df["PaymentMethod"].str.contains("automatic", case=False, na=False).astype(int)

    # 6. Contract risk flag (Month-to-month is highest churn risk)
    df["IsMonthToMonth"] = (df["Contract"] == "Month-to-month").astype(int)

    # 7. Tenure cohort buckets
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 100],
        labels=["0-12m", "13-24m", "25-48m", "49m+"]
    ).astype(str)

    # Drop non-predictive identifier
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    return df


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Build reusable Scikit-learn ColumnTransformer for scaling and encoding."""
    numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor, numeric_features, categorical_features
